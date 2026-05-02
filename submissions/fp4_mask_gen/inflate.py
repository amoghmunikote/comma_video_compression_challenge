#!/usr/bin/env python
"""
fp4_mask_gen/inflate.py

Format on disk (all files inside archive/):
  model.pt.br  FP4-quantized + brotli'd state dict for the generator.
  mask.obu.br  Lossless AV1 OBU of MASK_STORE_W×MASK_STORE_H mask keyframes
               (1 per K_TEMPORAL pairs, 5-class gray).
  pose.bin.br  Brotli'd binary: 12 fp32 (per-dim mn, mx) + N_PAIRS*6 uint16.
"""
import io, os, sys, tempfile
from pathlib import Path

import av
import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

NET_W, NET_H     = 512, 384
OUT_W, OUT_H     = 1164, 874
N_PAIRS_PER_FILE = 600
COND_DIM         = 48
POSE_DIM         = 6
K_TEMPORAL       = 1
MASK_STORE_H     = 192
MASK_STORE_W     = 256


# ─── FP4 dequantization ───────────────────────────────────────────────────────
class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    @staticmethod
    def dequantize_from_nibbles(nibbles, scales, orig_shape):
        flat_n     = int(torch.tensor(orig_shape).prod().item())
        block_size = nibbles.numel() // scales.numel()
        nibbles    = nibbles.view(-1, block_size)
        signs      = (nibbles >> 3).to(torch.int64)
        mag_idx    = (nibbles & 0x7).to(torch.int64)
        levels     = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q          = torch.where(signs.bool(), -levels[mag_idx], levels[mag_idx])
        return (q * scales[:, None].float()).view(-1)[:flat_n].reshape(orig_shape)

def unpack_nibbles(packed, count):
    flat = packed.reshape(-1)
    out  = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = (flat >> 4) & 0x0F
    out[1::2] = flat & 0x0F
    return out[:count]

def get_decoded_state_dict(payload_data, device):
    data = torch.load(io.BytesIO(payload_data), map_location=device)
    sd   = {}
    for name, rec in data["quantized"].items():
        if rec["weight_kind"] == "fp4_packed":
            nibs = unpack_nibbles(rec["packed_weight"].to(device), rec["packed_weight"].numel() * 2)
            w    = FP4Codebook.dequantize_from_nibbles(nibs, rec["scales_fp16"].to(device), rec["weight_shape"])
        else:
            w = rec["weight_fp16"].to(device).float()
        sd[f"{name}.weight"] = w.float()
        if rec.get("bias_fp16") is not None:
            sd[f"{name}.bias"] = rec["bias_fp16"].to(device).float()
    for name, tensor in data["dense_fp16"].items():
        sd[name] = tensor.to(device).float() if torch.is_floating_point(tensor) else tensor.to(device)
    return sd


# ─── Pose codec (qpose14 fixed-scale) ─────────────────────────────────────────
def decode_pose_bin(payload: bytes) -> torch.Tensor:
    q = np.frombuffer(payload, dtype=np.uint16).reshape(-1, POSE_DIM).copy()
    out = np.empty(q.shape, dtype=np.float32)
    out[:, 0]  = q[:, 0].astype(np.float32) / 512.0 + 20.0
    out[:, 1:] = q[:, 1:].view(np.int16).astype(np.float32) / 2048.0
    return torch.from_numpy(out).float().contiguous()


# ─── Architecture (inference-only) — must match compress.py exactly ──────────
class QConv2d(nn.Conv2d):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)

class QEmbedding(nn.Embedding):
    def __init__(self, *args, block_size=32, quantize_weight=True, **kwargs):
        super().__init__(*args, **kwargs)

class SepConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid = in_ch * depth_mult
        self.dw   = QConv2d(in_ch, mid, k, stride=stride, padding=k//2, groups=in_ch, bias=False)
        self.pw   = QConv2d(mid, out_ch, 1, bias=True)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.norm(self.pw(self.dw(x))))

class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid, k, stride=stride, padding=k//2, groups=in_ch, bias=False)
        self.pw = QConv2d(mid, out_ch, 1, bias=True)
    def forward(self, x): return self.pw(self.dw(x))

class SepResBlock(nn.Module):
    def __init__(self, ch, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, depth_mult=depth_mult)
        self.conv2 = SepConv(ch, ch, depth_mult=depth_mult)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act   = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(x + self.norm2(self.conv2(self.conv1(x))))

class FiLMSepResBlock(nn.Module):
    def __init__(self, ch, cond_dim, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1     = SepConvGNAct(ch, ch, depth_mult=depth_mult)
        self.conv2     = SepConv(ch, ch, depth_mult=depth_mult)
        self.norm2     = nn.GroupNorm(2, ch)
        self.film_proj = nn.Linear(cond_dim, ch * 2)
        self.act       = nn.SiLU(inplace=True)
    def forward(self, x, cond):
        xb = self.norm2(self.conv2(self.conv1(x)))
        gamma, beta = self.film_proj(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return self.act(x + xb * (1.0 + gamma) + beta)

class SharedMaskDecoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=56, c2=64, depth_mult=1):
        super().__init__()
        self.embedding  = QEmbedding(num_classes, emb_dim)
        self.stem_conv  = SepConvGNAct(emb_dim + 2, c1, depth_mult=depth_mult)
        self.stem_block = SepResBlock(c1, depth_mult=depth_mult)
        self.down_conv  = SepConvGNAct(c1, c2, stride=2, depth_mult=depth_mult)
        self.down_block = SepResBlock(c2, depth_mult=depth_mult)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            SepConvGNAct(c2, c1, depth_mult=depth_mult),
        )
        self.fuse       = SepConvGNAct(c1 + c1, c1, depth_mult=depth_mult)
        self.fuse_block = SepResBlock(c1, depth_mult=depth_mult)

    def forward(self, mask2, coords):
        e2     = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up  = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)
        s      = self.stem_block(self.stem_conv(torch.cat([e2_up, coords], dim=1)))
        z      = self.up(self.down_block(self.down_conv(s)))
        return self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))

class FrameHead(nn.Module):
    def __init__(self, in_ch, cond_dim=COND_DIM, hidden=52, depth_mult=1):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre    = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head   = QConv2d(hidden, 3, 1)
    def forward(self, feat, cond):
        return torch.sigmoid(self.head(self.pre(self.block2(self.block1(feat, cond))))) * 255.0

class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=POSE_DIM, cond_dim=COND_DIM, depth_mult=1):
        super().__init__()
        self.shared_trunk = SharedMaskDecoder(num_classes, emb_dim=6, c1=56, c2=64, depth_mult=depth_mult)
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.frame1_head  = FrameHead(in_ch=56, cond_dim=cond_dim, hidden=52, depth_mult=depth_mult)
        self.frame2_head  = FrameHead(in_ch=56, cond_dim=cond_dim, hidden=52, depth_mult=depth_mult)

    def forward(self, mask2, pose6, pair_idx=None):
        coords    = make_coord_grid(mask2.shape[0], NET_H, NET_W, mask2.device, torch.float32)
        feat      = self.shared_trunk(mask2, coords)
        pose_cond = self.pose_mlp(pose6)
        return self.frame1_head(feat, pose_cond), self.frame2_head(feat, pose_cond)

def make_coord_grid(batch, height, width, device, dtype):
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width,  device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx*2-1, yy*2-1], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


def load_mask_video(path):
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        frames.append(np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4))
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()


def main():
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
        sys.exit(1)

    data_dir       = Path(sys.argv[1])
    out_dir        = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files  = [l.strip() for l in file_list_path.read_text().splitlines() if l.strip()]

    # Model is always in the archive; mask+pose may be embedded in this file.
    model_br = (data_dir / "model.pt.br").read_bytes()
    try:
        mask_br = _MASK_BR  # noqa: F821  – injected by compress.py embed step
        pose_br = _POSE_BR  # noqa: F821
    except NameError:
        # Fallback for standalone / test runs: read from data_dir
        mask_br = (data_dir / "mask.obu.br").read_bytes()
        pose_br = (data_dir / "pose.bin.br").read_bytes()

    weights_data = brotli.decompress(model_br)
    generator = JointFrameGenerator().to(device)
    generator.load_state_dict(get_decoded_state_dict(weights_data, device), strict=True)
    generator.eval()

    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(brotli.decompress(mask_br))
        tmp_path = tmp.name
    kf_masks_all = load_mask_video(tmp_path)  # [N_KF_TOTAL, MASK_STORE_H, MASK_STORE_W]
    os.remove(tmp_path)

    pose_frames_all = decode_pose_bin(brotli.decompress(pose_br))  # [N_PAIRS_TOTAL, 6]

    batch_size    = 4
    kf_cursor     = 0
    pose_cursor   = 0
    n_kf_per_file = N_PAIRS_PER_FILE // K_TEMPORAL

    with torch.inference_mode():
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            raw_out   = out_dir / f"{base_name}.raw"

            file_kf    = kf_masks_all[kf_cursor : kf_cursor + n_kf_per_file]
            file_masks = file_kf.repeat_interleave(K_TEMPORAL, dim=0)[:N_PAIRS_PER_FILE]
            file_poses = pose_frames_all[pose_cursor : pose_cursor + N_PAIRS_PER_FILE]
            kf_cursor   += n_kf_per_file
            pose_cursor += N_PAIRS_PER_FILE

            with open(raw_out, "wb") as f_out:
                for i in tqdm(range(0, N_PAIRS_PER_FILE, batch_size), desc=f"Decoding {file_name}"):
                    m    = file_masks[i:i+batch_size].to(device).long()
                    p    = file_poses[i:i+batch_size].to(device).float()
                    pidx = torch.arange(i, i + m.shape[0], device=device) % K_TEMPORAL

                    f1, f2 = generator(m, p, pidx)
                    f1_up  = F.interpolate(f1, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
                    f2_up  = F.interpolate(f2, (OUT_H, OUT_W), mode="bilinear", align_corners=False)

                    batch_comp = einops.rearrange(
                        torch.stack([f1_up, f2_up], dim=1),
                        "b t c h w -> (b t) h w c"
                    )
                    f_out.write(batch_comp.clamp(0, 255).round().to(torch.uint8).cpu().numpy().tobytes())

if __name__ == "__main__":
    main()
