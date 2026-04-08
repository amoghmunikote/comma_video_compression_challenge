#!/usr/bin/env python
"""Neural inflate: REN enhancement + SegNet gradient optimization on odd frames."""
import os, io, bz2, struct, sys, time, av, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── REN Model ────────────────────────────────────────────────────────────────

class REN(nn.Module):
    def __init__(self, features=32):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        self.body = nn.Sequential(
            nn.Conv2d(12, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, 12, 3, padding=1),
        )
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        x_norm = x / 255.0
        residual = self.up(self.body(self.down(x_norm)))
        return (x_norm + residual).clamp(0, 1) * 255.0


def _load_int8_bz2(path):
    with open(path, 'rb') as f:
        raw = bz2.decompress(f.read())
    buf = io.BytesIO(raw)
    n_tensors = struct.unpack('<I', buf.read(4))[0]
    sd = {}
    for _ in range(n_tensors):
        name_len = struct.unpack('<I', buf.read(4))[0]
        name = buf.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<I', buf.read(4))[0]
        shape = [struct.unpack('<I', buf.read(4))[0] for _ in range(n_dims)]
        scale = struct.unpack('<f', buf.read(4))[0]
        data_len = struct.unpack('<I', buf.read(4))[0]
        data = np.frombuffer(buf.read(data_len), dtype=np.int8)
        sd[name] = torch.from_numpy(data.astype(np.float32)).reshape(shape) * scale
    return sd


def _load_f16_bz2(path):
    with open(path, 'rb') as f:
        data = bz2.decompress(f.read())
    sd = torch.load(io.BytesIO(data), map_location=DEVICE, weights_only=True)
    return {k: v.float() for k, v in sd.items()}


_REN_MODEL = None

def get_ren(archive_dir=None):
    global _REN_MODEL
    if _REN_MODEL is not None:
        return _REN_MODEL
    for d in ([archive_dir] if archive_dir else []) + [str(HERE / 'archive'), str(HERE)]:
        for name, fmt in [('ren_model.int8.bz2', 'int8'), ('ren_model.pt.bz2', 'f16'), ('ren_model.pt', 'raw')]:
            path = os.path.join(d, name)
            if os.path.exists(path):
                _REN_MODEL = REN(features=32).to(DEVICE).eval()
                if fmt == 'int8':
                    _REN_MODEL.load_state_dict(_load_int8_bz2(path))
                elif fmt == 'f16':
                    _REN_MODEL.load_state_dict(_load_f16_bz2(path))
                else:
                    _REN_MODEL.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
                return _REN_MODEL
    raise FileNotFoundError("ren_model not found")


# ── SegNet Gradient Optimization ─────────────────────────────────────────────

def compute_segnet_labels(original_video_path, device, batch_size=16):
    """Compute SegNet labels from original video. Only odd-indexed frames."""
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    seg_h, seg_w = segnet_model_input_size[1], segnet_model_input_size[0]
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))

    fmt = 'hevc' if original_video_path.endswith('.hevc') else None
    container = av.open(original_video_path, format=fmt)
    stream = container.streams.video[0]

    all_labels = []
    frame_idx = 0
    batch_frames = []

    with torch.inference_mode():
        for frame in container.decode(stream):
            if frame_idx % 2 == 1:
                t = yuv420_to_rgb(frame)
                x = t.permute(2, 0, 1).unsqueeze(0).float()
                x = F.interpolate(x, size=(seg_h, seg_w), mode='bilinear', align_corners=False)
                batch_frames.append(x.squeeze(0))

                if len(batch_frames) == batch_size:
                    batch = torch.stack(batch_frames).to(device)
                    logits = segnet(batch)
                    all_labels.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
                    batch_frames = []
            frame_idx += 1

        if batch_frames:
            batch = torch.stack(batch_frames).to(device)
            logits = segnet(batch)
            all_labels.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))

    container.close()
    del segnet
    torch.cuda.empty_cache()
    return np.concatenate(all_labels, axis=0)


def optimize_frame_segnet(frame_chw, seg_label, segnet, device, n_steps=25, lr=2.0, reg=0.01):
    """Optimize a single frame's delta at SegNet resolution to match labels."""
    seg_h, seg_w = segnet_model_input_size[1], segnet_model_input_size[0]
    target_h, target_w = camera_size[1], camera_size[0]

    frame_gpu = frame_chw.unsqueeze(0).to(device)
    frame_small = F.interpolate(frame_gpu, size=(seg_h, seg_w), mode='bilinear', align_corners=False).squeeze(0)
    original_small = frame_small.clone().detach()
    target = torch.from_numpy(seg_label[np.newaxis].copy()).long().to(device)

    delta = torch.zeros_like(original_small, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        corrected = (original_small + delta).unsqueeze(0)
        logits = segnet(corrected)
        loss = F.cross_entropy(logits, target) + reg * (delta ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-20, 20)

    with torch.no_grad():
        delta_full = F.interpolate(delta.unsqueeze(0), size=(target_h, target_w),
                                    mode='bicubic', align_corners=False).squeeze(0)
        result = (frame_chw.to(device) + delta_full).clamp(0, 255).cpu()

    del delta, target, frame_gpu, frame_small, original_small
    return result


# ── Main Pipeline ────────────────────────────────────────────────────────────

def decode_and_resize_to_file(video_path: str, dst: str, video_name: str = None):
    target_w, target_h = camera_size
    t_start = time.time()

    # Step 1: Decode + REN enhance all frames
    print(f"  Step 1: Decode + REN enhance...", flush=True)
    fmt = 'hevc' if video_path.endswith('.hevc') else None
    container = av.open(video_path, format=fmt)
    stream = container.streams.video[0]
    ren = get_ren(os.path.dirname(video_path))

    frames = []
    for frame in container.decode(stream):
        t = yuv420_to_rgb(frame)
        H, W, _ = t.shape
        if H != target_h or W != target_w:
            pil = Image.fromarray(t.numpy())
            pil = pil.resize((target_w, target_h), Image.LANCZOS)
            x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                x = ren(x)
            t_out = x.clamp(0, 255).squeeze(0).cpu()
        else:
            t_out = t.permute(2, 0, 1).float()
        frames.append(t_out)
    container.close()
    print(f"    {len(frames)} frames decoded+enhanced ({time.time()-t_start:.1f}s)", flush=True)

    # Step 2: Compute SegNet labels from original video
    if video_name:
        original_path = str(ROOT / "videos" / video_name)
    else:
        original_path = str(ROOT / "videos" / "0.mkv")

    if os.path.exists(original_path):
        print(f"  Step 2: Computing SegNet labels from {original_path}...", flush=True)
        t_label = time.time()
        seg_labels = compute_segnet_labels(original_path, DEVICE)
        print(f"    {seg_labels.shape[0]} labels ({time.time()-t_label:.1f}s)", flush=True)

        # Step 3: Gradient-optimize odd frames for SegNet
        print(f"  Step 3: SegNet gradient optimization on odd frames...", flush=True)
        from modules import SegNet, segnet_sd_path
        from safetensors.torch import load_file
        segnet = SegNet().eval().to(DEVICE)
        segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))

        n_steps = 25 if DEVICE.type == 'cuda' else 15
        t_opt = time.time()
        pair_idx = 0
        for frame_idx in range(1, len(frames), 2):
            if pair_idx >= seg_labels.shape[0]:
                break
            frames[frame_idx] = optimize_frame_segnet(
                frames[frame_idx], seg_labels[pair_idx], segnet, DEVICE,
                n_steps=n_steps, lr=2.0, reg=0.01
            )
            pair_idx += 1
            if pair_idx % 100 == 0:
                print(f"    Optimized {pair_idx}/{seg_labels.shape[0]} frames", flush=True)

        del segnet
        torch.cuda.empty_cache()
        print(f"    Done ({time.time()-t_opt:.1f}s)", flush=True)
    else:
        print(f"  Step 2: Original video not found, skipping SegNet optimization", flush=True)

    # Step 4: Write output
    print(f"  Step 4: Writing output...", flush=True)
    with open(dst, 'wb') as f:
        for frame_chw in frames:
            t = frame_chw.permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())

    print(f"  Total: {len(frames)} frames, {time.time()-t_start:.1f}s", flush=True)
    return len(frames)


if __name__ == "__main__":
    video_path = sys.argv[1]
    dst = sys.argv[2]
    video_name = sys.argv[3] if len(sys.argv) > 3 else None
    n = decode_and_resize_to_file(video_path, dst, video_name)
    print(f"saved {n} frames")
