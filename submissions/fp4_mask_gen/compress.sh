#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting fp4_mask_gen compression pipeline..."
python3 "${HERE}/compress.py" "$@"

echo "Done. Archive: ${HERE}/archive.zip"
