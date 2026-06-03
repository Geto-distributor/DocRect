#!/bin/bash
# DocRect environment setup. Assumes Python 3.10-3.12 + an NVIDIA GPU (driver installed).
# Tuned for AutoDL (miniconda at /root/miniconda3); adjust paths for other hosts.
set -e
export PATH=/root/miniconda3/bin:$PATH

# 1) PaddlePaddle GPU — pick the index matching your CUDA (cu126 / cu123 / cu118).
python -m pip install --no-cache-dir paddlepaddle-gpu \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 2) PaddleX OCR/table/doc pipelines + service deps
python -m pip install --no-cache-dir "paddlex[ocr]" \
    fastapi "uvicorn[standard]" opencv-python-headless python-multipart pillow

# 3) Document segmentation (ISNet via rembg) + GPU ONNX runtime (also runs DocAligner)
python -m pip install --no-cache-dir rembg onnxruntime-gpu

# 4) Model weights:
#    - PaddleX models download lazily on first request (Baidu BOS).
#    - rembg ISNet (isnet-general-use.onnx) downloads on first use to ~/.u2net/.
#    - DocAligner weights: see models/README.md (Google Drive — download elsewhere if blocked).

echo "Done. Fetch the DocAligner model (models/README.md), then: bash scripts/start.sh"
