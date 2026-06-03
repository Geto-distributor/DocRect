# Model weights

## DocAligner (document-corner detector) — required for `/v1/doc-correct`

DocRect uses the `fastvit_sa24` heatmap model from
[DocAligner](https://github.com/DocsaidLab/DocAligner). Only the `.onnx` file is needed —
inference is ported into `app.py`, so the `docaligner-docsaid` / `capybara` packages are
**not** required at runtime.

Download `fastvit_sa24_h_e_bifpn_256_fp32.onnx` (~79 MB, Google Drive id
`14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl`) and place it here as
`models/docaligner_fastvit_sa24.onnx`:

```bash
pip install gdown
gdown 14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl -O models/docaligner_fastvit_sa24.onnx
```

If Google Drive is unreachable from the GPU host (e.g. mainland China), download it on a
machine that can reach Drive and `scp` it over. Override the path with
`DOCRECT_DOCALIGNER_MODEL` if you keep it elsewhere.

## Others (auto-downloaded, no action needed)

- **PaddleX** OCR / table / orientation models → `~/.paddlex/official_models/` on first request.
- **rembg ISNet** (`isnet-general-use.onnx`) → `~/.u2net/` on first `/v1/doc-correct` that hits the mask fallback.
