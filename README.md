# DocRect

GPU document-rectification + OCR service, built on **PaddleX** with a learned
document-corner detector (**DocAligner / FastViT**). Exposes three HTTP endpoints on
a single port (`:6006`); a thin C# layer in OmniX maps the raw output onto an external
("群杰") OCR contract.

```
client ──(群杰 contract)──> OmniX (C#)  ──(rich JSON)──> DocRect box (:6006, this repo)
                              maps schema                 runs the models (GPU)
```

## Endpoints (box service, `app.py`)

| Method / path | in | out |
|---|---|---|
| `GET  /health` | — | `{"status":"ok"}` |
| `POST /v1/ocr` | image (multipart `file` **or** raw binary body) | text recognition: `rec_texts / rec_scores / rec_polys / rec_boxes / dt_polys / text_word / text_word_boxes / doc_angle` |
| `POST /v1/table` | image | `tables[]` (`type` wired/wireless, `pred_html`, `cell_box_list`) + `overall_ocr` + `layout` |
| `POST /v1/doc-correct` | image + query `scan-m,bright,contrast,detail,enhanceMode,unwarp` | **rectified + enhanced JPEG** (binary) |

The box returns raw PaddleX-shaped JSON; all field renaming / `whole_text` assembly /
char-polygon geometry / HTML→grid parsing / borders is done by the OmniX C# layer
(`IOcrBackendClient` + `IOcrContractMapper` + the 群杰 controller), which lives in the
OmniX repo — it is not duplicated here.

## Document rectification pipeline (`/v1/doc-correct`)

Tiered, so it copes with clean docs, perspective shots, same-color backgrounds, soft
packages, and screenshots alike — and **never clips content**:

1. **DocAligner** learned 4-corner detector (GPU) — robust to same-color backgrounds &
   perspective. If it returns only 3 corners, the 4th is recovered with the
   **parallelogram prior** (the in-frame, most-rectangular completion).
2. Fallback: **4-border-line RANSAC fit → intersect** (recovers missing corners, ignores
   mask spikes), then convex-hull 4-vertex corners.
3. Candidate masks for the fallback: HSV bright/low-saturation + deep **ISNet** salient
   mask (rembg), pick the most rectangular; masks are spike-cleaned (morphological open).
4. Perspective warp → orientation (0/90/180/270) → **text-line deskew** (maximize the
   horizontal-projection variance of the ink map over small angles — a whole-page
   statistic, robust where a `minAreaRect` on dark pixels drifts) → **deblur** (auto:
   Richardson-Lucy deconvolution on luminance, only when the page is soft — classical,
   no hallucination) → **scan enhancement**: divide luminance by its background field to
   flatten shadows (paper → white), a tone curve that deepens ink and clips paper to pure
   white while keeping gradients (anti-aliased, no jaggies), hue-preserving recolor +
   near-white neutralization so the background is true white and **red stamps survive**.
5. No document found → enhance only (never distorts).

Query params: `enhanceMode` `0` color scan (default) · `1` grayscale · `2` binarized B/W
(Sauvola, integral-image — no scikit-image) · `3` binarized but red stamps kept in color.
`removeStamp=1` whitens red ink (seal + red handwriting). `deblur` `-1` auto (default,
deconvolve only soft pages) · `0` off · `1` force. `bright` / `contrast` / `detail` tune
brightness / ink darkness / sharpening. `scan-m=0` disables auto-crop; `unwarp=1` enables
UVDoc curved-page dewarp.

Detection runs on a downscaled copy; the full-res image is warped once. Warm latency is
sub-second to ~1s even on 24 MP photos (all inference on GPU).

## Setup

```bash
bash scripts/install.sh          # paddle, paddlex, fastapi, rembg, onnxruntime-gpu, ...
# then fetch the DocAligner weights — see models/README.md
bash scripts/start.sh            # serves on 0.0.0.0:6006
nohup bash scripts/watchdog.sh >/dev/null 2>&1 &   # optional keep-alive

python3 scripts/selftest.py --table --text *.jpg   # OCR quality A/B: original vs enhanced
```

`scripts/selftest.py` runs each image through `/v1/doc-correct` then OCRs the original
and the enhanced result side by side (lines / confidence / latency), and optionally checks
`/v1/table` — a quick regression/quality harness (pure stdlib, point `--base` at the box).

Requires an NVIDIA GPU. `scripts/start.sh` exports `LD_LIBRARY_PATH` so onnxruntime-gpu
finds paddle's bundled CUDA/cuDNN libs. The DocAligner weights path is configurable via
`DOCRECT_DOCALIGNER_MODEL` (defaults to `./models/docaligner_fastvit_sa24.onnx`).

## Known limitation

Residual paper **curvature** (wavy lines on a non-flat sheet) is not corrected — that
needs a dewarping network (DocTr++/DewarpNet). Perspective + deskew handle flat/angled
photos well; curvature is the open case.
