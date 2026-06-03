"""
OmniX OCR backend (PaddleX) — single FastAPI process on :6006.

Design: this service returns RAW PaddleX inference output (lightly slimmed).
All conversion into the external "群杰" contract (field renaming, whole_text
assembly, char-polygon geometry, HTML->grid parsing, borders/type) is done
downstream in OmniX (C#). The only pixel/inference-bound work that MUST live
here: running the models, and the OpenCV crop+enhance for /v1/doc-correct.

Endpoints:
  GET  /health
  POST /v1/ocr           image(multipart 'file' or raw binary) -> OCR JSON (incl. word/char boxes)
  POST /v1/table         image -> table_recognition_v2 JSON (cells, html, per-table wired/wireless)
  POST /v1/doc-correct   image + params -> rectified+cropped+enhanced JPEG (binary)
"""
import os
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from paddlex import create_pipeline, create_model

app = FastAPI(title="OmniX OCR Backend (PaddleX)", version="1.0.0")

DEVICE = "gpu:0"
_OCR = None
_TABLE = None
_TABLE_CLS = None
_ORI = None
_UV = None


def _pipelines():
    """Lazy singletons so the process can boot fast and warm up on first hit."""
    global _OCR, _TABLE, _TABLE_CLS
    if _OCR is None:
        _OCR = create_pipeline(pipeline="OCR", device=DEVICE)
    if _TABLE is None:
        _TABLE = create_pipeline(pipeline="table_recognition_v2", device=DEVICE)
    if _TABLE_CLS is None:
        _TABLE_CLS = create_model("PP-LCNet_x1_0_table_cls", device=DEVICE)
    return _OCR, _TABLE, _TABLE_CLS


def _get_ori():
    global _ORI
    if _ORI is None:
        _ORI = create_model("PP-LCNet_x1_0_doc_ori", device=DEVICE)
    return _ORI


def _get_uv():
    global _UV
    if _UV is None:
        _UV = create_model("UVDoc", device=DEVICE)
    return _UV


async def _read_bgr(request: Request, file: Optional[UploadFile]):
    """Accept both multipart 'file' and raw --data-binary body."""
    if file is not None:
        raw = await file.read()
    else:
        raw = await request.body()
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# ----------------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------------------------------------------------------
# OCR (general text recognition)
# ----------------------------------------------------------------------------
@app.post("/v1/ocr")
async def ocr(request: Request, file: Optional[UploadFile] = File(None)):
    t0 = time.time()
    img = await _read_bgr(request, file)
    if img is None:
        return JSONResponse({"error": "empty image"}, status_code=400)
    h, w = img.shape[:2]
    ocr_pl, _, _ = _pipelines()

    res = None
    for r in ocr_pl.predict(img, return_word_box=True):
        res = r.json["res"]
        break

    doc_angle = 0
    dp = res.get("doc_preprocessor_res") or {}
    if isinstance(dp, dict):
        doc_angle = dp.get("angle", 0)

    out = {
        "image_width": int(w),
        "image_height": int(h),
        "doc_angle": doc_angle,
        "rec_texts": res.get("rec_texts", []),
        "rec_scores": res.get("rec_scores", []),
        "rec_polys": res.get("rec_polys", []),          # per-line quad (4 pts x,y)
        "rec_boxes": res.get("rec_boxes", []),           # axis-aligned [x1,y1,x2,y2]
        "dt_polys": res.get("dt_polys", []),
        "textline_orientation_angles": res.get("textline_orientation_angles", []),
        "text_word": res.get("text_word", []),           # per-line list of words (CJK = per char)
        "text_word_boxes": res.get("text_word_boxes", []),  # per-line list of quad boxes
        "cost_ms": int((time.time() - t0) * 1000),
    }
    return JSONResponse(out)


# ----------------------------------------------------------------------------
# Table recognition v2
# ----------------------------------------------------------------------------
def _region_from_cells(cells):
    if not cells:
        return None
    xs1 = [c[0] for c in cells]; ys1 = [c[1] for c in cells]
    xs2 = [c[2] for c in cells]; ys2 = [c[3] for c in cells]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _classify_table(table_cls, img_bgr, region):
    h, w = img_bgr.shape[:2]
    if region is None:
        crop = img_bgr
    else:
        x1, y1, x2, y2 = [int(round(v)) for v in region]
        x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img_bgr
    try:
        r = next(iter(table_cls.predict(crop)))
        j = r.json["res"]
        idx = int(np.argmax(j["scores"]))
        return j["label_names"][idx], float(j["scores"][idx])
    except Exception as e:  # noqa: BLE001
        return "unknown", 0.0


@app.post("/v1/table")
async def table(request: Request, file: Optional[UploadFile] = File(None)):
    t0 = time.time()
    img = await _read_bgr(request, file)
    if img is None:
        return JSONResponse({"error": "empty image"}, status_code=400)
    h, w = img.shape[:2]
    _, table_pl, table_cls = _pipelines()

    res = None
    for r in table_pl.predict(img):
        res = r.json["res"]
        break

    tables = []
    for t in (res.get("table_res_list") or []):
        cells = t.get("cell_box_list") or []
        region = _region_from_cells(cells)
        label, score = _classify_table(table_cls, img, region)
        tables.append({
            "type": label,                # 'wired_table' | 'wireless_table' | 'unknown'
            "type_score": score,
            "region": region,             # [x1,y1,x2,y2] hull of cells
            "pred_html": t.get("pred_html", ""),
            "cell_box_list": cells,       # [[x1,y1,x2,y2], ...] (HTML <td> order)
        })

    overall = res.get("overall_ocr_res") or {}
    layout = []
    ld = res.get("layout_det_res") or {}
    for b in (ld.get("boxes") or []):
        layout.append({
            "label": b.get("label"),
            "score": b.get("score"),
            "coordinate": b.get("coordinate"),
        })

    out = {
        "image_width": int(w),
        "image_height": int(h),
        "tables": tables,
        "overall_ocr": {
            "rec_texts": overall.get("rec_texts", []),
            "rec_scores": overall.get("rec_scores", []),
            "rec_polys": overall.get("rec_polys", []),
            "rec_boxes": overall.get("rec_boxes", []),
        },
        "layout": layout,
        "cost_ms": int((time.time() - t0) * 1000),
    }
    return JSONResponse(out)


# ----------------------------------------------------------------------------
# Document correction (crop + rectify + enhance)
# ----------------------------------------------------------------------------
def _order_quad(pts):
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],   # top-left
        pts[np.argmin(d)],   # top-right
        pts[np.argmax(s)],   # bottom-right
        pts[np.argmax(d)],   # bottom-left
    ], dtype="float32")


def _warp_quad(img, quad):
    """Perspective-warp the quad to a front-on rectangle. Target size = the AVERAGE of
    each pair of opposite edges (top/bottom widths, left/right heights) — the trapezoid's
    representative dimension; cubic resampling keeps text edges clean. (A 4-corner homography
    can't fully recover the true aspect of a perspective shot without camera intrinsics;
    avg vs max differs <2% — corner accuracy, not this pick, drives any visible stretch.)"""
    (tl, tr, br, bl) = quad
    w_top = np.linalg.norm(tr - tl); w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl); h_right = np.linalg.norm(br - tr)
    mw = int(round((w_top + w_bot) / 2.0)); mh = int(round((h_left + h_right) / 2.0))
    if mw < 10 or mh < 10:
        return None
    dst = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, m, (mw, mh), flags=cv2.INTER_CUBIC)


def _plausible(out, w, h):
    """Reject blown-up / collapsed warps (rotation may swap w/h, so compare sorted)."""
    oh, ow = out.shape[:2]
    if max(ow, oh) > max(w, h) * 1.3 or min(ow, oh) > min(w, h) * 1.3:
        return False
    return ow >= w * 0.2 and oh >= h * 0.2


def _paper_mask(img):
    """Mask of bright, low-saturation regions = white paper on a darker/colored desk."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    _, v_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    s_mask = cv2.inRange(s, 0, 90)
    mask = cv2.bitwise_and(v_mask, s_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return mask


def _edge_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med)); hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi)
    return cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)


def _largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def _quad_from_contour(contour, img_area):
    peri = cv2.arcLength(contour, True)
    for eps in (0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx) \
                and cv2.contourArea(approx) > 0.2 * img_area:
            return approx
    return None


def _expand_quad(quad, frac):
    """Push the 4 corners outward from their centroid by `frac` (margin), so a
    legit document crop doesn't shave off text sitting right on the edge."""
    c = quad.mean(axis=0)
    return (c + (quad - c) * (1.0 + frac)).astype("float32")


# Document detection picks the contour with the highest RECTANGULARITY across two
# candidate masks: an HSV bright/low-saturation mask (best for paper on a desk) and a
# deep ISNet salient-object mask (best for object-like inputs such as a package). The
# page is then warped from its FOUR EXTREME CORNERS (outermost min/max of x±y), which
# circumscribe all content -> never clips, while following the page's tilt -> deskews.
_SEG = None


def _get_seg():
    global _SEG
    if _SEG is None:
        from rembg import new_session
        _SEG = new_session("isnet-general-use",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return _SEG


def _isnet_mask(img):
    try:
        from rembg import remove
        m = remove(img, session=_get_seg(), only_mask=True, post_process_mask=True)
        _, th = cv2.threshold(m, 100, 255, cv2.THRESH_BINARY)
        return th
    except Exception:  # noqa: BLE001
        return None


def _corners(contour):
    """Four extreme corners (ordered tl, tr, br, bl) — they circumscribe the contour."""
    pts = contour.reshape(-1, 2).astype("float32")
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], dtype="float32")


def _clean_mask(m, ref):
    """Open then close to remove thin spikes/tails (shadows, reflections) that would
    otherwise drag a corner off the page; keeps the solid sheet intact."""
    k = max(9, int(0.012 * ref))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)


def _quad_corners(contour):
    """Prefer the convex hull's 4-vertex polygon (true trapezoid corners, accurate
    under perspective and robust to a single stray point); fall back to the extreme
    corners. Always returned ordered tl, tr, br, bl."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return _order_quad(ap.reshape(4, 2))
    return _order_quad(_corners(contour))


def _best_doc_contour(img):
    """Most document-like contour (highest rectangularity) across candidate masks."""
    h, w = img.shape[:2]
    area = float(h * w)
    masks = [_paper_mask(img), _edge_mask(img)]
    isn = _isnet_mask(img)
    if isn is not None:
        masks.append(isn)

    ref = max(h, w)
    best, best_extent = None, 0.0
    for m in masks:
        c = _largest_contour(_clean_mask(m, ref))
        if c is None:
            continue
        cov = cv2.contourArea(c) / area
        if cov < 0.25 or cov > 0.985:      # too small, or whole frame (already a clean scan)
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) < 10:
            continue
        extent = cv2.contourArea(c) / (rw * rh)
        if extent > best_extent:
            best, best_extent = c, extent
    return best


def _fit_edge(P, vertical):
    """Robust line fit to one border (iterative outlier rejection). Returns (a,b,c)
    for a*x + b*y + c = 0. vertical → fit x=m*y+b, else y=m*x+b."""
    P = P.astype(np.float64)
    ind = P[:, 1] if vertical else P[:, 0]
    dep = P[:, 0] if vertical else P[:, 1]
    m, b = np.polyfit(ind, dep, 1)
    for _ in range(3):
        res = np.abs(dep - (m * ind + b))
        keep = res < max(2.0, 2.5 * np.median(res))
        if keep.sum() < 2 or keep.all():
            break
        ind, dep = ind[keep], dep[keep]
        m, b = np.polyfit(ind, dep, 1)
    return (1.0, -m, -b) if vertical else (-m, 1.0, -b)


def _intersect(l1, l2):
    a1, b1, c1 = l1; a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-6:
        return None
    return [(-c1 * b2 + c2 * b1) / d, (-a1 * c2 + a2 * c1) / d]


def _edgefit_corners(contour):
    """Recover the 4 corners by fitting the document's 4 border lines and intersecting
    them. Robust to a missing corner (recovered as the edges' intersection) and to mask
    spikes (rejected as outliers). Returns an ordered quad, or None if unreliable."""
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(contour)
    a = ang if ang < 45 else ang - 90
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), a, 1.0)
    pts = contour.reshape(-1, 2).astype(np.float64)
    rp = (M[:, :2] @ pts.T).T + M[:, 2]
    x0, x1 = rp[:, 0].min(), rp[:, 0].max()
    y0, y1 = rp[:, 1].min(), rp[:, 1].max()
    W, H = x1 - x0, y1 - y0
    band = 0.12
    groups = [rp[rp[:, 1] < y0 + band * H], rp[rp[:, 1] > y1 - band * H],
              rp[rp[:, 0] < x0 + band * W], rp[rp[:, 0] > x1 - band * W]]
    if min(len(g) for g in groups) < 5:
        return None
    lt, lb = _fit_edge(groups[0], False), _fit_edge(groups[1], False)
    ll, lr = _fit_edge(groups[2], True), _fit_edge(groups[3], True)
    corners = [_intersect(lt, ll), _intersect(lt, lr), _intersect(lb, lr), _intersect(lb, ll)]
    if any(p is None for p in corners):
        return None
    minv = cv2.invertAffineTransform(M)
    quad = (minv[:, :2] @ np.array(corners).T).T + minv[:, 2]
    return _order_quad(quad.astype("float32"))


def _valid_quad(quad, shape):
    h, w = shape[:2]
    area = cv2.contourArea(quad.astype("float32"))
    if area < 0.25 * h * w or area > 1.25 * h * w:
        return False
    return bool(cv2.isContourConvex(quad.astype(np.int32).reshape(-1, 1, 2)))


_DOCA = None
_DOCA_PATH = os.environ.get(
    "DOCRECT_DOCALIGNER_MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "docaligner_fastvit_sa24.onnx"),
)


def _get_doca():
    global _DOCA
    if _DOCA is None:
        import onnxruntime as ort
        _DOCA = ort.InferenceSession(
            _DOCA_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return _DOCA


def _complete_quad(pts3, shape):
    """Recover a missing 4th corner from 3 via the parallelogram prior. Of the three
    candidate completions, the bogus ones shoot far OUTSIDE the frame (that was the
    earlier garbage); the correct one stays inside and forms the most rectangular quad."""
    h, w = shape[:2]
    p = np.array(pts3, dtype="float32")
    candidates = [p[1] + p[2] - p[0], p[0] + p[2] - p[1], p[0] + p[1] - p[2]]
    best, best_score = None, -1.0
    for c in candidates:
        if not (-0.05 * w <= c[0] <= 1.05 * w and -0.05 * h <= c[1] <= 1.05 * h):
            continue  # completion outside the frame -> wrong candidate
        quad = _order_quad(np.vstack([p, c[None]]))
        if not cv2.isContourConvex(quad.astype(np.int32).reshape(-1, 1, 2)):
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(quad)
        if min(rw, rh) < 1:
            continue
        rectangularity = cv2.contourArea(quad) / (rw * rh)
        if rectangularity > best_score:
            best, best_score = quad, rectangularity
    return best


def _docaligner_corners(img):
    """Learned document-corner detector (DocAligner / FastViT heatmap). Robust to
    same-color backgrounds and perspective where mask/edge methods fail. Returns an
    ordered 4-point quad in img coordinates, or None if fewer than 4 corners found."""
    try:
        sess = _get_doca()
        iname = sess.get_inputs()[0].name
        oname = sess.get_outputs()[0].name
        nh, nw = img.shape[:2]
        inp = cv2.resize(img, (256, 256))
        inp = np.transpose(inp, (2, 0, 1)).astype("float32")[None] / 255.0
        pred = sess.run([oname], {iname: inp})[0]  # (1, 4, H, W) corner heatmaps
        pts = []
        for i in range(pred.shape[1]):
            p = cv2.resize(pred[0, i], (nw, nh))
            p[p < 0.3] = 0
            m = (p * 255).astype(np.uint8)
            _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            mo = cv2.moments(max(cs, key=cv2.contourArea))
            if mo["m00"] == 0:
                continue
            pts.append([mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]])
        if len(pts) == 4:
            return _order_quad(np.array(pts, dtype="float32"))
        if len(pts) == 3:
            return _complete_quad(pts, img.shape)
        return None
    except Exception:  # noqa: BLE001
        return None


def _rectify_document(img):
    """Crop + deskew. Tiered:
      1) learned corner detector (DocAligner) — best on same-color / perspective
      2) fit-the-4-border-lines + intersect — recovers missing corners, ignores spikes
      3) convex-hull 4-vertex corners
    Detection runs on a downscaled copy; corners scale back and the full-res image is
    warped once. Returns None when no document is found (caller just enhances)."""
    h, w = img.shape[:2]
    proc, scale = img, 1.0
    longest = max(h, w)
    if longest > 1600:
        scale = 1600.0 / longest
        proc = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

    quad = _docaligner_corners(proc)
    if quad is None or not _valid_quad(quad, proc.shape):
        contour = _best_doc_contour(proc)
        if contour is None:
            return None
        quad = _edgefit_corners(contour)
        if quad is None or not _valid_quad(quad, proc.shape):
            quad = _quad_corners(contour)

    quad = (quad / scale).astype("float32")
    warped = _warp_quad(img, quad)
    return warped if warped is not None and _plausible(warped, w, h) else None


def _deskew_textlines(img):
    """Fine residual deskew by maximizing the horizontal-projection variance of the ink
    map: when text lines are level the row-sum profile is sharply peaked (high variance),
    so we search small angles for the peak. Robust to noise/strokes/punctuation (whole-page
    statistic, unlike a minAreaRect on dark pixels). The angle search runs on a downscaled
    binary (cheap); the found angle is applied once to the full-res image. No-op when the
    page is already level, skipped when the result is implausible."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        s = 900.0 / max(h, w) if max(h, w) > 900 else 1.0
        small = cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))),
                           interpolation=cv2.INTER_AREA) if s < 1.0 else gray
        # project a Sauvola TEXT mask (not OTSU): excludes stamps / table fills / residual
        # shadow that would otherwise bias the variance toward a false peak.
        th = _sauvola_mask(small, window=25, k=0.2).astype(np.uint8) * 255
        th = cv2.medianBlur(th, 3)
        if cv2.countNonZero(th) < 30:
            return img
        sh, sw = th.shape[:2]
        center = (sw / 2.0, sh / 2.0)

        def variance_at(angle):
            m = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            rot = cv2.warpAffine(th, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
            proj = np.sum(rot, axis=1, dtype=np.float64)
            return float(np.var(proj))

        coarse = np.arange(-5.0, 5.001, 0.2)              # fine coarse step so the peak isn't missed
        best = max(coarse, key=variance_at)
        fine = np.arange(best - 0.6, best + 0.601, 0.02)  # 0.02 deg ~ 1px across a 3000px page
        best = float(max(fine, key=variance_at))
        if abs(best) < 0.02 or abs(best) > 10.0:          # already level, or implausible -> skip
            return img
        print(f"[deskew] angle={best:.3f} deg", flush=True)
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), best, 1.0)
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    except Exception:  # noqa: BLE001
        return img


def _estimate_background(lum):
    """Large-scale illumination field of a luminance image. Estimated on a 1/4-size copy
    (a big blur there ≈ a huge blur at full res, ~16x cheaper) then upsampled."""
    h, w = lum.shape[:2]
    sw, sh = max(1, w // 4), max(1, h // 4)
    small = cv2.resize(lum, (sw, sh), interpolation=cv2.INTER_AREA)
    k = max(3, (max(sw, sh) // 3) | 1)
    bg = cv2.GaussianBlur(small, (k, k), 0)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def _scan_color(img, bright, contrast):
    """The 'scan look', smooth (not bilevel) and hue-preserving:
      1) flatten uneven lighting by DIVIDING luminance by its background field, so paper
         normalizes to white regardless of shadow/gradient (multiplicative model);
      2) a tone curve deepens ink toward black and clips paper to pure white while keeping
         gradients, so edges stay anti-aliased (no jaggies);
      3) recolor by scaling B/G/R with the same per-pixel luminance gain, which preserves
         hue & saturation — red stamps stay red, they don't get blackened.
    """
    imgf = img.astype(np.float32)
    b, g, r = cv2.split(imgf)
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    bg = np.maximum(_estimate_background(lum), 1.0)
    flat = np.clip(lum / bg, 0.0, 1.3)                     # paper -> ~1.0, ink -> low

    # Confident-background mask from the CLEAN divided signal (taken BEFORE CLAHE, which would
    # otherwise amplify subtle sheen into false detail): pixels within ~10% of the local paper
    # level AND low-chroma = paper / haze / bleed-through / plastic-bag sheen. Forced to pure
    # white at the very end, so the background is clean regardless of substrate, while real ink
    # (much darker -> low flat) and colored stamps (high chroma) are left untouched.
    chroma0 = imgf.max(axis=2) - imgf.min(axis=2)
    bg_mask = (flat > 0.90) & (chroma0 < 40.0)

    # Local contrast equalization (CLAHE): lift faint/soft regions — e.g. the foreshortened,
    # softly-focused top of an angled photo — to the contrast of the sharp regions, so
    # light-gray text reads as black instead of washing out. Per-tile, so it adapts to
    # spatially varying ink density; the background mottling it adds is wiped by the
    # near-white neutralization below.
    f8 = np.clip(flat * 200.0, 0, 255).astype(np.uint8)
    f8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(f8)
    flat = f8.astype(np.float32) / 200.0

    pivot = 0.70 - contrast / 500.0                        # below pivot darkens, above whitens
    strength = 3.6 + contrast / 40.0                       # stronger base -> bold, ink-black text
    new_lum = np.clip((flat - pivot) * strength + 1.0, 0.0, 1.0) * 255.0 + float(bright)
    new_lum = np.clip(new_lum, 0.0, 255.0)

    # Keep colored ink (red seal / signature) BRIGHT instead of darkening it like black text:
    # the tone curve above would mud the stamp. LAB a* (red axis) flags red regardless of
    # brightness, so for those pixels target a lifted luminance -> the seal stays vivid, not dark.
    a_red = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32)
    colorness = np.clip((a_red - 138.0) / 30.0, 0.0, 1.0)
    lum_keep = np.clip(lum * 1.15 + 45.0, 0.0, 255.0)
    new_lum = new_lum * (1.0 - colorness) + lum_keep * colorness

    gain = np.clip(new_lum / np.maximum(lum, 1.0), 0.0, 4.0)
    out = cv2.merge([np.clip(b * gain, 0, 255),
                     np.clip(g * gain, 0, 255),
                     np.clip(r * gain, 0, 255)]).astype(np.float32)

    # Neutralize the paper: bright + low-chroma pixels are blended toward pure white,
    # killing any residual color cast so the background reads as true white. Keyed on the
    # MIN channel, so saturated ink (red stamp = low B/G) and dark text are left alone.
    mx = out.max(axis=2); mn = out.min(axis=2)
    lum2 = 0.114 * out[..., 0] + 0.587 * out[..., 1] + 0.299 * out[..., 2]
    chroma = mx - mn
    bright = np.clip((lum2 - 185.0) / 55.0, 0.0, 1.0)
    achroma = np.clip((60.0 - chroma) / 60.0, 0.0, 1.0)    # 1 = neutral paper, 0 = saturated ink
    wpaper = (bright * achroma)[..., None]                 # whiten bright low-chroma paper only
    out = out * (1.0 - wpaper) + 255.0 * wpaper            # (slightly cyan/yellow paper -> white)
    out[bg_mask] = 255.0                                   # hard-whiten confident background: wipes
    out = np.clip(out, 0, 255).astype(np.uint8)            # haze / bleed-through / bag sheen

    # Vivid color (scan-app 'magic color' look): boost saturation so the red seal / signature
    # pop. Neutral ink & paper have ~0 saturation, so black text and the white background are
    # untouched — only genuinely colored pixels get more saturated.
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.85, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _sauvola_mask(gray, window=25, k=0.2, r=128.0):
    """Sauvola local binarization via integral images (cv2.boxFilter); no scikit-image.
    Returns a bool mask, True where the pixel is INK (below the local adaptive threshold)."""
    g = gray.astype(np.float32)
    win = (window, window)
    mean = cv2.boxFilter(g, -1, win, normalize=True, borderType=cv2.BORDER_REPLICATE)
    sqmean = cv2.boxFilter(g * g, -1, win, normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = cv2.sqrt(np.maximum(sqmean - mean * mean, 0.0))
    thresh = mean * (1.0 + k * (std / r - 1.0))
    return g < thresh


def _red_mask(img):
    """Red ink (seal + red handwriting) via HSV hue. Robust to DARK / low-saturation reds
    that a raw BGR channel-difference test misses: a photographed stamp can be dark red
    (R~80, B~50) where r-b is only ~30, but its hue is unmistakably red."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # LAB a* (red-green axis, 128 = neutral) is the robust redness signal: a dark maroon
    # stamp core keeps a high a* even though its HSV saturation is low, so a* catches the
    # part hue/saturation misses — while neutral black ink (a* ~128) is left alone. Combine
    # with an HSV hue test for bright reds. v>=40 excludes near-black so real ink survives.
    a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 1]
    hue_red = ((h <= 12) | (h >= 168)) & (s >= 40)
    m = (((hue_red) | (a >= 138)) & (v >= 40)).astype(np.uint8) * 255
    # NO open (it erodes thin strokes). a* already catches the maroon core directly, so only a
    # moderate close is needed to bridge the ring's strokes — kept small to limit collateral
    # whitening of black text the stamp overlaps. Dilate covers the anti-aliased halo.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    return cv2.dilate(m, np.ones((3, 3), np.uint8))


def _enhance(img, bright, contrast, detail, enhance_mode, remove_red=False):
    """enhanceMode: 0=color scan (default), 1=grayscale, 2=binarized B/W, 3=binarized + red
    stamp kept. removeStamp=1 whitens all red ink (seal + red handwriting) in any mode."""
    out = _scan_color(img, bright, contrast)

    # detail / sharpen via unsharp mask (-1 = auto; stronger default for crisp text edges)
    amount = 0.9 if detail == -1 else max(0.0, detail / 100.0)
    if amount > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 3)
        out = cv2.addWeighted(out, 1 + amount, blur, -amount, 0)

    red = _red_mask(img) > 0
    if remove_red:
        out[red] = 255          # whiten red BEFORE binarizing, so the seal can't become black ink

    if enhance_mode == 1:
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    elif enhance_mode in (2, 3):
        # Binarize on the (already background-cleaned) scan. k=0.20 commits faint strokes so
        # they stay connected; a light close bridges 1px gaps for stroke continuity. The
        # upstream background whitening removed the smudges that used to turn into speckle,
        # so this stays clean despite the looser k. CC-area filter drops isolated dust.
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        win = max(31, (min(gray.shape[:2]) // 40) | 1)
        ink = _sauvola_mask(gray, window=win, k=0.20).astype(np.uint8)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))  # bridge stroke gaps
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        min_area = max(4, (ink.shape[0] * ink.shape[1]) // 400000)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False                                    # label 0 is the background
        bw = np.full_like(out, 255)
        bw[keep[labels]] = (0, 0, 0)
        bw = cv2.GaussianBlur(bw, (0, 0), 0.6)             # symmetric anti-alias (no weight bias)
        out = bw

    if not remove_red and enhance_mode == 3:               # overlay crisp red stamp from original
        out[red] = img[red]
    return out


def _orient(img):
    """Detect 0/90/180/270 rotation and bring the page upright (cv2 rotate)."""
    try:
        r = next(iter(_get_ori().predict(img)))
        angle = int(r.json["res"]["label_names"][0])
    except Exception:  # noqa: BLE001
        angle = 0
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _unwarp(img):
    """UVDoc geometric dewarp (for curved-page photos). Returns clean array."""
    try:
        r = next(iter(_get_uv().predict(img)))
        arr = r.img.get("res")
        if arr is not None:
            return np.ascontiguousarray(np.array(arr))
    except Exception:  # noqa: BLE001
        pass
    return img


@app.post("/v1/doc-correct")
async def doc_correct(
    request: Request,
    file: Optional[UploadFile] = File(None),
    scan_m: int = Query(-1, alias="scan-m"),
    bright: int = Query(0),
    contrast: int = Query(0),
    detail: int = Query(-1),
    enhance_mode: int = Query(0, alias="enhanceMode"),
    unwarp: int = Query(0),  # 1 = enable UVDoc dewarp (only for curved-page photos)
    remove_red: int = Query(0, alias="removeStamp"),  # 1 = whiten red ink (seal + red handwriting)
):
    img = await _read_bgr(request, file)
    if img is None:
        return JSONResponse({"error": "empty image"}, status_code=400)

    # 1) document detection: crop + deskew (scan_m == 0 disables auto-crop)
    if scan_m != 0:
        rectified = _rectify_document(img)
        if rectified is not None:
            img = rectified

    # 2) orientation correction (cheap 0/90/180/270 rotate, always on).
    img = _orient(img)
    # optional UVDoc geometric dewarp — OFF by default; on flat docs it distorts.
    if unwarp == 1:
        img = _unwarp(img)

    # 2.5) fine residual text-line deskew (make lines truly horizontal)
    img = _deskew_textlines(img)

    # 3) photometric enhancement
    img = _enhance(img, bright, contrast, detail, enhance_mode, remove_red=bool(remove_red))

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return JSONResponse({"error": "encode failed"}, status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg")
