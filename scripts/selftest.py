#!/usr/bin/env python3
"""DocRect self-test — horizontal OCR/quality comparison.

For each input image, runs it through the box's `/v1/doc-correct` (rectify + enhance),
then OCRs BOTH the original and the enhanced result and prints a side-by-side comparison:
line count, mean recognition confidence, and latency. The enhanced column should show
equal-or-higher confidence, fewer fragmented lines, and lower latency on photographed
documents; with `--text` you can also eyeball the reading order (a skewed original often
comes back reverse-ordered, the enhanced one top-to-bottom). `--table` additionally checks
`/v1/table` on the enhanced image — paragraph-only pages should detect zero tables.

Pure stdlib (urllib) — no dependencies. Point `--base` at the box.

Usage:
  python3 scripts/selftest.py contract.jpg page.png
  python3 scripts/selftest.py --base http://127.0.0.1:6006 --table --text *.jpg
  python3 scripts/selftest.py --params "enhanceMode=0&removeStamp=1" stamped.jpg
"""
import argparse
import json
import urllib.request


def _post(url, data, ctype="application/octet-stream", timeout=180):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _ocr(base, img_bytes):
    d = json.loads(_post(base + "/v1/ocr", img_bytes))
    texts, scores = d.get("rec_texts", []), d.get("rec_scores", [])
    return {
        "lines": len(texts),
        "score": (sum(scores) / len(scores) if scores else 0.0),
        "cost": d.get("cost_ms"),
        "text": "".join(texts),
    }


def main():
    ap = argparse.ArgumentParser(description="DocRect OCR self-test / comparison")
    ap.add_argument("images", nargs="+", help="image files to test")
    ap.add_argument("--base", default="http://127.0.0.1:6006", help="box base URL")
    ap.add_argument("--params", default="enhanceMode=0", help="doc-correct query string")
    ap.add_argument("--table", action="store_true", help="also run /v1/table on the enhanced image")
    ap.add_argument("--text", action="store_true", help="print recognized text per image")
    a = ap.parse_args()

    try:  # warm up the OCR pipeline once (first call loads the models)
        _post(a.base + "/v1/ocr", open(a.images[0], "rb").read())
    except Exception:  # noqa: BLE001
        pass

    print("%-22s | ORIGINAL lines/score/ms  | ENHANCED lines/score/ms" % "image")
    print("-" * 76)
    for path in a.images:
        raw = open(path, "rb").read()
        enhanced = _post(a.base + "/v1/doc-correct?" + a.params, raw)
        o, e = _ocr(a.base, raw), _ocr(a.base, enhanced)
        name = path.rsplit("/", 1)[-1]
        print("%-22s | %3d / %.3f / %5sms  | %3d / %.3f / %5sms" % (
            name[:22], o["lines"], o["score"], o["cost"], e["lines"], e["score"], e["cost"]))
        if a.table:
            tabs = json.loads(_post(a.base + "/v1/table", enhanced)).get("tables", [])
            for i, tb in enumerate(tabs):
                print("    table#%d type=%s cells=%d" % (i, tb.get("type"), len(tb.get("cell_box_list", []))))
            if not tabs:
                print("    (no table detected)")
        if a.text:
            print("    ENHANCED:", e["text"][:200])


if __name__ == "__main__":
    main()
