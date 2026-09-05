#!/usr/bin/env python3
"""
detect_and_build.py -- two-phase pipeline around the editable box store.

  detect  scan image -> Vision word boxes -> merged into boxes/<page>.json
          (also caches the PROCESSED page to pages/<page>.png)
  build   boxes/*.json -> word crops + manifest.csv + labels_auto.csv

Split in two because the box store is hand-editable in between. `detect` never
destroys manual work (see boxstore.merge_detection), and `build` is a pure
function of the box store, so re-running it after an edit is always safe and
cheap.

pages/<page>.png matters more than it looks: the editor overlays boxes on that
exact image, so box coordinates and displayed pixels can never drift apart. The
processing (rotate -> page-crop -> downscale -> background-normalise) is
deterministic, so the cache always reproduces.
"""

import argparse
import csv
import glob
import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boxstore  # noqa: E402
from segment import (  # noqa: E402
    load_image_any, downscale_if_needed, detect_page_region, normalize_background,
)
from vision_segment import (  # noqa: E402
    vision_recognize, dominant_column, clean_word, load_rotations, apply_rotation,
    SUPPORTED_EXTS,
)

PAGES_DIRNAME = "pages"


def processed_page(path, rotations):
    """Rotate -> page-crop -> downscale -> background-normalise. Deterministic."""
    img = load_image_any(path)
    if img is None:
        return None, None
    deg = int((rotations or {}).get(os.path.basename(path), 0))
    if deg:
        img = apply_rotation(img, deg)
    region = detect_page_region(img)
    if region is not None:
        x, y, w, h = region
        img = img[y:y + h, x:x + w]
    img = downscale_if_needed(img)
    gray = normalize_background(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return img, gray


def detect_page(path, root, rotations, min_conf, min_words):
    color, gray = processed_page(path, rotations)
    if color is None:
        return None
    H, W = gray.shape[:2]

    pages = os.path.join(root, PAGES_DIRNAME)
    os.makedirs(pages, exist_ok=True)
    cv2.imwrite(os.path.join(pages, boxstore.page_key(path) + ".png"), gray)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp = tf.name
    cv2.imwrite(tmp, color)
    try:
        obs, VW, VH = vision_recognize(tmp)
    finally:
        os.unlink(tmp)
    obs = dominant_column(obs, VW)

    detected = []
    for li, o in enumerate(obs, start=1):
        if o["conf"] < min_conf or len(o["words"]) < min_words:
            continue
        for wi, (word, x, y, w, h) in enumerate(o["words"], start=1):
            token = clean_word(word)
            if not token:
                continue
            pad = max(2, int(0.06 * h))
            detected.append(boxstore.make_box(
                max(0, x - pad), max(0, y - pad),
                min(W - max(0, x - pad), w + 2 * pad),
                min(H - max(0, y - pad), h + 2 * pad),
                text=token, conf=o["conf"], line=li, word=wi, source="vision"))

    doc = boxstore.load(root, path) or boxstore.empty_doc(
        path, W, H, int((rotations or {}).get(os.path.basename(path), 0)))
    doc["width"], doc["height"] = W, H
    doc["rotation"] = int((rotations or {}).get(os.path.basename(path), 0))
    stats = boxstore.merge_detection(doc, detected)
    boxstore.assign_reading_order(doc)
    boxstore.save(root, path, doc)
    return stats, len(boxstore.active(doc))


def build_page(path, root, rotations, out_dir, min_h, min_w):
    doc = boxstore.load(root, path)
    if not doc:
        return [], 0
    _, gray = processed_page(path, rotations)
    if gray is None:
        return [], 0
    H, W = gray.shape[:2]

    for f in glob.glob(os.path.join(out_dir, boxstore.page_key(path) + "__*.png")):
        os.remove(f)

    rows = []
    for b in sorted(boxstore.active(doc), key=lambda b: (b["line"], b["word"])):
        x0, y0 = max(0, b["x"]), max(0, b["y"])
        x1, y1 = min(W, b["x"] + b["w"]), min(H, b["y"] + b["h"])
        if x1 - x0 < min_w or y1 - y0 < min_h:
            continue
        name = boxstore.crop_name(path, b)
        fpath = os.path.join(out_dir, name)
        cv2.imwrite(fpath, gray[y0:y1, x0:x1])
        rows.append([fpath, path, b["id"], x0, y0, x1 - x0, y1 - y0,
                     b["line"], b["word"], b.get("text", ""), b.get("conf", 0.0),
                     b.get("source", "vision"), int(bool(b.get("edited")))])
    return rows, len(rows)


def main():
    ap = argparse.ArgumentParser(description="Detect into / build from the box store.")
    ap.add_argument("phase", choices=["detect", "build", "all"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--min-h", type=int, default=16)
    ap.add_argument("--min-w", type=int, default=10)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = os.path.join(root, "raw_scans")
    out_dir = os.path.join(root, "segmented")
    os.makedirs(out_dir, exist_ok=True)
    rotations = load_rotations(root)

    files = sorted(f for f in glob.glob(os.path.join(raw, "*"))
                   if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS)
    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
    if not files:
        print(f"No images in {raw}")
        sys.exit(0)

    if args.phase in ("detect", "all"):
        tot = {}
        for p in files:
            r = detect_page(p, root, rotations, args.min_conf, args.min_words)
            if not r:
                continue
            stats, n = r
            for k, v in stats.items():
                tot[k] = tot.get(k, 0) + v
            print(f"  [{os.path.basename(p)[:38]}] {n} boxes  {stats}")
        print(f"\ndetect: {tot}")

    if args.phase in ("build", "all"):
        all_rows = []
        for p in files:
            rows, n = build_page(p, root, rotations, out_dir, args.min_h, args.min_w)
            all_rows.extend(rows)
        # Only a full run owns the manifests.
        if not args.only:
            with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["crop_path", "source_scan", "box_id", "bbox_x", "bbox_y",
                            "bbox_w", "bbox_h", "line_idx", "word_idx", "auto_text",
                            "line_conf", "source", "edited"])
                w.writerows(all_rows)
            with open(os.path.join(root, "labels_auto.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_path", "text", "writer_id"])
                for r in all_rows:
                    w.writerow([r[0], r[9], "me"])
        print(f"\nbuild: {len(all_rows)} crops")


if __name__ == "__main__":
    main()
