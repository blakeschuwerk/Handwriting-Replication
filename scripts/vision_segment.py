#!/usr/bin/env python3
"""
vision_segment.py -- segmentation + auto-labeling in one pass using the macOS
Vision framework.

WHY THIS REPLACED THE CLASSICAL PIPELINE
----------------------------------------
scripts/segment.py (classical OpenCV) works on flat, clean scans but collapses
on this project's real input: phone photos of a CURVED, ruled notebook, shot at
an angle, then screenshotted out of a PDF viewer. On a representative page it
produced 4 usable crops out of 85 candidates, because the curvature broke every
geometric assumption -- ruled lines fragmented into arcs that evaded removal,
then bridged all text into one page-tall component.

Vision's learned text detector handles that page natively: 104 tight word boxes,
and it correctly declined to detect anything in the out-of-focus lower half of
the page that the classical gates kept trying (and failing) to reject.

Vision also RECOGNISES while it detects, so each word box arrives with proposed
text. That collapses "segment" and "label" into a single step and turns manual
transcription from ~12k boxes into reviewing ~20 line transcripts per page.

ACCURACY / TRUST
----------------
Recognition is roughly 75-85% word-accurate on this handwriting. It is NOT
trustworthy enough to train on unreviewed: observed errors include
"competition"->"comprition" and "similar product"->"simial prout", and Vision
reports confidence 1.00 for several of them, so its confidence is a useful
signal for junk lines but NOT a per-word correctness guarantee.

Labels are therefore written to labels_auto.csv (a PROPOSAL), never straight to
labels.csv. Review happens at line granularity in the dashboard, then word
alignment back onto crops is mechanical.
"""

import argparse
import csv
import json
import glob
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment import (  # noqa: E402
    load_image_any, downscale_if_needed, detect_page_region, normalize_background,
)

import Vision  # noqa: E402
import Quartz  # noqa: E402
from Foundation import NSURL, NSRange  # noqa: E402

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".heic"}

ROTATION_FILE = "page_rotation.json"


def load_rotations(root):
    """Per-page rotation in degrees, keyed by scan filename.

    Some pages are photographed upside down. Vision reads them correctly anyway
    -- it normalises orientation internally -- so the TEXT is right while the
    cropped PIXELS are inverted, which would teach the model that upside-down
    glyphs spell "the".

    There is deliberately no auto-detection here. Three signals were measured
    against a known-flipped page and all failed: Vision's own confidence and
    word counts are identical in both orientations (it corrects internally),
    an explicit orientation hint to VNImageRequestHandler is ignored for the
    same reason, and per-crop ink centre-of-mass separates by under 0.01 --
    well inside page-to-page noise -- because Vision's tight boxes crop away
    the ascender/descender asymmetry the measure relies on.

    Glancing at a thumbnail grid is instant and exact, so orientation is a
    human control surfaced in the dashboard and recorded here.
    """
    p = os.path.join(root, ROTATION_FILE)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return {k: int(v) for k, v in json.load(f).items()}
    except Exception as e:
        print(f"[WARN] could not read {ROTATION_FILE}: {e}")
        return {}


def apply_rotation(img, deg):
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
def vision_recognize(path):
    """Run Vision text recognition; return (observations, width, height).

    Each observation is one detected text line:
        {"text": str, "conf": float, "words": [(word, x, y, w, h), ...]}
    with pixel coordinates in the given image's space, origin top-left.
    """
    url = NSURL.fileURLWithPath_(path)
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if src is None:
        raise RuntimeError(f"could not open {path}")
    cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    W, H = Quartz.CGImageGetWidth(cg), Quartz.CGImageGetHeight(cg)

    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    req.setRecognitionLanguages_(["en-US"])

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"Vision failed on {path}: {err}")

    out = []
    for obs in (req.results() or []):
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        cand = cands[0]
        text = cand.string()
        words, pos = [], 0
        for w in text.split():
            i = text.find(w, pos)
            pos = i + len(w)
            try:
                bb, _ = cand.boundingBoxForRange_error_(NSRange(i, len(w)), None)
                if bb is None:
                    continue
                r = bb.boundingBox()
                x = r.origin.x * W
                y = (1.0 - r.origin.y - r.size.height) * H  # flip to top-left
                words.append((w, int(round(x)), int(round(y)),
                              int(round(r.size.width * W)), int(round(r.size.height * H))))
            except Exception:
                continue
        if words:
            out.append({"text": text, "conf": float(cand.confidence()), "words": words})
    return out, W, H


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def dominant_column(observations, W, tol=0.18):
    """Keep only the main body column of text.

    A photo of an open notebook shows the facing page as a narrow strip of short,
    low-confidence fragments down one side. The body column is the widest dense
    cluster of lines, so anchor on the median left edge of multi-word lines and
    reject observations starting far outside it.
    """
    lefts = [min(w[1] for w in o["words"]) for o in observations if len(o["words"]) >= 3]
    if not lefts:
        return observations
    anchor = float(np.median(lefts))
    keep = []
    for o in observations:
        x0 = min(w[1] for w in o["words"])
        if abs(x0 - anchor) <= tol * W:
            keep.append(o)
    return keep


def clean_word(w):
    """Strip surrounding punctuation; keep internal apostrophes/hyphens."""
    return w.strip(".,;:!?()[]{}\"'`“”‘’").strip()


# ---------------------------------------------------------------------------
def process_page(path, out_dir, lines_dir, debug_dir, args, rows, line_rows, tally,
                 rotations=None):
    stem = os.path.splitext(os.path.basename(path))[0]
    img = load_image_any(path)
    if img is None:
        return 0

    deg = int((rotations or {}).get(os.path.basename(path), 0))
    if deg:
        img = apply_rotation(img, deg)
        print(f"  [{stem}] rotated {deg} deg")

    region = detect_page_region(img)
    if region is not None:
        x, y, w, h = region
        img = img[y:y + h, x:x + w]
    img = downscale_if_needed(img)

    # Vision needs a file; hand it the page-cropped frame so application window
    # chrome is gone before recognition rather than filtered after.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp = tf.name
    cv2.imwrite(tmp, img)
    try:
        obs, W, H = vision_recognize(tmp)
    finally:
        os.unlink(tmp)

    n_raw = len(obs)
    obs = dominant_column(obs, W)

    # Crops come from the background-normalised grey: near-uniform white paper
    # with dark ink, which is both cleaner and closer to the IAM-style images the
    # model was pretrained on.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = normalize_background(gray)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for f in glob.glob(os.path.join(out_dir, f"{stem}_L*.png")):
        os.remove(f)
    for f in glob.glob(os.path.join(lines_dir, f"{stem}_L*.png")):
        os.remove(f)

    kept = 0
    for li, o in enumerate(obs, start=1):
        if o["conf"] < args.min_conf or len(o["words"]) < args.min_words:
            tally["line_rejected"] = tally.get("line_rejected", 0) + 1
            for (_, x, y, w, h) in o["words"]:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 1)
            continue

        # line crop (used for review + as the HTR context unit)
        xs = [w[1] for w in o["words"]]
        ys = [w[2] for w in o["words"]]
        xe = [w[1] + w[3] for w in o["words"]]
        ye = [w[2] + w[4] for w in o["words"]]
        lx, ly = max(0, min(xs) - 4), max(0, min(ys) - 4)
        lx2, ly2 = min(W, max(xe) + 4), min(H, max(ye) + 4)
        lname = f"{stem}_L{li:03d}.png"
        cv2.imwrite(os.path.join(lines_dir, lname), gray[ly:ly2, lx:lx2])
        line_rows.append([os.path.join(lines_dir, lname), path, li, o["text"],
                          round(o["conf"], 3), len(o["words"])])

        for wi, (word, x, y, w, h) in enumerate(o["words"], start=1):
            pad = max(2, int(0.08 * h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            crop = gray[y0:y1, x0:x1]
            if crop.size == 0 or crop.shape[0] < args.min_h or crop.shape[1] < args.min_w:
                tally["too_small"] = tally.get("too_small", 0) + 1
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 1)
                continue
            token = clean_word(word)
            if not token:
                tally["empty_token"] = tally.get("empty_token", 0) + 1
                continue
            fname = f"{stem}_L{li:03d}_W{wi:03d}.png"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, crop)
            rows.append([fpath, path, x0, y0, x1 - x0, y1 - y0, li, wi,
                         token, round(o["conf"], 3)])
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
            kept += 1
            tally["kept"] = tally.get("kept", 0) + 1

    cv2.imwrite(os.path.join(debug_dir, f"{stem}_overlay.png"), overlay)
    print(f"  [{stem}] {n_raw} lines detected -> {len(obs)} in body column, {kept} word crops")
    return kept


def main():
    p = argparse.ArgumentParser(description="Vision-based segmentation + auto-labeling.")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--only", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min-conf", type=float, default=0.4,
                   help="Drop whole lines below this Vision confidence.")
    p.add_argument("--min-words", type=int, default=2,
                   help="Drop lines with fewer words (stray marks, page numbers).")
    p.add_argument("--min-h", type=int, default=16)
    p.add_argument("--min-w", type=int, default=10)
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = args.input_dir or os.path.join(root, "raw_scans")
    output_dir = args.output_dir or os.path.join(root, "segmented")
    lines_dir = os.path.join(output_dir, "lines")
    debug_dir = os.path.join(output_dir, "debug")
    for d in (output_dir, lines_dir, debug_dir):
        os.makedirs(d, exist_ok=True)

    files = sorted(f for f in glob.glob(os.path.join(input_dir, "*"))
                   if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS)
    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"No images in {input_dir}")
        sys.exit(0)

    rotations = load_rotations(root)
    if rotations:
        print(f"rotations configured for {len(rotations)} page(s)")
    rows, line_rows, tally = [], [], {}
    for path in files:
        print(f"Processing: {os.path.basename(path)}")
        process_page(path, output_dir, lines_dir, debug_dir, args, rows, line_rows,
                     tally, rotations)

    partial = bool(args.only or args.limit)
    if not partial:
        with open(os.path.join(output_dir, "manifest.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["crop_path", "source_scan", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                        "line_idx", "word_idx", "auto_text", "line_conf"])
            w.writerows(rows)
        with open(os.path.join(output_dir, "lines_manifest.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["line_path", "source_scan", "line_idx", "auto_text", "conf", "n_words"])
            w.writerows(line_rows)
        # A PROPOSAL, deliberately not labels.csv -- see module docstring.
        with open(os.path.join(root, "labels_auto.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "text", "writer_id"])
            for r in rows:
                w.writerow([r[0], r[8], "me"])

    print()
    print("=== Summary ===")
    print(f"Pages processed : {len(files)}")
    print(f"Word crops      : {tally.get('kept', 0)}")
    print(f"Line crops      : {len(line_rows)}")
    for k in sorted(tally, key=lambda k: -tally[k]):
        print(f"  {k:16s} {tally[k]}")
    if partial:
        print("\n[tuning run: manifests NOT rewritten]")
    else:
        print(f"\nProposed labels -> {os.path.join(root, 'labels_auto.csv')}")
        print("Review before promoting to labels.csv.")


if __name__ == "__main__":
    main()
