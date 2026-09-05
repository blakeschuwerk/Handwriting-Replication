#!/usr/bin/env python3
"""
segment.py -- classical OpenCV document segmentation for handwriting-synthesis
training data prep.

v2: quality-gated. The v1 pipeline (kept at segment_v1_backup.py) emitted every
connected component that cleared a fixed pixel-size floor, which on real photos
meant desk wood-grain, window chrome, ruled-line speckle and out-of-focus page
regions all became "words" -- ~600 crops/page with a median height of 8px.

v2 is deliberately AGGRESSIVE about rejection. Style fine-tuning needs on the
order of 1-2k clean word crops, not 12k noisy ones, and a mislabeled crop is
worse than a missing one. Every candidate box must survive:

  * page-region crop      -- discard anything outside the sheet of paper
  * absolute size floor   -- no 8px specks
  * page-relative size    -- height must sit near the page's own text height
  * ink fraction          -- not blank, not a solid shadow blob
  * stroke contrast       -- faint/washed-out regions are dropped
  * focus (Laplacian var) -- blurry regions of the photo are dropped

Rejected boxes are drawn in RED on the debug overlay, kept boxes in GREEN, so
threshold tuning is a visual loop: run with --only <page> and look at the overlay.

LINE CROPS: even in word granularity, every accepted line is also written to
segmented/lines/. Downstream auto-labeling transcribes those line images with an
HTR model and aligns the resulting words to the word crops of the same line
using (source_scan, line_idx, word_idx) from the manifest.

COORDINATE SPACE: bbox_* in manifest.csv are in the PROCESSED working-image
space (after page crop, downscale and deskew), consistent with the saved pixels.
"""

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_OK = True
except Exception:
    _HEIF_OK = False

from PIL import Image

# ---------------------------------------------------------------------------
# Tunables (all overridable from the CLI)
# ---------------------------------------------------------------------------
MAX_WORKING_DIM = 2500
SKEW_ANGLE_CAP_DEG = 30.0

LINE_DILATE_KERNEL = (45, 5)
MIN_LINE_HEIGHT = 14
MIN_LINE_AREA = 600

RULED_LINE_MIN_WIDTH_FRAC = 0.55
RULED_LINE_MAX_HEIGHT = 6

# Quality gates
DEF_MIN_H = 20          # px, absolute floor on crop height
DEF_MIN_W = 14          # px
DEF_MAX_H = 220         # px, above this it's a blob not a word
DEF_INK_MIN = 0.05      # foreground fraction: below = blank
DEF_INK_MAX = 0.65      # above = solid shadow / filled blob
DEF_CONTRAST_MIN = 45   # max-min grey spread within the crop
DEF_FOCUS_MIN = 55.0    # variance of Laplacian; below = out of focus
DEF_REL_H_LO = 0.55     # crop height vs page median text height
DEF_REL_H_HI = 2.20

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".heic"}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_image_any(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".heic":
            if not _HEIF_OK:
                print(f"[WARN] Skipping {path}: pillow-heif not available.")
                return None
            return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[WARN] Skipping unreadable file {path}: {e}")
        return None


def downscale_if_needed(img, max_dim=MAX_WORKING_DIM):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    s = max_dim / float(longest)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Page-region detection
# ---------------------------------------------------------------------------
def detect_page_region(bgr, min_area_frac=0.08, max_area_frac=0.98):
    """Find the sheet of paper and return its bbox in ORIGINAL coordinates.

    The paper is the largest bright region. This is what removes application
    window chrome, the desk surface, and anything else surrounding the page.
    Returns None when no confident region is found (caller then uses the whole
    image), so a weird photo degrades to v1 behaviour rather than losing data.
    """
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (600, int(600 * h / w)) if w >= h else (int(600 * w / h), 600),
                       interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (31, 31), 0)

    # Paper is the bright side of an Otsu split.
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, cw, ch, area = stats[idx]
    frac = area / float(sh * sw)
    if frac < min_area_frac or frac > max_area_frac:
        return None

    sx, sy = w / float(sw), h / float(sh)
    pad = 8
    X0 = max(0, int(x * sx) - pad)
    Y0 = max(0, int(y * sy) - pad)
    X1 = min(w, int((x + cw) * sx) + pad)
    Y1 = min(h, int((y + ch) * sy) + pad)
    if X1 - X0 < 50 or Y1 - Y0 < 50:
        return None
    return (X0, Y0, X1 - X0, Y1 - Y0)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def normalize_background(gray):
    """Flatten uneven lighting by dividing out an estimated paper background.

    The background is estimated with a morphological CLOSE using a kernel larger
    than any glyph, which erases the dark text and leaves only the illumination
    profile.

    A plain Gaussian blur is NOT sufficient here and fails in an asymmetric way:
    inside a dense paragraph the blur average is dragged down by the surrounding
    ink, so the local "background" comes out nearly as dark as the text and the
    whole paragraph normalizes to white, while sparse headings sitting on open
    paper survive. That produced pages where only the title line was detected
    and every body paragraph vanished before segmentation ever saw it.
    """
    h, w = gray.shape[:2]
    k = int(round(min(h, w) * 0.035))
    k = max(31, k | 1)  # odd, and comfortably larger than a glyph
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=k / 3.0)
    bg = np.maximum(bg, 1).astype(np.float32)
    norm = (gray.astype(np.float32) / bg) * 220.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def estimate_and_correct_skew(gray, angle_cap=SKEW_ANGLE_CAP_DEG):
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] < 50:
        return gray, 0.0
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > angle_cap or abs(angle) < 0.3:
        return gray, 0.0
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE), angle


def binarize(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, blockSize=35, C=15)


def remove_ruled_lines(mask):
    """Erase notebook rules, margin lines and page edges.

    Anything surviving a long horizontal (or vertical) morphological OPEN is by
    construction a continuous straight run far longer than any pen stroke, so it
    is removed outright rather than being further filtered on how much of the
    page it spans.

    The span test used previously failed on this data because the notebook is
    photographed CURVED: each rule breaks into several shorter arcs, none of
    which covers the 55% of page width the filter demanded. The rules therefore
    survived and went on to bridge every text line into a single component.
    """
    h, w = mask.shape[:2]
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 25), 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 25))))
    ruled = cv2.bitwise_or(horiz, vert)
    # Grow slightly so the rule's anti-aliased shoulders go with it.
    ruled = cv2.dilate(ruled, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return cv2.bitwise_and(mask, cv2.bitwise_not(ruled))


def drop_large_structures(mask, max_h_frac=0.45, max_w_frac=0.75, max_area_frac=0.04):
    """Remove page outlines, binder shadows and similar non-text blobs.

    A component this large is never a word, but it will happily connect
    otherwise-separate text lines into one giant component and defeat any
    connectivity-based line finder.
    """
    H, W = mask.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = mask.copy()
    for i in range(1, n):
        _, _, cw, ch, area = stats[i]
        if ch > max_h_frac * H or cw > max_w_frac * W or area > max_area_frac * H * W:
            out[labels == i] = 0
    return out


# ---------------------------------------------------------------------------
# Box finding
# ---------------------------------------------------------------------------
def find_line_boxes(mask, max_line_h_frac=0.25):
    """Locate text lines from the horizontal ink projection.

    Connected-component line finding is fragile on photographed pages: a single
    leftover vertical structure merges every line on the sheet into one
    component -- observed on this data as a "line" 1852px tall spanning the
    entire page, which then handed find_word_boxes an 833px dilation kernel and
    smeared all the body text into a couple of unusable blobs.

    A row-wise projection is a statistic rather than a connectivity walk, so no
    stray connector can collapse the page.
    """
    H, W = mask.shape[:2]
    prof = (mask > 0).sum(axis=1).astype(np.float32)
    if prof.max() <= 0:
        return []
    k = max(3, int(H * 0.004) | 1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, k), 0).ravel()
    on = prof > max(3.0, 0.06 * float(prof.max()))

    boxes, y = [], 0
    while y < H:
        if not on[y]:
            y += 1
            continue
        y0 = y
        while y < H and on[y]:
            y += 1
        lh = y - y0
        if lh < MIN_LINE_HEIGHT or lh > max_line_h_frac * H:
            continue
        cols = np.nonzero((mask[y0:y0 + lh] > 0).sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        if (x1 - x0) * lh < MIN_LINE_AREA:
            continue
        boxes.append((x0, y0, x1 - x0, lh))
    return boxes


def find_word_boxes(line_mask, line_h):
    """Split a line into words using a dilation kernel scaled to the line height.

    A fixed kernel (v1 used 12px) over-splits large writing and under-splits
    small writing. Inter-word gaps scale with text size, so the kernel should
    too: ~45% of line height reliably bridges intra-word gaps without bridging
    the wider between-word gaps.
    """
    kw = int(max(8, round(0.45 * line_h)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
    dil = cv2.dilate(line_mask, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    boxes = [tuple(stats[i][:4]) for i in range(1, n) if stats[i][4] >= 40]
    boxes.sort(key=lambda b: b[0])
    return boxes


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def focus_score(crop_gray):
    """Variance of the Laplacian: the standard sharpness measure.

    This is the gate that matters most for phone photos of a curved notebook --
    the part of the page that drifts out of the focal plane produces smooth,
    low-variance crops even when it still has ink in it.
    """
    if crop_gray.size < 64 or min(crop_gray.shape[:2]) < 5:
        return 0.0
    return float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())


def ink_stats(crop_gray):
    """Return (ink_fraction, contrast) using an Otsu split inside the crop."""
    if crop_gray.size == 0:
        return 0.0, 0.0
    lo, hi = float(crop_gray.min()), float(crop_gray.max())
    contrast = hi - lo
    if contrast < 10:
        return 0.0, contrast
    thr, _ = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = float((crop_gray < thr).mean())
    return ink, contrast


def quality_score(ink, contrast, focus):
    f = min(focus / 250.0, 1.0)
    c = min(contrast / 140.0, 1.0)
    # ink centred around ~0.25 is the healthy band for a word crop
    i = max(0.0, 1.0 - abs(ink - 0.25) / 0.35)
    return round(0.5 * f + 0.3 * c + 0.2 * i, 4)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------
def clear_prior_outputs(out_dir, lines_dir, debug_dir, stem):
    for pat in (os.path.join(out_dir, f"{stem}_L*.png"),
                os.path.join(lines_dir, f"{stem}_L*.png")):
        for f in glob.glob(pat):
            os.remove(f)
    ov = os.path.join(debug_dir, f"{stem}_overlay.png")
    if os.path.exists(ov):
        os.remove(ov)


def process_image(path, out_dir, lines_dir, debug_dir, args, manifest_rows, lines_rows, tally):
    stem = os.path.splitext(os.path.basename(path))[0]
    img = load_image_any(path)
    if img is None:
        return 0, 0

    if args.crop_page:
        region = detect_page_region(img)
        if region is not None:
            x, y, w, h = region
            img = img[y:y + h, x:x + w]
            print(f"  [{stem}] page region: {w}x{h}")
        else:
            print(f"  [{stem}] page region: not found, using full frame")

    img = downscale_if_needed(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)
    gray = normalize_background(gray)
    gray, angle = estimate_and_correct_skew(gray)
    if angle:
        print(f"  [{stem}] deskew: {angle:.2f} deg")

    mask = binarize(gray)
    if args.remove_lines:
        mask = remove_ruled_lines(mask)
    mask = drop_large_structures(mask)

    proc_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = proc_bgr.copy()
    clear_prior_outputs(out_dir, lines_dir, debug_dir, stem)

    line_boxes = find_line_boxes(mask)
    if not line_boxes:
        print(f"  [{stem}] WARNING: no text lines detected.")
        return 0, 0

    # ---- pass 1: gather candidates with metrics --------------------------
    cands = []
    for li, (lx, ly, lw, lh) in enumerate(line_boxes, start=1):
        if args.granularity == "line":
            boxes = [(0, 0, lw, lh)]
        else:
            boxes = find_word_boxes(mask[ly:ly + lh, lx:lx + lw], lh)
        for wi, (wx, wy, ww, wh) in enumerate(boxes, start=1):
            ax, ay = lx + wx, ly + wy
            crop = gray[ay:ay + wh, ax:ax + ww]
            if crop.size == 0:
                continue
            ink, contrast = ink_stats(crop)
            focus = focus_score(crop)
            cands.append({
                "li": li, "wi": wi, "x": ax, "y": ay, "w": ww, "h": wh,
                "ink": ink, "contrast": contrast, "focus": focus,
                "lbox": (lx, ly, lw, lh),
            })

    if not cands:
        return 0, 0

    # ---- page-relative height reference ----------------------------------
    # Median over boxes that already clear the absolute floor, so that noise
    # specks cannot drag the page's notion of "normal text height" down.
    plausible = [c["h"] for c in cands if c["h"] >= args.min_h and c["w"] >= args.min_w]
    med_h = float(np.median(plausible)) if plausible else 0.0

    # ---- pass 2: filter ---------------------------------------------------
    kept, kept_lines = 0, set()
    for c in cands:
        reason = None
        if c["h"] < args.min_h or c["w"] < args.min_w:
            reason = "too_small"
        elif c["h"] > args.max_h:
            reason = "too_tall"
        elif med_h > 0 and not (args.rel_h_lo * med_h <= c["h"] <= args.rel_h_hi * med_h):
            reason = "off_page_scale"
        elif not (args.ink_min <= c["ink"] <= args.ink_max):
            reason = "bad_ink"
        elif c["contrast"] < args.contrast_min:
            reason = "low_contrast"
        elif c["focus"] < args.focus_min:
            reason = "blurry"

        if reason:
            tally[reason] = tally.get(reason, 0) + 1
            cv2.rectangle(overlay, (c["x"], c["y"]),
                          (c["x"] + c["w"], c["y"] + c["h"]), (0, 0, 255), 1)
            continue

        if args.granularity == "line":
            fname = f"{stem}_L{c['li']:03d}.png"
        else:
            fname = f"{stem}_L{c['li']:03d}_W{c['wi']:03d}.png"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, gray[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]])
        manifest_rows.append([
            fpath, path, c["x"], c["y"], c["w"], c["h"], c["li"], c["wi"],
            round(c["ink"], 4), round(c["contrast"], 1), round(c["focus"], 1),
            quality_score(c["ink"], c["contrast"], c["focus"]),
        ])
        cv2.rectangle(overlay, (c["x"], c["y"]),
                      (c["x"] + c["w"], c["y"] + c["h"]), (0, 255, 0), 2)
        kept += 1
        kept_lines.add(c["li"])
        tally["kept"] = tally.get("kept", 0) + 1

    # ---- write line crops for the lines that survived --------------------
    # These feed HTR auto-labeling: transcribe the line, then align its words to
    # the word crops sharing the same (source_scan, line_idx).
    n_lines = 0
    if args.granularity == "word":
        for li, (lx, ly, lw, lh) in enumerate(line_boxes, start=1):
            if li not in kept_lines:
                continue
            lname = f"{stem}_L{li:03d}.png"
            lpath = os.path.join(lines_dir, lname)
            cv2.imwrite(lpath, gray[ly:ly + lh, lx:lx + lw])
            lines_rows.append([lpath, path, lx, ly, lw, lh, li])
            cv2.rectangle(overlay, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 1)
            n_lines += 1

    cv2.imwrite(os.path.join(debug_dir, f"{stem}_overlay.png"), overlay)
    print(f"  [{stem}] kept {kept}/{len(cands)} candidates "
          f"({100.0 * kept / max(1, len(cands)):.0f}%), {n_lines} lines, med_h={med_h:.0f}px")
    return kept, n_lines


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Quality-gated handwriting segmentation.")
    p.add_argument("--granularity", choices=["word", "line"], default="word")
    p.add_argument("--remove-lines", dest="remove_lines", action="store_true", default=True)
    p.add_argument("--no-remove-lines", dest="remove_lines", action="store_false")
    p.add_argument("--crop-page", dest="crop_page", action="store_true", default=True,
                   help="Crop to the detected sheet of paper first (default on).")
    p.add_argument("--no-crop-page", dest="crop_page", action="store_false")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--only", default=None, help="Process only files whose name contains this.")
    p.add_argument("--limit", type=int, default=0, help="Process at most N images (0 = all).")
    # quality gates
    p.add_argument("--min-h", type=int, default=DEF_MIN_H)
    p.add_argument("--min-w", type=int, default=DEF_MIN_W)
    p.add_argument("--max-h", type=int, default=DEF_MAX_H)
    p.add_argument("--ink-min", type=float, default=DEF_INK_MIN)
    p.add_argument("--ink-max", type=float, default=DEF_INK_MAX)
    p.add_argument("--contrast-min", type=float, default=DEF_CONTRAST_MIN)
    p.add_argument("--focus-min", type=float, default=DEF_FOCUS_MIN)
    p.add_argument("--rel-h-lo", type=float, default=DEF_REL_H_LO)
    p.add_argument("--rel-h-hi", type=float, default=DEF_REL_H_HI)
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
        print(f"No image files found in {input_dir}.")
        sys.exit(0)

    manifest_rows, lines_rows, tally = [], [], {}
    total_crops = total_lines = 0
    for path in files:
        print(f"Processing: {os.path.basename(path)}")
        k, nl = process_image(path, output_dir, lines_dir, debug_dir, args,
                              manifest_rows, lines_rows, tally)
        total_crops += k
        total_lines += nl

    # Only rewrite the manifests on a full run; a --only/--limit run is a tuning
    # pass and must not clobber the manifest for pages it never looked at.
    partial = bool(args.only or args.limit)
    if not partial:
        with open(os.path.join(output_dir, "manifest.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["crop_path", "source_scan", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                        "line_idx", "word_idx", "ink", "contrast", "focus", "quality"])
            w.writerows(manifest_rows)
        with open(os.path.join(output_dir, "lines_manifest.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["line_path", "source_scan", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "line_idx"])
            w.writerows(lines_rows)

    print()
    print("=== Summary ===")
    print(f"Images processed : {len(files)}")
    print(f"Word crops kept  : {total_crops}")
    print(f"Line crops kept  : {total_lines}")
    considered = sum(tally.values())
    print(f"Candidates seen  : {considered}")
    if considered:
        for k in sorted(tally, key=lambda k: -tally[k]):
            print(f"  {k:16s} {tally[k]:7d}  ({100.0 * tally[k] / considered:5.1f}%)")
    if partial:
        print("\n[tuning run: manifest.csv NOT rewritten]")
    print(f"Debug overlays   : {debug_dir}")


if __name__ == "__main__":
    main()
