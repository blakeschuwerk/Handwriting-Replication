#!/usr/bin/env python3
"""
segment.py -- classical OpenCV document-segmentation pipeline for
handwriting-synthesis training data prep.

Pipeline (see README_segmentation.md for rationale):
  1. Read every image in raw_scans/ (png/jpg/jpeg/tif/bmp/heic).
  2. Downscale to a max working dimension.
  3. Grayscale + light denoise.
  4. Deskew (capped, sign-safe angle correction).
  5. Ruled-line removal (optional, default on).
  6. Binarize (adaptive threshold).
  7. Line segmentation (horizontal dilation + connected components).
  8. Word segmentation within each line (smaller dilation + gap analysis).
  9. Write crops + manifest.csv + debug overlays.

DEFAULT GRANULARITY: word. Line-level is available via --granularity line
as a robust fallback for cursive/joined handwriting where word boundaries
are ambiguous.

COORDINATE SPACE: all bbox_x/y/w/h in manifest.csv are in the PROCESSED
working-image space (i.e. after downscaling and after deskew rotation),
NOT the original source-image pixel space. Crops are saved from that same
processed image, so crop pixels and manifest bboxes are always consistent
with each other.
"""

import argparse
import csv
import os
import sys
import glob

import numpy as np
import cv2

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_OK = True
except Exception:
    _HEIF_OK = False

from PIL import Image

# ---------------------------------------------------------------------------
# Tunables (also exposed as CLI args below where useful)
# ---------------------------------------------------------------------------
MAX_WORKING_DIM = 2500          # longest side cap for processing, px
SKEW_ANGLE_CAP_DEG = 30.0       # never rotate more than this; skip instead
LINE_DILATE_KERNEL = (45, 5)    # (w, h) kernel to merge glyphs into line blobs
WORD_DILATE_KERNEL = (12, 5)    # (w, h) kernel to merge glyphs into word blobs within a line
MIN_LINE_HEIGHT = 12            # px, filters noise specks / stray marks
MIN_LINE_AREA = 400             # px^2
MIN_WORD_WIDTH = 8               # px
MIN_WORD_AREA = 60               # px^2
RULED_LINE_MIN_WIDTH_FRAC = 0.55  # horizontal line candidate must span this fraction of image width
RULED_LINE_MAX_HEIGHT = 6         # px, ruled lines are thin

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".heic"}


def load_image_any(path):
    """Load an image file into a BGR numpy array (OpenCV convention).
    Returns None (and prints a warning) on failure instead of raising."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".heic":
            if not _HEIF_OK:
                print(f"[WARN] Skipping {path}: HEIC support (pillow-heif) not available.")
                return None
            pil_img = Image.open(path).convert("RGB")
            arr = np.array(pil_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                return img
            # Fallback via PIL for formats cv2 might choke on
            pil_img = Image.open(path).convert("RGB")
            arr = np.array(pil_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[WARN] Skipping unreadable file {path}: {e}")
        return None


def downscale_if_needed(img, max_dim=MAX_WORKING_DIM):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / float(longest)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def denoise_gray(gray):
    # Light bilateral filter: smooths noise while preserving text edges.
    return cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)


def estimate_and_correct_skew(gray, angle_cap=SKEW_ANGLE_CAP_DEG):
    """Estimate skew via minAreaRect on the text mask and rotate to correct.
    BUG GUARD: cv2.minAreaRect returns angle in [-90, 0). Normalize so a
    near-vertical rect (angle close to -90) doesn't get treated as a huge
    rotation, and cap the correction so we never flip the page sideways
    if the estimate is garbage (e.g. dominated by noise/lines, not text)."""
    # Build a text mask: text becomes white on black bg via inverted Otsu.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] < 50:
        # Not enough foreground pixels to estimate a reliable skew; skip.
        return gray, 0.0, mask

    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))  # (x,y) order
    angle = rect[-1]

    # Normalize angle from cv2's [-90, 0) convention to a signed small angle.
    if angle < -45:
        angle = 90 + angle
    # angle is now roughly in (-45, 45]

    if abs(angle) > angle_cap:
        # Out-of-range estimate -- likely not real page skew; don't rotate.
        return gray, 0.0, mask

    if abs(angle) < 0.3:
        # Negligible skew, don't bother rotating (avoids resample blur).
        return gray, 0.0, mask

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle, mask


def remove_ruled_lines(binary_mask, min_width_frac=RULED_LINE_MIN_WIDTH_FRAC,
                        max_line_thickness=RULED_LINE_MAX_HEIGHT):
    """Detect and remove long thin horizontal (and vertical) ruled lines from
    a binary text mask (foreground=white=text/lines on black bg), without
    eating descenders/ascenders of real glyphs. Only components with a very
    high aspect ratio AND spanning most of the image width/height are
    considered ruled lines."""
    h, w = binary_mask.shape[:2]

    # Horizontal ruled-line detector: wide-short kernel isolates long
    # horizontal strokes (table/notebook rule lines), not individual glyphs.
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 15), 1))
    horiz_lines = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, horiz_kernel)

    # Vertical ruled-line detector (e.g. graph/margin lines): tall-thin kernel.
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 15)))
    vert_lines = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, vert_kernel)

    ruled = cv2.bitwise_or(horiz_lines, vert_lines)

    # Only keep components that actually span most of the width/height and
    # are thin, to avoid removing thick blobs of connected handwriting.
    cleaned_ruled = np.zeros_like(ruled)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ruled, connectivity=8)
    for i in range(1, n_labels):
        x, y, cw, ch, area = stats[i]
        spans_width = cw >= min_width_frac * w and ch <= max_line_thickness * 3
        spans_height = ch >= min_width_frac * h and cw <= max_line_thickness * 3
        if spans_width or spans_height:
            cleaned_ruled[labels == i] = 255

    result = cv2.bitwise_and(binary_mask, cv2.bitwise_not(cleaned_ruled))
    return result


def binarize(gray):
    # Adaptive threshold handles uneven lighting/gradient background well.
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=35, C=15,
    )
    return mask


def find_line_boxes(binary_mask, dilate_kernel=LINE_DILATE_KERNEL,
                     min_height=MIN_LINE_HEIGHT, min_area=MIN_LINE_AREA):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    boxes = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if h < min_height or area < min_area:
            continue
        boxes.append((x, y, w, h))

    # Sort top-to-bottom
    boxes.sort(key=lambda b: b[1])
    return boxes


def find_word_boxes(line_roi_mask, dilate_kernel=WORD_DILATE_KERNEL,
                     min_width=MIN_WORD_WIDTH, min_area=MIN_WORD_AREA):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    dilated = cv2.dilate(line_roi_mask, kernel, iterations=1)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    boxes = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if w < min_width or area < min_area:
            continue
        boxes.append((x, y, w, h))

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])
    return boxes


def clear_prior_outputs(out_dir, debug_dir, source_stem):
    for f in glob.glob(os.path.join(out_dir, f"{source_stem}_L*_W*.png")):
        os.remove(f)
    for f in glob.glob(os.path.join(out_dir, f"{source_stem}_L*.png")):
        # covers line-granularity crops too (no _W suffix)
        os.remove(f)
    overlay_path = os.path.join(debug_dir, f"{source_stem}_overlay.png")
    if os.path.exists(overlay_path):
        os.remove(overlay_path)


def process_image(path, out_dir, debug_dir, granularity, remove_lines, manifest_rows):
    source_stem = os.path.splitext(os.path.basename(path))[0]
    img = load_image_any(path)
    if img is None:
        return 0

    img = downscale_if_needed(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = denoise_gray(gray)

    gray, skew_angle, _ = estimate_and_correct_skew(gray)
    if skew_angle != 0.0:
        print(f"  [{source_stem}] corrected skew: {skew_angle:.2f} deg")
    else:
        print(f"  [{source_stem}] no skew correction applied")

    binary_mask = binarize(gray)

    if remove_lines:
        binary_mask = remove_ruled_lines(binary_mask)

    # Processed color image (for overlay + crops), same space as binary_mask.
    processed_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = processed_bgr.copy()

    clear_prior_outputs(out_dir, debug_dir, source_stem)

    line_boxes = find_line_boxes(binary_mask)

    crop_count = 0

    if not line_boxes:
        print(f"  [{source_stem}] WARNING: no text lines detected.")

    for li, (lx, ly, lw, lh) in enumerate(line_boxes, start=1):
        # Draw line box in blue on overlay regardless of granularity.
        cv2.rectangle(overlay, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)

        if granularity == "line":
            crop = processed_bgr[ly:ly + lh, lx:lx + lw]
            fname = f"{source_stem}_L{li:02d}.png"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, crop)
            manifest_rows.append([fpath, path, lx, ly, lw, lh])
            cv2.rectangle(overlay, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
            crop_count += 1
            continue

        # word granularity: crop the line ROI from the binary mask, find
        # word boxes within it, then map back to full-image coords.
        line_roi_mask = binary_mask[ly:ly + lh, lx:lx + lw]
        word_boxes = find_word_boxes(line_roi_mask)

        if not word_boxes:
            continue

        for wi, (wx, wy, ww, wh) in enumerate(word_boxes, start=1):
            abs_x, abs_y = lx + wx, ly + wy
            crop = processed_bgr[abs_y:abs_y + wh, abs_x:abs_x + ww]
            if crop.size == 0:
                continue
            fname = f"{source_stem}_L{li:02d}_W{wi:02d}.png"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, crop)
            manifest_rows.append([fpath, path, abs_x, abs_y, ww, wh])
            cv2.rectangle(overlay, (abs_x, abs_y), (abs_x + ww, abs_y + wh), (0, 255, 0), 2)
            crop_count += 1

    overlay_path = os.path.join(debug_dir, f"{source_stem}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    return crop_count


def main():
    parser = argparse.ArgumentParser(description="Classical OpenCV handwriting segmentation pipeline.")
    parser.add_argument("--granularity", choices=["word", "line"], default="word",
                         help="Emit word-level crops (default) or line-level crops (cursive fallback).")
    parser.add_argument("--engine", choices=["classical", "doctr"], default="classical",
                         help="Segmentation engine. 'classical' (default) needs no DL stack. "
                              "'doctr' is an optional DL fallback (requires python-doctr installed).")
    parser.add_argument("--remove-lines", dest="remove_lines", action="store_true", default=True,
                         help="Remove ruled/notebook lines before segmentation (default on).")
    parser.add_argument("--no-remove-lines", dest="remove_lines", action="store_false",
                         help="Disable ruled-line removal.")
    parser.add_argument("--input-dir", default=None, help="Override raw_scans/ directory.")
    parser.add_argument("--output-dir", default=None, help="Override segmented/ directory.")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = args.input_dir or os.path.join(repo_root, "raw_scans")
    output_dir = args.output_dir or os.path.join(repo_root, "segmented")
    debug_dir = os.path.join(output_dir, "debug")

    if args.engine == "doctr":
        try:
            import doctr  # noqa: F401
        except ImportError:
            print("docTR not installed, install with pip install python-doctr")
            sys.exit(1)
        print("docTR engine is not yet implemented in this pipeline (classical is primary). Exiting.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    all_files = sorted(
        f for f in glob.glob(os.path.join(input_dir, "*"))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    )

    if not all_files:
        print(f"No image files found in {input_dir}.")
        print("Run scripts/make_test_image.py to generate a synthetic test scan, "
              "or drop a scan (.png/.jpg/.jpeg/.tif/.bmp/.heic) into raw_scans/.")
        sys.exit(0)

    manifest_rows = []
    total_crops = 0
    processed_count = 0

    for path in all_files:
        print(f"Processing: {path}")
        n = process_image(path, output_dir, debug_dir, args.granularity, args.remove_lines, manifest_rows)
        total_crops += n
        processed_count += 1

    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_path", "source_scan", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        writer.writerows(manifest_rows)

    print()
    print("=== Summary ===")
    print(f"Source images processed: {processed_count}")
    print(f"Crops written: {total_crops}")
    print(f"Granularity: {args.granularity}")
    print(f"Manifest: {manifest_path}")
    print(f"Debug overlays: {debug_dir}")


if __name__ == "__main__":
    main()
