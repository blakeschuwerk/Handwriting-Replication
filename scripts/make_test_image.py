#!/usr/bin/env python3
"""
make_test_image.py

Generates a synthetic "handwritten scan" test image so the segmentation
pipeline (segment.py) can be exercised end-to-end without waiting on a
real phone photo. Simulates:
  - a slightly uneven cream/off-white page (gradient + noise, like uneven
    phone-camera lighting)
  - 4-6 lines of text rendered in a macOS handwriting-style font
  - a small rotation (simulating a skewed phone photo)
  - mild gaussian noise (simulating camera sensor noise)

Output: raw_scans/sample_synth.png

Ground truth text is kept in GROUND_TRUTH_LINES below so a human (or a
future OCR-accuracy script) can compare recognized text against it.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Ground truth content (inspectable) -- what the rendered page "says".
# ---------------------------------------------------------------------------
GROUND_TRUTH_LINES = [
    "The quick brown fox jumps over the lazy dog.",
    "Handwriting synthesis needs clean word crops.",
    "Segmentation must survive skewed phone photos.",
    "Ruled lines and noise should not break detection.",
    "Every word box should land on an actual word.",
]

# Candidate handwriting-style fonts available on this machine (macOS).
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
    "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
    "/System/Library/Fonts/Supplemental/Brush Script.ttf",
]

FONT_SIZE = 48
PAGE_W, PAGE_H = 1700, 2200  # roughly a letter-page aspect ratio
LINE_SPACING = 90
LEFT_MARGIN = 140
TOP_MARGIN = 260
ROTATION_DEG = 4.5          # small skew simulating a hand-held photo
GAUSSIAN_NOISE_SIGMA = 6.0  # mild sensor-noise-like grain

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_scans")
OUT_PATH = os.path.join(OUT_DIR, "sample_synth.png")


def pick_font():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, FONT_SIZE), path
            except Exception:
                continue
    # Fallback to PIL default bitmap font (ugly, but keeps pipeline runnable).
    return ImageFont.load_default(), None


def make_cream_background(w, h, rng):
    """Cream/off-white page with a slight gradient + fine noise to mimic
    uneven phone-camera lighting."""
    base = np.zeros((h, w, 3), dtype=np.float32)
    # Base cream color
    cream = np.array([250, 246, 235], dtype=np.float32)
    base[:, :] = cream

    # Add a soft diagonal lighting gradient (darker corner -> lighter corner)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (xx / w * 0.6 + yy / h * 0.4)
    grad = (grad - grad.min()) / (grad.max() - grad.min())
    shading = (1.0 - 0.12 * grad)  # up to ~12% darkening across the page
    base *= shading[:, :, None]

    # Fine per-pixel noise to simulate uneven lighting/texture
    noise = rng.normal(0, 4.0, size=(h, w, 1)).astype(np.float32)
    base += noise

    base = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(base, mode="RGB")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    font, font_path = pick_font()

    img = make_cream_background(PAGE_W, PAGE_H, rng)
    draw = ImageDraw.Draw(img)

    ink_color = (35, 35, 45)  # dark blue-black ink
    y = TOP_MARGIN
    for line in GROUND_TRUTH_LINES:
        # tiny per-line jitter in x/y to look less robotic
        jx = int(rng.normal(0, 3))
        jy = int(rng.normal(0, 2))
        draw.text((LEFT_MARGIN + jx, y + jy), line, font=font, fill=ink_color)
        y += LINE_SPACING

    # Rotate slightly to simulate a skewed phone photo. Expand + fill with
    # background-like color so the crop still looks like a page.
    img = img.rotate(ROTATION_DEG, expand=True, fillcolor=(248, 244, 232), resample=Image.BICUBIC)

    # Add mild gaussian noise (sensor grain) on top of everything.
    arr = np.array(img).astype(np.float32)
    noise = rng.normal(0, GAUSSIAN_NOISE_SIGMA, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    img.save(OUT_PATH)
    print(f"Wrote synthetic test scan to: {OUT_PATH}")
    print(f"Page size: {img.size[0]}x{img.size[1]}, rotation: {ROTATION_DEG} deg")
    print(f"Font used: {font_path or '(PIL default bitmap font - no system font found)'}")
    print("Ground truth lines:")
    for i, line in enumerate(GROUND_TRUTH_LINES, 1):
        print(f"  L{i}: {line}")


if __name__ == "__main__":
    main()
