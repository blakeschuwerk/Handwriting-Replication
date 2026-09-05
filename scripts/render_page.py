#!/usr/bin/env python3
"""
render_page.py -- C8 output-rendering component.

Stitches individual word-image PNGs (produced by C7's generate.py) into
full readable lines/pages of "handwriting" and exports as an image or PDF.

This is deliberately a plain raster-layout job using Pillow only -- no
custom layout engine, no ML, no cleverness. Two input modes:

  1. --words <path> [<path> ...]
       An explicit ordered list of word-image PNGs. Order is preserved
       exactly as given (never sorted). Each image may have an arbitrary
       height and is resized to a common target height on load.

  2. --text "<string>"
       The script invokes C7's generate.py (in a fresh scratch output
       directory) to synthesize word images for the given text, then
       reads them back in word order and lays them out the same way.

Layout: words are packed left-to-right with a fixed gap, wrapping to a
new line when a word would overflow the usable page width, and
paginating (starting a new page) when a line would overflow the usable
page height. Oversized single words (wider than the whole usable width)
are shrunk to fit rather than allowed to overflow.

Output: a single-page image, a multi-page image (one file per page), or
a single multi-page PDF, depending on the --output extension and how
many pages were needed.
"""

import argparse
import os
import subprocess
import sys
import time
from glob import glob

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# This script lives at PROJECT/scripts/render_page.py -- anchor all relative
# paths (styles, ckpt, output dirs, --words paths, etc.) to the project root
# rather than to the caller's current working directory.
PROJECT = "/Users/blakey5aces/Handwriting Analysis"
GENERATE_PY = os.path.join(PROJECT, "scripts", "generate.py")

# Explicit venv interpreter. sys.executable is wrong here because this
# script may be invoked by a different Python than the shared project venv;
# generate.py's dependencies (torch, etc.) only exist in PROJECT/.venv.
VENV_PYTHON = os.path.join(PROJECT, ".venv", "bin", "python3")

WHITE = 255  # 'L' mode background/fill value


def resolve(path):
    """Resolve a possibly-relative path against the project root."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(PROJECT, path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Lay out word-image PNGs into page(s) of handwriting output."
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--words", nargs="+", metavar="PNG",
        help="Explicit ordered list of word-image PNG paths (order preserved exactly).",
    )
    mode.add_argument(
        "--text", type=str,
        help="Text string to synthesize via generate.py, then lay out.",
    )

    p.add_argument("--output", default="output/page.pdf",
                    help="Output path. '.pdf' extension -> multi-page PDF; "
                         "otherwise treated as an image (e.g. .png).")
    p.add_argument("--page-width", type=int, default=1200)
    p.add_argument("--page-height", type=int, default=1600)
    p.add_argument("--margin", type=int, default=60)
    p.add_argument("--word-gap", type=int, default=24,
                    help="Horizontal pixel gap between words on a line.")
    p.add_argument("--line-gap", type=int, default=28,
                    help="Vertical pixel gap between lines.")
    p.add_argument("--word-height", type=int, default=64,
                    help="Resize target height (px) for each word image.")
    p.add_argument("--keep-background", action="store_true",
                    help="Paste each word with its original grey background instead "
                         "of matting it out (for before/after comparison).")

    # generate.py passthroughs (only used in --text mode)
    p.add_argument("--style", nargs="+", metavar="PNG",
                    help="Style-reference crop(s), passed through to generate.py.")
    p.add_argument("--ckpt", type=str, help="Checkpoint path, passed through to generate.py.")
    p.add_argument("--device", type=str, default="mps",
                    help="Device for generate.py (mps/cpu/cuda).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# --text mode: invoke generate.py and collect resulting word images
# ---------------------------------------------------------------------------

def run_generate_and_collect_words(args):
    """Run C7's generate.py into a fresh scratch subdir and return the
    ordered list of word-image paths it produced (excluding preview files)."""

    # Fresh per-run subdir avoids scraping stale files from previous runs
    # (including other agents' test outputs sitting in output/) and avoids
    # clobbering a concurrently-running invocation.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    subdir_name = f"_render_words_{timestamp}_{os.getpid()}"
    subdir = os.path.join(PROJECT, "output", subdir_name)
    os.makedirs(subdir, exist_ok=True)

    cmd = [
        VENV_PYTHON, GENERATE_PY,
        "--text", args.text,
        "--device", args.device,
        "--output-dir", subdir,
    ]
    if args.style:
        cmd.append("--style")
        cmd.extend(resolve(s) for s in args.style)
    if args.ckpt:
        cmd.extend(["--ckpt", resolve(args.ckpt)])

    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}

    print(f"[render_page] Running generate.py: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Surface generate.py's own output for visibility/debugging.
    if result.stdout:
        print("[generate.py stdout]\n" + result.stdout)
    if result.returncode != 0:
        print("[generate.py stderr]\n" + result.stderr, file=sys.stderr)
        sys.exit(1)
    elif result.stderr:
        # Non-fatal stderr (warnings etc.) -- still show it.
        print("[generate.py stderr]\n" + result.stderr, file=sys.stderr)

    # generate.py names word files "{idx:03d}_{safe}.png" enumerated in word
    # order, plus a "_line.png" preview strip that is NOT a word and must be
    # excluded. Any other underscore-prefixed file is excluded defensively
    # for the same reason. Lexical sort of the zero-padded prefix == word order.
    all_pngs = sorted(glob(os.path.join(subdir, "*.png")))
    word_pngs = [
        f for f in all_pngs
        if not os.path.basename(f).startswith("_")
    ]

    if not word_pngs:
        print(f"[render_page] ERROR: generate.py produced no word images in {subdir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[render_page] Collected {len(word_pngs)} word image(s) from {subdir}")
    return word_pngs


# ---------------------------------------------------------------------------
# Loading word images
# ---------------------------------------------------------------------------

# Matte shaping constants, chosen by measuring a full generated page rather
# than by eye. Two earlier attempts were measured and rejected: a local
# background estimate looked better but scored 48% WORSE (it tracked the paper
# texture instead of ignoring it), and a gentler toe left a visible film
# because the first metric only sampled pixels very close to paper and very
# deep in ink -- the haze lives between those, so it went unmeasured.
# Scored against the honest question "what share of the rectangle carries any
# cover at all": 20.3% vs 19.9% of pixels that are genuinely ink, with cover on
# the darkest pixels at 0.995. Effectively the strokes and nothing else.
BG_PCT = 96   # percentile taken as this word's paper level
TOE = 0.30    # cover below this is paper film, not ink -- this is what kills the box
GAIN = 2.2    # re-solidify strokes after the cut


def ink_alpha(im, keep_texture=True):
    """Build a transparency mask that isolates the ink from its background.

    THE PROBLEM THIS SOLVES
    -----------------------
    The generator emits a plain grayscale rectangle with NO alpha channel, and
    its background sits around 193/255 -- light grey, not white. Pasting that
    rectangle onto the page pasted the background with it, so every word sat in
    a visible grey box.

    WHY A SOFT MATTE AND NOT A THRESHOLD
    ------------------------------------
    Hard-thresholding would cut the strokes out with jagged binary edges and
    throw away the antialiased halo that makes the writing read as pencil. So
    brightness is mapped CONTINUOUSLY onto alpha: paper-level pixels go fully
    transparent, the darkest ink goes fully opaque, and everything between
    keeps a proportional amount of cover. The original grey values are left
    untouched underneath -- whatever texture the model produced, pencil or pen,
    survives exactly as generated. This decides only what shows through, never
    what colour it is.

    Levels are measured per word, because each generated image carries its own
    background level rather than a fixed one.
    """
    a = np.asarray(im, dtype=np.float32)
    bg = float(np.percentile(a, BG_PCT))  # this word's paper level
    ink = float(np.percentile(a, 2))     # this word's darkest ink
    span = max(8.0, bg - ink)            # guard: a blank image must not blow up
    alpha = np.clip((bg - a) / span, 0.0, 1.0)

    # TOE + GAIN. A flat background level alone still left a faint rectangular
    # haze: the model paints paper texture (shading, ruled-line ghosts) sitting
    # just below paper level, so a little cover survived across the whole
    # rectangle and read as a grey box. The haze is thin, even cover while a
    # real stroke edge carries far more, so cutting low cover removes the film
    # and leaves the antialiasing that makes it read as pencil, not a cutout.
    alpha = np.clip((alpha - TOE) / (1.0 - TOE), 0.0, 1.0) * GAIN
    alpha = np.clip(alpha, 0.0, 1.0)

    if not keep_texture:
        alpha = (alpha > 0.35).astype(np.float32)
    return Image.fromarray((alpha * 255.0).astype(np.uint8), mode="L")


def load_word_images(paths, word_height, matte=True):
    """Open each path, convert to grayscale, resize to a common target
    height (preserving aspect ratio). Skips (with a warning) any file that
    fails to open, rather than crashing the whole run."""
    images = []
    for path in paths:
        try:
            im = Image.open(path).convert("L")
        except Exception as e:
            print(f"[render_page] WARNING: could not open '{path}' ({e}); skipping.",
                  file=sys.stderr)
            continue

        w, h = im.size
        if h <= 0:
            print(f"[render_page] WARNING: '{path}' has invalid height {h}; skipping.",
                  file=sys.stderr)
            continue

        alpha = ink_alpha(im) if matte else None

        new_w = max(1, round(w * word_height / h))
        if (new_w, word_height) != (w, h):
            im = im.resize((new_w, word_height), Image.LANCZOS)
            if alpha is not None:
                # resized with the image, or the mask no longer lines up
                alpha = alpha.resize((new_w, word_height), Image.LANCZOS)

        images.append((path, im, alpha))

    return images


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout_pages(word_images, page_width, page_height, margin, word_gap, line_gap, word_height):
    """Pack word images left-to-right, wrapping lines and paginating as
    needed. Returns a list of PIL 'L' page canvases."""

    usable_width = page_width - 2 * margin
    if usable_width < 1:
        print(f"[render_page] ERROR: usable_width ({usable_width}) < 1 -- "
              f"page-width {page_width} too small for margin {margin}.", file=sys.stderr)
        sys.exit(1)

    line_height = word_height  # nominal line height; shrunk words paste top-aligned within it

    def new_page():
        return Image.new("L", (page_width, page_height), color=WHITE)

    pages = [new_page()]
    x, y = margin, margin

    for path, img, alpha in word_images:
        w, h = img.size

        # Oversized-word guard: never let a single word exceed the usable
        # width. Shrink it further (preserving aspect ratio) so it fits.
        if w > usable_width:
            print(f"[render_page] WARNING: word image '{path}' width {w}px exceeds "
                  f"usable page width {usable_width}px; shrinking to fit.", file=sys.stderr)
            new_w = usable_width
            new_h = max(1, round(h * new_w / w))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            if alpha is not None:
                alpha = alpha.resize((new_w, new_h), Image.LANCZOS)
            w, h = img.size

        # Wrap to a new line if this word won't fit after the current one
        # (only applies if this isn't already the first word on the line --
        # a first word either fits by construction above, or was shrunk to).
        if x != margin and (x + word_gap + w) > (page_width - margin):
            x = margin
            y += line_height + line_gap

        # Paginate if this line doesn't fit vertically on the current page.
        if y + line_height > page_height - margin:
            pages.append(new_page())
            x, y = margin, margin

        if x != margin:
            x += word_gap

        # the mask is what keeps the grey box off the page
        pages[-1].paste(img, (x, y), alpha)
        x += w

    return pages


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_pages(pages, output_path):
    """Save the page canvases according to the --output extension:
    '.pdf' -> one multi-page PDF; otherwise -> image file(s), with
    '_p01', '_p02', ... suffixes if there's more than one page."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()

    saved_paths = []

    if ext == ".pdf":
        rgb_pages = [p.convert("RGB") for p in pages]
        rgb_pages[0].save(output_path, save_all=True, append_images=rgb_pages[1:])
        saved_paths.append(output_path)
    else:
        if len(pages) == 1:
            pages[0].save(output_path)
            saved_paths.append(output_path)
        else:
            stem, ext2 = os.path.splitext(output_path)
            for i, page in enumerate(pages, start=1):
                page_path = f"{stem}_p{i:02d}{ext2}"
                page.save(page_path)
                saved_paths.append(page_path)

    return saved_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_path = resolve(args.output)

    if args.words:
        word_paths = [resolve(p) for p in args.words]
    else:
        word_paths = run_generate_and_collect_words(args)

    word_images = load_word_images(word_paths, args.word_height,
                                   matte=not args.keep_background)

    if not word_images:
        print("[render_page] ERROR: no valid word images to lay out (all failed to load).",
              file=sys.stderr)
        sys.exit(1)

    pages = layout_pages(
        word_images,
        page_width=args.page_width,
        page_height=args.page_height,
        margin=args.margin,
        word_gap=args.word_gap,
        line_gap=args.line_gap,
        word_height=args.word_height,
    )

    saved_paths = save_pages(pages, output_path)

    print(f"[render_page] Done. {len(pages)} page(s) written:")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
