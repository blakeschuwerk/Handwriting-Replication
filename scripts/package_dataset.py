#!/usr/bin/env python3
"""
package_dataset.py

Packages `labels.csv` (image_path,text,writer_id) + the crop images it
references into `dataset/dataset.h5`, matching the FW-GAN Hdf5Dataset
schema EXACTLY (reverse-engineered from models/FW_GAN/data/test.hdf5,
HF rev 44244c6):

    imgs           (32, sum(W))  uint8   all word images concatenated
                                         horizontally, height fixed at 32
    img_lens       (N,)         int16   width of each sample
    img_seek_idxs  (N,)         int64   start column of each sample in imgs
    lbs            (sum(L),)   int32   all label texts concatenated as
                                         Unicode code points (ord(char))
    lb_lens        (N,)         int16   character length of each label
    lb_seek_idxs   (N,)         int64   start index of each label in lbs
    wids           (N,)         int16   writer id per sample (0-indexed)

Polarity: white background (255), dark ink (low values) — matches
test.hdf5 (mean ~217.6). Stored as-is; only inverted defensively if a
crop's grayscale mean < 110 (should never trigger on these crops).

This script performs a SINGLE lockstep pass: for every kept sample, the
image array, label codepoints, and writer id are appended together in
the same loop iteration, so there is no possibility of img/label/wid
arrays drifting out of alignment relative to each other.

Usage:
    ./.venv/bin/python scripts/package_dataset.py
"""
import csv
import os
import sys

import h5py
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH = os.path.join(ROOT, "labels.csv")
DATASET_DIR = os.path.join(ROOT, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "dataset.h5")
DATASET_TMP_PATH = DATASET_PATH + ".tmp"
SCHEMA_DOC_PATH = os.path.join(DATASET_DIR, "DATASET_SCHEMA.md")

sys.path.insert(0, os.path.join(ROOT, "models", "FW_GAN"))
from lib.alphabet import Alphabets  # noqa: E402

ALPHABET = Alphabets["all"]
ALPHABET_SET = set(ALPHABET)

TARGET_HEIGHT = 32
INT16_MAX = 32767

# Typographic-variant normalization map (documented per instructions).
NORMALIZATION_MAP = {
    "’": "'",  # right single quotation mark
    "‘": "'",  # left single quotation mark
    "“": '"',  # left double quotation mark
    "”": '"',  # right double quotation mark
    "–": "-",  # en dash
    "—": "-",  # em dash
    "…": "...",  # ellipsis
}


def normalize_text(text):
    out = []
    for ch in text:
        out.append(NORMALIZATION_MAP.get(ch, ch))
    return "".join(out)


def load_and_prepare_image(path, log):
    """Load crop as grayscale uint8 numpy array, apply polarity guard,
    resize to fixed height 32 preserving aspect ratio. Returns (arr, note)
    where arr has shape (32, new_w) dtype uint8."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    note = None
    if arr.mean() < 110:
        arr = 255 - arr
        note = "inverted (mean<110)"

    h, w = arr.shape
    new_w = max(8, round(w * TARGET_HEIGHT / h))
    img_resized = Image.fromarray(arr, mode="L").resize(
        (new_w, TARGET_HEIGHT), Image.Resampling.LANCZOS
    )
    arr_resized = np.array(img_resized, dtype=np.uint8)
    return arr_resized, note


def main():
    if not os.path.isfile(LABELS_PATH):
        print(f"ERROR: labels.csv not found at {LABELS_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(LABELS_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("ERROR: labels.csv has no data rows", file=sys.stderr)
        sys.exit(1)

    # Deterministic writer_id mapping: sorted-unique string -> int index.
    unique_writers = sorted({row["writer_id"] for row in rows})
    writer_to_id = {w: i for i, w in enumerate(unique_writers)}
    for wid_int in writer_to_id.values():
        assert wid_int < INT16_MAX, "writer id overflow"

    img_arrays = []
    widths = []
    all_codes = []
    text_lens = []
    wids = []

    packaged = 0
    skipped = []  # list of (path, reason)

    for row in rows:
        image_path = row["image_path"]
        raw_text = row["text"]
        writer_str = row["writer_id"]
        basename = os.path.basename(image_path)

        if not os.path.isfile(image_path):
            skipped.append((basename, f"file not found: {image_path}"))
            continue

        # --- text validation/normalization ---
        norm_text = normalize_text(raw_text)

        if norm_text.strip() == "":
            skipped.append((basename, "empty/whitespace-only label after normalization"))
            continue

        offending = sorted({ch for ch in norm_text if ch not in ALPHABET_SET})
        if offending:
            skipped.append(
                (basename, f"chars not in alphabet: {offending!r} (text={norm_text!r})")
            )
            continue

        if len(norm_text) >= INT16_MAX:
            skipped.append((basename, "label length exceeds int16 range"))
            continue

        # --- image load/prepare ---
        try:
            arr, note = load_and_prepare_image(image_path, skipped)
        except Exception as e:  # noqa: BLE001
            skipped.append((basename, f"image load/resize error: {e}"))
            continue

        new_w = arr.shape[1]
        if new_w >= INT16_MAX:
            skipped.append((basename, "image width exceeds int16 range"))
            continue

        wid_int = writer_to_id[writer_str]

        # --- lockstep append: image, label codes, wid together ---
        img_arrays.append(arr)
        widths.append(new_w)
        codes = [ord(c) for c in norm_text]
        all_codes.extend(codes)
        text_lens.append(len(norm_text))
        wids.append(wid_int)

        packaged += 1
        if note:
            print(f"  [note] {basename}: {note}")

    if packaged == 0:
        print("ERROR: 0 valid samples after filtering; not writing dataset.h5", file=sys.stderr)
        for basename, reason in skipped:
            print(f"  SKIPPED {basename}: {reason}", file=sys.stderr)
        sys.exit(1)

    imgs = np.concatenate(img_arrays, axis=1).astype(np.uint8)
    img_lens = np.array(widths, dtype=np.int16)
    img_seek_idxs = np.concatenate(
        [[0], np.cumsum(img_lens.astype(np.int64))[:-1]]
    ).astype(np.int64)

    lbs = np.array(all_codes, dtype=np.int32)
    lb_lens = np.array(text_lens, dtype=np.int16)
    lb_seek_idxs = np.concatenate(
        [[0], np.cumsum(lb_lens.astype(np.int64))[:-1]]
    ).astype(np.int64)

    wids_arr = np.array(wids, dtype=np.int16)

    # Sanity asserts before write.
    assert imgs.dtype == np.uint8
    assert imgs.shape[0] == TARGET_HEIGHT
    assert int(img_seek_idxs[-1]) + int(img_lens[-1]) == imgs.shape[1]
    assert int(lb_seek_idxs[-1]) + int(lb_lens[-1]) == lbs.shape[0]
    n = packaged
    assert len(img_lens) == n
    assert len(img_seek_idxs) == n
    assert len(lb_lens) == n
    assert len(lb_seek_idxs) == n
    assert len(wids_arr) == n

    os.makedirs(DATASET_DIR, exist_ok=True)

    # Atomic write: tmp file then os.replace.
    if os.path.exists(DATASET_TMP_PATH):
        os.remove(DATASET_TMP_PATH)
    with h5py.File(DATASET_TMP_PATH, "w") as h5f:
        h5f.create_dataset("imgs", data=imgs, dtype=np.uint8)
        h5f.create_dataset("img_lens", data=img_lens, dtype=np.int16)
        h5f.create_dataset("img_seek_idxs", data=img_seek_idxs, dtype=np.int64)
        h5f.create_dataset("lbs", data=lbs, dtype=np.int32)
        h5f.create_dataset("lb_lens", data=lb_lens, dtype=np.int16)
        h5f.create_dataset("lb_seek_idxs", data=lb_seek_idxs, dtype=np.int64)
        h5f.create_dataset("wids", data=wids_arr, dtype=np.int16)
    os.replace(DATASET_TMP_PATH, DATASET_PATH)

    # Write schema doc.
    schema_lines = [
        "# dataset/dataset.h5 — Schema Documentation",
        "",
        "Reverse-engineered from the real `models/FW_GAN/data/test.hdf5`",
        "(HuggingFace revision 44244c6), loaded via FW-GAN's own",
        "`lib.datasets.Hdf5Dataset`. This file matches that schema exactly",
        "so it can be loaded through the unmodified FW-GAN loader.",
        "",
        "## Datasets (7 total)",
        "",
        "| dataset | shape | dtype | meaning |",
        "|---|---|---|---|",
        f"| `imgs` | `(32, {imgs.shape[1]})` | uint8 | All word images concatenated horizontally, fixed height 32. |",
        "| `img_lens` | `(N,)` | int16 | Width (columns) of each sample. |",
        "| `img_seek_idxs` | `(N,)` | int64 | Start column of each sample in `imgs` (prefix sum of `img_lens`). |",
        "| `lbs` | `(sum(L),)` | int32 | All label texts concatenated as Unicode code points (`ord(char)`). |",
        "| `lb_lens` | `(N,)` | int16 | Character length of each label. |",
        "| `lb_seek_idxs` | `(N,)` | int64 | Start index of each label in `lbs` (prefix sum of `lb_lens`). |",
        "| `wids` | `(N,)` | int16 | Writer id per sample, 0-indexed integer. |",
        "",
        "## Polarity",
        "",
        "White background (255), dark ink (low values) — NOT inverted.",
        "Matches reference `test.hdf5` (mean ~217.6, mostly-white background).",
        "Defensive guard: any crop with grayscale mean < 110 is inverted and",
        "logged (did not trigger for this dataset's crops).",
        "",
        "## Alphabet",
        "",
        f"Fixed 81-char alphabet from `models/FW_GAN/lib/alphabet.py` `Alphabets['all']`:",
        f"`{ALPHABET}`",
        "",
        "## Label provenance — IMPORTANT HONESTY NOTE",
        "",
        "The text labels in this dataset are PLACEHOLDER labels generated by",
        "`scripts/generate_placeholder_labels.py`. There is no ground-truth",
        "human transcription of the handwriting crops (C3's word-segmentation",
        "does not map 1:1 onto actual source words). This dataset packaging",
        "proves the HDF5 plumbing / FW-GAN loader compatibility, NOT",
        "transcription accuracy.",
        "",
        "## Packaging run summary",
        "",
        f"- Samples packaged: {packaged}",
        f"- Samples skipped: {len(skipped)}",
    ]
    if skipped:
        schema_lines.append("")
        schema_lines.append("### Skip reasons")
        schema_lines.append("")
        for basename, reason in skipped:
            schema_lines.append(f"- `{basename}`: {reason}")
    schema_lines.append("")
    schema_lines.append(f"- Final `imgs` shape: {imgs.shape}, dtype: {imgs.dtype}")
    schema_lines.append("")

    with open(SCHEMA_DOC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(schema_lines))

    # Print summary.
    print()
    print("=" * 60)
    print("PACKAGING SUMMARY")
    print("=" * 60)
    print(f"Packaged: {packaged}")
    print(f"Skipped:  {len(skipped)}")
    for basename, reason in skipped:
        print(f"  SKIPPED {basename}: {reason}")
    print(f"imgs shape: {imgs.shape}, dtype: {imgs.dtype}")
    print(f"Wrote: {DATASET_PATH}")
    print(f"Wrote: {SCHEMA_DOC_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
