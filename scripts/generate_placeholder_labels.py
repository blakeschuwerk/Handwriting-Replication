#!/usr/bin/env python3
"""
generate_placeholder_labels.py

Generates project-root `labels.csv` with PLACEHOLDER text labels for every
crop listed in `segmented/manifest.csv`.

IMPORTANT / HONESTY NOTE:
There is no ground-truth human transcription of the handwriting crops
produced by C3's segmentation step. C3's word-segmentation does not map
1:1 onto actual source words (it is a bounding-box heuristic), so we
cannot honestly claim these labels represent what is actually written in
each crop. Instead we deterministically assign a simple lowercase
dictionary word (guaranteed alphabet-safe) to each crop, keyed by the
crop's filename so the mapping is stable and reproducible. This lets
`scripts/package_dataset.py` and `scripts/validate_dataset.py` exercise
the full HDF5 packaging + FW-GAN loader PLUMBING end-to-end. It proves
the pipeline works, NOT transcription accuracy.

Usage:
    ./.venv/bin/python scripts/generate_placeholder_labels.py
"""
import csv
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "segmented", "manifest.csv")
WORDS_PATH = os.path.join(ROOT, "models", "FW_GAN", "data", "english_words.txt")
OUT_PATH = os.path.join(ROOT, "labels.csv")


def load_lowercase_words(path):
    """Load words consisting only of lowercase a-z (guaranteed alphabet-safe,
    non-empty). The source file is NOT guaranteed to be pure lowercase a-z
    (it contains entries like '10-point', '2D', etc.), so we filter here."""
    words = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            if w and w.isalpha() and w.islower() and w.isascii():
                words.append(w)
    if not words:
        raise RuntimeError(f"No usable lowercase words found in {path}")
    return words


def main():
    if not os.path.isfile(MANIFEST_PATH):
        print(f"ERROR: manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)

    words = load_lowercase_words(WORDS_PATH)
    print(f"Loaded {len(words)} candidate lowercase placeholder words from {WORDS_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("ERROR: manifest.csv has no data rows", file=sys.stderr)
        sys.exit(1)

    out_rows = []
    for row in rows:
        crop_path = row["crop_path"]
        # Deterministic word choice keyed by filename (stable across reruns).
        basename = os.path.basename(crop_path)
        digest = hashlib.sha256(basename.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % len(words)
        word = words[idx]
        out_rows.append({"image_path": crop_path, "text": word, "writer_id": "me"})

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text", "writer_id"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} placeholder label rows to {OUT_PATH}")
    print("NOTE: these are PLACEHOLDER labels (no ground-truth transcription exists).")


if __name__ == "__main__":
    main()
