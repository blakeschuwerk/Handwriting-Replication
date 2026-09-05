#!/usr/bin/env python3
"""
validate_dataset.py — ACCEPTANCE TEST for dataset/dataset.h5

Loads dataset/dataset.h5 through FW-GAN's OWN loader class
(lib.datasets.Hdf5Dataset, unmodified, imported with models/FW_GAN on
sys.path) and checks:

  1. len(ds) >= 3
  2. For at least 3 samples: img shape (1,32,W), W>0, float dtype,
     values within [-1.001, 1.001] (Normalize([0.5],[0.5]) range).
  3. Label indices in range(len(alphabet)); len(lb) > 0.
  4. ROUND-TRIP TEXT CHECK: decode lb back to text via the alphabet and
     assert it equals the ORIGINAL label text read independently from
     labels.csv, using the exact same build order package_dataset.py
     used (so index i in the dataset lines up with the i-th KEPT row
     in labels.csv). This is the anti-misalignment guard.
  5. wid is a non-negative integer.
  6. DataLoader + Hdf5Dataset.collect_fn batch check: imgs 4-D,
     shape[1]==1, shape[2]==32; lbs.shape[0]==imgs.shape[0]; no NaN/Inf.

Prints VALIDATION PASSED and exits 0 on success; prints failure details
and exits non-zero otherwise.

Usage:
    cd "/Users/blakey5aces/Handwriting Analysis"
    PYTORCH_ENABLE_MPS_FALLBACK=1 ./.venv/bin/python scripts/validate_dataset.py
"""
import csv
import os
import sys
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models", "FW_GAN"))

LABELS_PATH = os.path.join(ROOT, "labels.csv")

# Same normalization map + alphabet-filter logic as package_dataset.py,
# duplicated here (independently) so we can reconstruct, from labels.csv
# alone, the exact ordered list of labels that ended up KEPT in the h5 file.
NORMALIZATION_MAP = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
}


def normalize_text(text):
    return "".join(NORMALIZATION_MAP.get(ch, ch) for ch in text)


def main():
    failures = []

    try:
        from torchvision.transforms import Compose, ToTensor, Normalize
        from lib.datasets import Hdf5Dataset
        from lib.alphabet import Alphabets
        import torch
        from torch.utils.data import DataLoader
        import numpy as np
    except Exception:
        print("FAILED to import required modules:")
        traceback.print_exc()
        sys.exit(1)

    alphabet = Alphabets["all"]
    alphabet_set = set(alphabet)

    # Reconstruct expected kept-label order from labels.csv independently.
    if not os.path.isfile(LABELS_PATH):
        print(f"FAILED: labels.csv not found at {LABELS_PATH}")
        sys.exit(1)

    with open(LABELS_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    expected_texts = []
    for row in rows:
        raw_text = row["text"]
        image_path = row["image_path"]
        norm_text = normalize_text(raw_text)
        if norm_text.strip() == "":
            continue
        if any(ch not in alphabet_set for ch in norm_text):
            continue
        if not os.path.isfile(image_path):
            continue
        expected_texts.append(norm_text)

    try:
        ds = Hdf5Dataset(
            root=os.path.join(ROOT, "dataset"),
            split="dataset.h5",
            transforms=Compose([ToTensor(), Normalize([0.5], [0.5])]),
            alphabet_key="all",
        )
    except Exception:
        print("FAILED to construct Hdf5Dataset:")
        traceback.print_exc()
        sys.exit(1)

    # Check 1: length
    n = len(ds)
    print(f"Dataset length: {n}")
    if n < 3:
        failures.append(f"len(ds)={n} < 3")

    if len(expected_texts) != n:
        failures.append(
            f"expected_texts from labels.csv ({len(expected_texts)}) != len(ds) ({n}) "
            "-- kept-row reconstruction mismatch"
        )

    # Checks 2-5: per-sample
    num_to_check = min(n, max(3, n))  # check all samples (dataset is small, ~43)
    checked = 0
    for i in range(num_to_check):
        try:
            img, lb, wid = ds[i]
        except Exception:
            failures.append(f"sample {i}: ds[{i}] raised an exception:\n{traceback.format_exc()}")
            continue

        # img checks
        if not hasattr(img, "shape") or len(img.shape) != 3:
            failures.append(f"sample {i}: img has unexpected shape {getattr(img, 'shape', None)}")
        else:
            c, h, w = img.shape
            if c != 1 or h != 32 or w <= 0:
                failures.append(f"sample {i}: img shape {img.shape} != (1,32,W>0)")
            if not torch.is_floating_point(img):
                failures.append(f"sample {i}: img dtype {img.dtype} is not floating point")
            img_min, img_max = float(img.min()), float(img.max())
            if img_min < -1.001 or img_max > 1.001:
                failures.append(f"sample {i}: img range [{img_min},{img_max}] outside [-1.001,1.001]")

        # label checks
        if not isinstance(lb, list) or len(lb) == 0:
            failures.append(f"sample {i}: lb is not a non-empty list: {lb!r}")
        else:
            if not all(isinstance(x, int) and 0 <= x < len(alphabet) for x in lb):
                failures.append(f"sample {i}: lb has out-of-range index: {lb!r}")

            # Round-trip text check (anti-misalignment guard).
            decoded = "".join(alphabet[idx] for idx in lb)
            expected = expected_texts[i] if i < len(expected_texts) else None
            if decoded != expected:
                failures.append(
                    f"sample {i}: ROUND-TRIP MISMATCH decoded={decoded!r} expected={expected!r}"
                )

        # wid check
        wid_val = int(wid)
        if wid_val < 0:
            failures.append(f"sample {i}: wid={wid_val} is negative")

        checked += 1

    print(f"Checked {checked} samples individually (shape/range/label-range/round-trip/wid).")

    # Check 6: DataLoader + collect_fn batch check.
    try:
        dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=Hdf5Dataset.collect_fn)
        imgs, img_lens, lbs, lb_lens, wids = next(iter(dl))

        if imgs.ndim != 4:
            failures.append(f"batch imgs.ndim={imgs.ndim} != 4")
        if imgs.shape[1] != 1:
            failures.append(f"batch imgs.shape[1]={imgs.shape[1]} != 1")
        if imgs.shape[2] != 32:
            failures.append(f"batch imgs.shape[2]={imgs.shape[2]} != 32")
        if lbs.shape[0] != imgs.shape[0]:
            failures.append(f"batch lbs.shape[0]={lbs.shape[0]} != imgs.shape[0]={imgs.shape[0]}")
        if torch.isnan(imgs).any():
            failures.append("batch imgs contain NaN")
        if torch.isinf(imgs).any():
            failures.append("batch imgs contain Inf")

        print(
            f"Batch check: imgs.shape={tuple(imgs.shape)}, "
            f"lbs.shape={tuple(lbs.shape)}, wids.shape={tuple(wids.shape)}"
        )
    except Exception:
        failures.append(f"DataLoader/collect_fn batch check raised:\n{traceback.format_exc()}")

    print()
    if failures:
        print("VALIDATION FAILED")
        print(f"{len(failures)} failure(s):")
        for f_ in failures:
            print(f" - {f_}")
        sys.exit(1)
    else:
        print("VALIDATION PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
