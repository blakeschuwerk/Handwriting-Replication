#!/usr/bin/env python3
"""
label_tool.py

Standard-library CLI for transcribing handwriting crops produced by the
C1/C3 segmentation pipeline (segmented/manifest.csv) into a labels.csv
suitable for downstream handwriting-synthesis training.

LABELING MODE (default):
    Reads segmented/manifest.csv, and for every crop_path not already
    present in labels.csv (matched by absolute path -- the RESUME set),
    displays the crop image (macOS `open`) and prompts the human to type
    the transcription. Each answer is appended to labels.csv IMMEDIATELY
    and flushed, so an interrupted run (Ctrl-C, crash, closed terminal)
    loses at most the single in-progress entry -- re-running the tool
    picks up exactly where it left off, never re-prompting for crops
    that already have a row in labels.csv and never producing duplicate
    rows for the same image_path.

    An empty line is accepted as a valid label: some crops are spurious
    noise fragments (stray marks, ruled-line artifacts) with no text, and
    forcing a human to type something for those would corrupt the dataset
    more than an honest empty string does.

QA MODE (--qa):
    Loads labels.csv, randomly samples ~10% of rows (minimum 1 row when
    at least one row exists), re-displays each sampled crop, and asks the
    human to re-type the transcription from scratch. The re-typed answer
    is compared to the originally stored answer with EXACT string
    equality (no trimming/casefolding/normalization) -- the point of QA
    is to check whether the same human would type the exact same thing
    twice, which is a proxy for "is this transcription actually legible
    and unambiguous". Any mismatch is printed to the console and written
    to qa_report.csv. QA is read-only with respect to labels.csv: it
    never edits or "corrects" a stored label, it only reports.

Both modes support --no-display (skip opening the image viewer -- useful
for automated testing) and --manifest / --labels path overrides.
"""

import argparse
import csv
import os
import random
import subprocess
import sys

DEFAULT_MANIFEST = "/Users/blakey5aces/Handwriting Analysis/segmented/manifest.csv"
DEFAULT_LABELS = "/Users/blakey5aces/Handwriting Analysis/labels.csv"
DEFAULT_QA_REPORT = "/Users/blakey5aces/Handwriting Analysis/qa_report.csv"

LABELS_HEADER = ["image_path", "text", "writer_id"]
QA_HEADER = ["image_path", "original_text", "reentered_text", "status"]


def display_crop(image_path, no_display):
    """Open the crop image in the default macOS viewer. Never crashes --
    a failure to display (missing `open`, headless env, etc.) is only a
    warning; labeling can still proceed blind if the human insists."""
    if no_display:
        return
    try:
        subprocess.run(["open", image_path], check=False)
    except Exception as exc:  # noqa: BLE001 -- display failure must never crash
        print(f"WARNING: could not display '{image_path}' ({exc})")


def read_manifest(manifest_path):
    """Return the list of abspath-normalized crop_path values from the
    manifest, in manifest order."""
    crops = []
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop_path = row.get("crop_path")
            if crop_path:
                crops.append(os.path.abspath(crop_path))
    return crops


def read_labeled_set(labels_path):
    """Return the set of abspath-normalized image_path values already
    present in labels.csv (the RESUME set). Empty set if the file does
    not exist yet."""
    labeled = set()
    if not os.path.exists(labels_path):
        return labeled
    with open(labels_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row.get("image_path")
            if image_path:
                labeled.add(os.path.abspath(image_path))
    return labeled


def run_labeling(manifest_path, labels_path, no_display, input_fn=input):
    """Label every crop in the manifest not already present in labels.csv.

    input_fn is injectable so an automated test can supply a scripted
    "user" instead of reading real stdin.
    """
    crops = read_manifest(manifest_path)
    if not crops:
        print(f"Manifest '{manifest_path}' has no crops. Nothing to label. Exiting.")
        return

    labeled_set = read_labeled_set(labels_path)
    pending = [c for c in crops if c not in labeled_set]

    if not pending:
        print(
            f"All {len(crops)} crops already have labels in '{labels_path}'. "
            "Nothing to do."
        )
        return

    file_exists = os.path.exists(labels_path)
    need_header = not file_exists or os.path.getsize(labels_path) == 0

    print(
        f"Labeling {len(pending)} of {len(crops)} crops "
        f"({len(crops) - len(pending)} already done, resuming)."
    )

    for crop_path in pending:
        if not os.path.exists(crop_path):
            print(f"WARNING: crop file does not exist, skipping: {crop_path}")
            continue

        display_crop(crop_path, no_display)

        try:
            text = input_fn(f"Transcribe [{os.path.basename(crop_path)}]: ")
        except EOFError:
            print("\nInput closed early; stopping labeling run.")
            break

        # Append immediately and flush so a crash loses at most this row.
        with open(labels_path, "a", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(LABELS_HEADER)
                need_header = False
            writer.writerow([crop_path, text, "me"])
            f.flush()
            os.fsync(f.fileno())

    print(f"Labeling session complete. Labels stored in '{labels_path}'.")


def read_all_labels(labels_path):
    """Return the list of label rows (dicts) from labels.csv, or an empty
    list if the file does not exist / has no rows."""
    if not os.path.exists(labels_path):
        return []
    with open(labels_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_qa(labels_path, qa_report_path, no_display, input_fn=input, rng=random):
    """Sample ~10% (>=1) of labeled rows, re-prompt for the transcription,
    and flag any exact-string mismatch to console + qa_report.csv."""
    rows = read_all_labels(labels_path)
    if not rows:
        print(f"No labels found in '{labels_path}'. Nothing to QA.")
        return

    n = len(rows)
    sample_size = max(1, round(0.10 * n))
    sample_size = min(sample_size, n)
    sampled = rng.sample(rows, sample_size)

    print(f"QA: sampling {sample_size} of {n} labeled rows.")

    qa_rows = []
    mismatch_count = 0

    for row in sampled:
        image_path = row["image_path"]
        original_text = row["text"]

        if not os.path.exists(image_path):
            print(f"WARNING: crop file does not exist, skipping QA for: {image_path}")
            continue

        display_crop(image_path, no_display)

        try:
            reentered = input_fn(
                f"Re-transcribe [{os.path.basename(image_path)}]: "
            )
        except EOFError:
            print("\nInput closed early; stopping QA run.")
            break

        if reentered == original_text:
            status = "OK"
        else:
            status = "MISMATCH"
            mismatch_count += 1
            print(
                f"MISMATCH: {image_path}\n"
                f"  original : {original_text!r}\n"
                f"  reentered: {reentered!r}"
            )

        qa_rows.append([image_path, original_text, reentered, status])

    with open(qa_report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(QA_HEADER)
        writer.writerows(qa_rows)

    print(
        f"QA complete. {mismatch_count} mismatch(es) out of {len(qa_rows)} "
        f"sampled row(s). Report written to '{qa_report_path}'."
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Transcribe/QA handwriting crops into labels.csv."
    )
    parser.add_argument(
        "--qa", action="store_true", help="Run QA mode instead of labeling mode."
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not attempt to open crop images (useful for automated tests).",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Path to segmented manifest.csv (labeling mode).",
    )
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help="Path to labels.csv (read/write).",
    )
    parser.add_argument(
        "--qa-report",
        default=DEFAULT_QA_REPORT,
        help="Path to qa_report.csv (QA mode output).",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.qa:
        run_qa(args.labels, args.qa_report, args.no_display)
    else:
        run_labeling(args.manifest, args.labels, args.no_display)


if __name__ == "__main__":
    main()
