#!/usr/bin/env python3
"""
test_label_tool.py

Automated stdlib test for scripts/label_tool.py, proving the plumbing
required by the C4 acceptance criteria:

  (a) A full labeling run (via run_labeling with an injected input_fn,
      --no-display semantics) produces a correct labels.csv: exact
      header columns, one row per manifest crop, values equal to the
      scripted "user" input.
  (b) Killing/restarting mid-run resumes without re-asking already
      labeled rows: labeling only the first K crops, then re-running,
      only prompts for the remaining crops and does not duplicate or
      re-prompt the first K.
  (c) --qa (run_qa) correctly flags a deliberately-injected mismatch
      (false-negative direction: a wrong re-entry IS flagged) AND does
      NOT flag a correct re-entry (false-positive direction: a correct
      re-entry is NOT flagged). Uses a fixed random seed so the sample
      is deterministic.

IMPORTANT HONESTY NOTE ON WHAT THIS TEST PROVES:
This test uses a small synthetic manifest of real, on-disk 1x1 PNG
files with DETERMINISTIC PLACEHOLDER transcription strings keyed off
each crop's filename (e.g. "label_for_crop_0.png"). These placeholder
strings are NOT real handwriting transcriptions, and the real 43-crop
segmented/manifest.csv from C3 does not map 1:1 onto real dictionary
words (C3's heuristic segmentation merges/splits words arbitrarily).
This test therefore proves the tool's PLUMBING is correct -- file I/O,
CSV schema/quoting, resume-by-abspath semantics, crash-safety framing,
and QA exact-match mismatch detection -- it does NOT prove or attempt
to prove transcription accuracy against ground truth, because no such
ground truth exists for the real crops.
"""

import csv
import os
import random
import shutil
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import label_tool  # noqa: E402


# A minimal valid 1x1 PNG (67 bytes), used so os.path.exists() checks in
# label_tool succeed without needing Pillow or any image library.
MINIMAL_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844440000000100000001080600000"
    "01f15c4890000000a49444154789c6360000002000155a3c8a8000000"
    "0049454e44ae426082"
)


def make_synthetic_crop_set(root_dir, count=6):
    """Create `count` real tiny PNG files plus a manifest.csv referencing
    them via absolute paths, mirroring the columns produced by C3."""
    crops_dir = os.path.join(root_dir, "segmented")
    os.makedirs(crops_dir, exist_ok=True)

    manifest_path = os.path.join(root_dir, "manifest.csv")
    crop_paths = []
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["crop_path", "source_scan", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        )
        for i in range(count):
            crop_path = os.path.join(crops_dir, f"crop_{i}.png")
            with open(crop_path, "wb") as img:
                img.write(MINIMAL_PNG_BYTES)
            crop_paths.append(os.path.abspath(crop_path))
            writer.writerow(
                [crop_path, "/fake/source_scan.png", 0, 0, 10, 10]
            )
    return manifest_path, crop_paths


def placeholder_text(crop_path):
    """Deterministic placeholder "transcription" keyed by filename --
    NOT real transcription, see module docstring."""
    return f"label_for_{os.path.basename(crop_path)}"


def scripted_input_fn(answers):
    """Return an input_fn that pops answers off a queue in order,
    ignoring the prompt text."""
    it = iter(answers)

    def _input(prompt):
        return next(it)

    return _input


class TestLabelToolPlumbing(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="label_tool_test_")
        self.manifest_path, self.crop_paths = make_synthetic_crop_set(
            self.tmp_dir, count=6
        )
        self.labels_path = os.path.join(self.tmp_dir, "labels.csv")
        self.qa_report_path = os.path.join(self.tmp_dir, "qa_report.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # (a) full labeling run produces a correct labels.csv
    # ------------------------------------------------------------------
    def test_a_full_labeling_run_produces_correct_labels_csv(self):
        answers = [placeholder_text(p) for p in self.crop_paths]
        input_fn = scripted_input_fn(answers)

        label_tool.run_labeling(
            self.manifest_path, self.labels_path, no_display=True, input_fn=input_fn
        )

        self.assertTrue(os.path.exists(self.labels_path))
        with open(self.labels_path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        header, data_rows = rows[0], rows[1:]
        self.assertEqual(header, ["image_path", "text", "writer_id"])
        self.assertEqual(len(data_rows), len(self.crop_paths))

        for crop_path, row in zip(self.crop_paths, data_rows):
            image_path, text, writer_id = row
            self.assertEqual(os.path.abspath(image_path), crop_path)
            self.assertEqual(text, placeholder_text(crop_path))
            self.assertEqual(writer_id, "me")

    # ------------------------------------------------------------------
    # (b) mid-run resume: no re-prompting / no duplicate rows
    # ------------------------------------------------------------------
    def test_b_resume_does_not_reprompt_or_duplicate(self):
        K = 3
        first_batch_crops = self.crop_paths[:K]
        first_answers = [placeholder_text(p) for p in first_batch_crops]

        # Simulate a run that only manages to label the first K crops
        # before being "killed": we feed exactly K answers, and the
        # (K+1)th input_fn call raises EOFError, mimicking stdin closing
        # (Ctrl-C / killed terminal) mid-run. run_labeling must catch
        # this and stop cleanly, having already flushed the first K rows.
        answers_iter = iter(first_answers)

        def killed_input_fn(prompt):
            try:
                return next(answers_iter)
            except StopIteration:
                raise EOFError("simulated kill mid-run")

        label_tool.run_labeling(
            self.manifest_path,
            self.labels_path,
            no_display=True,
            input_fn=killed_input_fn,
        )

        with open(self.labels_path, "r", newline="") as f:
            rows_after_first_run = list(csv.DictReader(f))
        self.assertEqual(len(rows_after_first_run), K)
        labeled_paths_after_first_run = {
            os.path.abspath(r["image_path"]) for r in rows_after_first_run
        }
        self.assertEqual(labeled_paths_after_first_run, set(first_batch_crops))

        # Now "restart": re-run with an input_fn that would fail the
        # test immediately if it were asked about any of the first K
        # crops (proving they are not re-prompted), and provide answers
        # only for the remaining crops.
        remaining_crops = self.crop_paths[K:]
        remaining_answers = [placeholder_text(p) for p in remaining_crops]
        remaining_iter = iter(remaining_answers)
        prompts_seen = []

        def resume_input_fn(prompt):
            prompts_seen.append(prompt)
            return next(remaining_iter)

        label_tool.run_labeling(
            self.manifest_path,
            self.labels_path,
            no_display=True,
            input_fn=resume_input_fn,
        )

        # Exactly len(remaining_crops) prompts were issued -- none for
        # the already-labeled first K.
        self.assertEqual(len(prompts_seen), len(remaining_crops))
        for crop_path in first_batch_crops:
            basename = os.path.basename(crop_path)
            self.assertFalse(
                any(basename in p for p in prompts_seen),
                f"{basename} was re-prompted after already being labeled",
            )

        with open(self.labels_path, "r", newline="") as f:
            final_rows = list(csv.DictReader(f))

        # No duplicates, one row per crop total.
        self.assertEqual(len(final_rows), len(self.crop_paths))
        final_paths = [os.path.abspath(r["image_path"]) for r in final_rows]
        self.assertEqual(len(final_paths), len(set(final_paths)))
        self.assertEqual(set(final_paths), set(self.crop_paths))

        for row in final_rows:
            self.assertEqual(row["text"], placeholder_text(row["image_path"]))

    # ------------------------------------------------------------------
    # (c) QA flags a real mismatch and does NOT flag a correct re-entry
    # ------------------------------------------------------------------
    def test_c_qa_flags_mismatch_and_not_correct_reentry(self):
        answers = [placeholder_text(p) for p in self.crop_paths]
        label_tool.run_labeling(
            self.manifest_path,
            self.labels_path,
            no_display=True,
            input_fn=scripted_input_fn(answers),
        )

        # Fixed seed -> deterministic sample of 6 rows: max(1, round(0.6)) = 1.
        rng = random.Random(1234)
        rows = label_tool.read_all_labels(self.labels_path)
        n = len(rows)
        sample_size = max(1, round(0.10 * n))
        sampled_preview = random.Random(1234).sample(rows, sample_size)
        sampled_path = sampled_preview[0]["image_path"]
        correct_text = sampled_preview[0]["text"]
        wrong_text = correct_text + "_DELIBERATE_MISMATCH"

        qa_input_fn = scripted_input_fn([wrong_text])
        label_tool.run_qa(
            self.labels_path,
            self.qa_report_path,
            no_display=True,
            input_fn=qa_input_fn,
            rng=rng,
        )

        with open(self.qa_report_path, "r", newline="") as f:
            qa_rows = list(csv.DictReader(f))

        self.assertEqual(len(qa_rows), 1)
        self.assertEqual(qa_rows[0]["image_path"], sampled_path)
        self.assertEqual(qa_rows[0]["original_text"], correct_text)
        self.assertEqual(qa_rows[0]["reentered_text"], wrong_text)
        self.assertEqual(qa_rows[0]["status"], "MISMATCH")

        # labels.csv itself must be untouched by QA (QA only reports).
        with open(self.labels_path, "r", newline="") as f:
            labels_after_qa = list(csv.DictReader(f))
        self.assertEqual(len(labels_after_qa), len(self.crop_paths))
        for row in labels_after_qa:
            self.assertEqual(row["text"], placeholder_text(row["image_path"]))

        # --- False-positive direction: correct re-entry must NOT be flagged ---
        rng2 = random.Random(1234)
        qa_input_fn_correct = scripted_input_fn([correct_text])
        qa_report_path_2 = os.path.join(self.tmp_dir, "qa_report_2.csv")
        label_tool.run_qa(
            self.labels_path,
            qa_report_path_2,
            no_display=True,
            input_fn=qa_input_fn_correct,
            rng=rng2,
        )

        with open(qa_report_path_2, "r", newline="") as f:
            qa_rows_2 = list(csv.DictReader(f))

        self.assertEqual(len(qa_rows_2), 1)
        self.assertEqual(qa_rows_2[0]["image_path"], sampled_path)
        self.assertEqual(qa_rows_2[0]["status"], "OK")

    # ------------------------------------------------------------------
    # Extra plumbing checks called out by the spec (kept lightweight)
    # ------------------------------------------------------------------
    def test_empty_text_row_counts_as_labeled_on_resume(self):
        # First crop gets an empty-string label (noise-fragment crop).
        empty_and_rest = [""] + [
            placeholder_text(p) for p in self.crop_paths[1:]
        ]
        label_tool.run_labeling(
            self.manifest_path,
            self.labels_path,
            no_display=True,
            input_fn=scripted_input_fn(empty_and_rest),
        )

        prompts_seen = []

        def fail_if_called(prompt):
            prompts_seen.append(prompt)
            raise AssertionError("should not be re-prompted; nothing pending")

        label_tool.run_labeling(
            self.manifest_path,
            self.labels_path,
            no_display=True,
            input_fn=fail_if_called,
        )
        self.assertEqual(prompts_seen, [])

        with open(self.labels_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["text"], "")

    def test_missing_crop_file_warns_and_skips_without_crash(self):
        # Remove one crop file on disk while leaving it in the manifest.
        missing_crop = self.crop_paths[0]
        os.remove(missing_crop)

        answers = [
            placeholder_text(p) for p in self.crop_paths if p != missing_crop
        ]
        # Should not raise.
        label_tool.run_labeling(
            self.manifest_path,
            self.labels_path,
            no_display=True,
            input_fn=scripted_input_fn(answers),
        )

        with open(self.labels_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        labeled_paths = {os.path.abspath(r["image_path"]) for r in rows}
        self.assertNotIn(missing_crop, labeled_paths)
        self.assertEqual(len(rows), len(self.crop_paths) - 1)

    def test_empty_manifest_exits_gracefully(self):
        empty_manifest_path = os.path.join(self.tmp_dir, "empty_manifest.csv")
        with open(empty_manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["crop_path", "source_scan", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
            )

        def fail_if_called(prompt):
            raise AssertionError("input_fn should never be called for empty manifest")

        # Should not raise.
        label_tool.run_labeling(
            empty_manifest_path,
            self.labels_path,
            no_display=True,
            input_fn=fail_if_called,
        )
        self.assertFalse(os.path.exists(self.labels_path))

    def test_qa_on_empty_labels_exits_gracefully(self):
        empty_labels_path = os.path.join(self.tmp_dir, "no_labels.csv")

        def fail_if_called(prompt):
            raise AssertionError("input_fn should never be called for empty labels")

        # Should not raise.
        label_tool.run_qa(
            empty_labels_path,
            self.qa_report_path,
            no_display=True,
            input_fn=fail_if_called,
            rng=random.Random(1),
        )
        self.assertFalse(os.path.exists(self.qa_report_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
