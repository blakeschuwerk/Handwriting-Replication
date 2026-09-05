# Handwriting Replication

A local pipeline that learns one person's handwriting from photos of their own
notebook pages, and generates new text in that handwriting — laid out onto a
photo of real notebook paper. Runs entirely on-device (macOS, Apple Silicon);
no cloud API calls, no third-party handwriting data.

## Pipeline overview

```
raw_scans/ (your photos)
      │  scripts/vision_segment.py  — Apple Vision: detect + OCR every word
      ▼
boxes/*.json                         — one file per page, stable UUID per box
      │  dashboard's Edit Boxes tab  — you review, fix, or accept flagged boxes
      ▼
scripts/build_labels.py              — boxes → labels.csv (only reviewed words)
      │
      ▼
scripts/package_dataset.py           — labels.csv + crops → dataset/dataset.h5
      │
      ▼
scripts/finetune.py                  — fine-tunes FW-GAN on your handwriting
      │
      ▼
scripts/generate.py                  — arbitrary text → generated word images
      │
      ▼
scripts/render_page.py               — words laid out onto a page, boxes matted out
```

Everything above is driven from one dashboard: `dashboard/server.py` (FastAPI)
+ `dashboard/index.html` (single-page UI, no build step). Launch it with
`dashboard/start.command` or `python3 dashboard/server.py`.

## Why each stage exists

**Detection (`vision_segment.py`).** Apple's Vision framework does text
detection and OCR in one pass, fully offline. Its box placement is reliable,
but word-level transcription on isolated crops runs closer to 35% accurate —
line-level context gets it to roughly 75–80%. Either way, its output is a
*proposal*, never trusted as a label until a human looks at it.

**Stable box identity (`boxstore.py`).** Boxes carry a UUID, not a line/word
index. Re-running detection after you've hand-edited a page merges the new
pass in by overlap (IoU) and never overwrites a box you've touched — otherwise
every edit would be destroyed the next time detection ran.

**Review, not blind training (`build_labels.py`).** A document-aware spell
checker (`wordcheck.py`) flags anything that doesn't look like a word this
specific document actually uses — ranking corrections by how often *this
writer* uses a word, not a generic dictionary. Vision's confidence score is
close to useless for this (it reports 1.00 on plenty of wrong reads), so
flagging is corpus-based, not confidence-based. Only words that are unflagged
or explicitly accepted as-written reach `labels.csv` — genuine misspellings in
the handwriting are kept if you confirm them, since the label must match the
image, not a dictionary.

**Dataset packaging (`package_dataset.py`).** Packs the reviewed crops into
the exact HDF5 layout the model's `Hdf5Dataset` loader expects (schema
reverse-engineered from its own test fixture — see
`dataset/DATASET_SCHEMA.md`).

**Fine-tuning (`finetune.py`).** Starts from FW-GAN's pretrained weights
(trained on public handwriting datasets — see Model & data provenance) and
adapts them toward this one writer's strokes. Runs on Apple Silicon via MPS.

**Generation & rendering (`generate.py`, `render_page.py`).** Generates each
word as a small grayscale image, then lays them onto a page. The generator has
no alpha channel and a non-white background level, so `render_page.py` mattes
each word (luminance → transparency, with a tuned toe + gain curve — see the
comments in `ink_alpha()`) rather than pasting the raw rectangle. This was
measured, not eyeballed: two earlier matte designs that looked plausible were
rejected because they scored worse on actual pixel coverage.

## Model & data provenance

- **Base model**: [FW-GAN](https://github.com/Data-Driven-AI-Research/FW-GAN)
  (DAIR Group, MIT licensed) — not vendored in this repo; clone it separately
  into `models/FW_GAN/` per its own instructions, and download the pretrained
  checkpoint per its README.
- **Fine-tuning data**: only this user's own handwritten notebook pages,
  reviewed word-by-word through the Edit Boxes tab before being added to
  `labels.csv`. No other handwriting samples are mixed into the dataset.
- **What's excluded from this repo**: the actual scanned photos, crops, and
  transcriptions are personal and are gitignored — see `CLAUDE.md` for the
  full list and why. Anyone using this pipeline supplies their own scans.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Clone the base model separately (not included in this repo):
git clone https://github.com/Data-Driven-AI-Research/FW-GAN.git models/FW_GAN
# then follow models/FW_GAN's own README to download its pretrained weights
# into models/weights/FW-GAN.pth

mkdir -p raw_scans
# drop your own handwriting photos into raw_scans/, then:
python3 dashboard/server.py
# open http://127.0.0.1:8765
```

`dashboard/start.command` does the same thing via double-click on macOS, but
has a hardcoded path (`/Users/blakey5aces/Handwriting Analysis`) — edit the
`PROJECT` variable at the top if you're running this from somewhere else.

## Dashboard tabs

1. **Upload & Detect** — add scans, run Vision detection.
2. **Edit Boxes** — drag/resize/add/delete word boxes; a review sidebar lists
   every flagged word with a one-key "accept as written" override; full
   undo/redo.
3. **Build Dataset** — pick which pages to train on (fully-reviewed vs. still
   has flagged words), see a live word/character count and a sizing
   recommendation measured against FW-GAN's own reference data, then pack
   `dataset.h5`.
4. **Train** — an XY pad maps learning-rate × epochs to one drag instead of
   raw hyperparameters, with plain-English explanations of what each control
   actually does.
5. **Generate & Render** — type any text, generate it in the fine-tuned
   handwriting, lay it onto a page.

## Repo layout

```
scripts/       pipeline stages (detection, labeling, packaging, training, generation)
dashboard/     FastAPI server + single-page UI
dataset/       DATASET_SCHEMA.md documents the HDF5 layout (data itself gitignored)
CLAUDE.md      standing rules for AI-assisted work in this repo
```
