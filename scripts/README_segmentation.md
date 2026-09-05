# Segmentation Pipeline (C3-preprocessing-segmentation)

Classical OpenCV document-segmentation pipeline that turns raw scans/photos
of handwritten pages into word- (or line-) level crops for handwriting
synthesis training data.

## How to run

```bash
cd "/Users/blakey5aces/Handwriting Analysis"
source .venv/bin/activate

# 1. Generate a synthetic test scan (only needed once, or to regenerate):
python scripts/make_test_image.py

# 2. Run segmentation (default: word-level crops)
python scripts/segment.py

# Line-level fallback (recommended for heavily cursive/joined handwriting):
python scripts/segment.py --granularity line

# Disable ruled-line removal (e.g. for blank/unlined paper, to save time):
python scripts/segment.py --no-remove-lines
```

Venv: created with Python 3.11.9
(`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11`).
Dependencies pinned in `requirements-segmentation.txt`.

## Granularity: word (default), line (fallback)

Word-level crops are the primary output because downstream handwriting
synthesis training typically wants word-aligned image/text pairs. However,
cursive or heavily joined handwriting makes word boundaries genuinely
ambiguous from pixels alone (letters within AND between words can touch).
`--granularity line` skips word splitting and emits one crop per detected
text line instead -- a robust fallback when word segmentation is
unreliable for a given writer's style. On the bundled synthetic test image
(a semi-cursive font), line-level segmentation was visibly cleaner (5
clean line boxes + 2 tiny stray-mark fragments) than word-level (43 crops
across 5 lines, with occasional merges like "brown fox" -> one box, and a
couple of spurious single-pixel-blob fragments from disconnected dots/
strokes). Use line mode first when validating a new source's handwriting
style, then try word mode.

## Coordinate space of bboxes

All `bbox_x, bbox_y, bbox_w, bbox_h` values in `segmented/manifest.csv` are
in the **processed working-image space**: after downscaling (if the source
exceeded `MAX_WORKING_DIM`, currently 2500px on the longest side) and AFTER
deskew rotation. Crops are saved from that exact same processed image
buffer, so crop pixel content and manifest bboxes are always mutually
consistent. They are NOT coordinates in the original source file's pixel
space -- if you need original-space coordinates you must re-derive them
via the recorded downscale factor and rotation matrix (not currently
persisted per-row; add if needed downstream).

## QA artifact: debug overlays

`segmented/debug/<source_stem>_overlay.png` draws every detected line box
(blue) and every emitted crop box (green, word or line depending on
granularity) on top of the deskewed/processed image. This is the primary
visual acceptance artifact -- always inspect it after a run on a new
source to catch silent segmentation failures (missed words, wildly
misaligned boxes, whole lines dropped) before trusting the crops.

## Known weaknesses / caveats

- Word splitting can merge short adjacent words (e.g. "brown fox", "over
  the") when the gap between them is comparable to normal intra-word
  letter spacing in a given font/writer.
- Isolated diacritic-like marks (e.g. dots left behind after morphological
  processing near letters like "j", "i") can occasionally form tiny
  spurious single-digit-pixel-count boxes; `MIN_WORD_AREA`/`MIN_WORD_WIDTH`
  filter most but not all of these.
- Deskew is capped at +/-30 degrees and skipped entirely if the estimated
  angle is out of range or foreground pixel count is too low to trust --
  this avoids the classic `cv2.minAreaRect` sign/quadrant bug flipping a
  page sideways, at the cost of occasionally not correcting a real skew
  if the text mask is too sparse/noisy.
- Ruled-line removal only strips components that are both very high
  aspect ratio AND span most of the page width/height, to avoid eating
  descenders/ascenders; on lightly ruled or graph paper this is
  conservative (may leave short line fragments) rather than aggressive.
- `--engine doctr` is a stub that checks for `python-doctr` and exits
  with a clear message if absent; it is NOT implemented as a working
  detector in this pass. The classical pipeline is the supported default
  and requires no DL/torch stack.
