#!/usr/bin/env python3
"""
page_reader.py -- read a whole page with Chandra OCR 2, running locally.

WHY CHANDRA
-----------
Benchmarked against a hand-written answer key for three notebook pages
(557 words), reading the page text:

    Apple Vision (current)          63.0% of words right
    Vision + wordcheck fixes        67.5%
    PaddleOCR-VL 1.6, whole page    73.6%
    PaddleOCR-VL 1.6, per line      88.0%  (but 331 duplicated/wrong extras)
    Chandra OCR 2, whole page       96.8%  (12 wrong/extra words)

Chandra's misses were mostly circled list numbers, plurals and two digits.
Its one systematic habit is CORRECTING spelling ("parrellell" -> "parallel"),
which is worth knowing: labels should match the letters on the page.

It gives no word positions, so it never replaces Vision -- transcript_align.py
puts its words onto Vision's boxes.

SETUP
-----
    pip install mlx-vlm
    HF_HUB_DISABLE_XET=1 hf download mlx-community/chandra-ocr-2-oQ8 \\
        --local-dir models/ocr/chandra-ocr-2-oQ8

(The Xet transfer backend stalled at 0 bytes on this machine; plain HTTP ran
at ~12MB/s.) Runs on Apple Silicon via MLX: ~5GB in memory, ~80s per page on
an M4. Weights: datalab-to/chandra-ocr-2, OpenRAIL licence.

Results are cached next to the processed page (pages/<key>.chandra.json),
keyed on a hash of the exact pixels read, so re-running detection after a box
edit costs nothing and a re-rotated page is read again.
"""

import hashlib
import json
import os
import tempfile

import cv2

MODEL_DIRNAME = os.path.join("models", "ocr", "chandra-ocr-2-oQ8")

# Chandra's own OCR prompt, verbatim from github.com/datalab-to/chandra
# (chandra/prompts.py, OCR_PROMPT; Apache-2.0). The model was trained on it;
# a shorter prompt still answers but drifts from the trained format.
_TAGS = ["math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td", "p", "th",
         "div", "pre", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li", "input", "a", "span",
         "img", "hr", "tbody", "small", "caption", "strong", "thead", "big", "code", "chem"]
_ATTRS = ["class", "colspan", "rowspan", "display", "checked", "type", "border", "value",
          "style", "href", "alt", "align", "data-bbox", "data-label"]
OCR_PROMPT = f"""OCR this image to HTML.

Only use these tags {_TAGS}, and these attributes {_ATTRS}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural."""

_MODEL = {}


def available(root):
    return os.path.isfile(os.path.join(root, MODEL_DIRNAME, "config.json"))


def _load(root):
    if "m" not in _MODEL:
        from mlx_vlm import load
        _MODEL["m"] = load(os.path.join(root, MODEL_DIRNAME), trust_remote_code=True)
    return _MODEL["m"]


def read(root, bgr, cache_path=None, max_tokens=3000):
    """Raw Chandra output (HTML) for a BGR page image, cached by pixel hash."""
    digest = hashlib.sha1(bgr.tobytes()).hexdigest()
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                hit = json.load(f)
            if hit.get("sha1") == digest:
                return hit["raw"]
        except (OSError, ValueError, KeyError):
            pass

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, proc = _load(root)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp = tf.name
    try:
        cv2.imwrite(tmp, bgr)
        prompt = apply_chat_template(proc, model.config, OCR_PROMPT, num_images=1)
        raw = generate(model, proc, prompt, image=[tmp],
                       max_tokens=max_tokens, temperature=0.0).text
    finally:
        os.unlink(tmp)

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump({"sha1": digest, "model": MODEL_DIRNAME, "raw": raw}, f)
    return raw
