#!/usr/bin/env python3
"""
build_labels.py -- turn the reviewed box store into labels.csv.

This replaces curate_labels.py as the path into training.

WHY THE CHANGE
--------------
curate_labels.py decided what to train on by re-judging every word against the
system dictionary. That made sense when the labels were raw Vision output that
nobody had looked at. It stopped making sense once the box editor gained a
review queue: a human now looks at the crop, sees the word in its sentence, and
either fixes it or marks it correct as written.

Re-judging after that is not just redundant, it is wrong. These are handwritten
notes containing genuine misspellings -- "efficency", "tradeoff", "Intro" -- and
the label has to match the image, not the dictionary. The old path silently
dropped exactly the words a human had deliberately confirmed, because Webster's
2nd has never heard of them.

So the rule is now:

  ACCEPT   the box carries no objection, OR a human explicitly accepted it
  REVIEW   still flagged and nobody has signed off on it

Human judgement is the filter. The spell check only decides what to *ask* about.
"""

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boxstore  # noqa: E402
import wordcheck  # noqa: E402


def page_ink(root, key, boxes):
    """Letter-ink fraction per box, so an empty box can't reach the dataset."""
    import cv2
    p = os.path.join(root, "pages", f"{key}.png")
    gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
    out = {}
    for b in boxes:
        if gray is None:
            out[b["id"]] = None
            continue
        r = gray[max(0, b["y"]):b["y"] + b["h"], max(0, b["x"]):b["x"] + b["w"]]
        if r.size < 100:
            out[b["id"]] = 0.0
            continue
        bw = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
        n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
        H, keep = r.shape[0], 0
        for i in range(1, n):
            _, _, w, h, a = st[i]
            if h < max(3, 0.14 * H) or a < 12:
                continue
            if w > 6 * h and h < 0.30 * H:
                continue
            keep += a
        out[b["id"]] = keep / float(r.size)
    return out


def main():
    ap = argparse.ArgumentParser(description="Build labels.csv from the reviewed box store.")
    ap.add_argument("--min-len", type=int, default=1,
                    help="Drop labels shorter than this many characters.")
    ap.add_argument("--pages", default="",
                    help="Comma-separated page keys to include. Empty = all pages. "
                         "Lets the Build Dataset tab train on a chosen subset.")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = os.path.join(root, "segmented", "manifest.csv")
    if not os.path.exists(manifest):
        print(f"No manifest at {manifest}. Rebuild crops first.")
        sys.exit(1)

    lex = wordcheck.Lexicon.build(root)
    only = {k for k in args.pages.split(",") if k} or None

    # box_id -> (text, accepted, flag)
    info = {}
    raw = os.path.join(root, "raw_scans")
    for fn in sorted(os.listdir(raw)) if os.path.isdir(raw) else []:
        if only and boxstore.page_key(fn) not in only:
            continue
        doc = boxstore.load(root, fn)
        if not doc:
            continue
        boxes = boxstore.active(doc)
        inks = page_ink(root, boxstore.page_key(fn), boxes)
        for b in boxes:
            text = (b.get("text") or "").strip()
            if b.get("accepted"):
                flag = None
            else:
                flag, _s, _src = wordcheck.judge(text, lex, inks.get(b["id"]))
            info[b["id"]] = (text, bool(b.get("accepted")), flag)

    accepted, review = [], []
    stats = Counter()
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bid = r.get("box_id", "")
            if bid not in info:
                # page not selected for this build (or box no longer exists)
                stats["skipped_unselected"] += 1
                continue
            text, was_accepted, flag = info[bid]
            path = r["crop_path"]
            if not text or len(text) < args.min_len:
                stats["empty"] += 1
                review.append([path, text, "empty"])
            elif flag is None:
                stats["accepted_override" if was_accepted else "accepted_clean"] += 1
                accepted.append([path, text, "me"])
            else:
                stats[f"review_{flag}"] += 1
                review.append([path, text, flag])

    with open(os.path.join(root, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "text", "writer_id"])
        w.writerows(accepted)
    with open(os.path.join(root, "labels_review.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "text", "reason"])
        w.writerows(review)

    words = Counter(t.lower() for _p, t, _w in accepted)
    chars = sorted({c for _p, t, _w in accepted for c in t})
    if only:
        print(f"pages selected    : {len(only)}")
    print(f"labels.csv        : {len(accepted)} training pairs")
    print(f"characters covered: {len(chars)} -> {''.join(chars)}")
    print(f"labels_review.csv : {len(review)} still needing review")
    print(f"unique words      : {len(words)}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<24} {v}")


if __name__ == "__main__":
    main()
