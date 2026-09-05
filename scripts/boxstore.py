#!/usr/bin/env python3
"""
boxstore.py -- the authoritative, editable record of word boxes per page.

WHY THIS EXISTS
---------------
Crops used to be named {page}_L003_W005.png, with labels.csv keyed by that
filename. That scheme cannot survive manual box editing: deleting one box
renumbers every box after it, so labels silently re-point to the wrong images --
and re-running detection would wipe manual edits entirely.

So identity moves off the line/word index and onto a STABLE ID. A box keeps its
id for life; crops are named {page}__{id}.png. Line and word indices are still
recorded, because reading order matters for review and for aligning a line's
transcript to its words, but they are ORDERING metadata, never identity.

MERGE RULE
----------
Re-running detection must never destroy human work, so merge_detection() keeps
every box the user added or edited and only refreshes untouched detector boxes.
A box is protected once its `source` is "manual" or its `edited` flag is set.

One JSON per page under boxes/, so a page can be re-detected or hand-edited
without touching any other page.
"""

import json
import os
import uuid

BOXES_DIRNAME = "boxes"


def boxes_dir(root):
    d = os.path.join(root, BOXES_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def page_key(scan_filename):
    """Filesystem-safe stem for a scan (screenshot names contain U+202F)."""
    stem = os.path.splitext(os.path.basename(scan_filename))[0]
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)


def path_for(root, scan_filename):
    return os.path.join(boxes_dir(root), page_key(scan_filename) + ".json")


def new_id():
    return uuid.uuid4().hex[:10]


def make_box(x, y, w, h, text="", conf=0.0, line=0, word=0, source="vision"):
    return {
        "id": new_id(),
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "text": text, "conf": float(conf),
        "line": int(line), "word": int(word),
        "source": source,      # "vision" | "manual"
        "edited": False,       # set once a human moves/resizes/retypes it
        "deleted": False,      # soft delete, so an accidental delete is undoable
    }


def load(root, scan_filename):
    p = path_for(root, scan_filename)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save(root, scan_filename, doc):
    p = path_for(root, scan_filename)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    return p


def empty_doc(scan_filename, width, height, rotation=0):
    return {
        "scan": os.path.basename(scan_filename),
        "width": int(width),
        "height": int(height),
        "rotation": int(rotation),
        "boxes": [],
    }


def active(doc):
    return [b for b in doc.get("boxes", []) if not b.get("deleted")]


def _iou(a, b):
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    iy = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / float(union) if union > 0 else 0.0


def merge_detection(doc, detected, iou_thresh=0.5):
    """Fold a fresh detector pass into an existing page.

    Protected (kept verbatim): manual boxes and any box a human edited.
    Refreshed: untouched detector boxes matched by overlap.
    Added: detections that match nothing.
    Retired: unmatched, unedited detector boxes (soft-deleted, so recoverable).

    Returns a stats dict for reporting.
    """
    existing = doc.get("boxes", [])
    protected = [b for b in existing if b.get("source") == "manual" or b.get("edited")]
    refreshable = [b for b in existing
                   if not (b.get("source") == "manual" or b.get("edited")) and not b.get("deleted")]

    stats = {"kept_protected": len(protected), "refreshed": 0, "added": 0, "retired": 0}
    used = set()
    out = list(protected)

    for old in refreshable:
        best, best_iou = None, 0.0
        for i, det in enumerate(detected):
            if i in used:
                continue
            v = _iou(old, det)
            if v > best_iou:
                best, best_iou = i, v
        if best is not None and best_iou >= iou_thresh:
            det = detected[best]
            used.add(best)
            old.update({"x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"],
                        "text": det["text"], "conf": det["conf"],
                        "line": det["line"], "word": det["word"]})
            out.append(old)
            stats["refreshed"] += 1
        else:
            old["deleted"] = True
            out.append(old)
            stats["retired"] += 1

    for i, det in enumerate(detected):
        if i in used:
            continue
        # Don't re-add a detection that just lands on top of a protected box;
        # the human's version of that word wins.
        if any(_iou(det, p) >= iou_thresh for p in protected):
            continue
        out.append(det)
        stats["added"] += 1

    doc["boxes"] = out
    return stats


def assign_reading_order(doc, line_tol_frac=0.6):
    """Recompute line/word indices from geometry, top-to-bottom, left-to-right.

    Needed after manual edits: a hand-drawn box has no line number, and a moved
    box may belong to a different line than it did. Rows are grouped by vertical
    overlap against the median box height, which tolerates the baseline drift of
    a photographed, curved page better than a fixed pixel threshold.
    """
    bs = active(doc)
    if not bs:
        return
    heights = sorted(b["h"] for b in bs)
    med_h = heights[len(heights) // 2] or 1
    tol = max(4, int(line_tol_frac * med_h))

    for b in bs:
        b["_cy"] = b["y"] + b["h"] / 2.0
    bs.sort(key=lambda b: b["_cy"])

    lines, cur = [], [bs[0]]
    for b in bs[1:]:
        if abs(b["_cy"] - cur[-1]["_cy"]) <= tol:
            cur.append(b)
        else:
            lines.append(cur)
            cur = [b]
    lines.append(cur)

    for li, ln in enumerate(lines, start=1):
        ln.sort(key=lambda b: b["x"])
        for wi, b in enumerate(ln, start=1):
            b["line"], b["word"] = li, wi
    for b in bs:
        b.pop("_cy", None)


def crop_name(scan_filename, box):
    return f"{page_key(scan_filename)}__{box['id']}.png"
