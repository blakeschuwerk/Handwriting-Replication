#!/usr/bin/env python3
"""
transcript_align.py -- put a page transcript's words onto Vision's word boxes.

WHY
---
Vision is good at WHERE words are and bad at WHAT they say: measured against a
hand-checked answer key for three notebook pages it read 63% of words right,
and dropped whole stretches of lines it found but scored as low-confidence.
A page reader such as Chandra OCR 2 has the opposite profile -- 96.8% of words
right on the same pages, but no word positions at all.

So each tool does the half it is good at: Vision's boxes, the reader's text.
This module is the join. For every Vision line it finds the stretch of the
transcript that line most resembles (spelling similarity, local alignment), and
hands each box the transcript word it lines up with.

What it deliberately does NOT trust:
  * Vision's line ORDER. On a curled page Vision's lines come back slightly out
    of order, so each line is placed independently, not by a global sweep.
  * A match that does not look alike. Each box carries the similarity between
    Vision's reading and the transcript word; a weak match is flagged for a
    human rather than silently relabelled, because a wrong label teaches the
    model a wrong letter.
  * Vision's word SPLITS. Vision sometimes cuts one word into two boxes
    ("R" + "sht" for "Right"). Two boxes that only match together are both
    flagged, so the box editor shows them for merging instead of training on
    half a word labelled as a whole one.
"""

import difflib
import html
import re

# Alignment scores. A box/word pair earns sim - MATCH_BAR, so pairs that look
# less than ~40% alike cost more than skipping them.
MATCH_BAR = 0.4
SKIP_BOX = -0.35     # a Vision box with no transcript word (smudge, stray mark)
SKIP_WORD = -0.25    # a transcript word Vision never boxed
# Below this similarity an assigned label is flagged for review.
TRUST_SIM = 0.5
# Two boxes are treated as one split word only when joined they match the
# word this much better than the better half does on its own.
SPLIT_GAIN = 0.1

CIRCLED = {chr(0x2460 + i): str(i + 1) for i in range(20)}  # ① .. ⑳


def clean_transcript(raw):
    """Reader output (HTML/markdown) -> plain text lines."""
    s = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</h\d>|</tr>", "\n", raw)
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    s = "".join(CIRCLED.get(c, c) for c in s)
    return [ln.strip() for ln in s.splitlines() if ln.strip()]


def _norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _digits(t):
    return any(c.isdigit() for c in t)


def similarity(a, b):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def transcript_tokens(lines):
    """Words with attached punctuation stripped; pure-punctuation tokens dropped."""
    out = []
    for ln in lines:
        for t in ln.split():
            t = t.strip(".,;:!?()[]{}\"'`“”‘’-–—")
            if _norm(t):
                out.append(t)
    return out


def align_line(boxes_text, words):
    """Local alignment of one line's box texts against the transcript words.

    Returns (score, start, assignment) where assignment[i] is
    (word_index | None, sim, kind) for box i; kind is "1:1", "split" (box is
    half of a word another box shares) or "unmatched".
    """
    n, m = len(boxes_text), len(words)
    NEG = float("-inf")
    # S[i][j]: best score of an alignment ending after box i-1 and word j-1.
    S = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best = (0.0, 0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cands = [(0.0, None)]  # local: may start fresh here
            s11 = similarity(boxes_text[i - 1], words[j - 1])
            cands.append((S[i - 1][j - 1] + s11 - MATCH_BAR, ("11", s11)))
            if i >= 2:
                # Only a real split if the halves match clearly better together
                # than either does alone. Without this, "-" + "Lean" scores as
                # well as "Lean" (punctuation carries no letters) and every
                # bullet dash got glued onto the word after it.
                s21 = similarity(boxes_text[i - 2] + boxes_text[i - 1], words[j - 1])
                alone = max(s11, similarity(boxes_text[i - 2], words[j - 1]))
                if s21 >= alone + SPLIT_GAIN:
                    cands.append((S[i - 2][j - 1] + 2 * (s21 - MATCH_BAR) - 0.1, ("21", s21)))
            cands.append((S[i - 1][j] + SKIP_BOX, ("skipbox", 0.0)))
            cands.append((S[i][j - 1] + SKIP_WORD, ("skipword", 0.0)))
            S[i][j], back[i][j] = max(cands, key=lambda c: c[0])
            if S[i][j] > best[0]:
                best = (S[i][j], i, j)
    score, i, j = best
    assign = [(None, 0.0, "unmatched")] * n
    while i > 0 and j > 0 and back[i][j] is not None and S[i][j] > 0:
        op, sim = back[i][j]
        if op == "11":
            assign[i - 1] = (j - 1, sim, "1:1"); i, j = i - 1, j - 1
        elif op == "21":
            assign[i - 1] = (j - 1, sim, "split"); assign[i - 2] = (j - 1, sim, "split")
            i, j = i - 2, j - 1
        elif op == "skipbox":
            i -= 1
        else:
            j -= 1
    start = j
    return score, start, assign


def align_page(vision_lines, transcript_lines):
    """vision_lines: [[box_text, ...], ...] in Vision's order.

    Returns one list per line of dicts {text, sim, flag} -- text is the label
    to use, flag is None or a reason the box needs a human look.
    """
    words = transcript_tokens(transcript_lines)
    out = []
    for line in vision_lines:
        _, _, assign = align_line(line, words)
        res = []
        for vis, (wi, sim, kind) in zip(line, assign):
            if wi is None:
                res.append({"text": vis, "sim": 0.0, "flag": "not_in_transcript"})
            elif kind == "split":
                res.append({"text": words[wi], "sim": sim, "flag": "split_word"})
            elif _digits(vis) and _digits(words[wi]) and _norm(vis) != _norm(words[wi]):
                # Both read a number and they disagree. Spelling similarity
                # can't arbitrate digits ("800" vs "500" look 67% alike), and
                # the reader is not reliably the right one: on the test pages
                # Vision had 800 correct where Chandra said 500.
                res.append({"text": words[wi], "sim": sim, "flag": "number_mismatch"})
            else:
                res.append({"text": words[wi], "sim": sim,
                            "flag": None if sim >= TRUST_SIM else "weak_match"})
        out.append(res)
    return out
