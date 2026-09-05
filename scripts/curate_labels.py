#!/usr/bin/env python3
"""
curate_labels.py -- turn Vision's label PROPOSALS into a trustworthy training set.

Vision transcribes this handwriting at roughly 75-80% word accuracy, and reports
confidence 1.00 for many of its mistakes, so its own confidence cannot be used to
separate right from wrong at the word level. Training directly on labels_auto.csv
would teach the model that e.g. the glyphs for "competition" spell "comprition" --
a silent corruption that no loss curve would reveal.

The saving grace is that the errors are near-misses of real English words. So:

  ACCEPT    token is a real word as-is
  CORRECT   token is exactly ONE edit away from a real word -> take that word.
            Requiring a UNIQUE candidate is the safety property: "sustainabirity"
            has only one edit-1 neighbour ("sustainability") and is safe, whereas
            an ambiguous token with several neighbours is a coin flip and is not.
  REVIEW    everything else -> labels_review.csv, for a human to fix or discard

Only ACCEPT + CORRECT reach labels.csv. The dataset ends up smaller and clean
rather than larger and quietly wrong, which is the right trade for style
fine-tuning: a few thousand correct pairs is plenty, and a wrong pair is
actively harmful.
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter

DICT_PATHS = ["/usr/share/dict/words", "/usr/share/dict/web2"]
ALPHA = "abcdefghijklmnopqrstuvwxyz"

# Short function words that a plain dictionary file often omits but which are
# extremely common in real prose, plus the ones we never want to drop.
EXTRA_OK = {
    "a", "i", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is",
    "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
    "the", "and", "but", "for", "not", "you", "all", "can", "her", "was", "one",
    "our", "out", "has", "had", "his", "how", "its", "who", "did", "get", "him",
    "she", "too", "use", "way", "why", "new", "now", "old", "see", "two", "may",
    "are", "were", "been", "have", "this", "that", "with", "they", "から",
    "will", "from", "them", "than", "then", "when", "what", "your", "into",
    "also", "more", "most", "must", "such", "some", "each", "very", "over",
    "om",  # domain: "operations management" abbreviation used throughout
}


def load_dictionary():
    words = set()
    for p in DICT_PATHS:
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        words.add(w)
            break
    words |= EXTRA_OK
    return words


def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    out = set()
    for L, R in splits:
        if R:
            out.add(L + R[1:])                       # delete
        if len(R) > 1:
            out.add(L + R[1] + R[0] + R[2:])         # transpose
        for c in ALPHA:
            if R:
                out.add(L + c + R[1:])               # replace
            out.add(L + c + R)                       # insert
    out.discard(word)
    return out


def inflection_ok(stem, vocab):
    """True if `stem` is a regular inflection of a dictionary word.

    /usr/share/dict/words on macOS is Webster's 2nd (1934), a BASE-FORM word
    list: it has "operation" but not "operations", "decide" but not "decisions".
    Without this check every plural and past tense looks unknown, and the edit-1
    corrector then "fixes" a perfectly correct "operations" into "operation" --
    writing a label that disagrees with the image it is attached to.
    """
    if stem in vocab:
        return True
    if stem.endswith("ies") and (stem[:-3] + "y") in vocab:
        return True
    if stem.endswith("es") and (stem[:-2] in vocab or stem[:-1] in vocab):
        return True
    if stem.endswith("s") and stem[:-1] in vocab:
        return True
    for suf in ("ed", "ing", "ly", "er", "est", "ness", "ment", "tion"):
        if stem.endswith(suf):
            b = stem[: -len(suf)]
            if b in vocab or (b + "e") in vocab:
                return True
            if b.endswith("i") and (b[:-1] + "y") in vocab:
                return True
            # consonant doubling: "planning" -> "plan"
            if len(b) > 2 and b[-1] == b[-2] and b[:-1] in vocab:
                return True
    return False


def classify(token, vocab):
    """Return (status, final_text)."""
    t = token.strip()
    if not t:
        return "reject_empty", ""

    # keep numbers and simple numeric forms as-is
    if re.fullmatch(r"[0-9]+([./-][0-9]+)*", t):
        return "accept_numeric", t

    core = t.lower()
    poss = core.endswith("'s")
    stem = core[:-2] if poss else core

    if not re.fullmatch(r"[a-z][a-z'-]*", stem):
        return "review_nonword", t
    if len(stem) < 2 and stem not in EXTRA_OK:
        return "reject_tooshort", t

    if inflection_ok(stem, vocab):
        return "accept", t

    # hyphenated / possessive compounds: every part must be real
    parts = [p for p in re.split(r"[-']", stem) if p]
    if len(parts) > 1 and all(p in vocab for p in parts):
        return "accept", t

    cands = edits1(stem) & vocab
    if len(cands) == 1:
        fixed = next(iter(cands))
        # preserve the original capitalisation pattern
        if t[0].isupper():
            fixed = fixed.capitalize()
        if poss:
            fixed += "'s"
        return "correct", fixed

    return "review_unknown", t


def main():
    ap = argparse.ArgumentParser(description="Curate Vision labels into a clean training set.")
    ap.add_argument("--min-conf", type=float, default=0.3,
                    help="Drop crops whose source line scored below this.")
    # OFF by default. Even with a unique edit-1 neighbour the corrector is not
    # trustworthy here: Webster's 2nd is full of archaic entries that make
    # plausible-looking but wrong targets ("avarity" -> "amarity"). Corrections
    # recovered only ~7% of crops while risking labels that contradict the
    # image, so they now go to review instead. Opt in with --allow-correct.
    ap.add_argument("--allow-correct", dest="allow_correct", action="store_true", default=False,
                    help="Also accept unique edit-1 corrections (risky; default off).")
    ap.add_argument("--no-correct", dest="allow_correct", action="store_false")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = os.path.join(root, "segmented", "manifest.csv")
    if not os.path.exists(manifest):
        print(f"No manifest at {manifest}. Run scripts/vision_segment.py first.")
        sys.exit(1)

    vocab = load_dictionary()
    print(f"dictionary: {len(vocab)} words")

    rows = list(csv.DictReader(open(manifest)))
    accepted, review = [], []
    stats = Counter()

    for r in rows:
        if float(r.get("line_conf") or 0) < args.min_conf:
            stats["reject_lowconf"] += 1
            continue
        status, text = classify(r["auto_text"], vocab)
        stats[status] += 1
        if status.startswith("accept") or (status == "correct" and args.allow_correct):
            accepted.append([r["crop_path"], text, "me"])
        elif status.startswith("review") or status == "correct":
            review.append([r["crop_path"], r["auto_text"], text, status,
                           r["source_scan"], r["line_idx"], r["word_idx"]])

    with open(os.path.join(root, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "text", "writer_id"])
        w.writerows(accepted)
    with open(os.path.join(root, "labels_review.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "auto_text", "suggestion", "status",
                    "source_scan", "line_idx", "word_idx"])
        w.writerows(review)

    print()
    print("=== Curation ===")
    total = sum(stats.values())
    for k, v in stats.most_common():
        print(f"  {k:18s} {v:5d}  ({100.0 * v / max(1, total):5.1f}%)")
    print()
    print(f"labels.csv        : {len(accepted)} training pairs")
    print(f"labels_review.csv : {len(review)} needing review")
    uniq = len({a[1].lower() for a in accepted})
    print(f"unique words      : {uniq}")


if __name__ == "__main__":
    main()
