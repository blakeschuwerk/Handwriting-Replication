#!/usr/bin/env python3
"""
wordcheck.py -- decide whether a proposed label is plausible, and if not, what
it probably should have been.

WHY NOT JUST A SPELL CHECKER
----------------------------
A plain dictionary answers "is this a word", which flags "Manasement" but has no
opinion on what to replace it with -- and against Webster's 2nd (the macOS word
list) "Manasement" is one edit from *several* archaic entries, so a generic
corrector picks badly.

The trick is that these pages are one document about one subject. "management"
and "operations" appear dozens of times, and most of them Vision read correctly.
So the corpus itself is a far better prior than any general dictionary:

    Manasement -> management   (edit 1, seen 51x in this document)
    orerations -> operations   (edit 1, seen 44x)
    regources  -> resources    (edit 1, seen  9x)

A candidate that is one edit away from a word this writer demonstrably uses over
and over is a much safer bet than one edit away from a word in a 1934 lexicon.
That is the whole idea: rank corrections by DOCUMENT frequency first, dictionary
membership second.

Suggestions are never auto-applied -- see curate_labels.py for what happened the
last time corrections were trusted blindly. They are offered to a human who is
already looking at the crop.

WHAT THIS CANNOT DO
-------------------
Real-word errors are invisible to it. Vision reading "work" as "wore" produces a
perfectly good English word attached to the wrong image, and nothing here will
object. Only looking at the crop catches those.
"""

import csv
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curate_labels import (  # noqa: E402
    load_dictionary, edits1, EXTRA_OK,
)

# TRUE inflections only. curate_labels.inflection_ok also strips the
# derivational suffixes -ment/-tion/-ness, which is too generous for judging a
# single word: "mangement" gets accepted as mange + -ment, so the one error the
# whole feature exists to catch slips through silently. Real derived nouns like
# "management" are in the dictionary in their own right, so nothing is lost.
_SUFFIX = ("ed", "ing", "ly", "er", "est")
_MIN_BASE = 3     # without this, "ving" = "v" + -ing passes ("v" is in Webster's)

# Suffixes that Webster's also lists as standalone entries. Without excluding
# them the compound rule reads "mangement" as mange + ment and accepts it.
_BOUND = {"ment", "tion", "sion", "ness", "able", "ible", "ance", "ence",
          "ing", "ment", "ally", "ity", "ive", "ous", "ful", "less"}

# Below this, a box holds no letter-shaped ink worth training on. Deliberately
# conservative: the page is dense and textured (ruled lines, spiral shadow), so
# a tighter threshold produces false alarms on legitimately sparse crops.
BLANK_INK = 0.03


def _inflected(stem, vocab):
    """True if `stem` is a dictionary word or a regular inflection of one."""
    if stem in vocab or stem in EXTRA_OK:
        return True
    if stem.endswith("ies") and len(stem) > 5 and (stem[:-3] + "y") in vocab:
        return True
    if stem.endswith("es") and len(stem) > 4 and (stem[:-2] in vocab or stem[:-1] in vocab):
        return True
    if stem.endswith("s") and len(stem) > 3 and stem[:-1] in vocab:
        return True
    for suf in _SUFFIX:
        if not stem.endswith(suf):
            continue
        b = stem[: -len(suf)]
        if len(b) < _MIN_BASE:
            continue
        if b in vocab or (b + "e") in vocab:
            return True
        if b.endswith("i") and (b[:-1] + "y") in vocab:
            return True
        if b[-1] == b[-2] and b[:-1] in vocab:      # "planning" -> "plan"
            return True
    return False


def _plural_ok(stem, vocab):
    """Narrower than _inflected(): only the plural/possessive rules.

    Used to accept a suggestion CANDIDATE, where _inflected()'s derivational
    suffixes (-ing, -est, ...) are too permissive -- they invent non-words
    like "forcest" (force+est) or "viling" (vile+ing) as if they were valid,
    which is fine for judging whether a human's own word is plausible but
    actively wrong when it lets a nonsense word win as the suggested fix.
    """
    if stem in vocab:
        return True
    if stem.endswith("ies") and len(stem) > 5 and (stem[:-3] + "y") in vocab:
        return True
    if stem.endswith("es") and len(stem) > 4 and (stem[:-2] in vocab or stem[:-1] in vocab):
        return True
    if stem.endswith("s") and len(stem) > 3 and stem[:-1] in vocab:
        return True
    return False


def _is_initialism(text):
    """HR, OM, KPI -- short and mostly capitals. Never a spelling mistake."""
    t = text.strip(".")
    return 1 < len(t) <= 4 and sum(c.isupper() for c in t) >= max(2, len(t) - 1)


def _dist_le(a, b, k):
    """Levenshtein distance if <= k, else None. Banded, so it exits early."""
    la, lb = len(a), len(b)
    if abs(la - lb) > k:
        return None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        lo, hi = max(1, i - k), min(lb, i + k)
        for j in range(1, lb + 1):
            if j < lo or j > hi:
                cur[j] = k + 1
                continue
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        if min(cur[lo:hi + 1] or [k + 1]) > k:
            return None
        prev = cur
    return prev[lb] if prev[lb] <= k else None


class Lexicon:
    """System dictionary + a frequency-weighted vocabulary of THIS document."""

    def __init__(self, vocab, doc_freq):
        self.vocab = vocab
        self.doc = doc_freq                      # lowercase word -> count
        # Words the document uses repeatedly AND that are real: the trusted core.
        self.strong = {w for w, c in doc_freq.items() if c >= 2}

    @classmethod
    def build(cls, root):
        vocab = load_dictionary()
        freq = Counter()
        man = os.path.join(root, "segmented", "manifest.csv")
        if os.path.exists(man):
            with open(man, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    t = (r.get("auto_text") or "").strip().lower()
                    t = t.strip(".,;:!?'\"")
                    if not re.fullmatch(r"[a-z][a-z'-]{1,}", t):
                        continue
                    # Only count it if it is genuinely a word; otherwise a
                    # repeated misreading would vote for itself.
                    if _inflected(t, vocab):
                        freq[t] += 1
        return cls(vocab, freq)

    def known(self, stem):
        return _inflected(stem, self.vocab) or stem in self.doc

    def compound(self, stem):
        """A run-together pair of substantial real words: workforce, eachother.

        Both halves must be >=4 letters. Short prefixes are what make spurious
        splits -- "forcast" is a genuine misreading, but "for"+"cast" are both
        words, so a laxer rule would wave it through as a compound.
        """
        for i in range(4, len(stem) - 3):
            a, b = stem[:i], stem[i:]
            if a in _BOUND or b in _BOUND:
                continue
            if a in self.vocab and b in self.vocab:
                return a, b
        return None

    def suggest(self, stem):
        """Best replacement for an unknown token, or None.

        Ordered by how much evidence backs the candidate, strongest first.
        """
        e1 = edits1(stem)
        # An edit-1 neighbour is only a candidate if IT is a real word. Testing
        # raw set membership here would miss inflected forms that are valid
        # only through _inflected() (e.g. "efficiencies", whose base form
        # "efficiency" is the dictionary entry) -- exactly the gap that let
        # "efficencies" fall through with no suggestion at all.
        e1 = {w for w in e1 if w in self.vocab or _plural_ok(w, self.vocab)}

        # 1. one edit from a word this document uses repeatedly -- the best signal
        hits = [(self.doc[w], w) for w in e1 & self.strong]
        if hits:
            return max(hits)[1], "doc"

        # 2. a UNIQUE dictionary neighbour (e1 is already filtered to real
        #    words). Ahead of the fuzzier searches below because it is exact:
        #    "forcast" -> "forecast" beats splitting it into "for cast", which
        #    is what a split-first order produced.
        cands = e1
        if len(cands) == 1:
            return next(iter(cands)), "dict"

        # 3. one edit from any word seen in the document, even once
        hits = [(self.doc[w], w) for w in e1 if w in self.doc]
        if hits:
            return max(hits)[1], "doc"

        # 4. two edits from a repeated document word (searched directly against
        #    the small doc vocabulary rather than generating ~290k strings)
        best = None
        for w in self.strong:
            d = _dist_le(stem, w, 2)
            if d is not None and (best is None or (d, -self.doc[w]) < best[0]):
                best = ((d, -self.doc[w]), w)
        if best:
            return best[1], "doc2"

        # 5. last resort: several dictionary neighbours, take the one this
        #    document actually uses, if any
        doc_cands = [(self.doc[w], w) for w in cands if w in self.doc]
        if doc_cands:
            return max(doc_cands)[1], "dict"
        return None, None


def _recase(src, repl):
    if src[:1].isupper():
        return " ".join(p.capitalize() for p in repl.split())
    return repl


def judge(text, lex, ink=None):
    """(flag, suggestion, source) -- flag is None when the label looks fine.

    `ink` is the letter-ink fraction of the crop when the caller has it; it is
    an independent check that the box actually contains something, which no
    amount of spelling analysis can tell you.
    """
    t = (text or "").strip()
    if ink is not None and ink < BLANK_INK:
        return "blank", None, None
    if not t:
        return "empty", None, None
    if re.fullmatch(r"[0-9]+([./-][0-9]+)*", t):
        return None, None, None

    core = t.lower().strip(".,;:!?'\"")
    poss = core.endswith("'s")
    stem = core[:-2] if poss else core

    if _is_initialism(t):
        return None, None, None
    if not re.fullmatch(r"[a-z][a-z'-]*", stem):
        return "not_a_word", None, None
    if len(stem) < 2 and stem not in EXTRA_OK:
        return "too_short", None, None
    if lex.known(stem):
        return None, None, None

    parts = [p for p in re.split(r"[-']", stem) if p]
    if len(parts) > 1 and all(lex.known(p) for p in parts):
        return None, None, None
    if lex.compound(stem):
        return None, None, None

    repl, src = lex.suggest(stem)
    if repl:
        repl = _recase(t, repl) + ("'s" if poss else "")
    return "misspelled", repl, src


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lex = Lexicon.build(root)
    print(f"dictionary {len(lex.vocab)}  document {len(lex.doc)} unique "
          f"({len(lex.strong)} seen 2+ times)\n")
    for t in sys.argv[1:]:
        print(f"  {t!r:>16} -> {judge(t, lex)}")
