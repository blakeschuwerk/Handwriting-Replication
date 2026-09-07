"""An editable page of handwriting.

Words stop being pixels burned into a photo and become records with stable
identities and positions in page space. That single change is what makes the
page editable -- dragging, deleting and reflowing are properties of the data
model, not of the rendering format, which is why none of this needs vectors.

Two rules the whole design rests on:

* **Identity is never positional.** ``scripts/boxstore.py`` learned this the
  hard way: when a crop's identity was its index in reading order, deleting one
  box silently re-pointed every later label at the wrong image. Words carry an
  opaque id for life and the cached PNG hangs off the record.

* **Positions are in page space**, ``(line, u)``, never pixels. The sheet can be
  refitted or hand-adjusted underneath a document and every word follows the
  paper. Since one unit of u is one line height, a word's width is
  ``aspect * scale`` on any line, so line breaking never has to know which line
  a word landed on before deciding.

Flow versus pinning
-------------------
Text operations behave like a word processor: delete a word and everything
after it flows back, cascading across line breaks. Dragging a word is not a
text operation -- it pins the word to a spot, and reflow packs around it. That
is the only way to have both "delete a word and the rest moves up" and "drag
two words without disturbing the paragraph", which a word processor cannot
actually do.
"""

import hashlib
import json
import math
import os
import uuid

VERSION = 1

DEFAULTS = {
    "scale": 1.05,
    "space_em": 0.28,
    "tab_em": 0.6,
    "start_line": 0,
    "darkness": 0.22,
    "humanize": 0.35,
}


def new_id():
    return uuid.uuid4().hex[:10]


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

def new_doc(photo, sheet=None, seed=None):
    return {
        "version": VERSION,
        "photo": photo,
        "seed": int(seed if seed is not None else uuid.uuid4().int % 1_000_000),
        "sheet": sheet.to_json() if sheet is not None else None,
        "style": {"crop": None, "ckpt": "pretrained"},
        "opts": dict(DEFAULTS),
        "words": [],
    }


def save(path, doc):
    """Atomic replace, so a crash mid-write cannot truncate the document."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh)
    os.replace(tmp, path)


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    if d.get("version", 1) > VERSION:
        raise ValueError(f"document is version {d['version']}, this build reads {VERSION}")
    return d


def new_word(text, ord_, nl=0, tabs=0):
    return {
        "id": new_id(), "text": text, "ord": int(ord_),
        "nl": int(nl), "tabs": int(tabs),
        "png": None, "aspect": None, "scale": 1.0, "slant": 0.0, "dark": 1.0,
        "pin": None, "placed": None, "overflow": False, "clipped": False,
        "gen": None,
    }


# ---------------------------------------------------------------------------
# Text <-> words
# ---------------------------------------------------------------------------

def parse_text(text):
    """Text -> [(word, newlines_before, tabs_before)], preserving structure."""
    out, pending_nl = [], 0
    for raw in text.split("\n"):
        stripped = raw.lstrip("\t")
        tabs = len(raw) - len(stripped)
        toks = stripped.split()
        if not toks:
            pending_nl += 1
            continue
        for k, w in enumerate(toks):
            out.append((w, pending_nl if k == 0 else 0, tabs if k == 0 else 0))
            pending_nl = 1
        pending_nl = 1
    return out


def _lcs_pairs(a, b):
    """Indices of a longest common subsequence between two token lists."""
    n, m = len(a), len(b)
    if not n or not m:
        return []
    # rolling DP; documents here are hundreds of words, not millions
    prev = [0] * (m + 1)
    table = [prev]
    for i in range(n):
        cur = [0] * (m + 1)
        ai = a[i]
        for j in range(m):
            cur[j + 1] = prev[j] + 1 if ai == b[j] else max(cur[j], prev[j + 1])
        table.append(cur)
        prev = cur
    out, i, j = [], n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and table[i][j] == table[i - 1][j - 1] + 1:
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def sync_text(doc, text):
    """Update the word list from edited text, keeping ids for words that stayed.

    Diffed rather than rebuilt: re-parsing into fresh ids would discard every
    pin and every cached image on each keystroke, so a single typo would
    regenerate the page and move everything the user had positioned by hand.
    """
    old = sorted(doc["words"], key=lambda w: w["ord"])
    tokens = parse_text(text)
    keep = dict(_lcs_pairs([w["text"] for w in old], [t[0] for t in tokens]))

    words, added, removed = [], 0, 0
    for j, (txt, nl, tabs) in enumerate(tokens):
        src = next((old[i] for i, jj in keep.items() if jj == j), None)
        if src is None:
            w = new_word(txt, j, nl, tabs)
            added += 1
        else:
            w = src
            w["ord"], w["nl"], w["tabs"] = j, nl, tabs
        words.append(w)
    removed = len(old) - (len(words) - added)
    doc["words"] = words
    doc["text"] = text
    return {"added": added, "removed": max(0, removed), "total": len(words)}


# ---------------------------------------------------------------------------
# Deterministic per-word variation
# ---------------------------------------------------------------------------

def _unit(seed, key):
    """A stable number in [0,1) from a document seed and a key."""
    h = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


def word_unit(doc, wid, salt):
    return _unit(doc.get("seed", 0), f"{wid}:{salt}")


def humanize_params(h):
    """Map one slider to the four things that actually vary in handwriting.

    Ranges chosen so the top of the slider still reads as a person writing
    quickly rather than as a rendering fault; h**1.3 keeps the low end gentle.
    """
    hh = max(0.0, min(1.0, float(h))) ** 1.3
    return {
        "baseline": 0.02 + 0.06 * hh,   # line heights
        "space_sd": 0.06 + 0.18 * hh,
        "size_sd": 0.01 + 0.05 * hh,
        "slant_sd": 0.5 + 2.5 * hh,     # degrees
    }


def baseline_drift(doc, line, u, amp):
    """Smooth drift along a line, as a function of position rather than of
    which words happen to be there.

    Real writing wanders off the rule gradually instead of hopping per word, so
    this interpolates value noise in u. Making it a function of u rather than of
    word index also means deleting a word does not change how its neighbours
    sit -- an index-based scheme quietly re-jitters the whole line.
    """
    i0 = math.floor(u)
    t = u - i0
    a = _unit(doc.get("seed", 0), f"drift:{line}:{i0}")
    b = _unit(doc.get("seed", 0), f"drift:{line}:{i0 + 1}")
    s = t * t * (3 - 2 * t)
    return ((a + (b - a) * s) * 2 - 1) * amp


# ---------------------------------------------------------------------------
# Reflow
# ---------------------------------------------------------------------------

def _free_intervals(sheet, blocked):
    """[u_left, u_right] minus the pinned words sitting on this line."""
    out, cur = [], sheet.u_left
    for a, b in sorted(blocked):
        if a > cur:
            out.append((cur, min(a, sheet.u_right)))
        cur = max(cur, b)
    if cur < sheet.u_right:
        out.append((cur, sheet.u_right))
    return [(a, b) for a, b in out if b - a > 1e-6]


def word_width(w, opts):
    return (w.get("aspect") or 3.0) * opts["scale"] * w.get("scale", 1.0)


def reflow(doc, sheet, opts=None):
    """Place every unpinned word, packing around the pinned ones.

    Words that run past the last rule are kept and flagged rather than dropped;
    losing text silently is worse than showing it as overflow.
    """
    o = dict(DEFAULTS)
    o.update(doc.get("opts") or {})
    o.update(opts or {})
    hp = humanize_params(o["humanize"])
    space = o["space_em"] * o["scale"]

    words = sorted(doc["words"], key=lambda w: w["ord"])
    first_line = sheet.i_first + int(o["start_line"])
    last_line = sheet.i_first + sheet.n_lines - 1

    # pinned words hold their place and become obstacles
    blocked = {}
    for w in words:
        p = w.get("pin")
        if not p:
            continue
        w["placed"] = {"line": int(p["line"]), "u": float(p["u"])}
        w["overflow"] = not (first_line <= p["line"] <= last_line)
        w["orphaned"] = not (sheet.i_first <= p["line"] <= last_line)
        blocked.setdefault(int(p["line"]), []).append(
            (p["u"] - space * 0.5, p["u"] + word_width(w, o) + space * 0.5))

    li = first_line
    spans = _free_intervals(sheet, blocked.get(li, []))
    si = 0
    u = spans[0][0] if spans else sheet.u_left
    first = True
    overflow, clipped = [], []

    def next_line(n=1):
        nonlocal li, spans, si, u
        li += n
        spans = _free_intervals(sheet, blocked.get(li, []))
        si = 0
        u = spans[0][0] if spans else sheet.u_left

    for w in words:
        if w.get("pin"):
            first = False
            continue
        if w["nl"] and not first:
            next_line(w["nl"])
        if w["tabs"]:
            u += w["tabs"] * o["tab_em"] * o["scale"]

        width = word_width(w, o)
        # find a gap this word fits in, spilling across intervals then lines
        guard = 0
        while li <= last_line and guard < 400:
            guard += 1
            if si < len(spans):
                lo, hi = spans[si]
                if u < lo:
                    u = lo
                if u + width <= hi or (u <= lo + 1e-9 and si == len(spans) - 1
                                       and hi - lo >= width):
                    break
                if u + width <= hi:
                    break
                si += 1
                if si < len(spans):
                    u = spans[si][0]
                    continue
            if si >= len(spans):
                next_line()
                continue
        if li > last_line:
            w["placed"] = None
            w["overflow"] = True
            overflow.append(w["id"])
            first = False
            continue

        lo, hi = spans[si] if si < len(spans) else (sheet.u_left, sheet.u_right)
        if width > hi - lo:
            w["clipped"] = True
            clipped.append(w["id"])
        else:
            w["clipped"] = False
        w["placed"] = {"line": int(li), "u": float(u)}
        w["overflow"] = False

        jitter = word_unit(doc, w["id"], "space") * 2 - 1
        u += width + space * (1 + jitter * hp["space_sd"])
        first = False

    return {"placed": sum(1 for w in words if w.get("placed") and not w["overflow"]),
            "total": len(words), "overflow": overflow, "clipped": clipped,
            "lines": (first_line, last_line)}


# ---------------------------------------------------------------------------
# Pinning
# ---------------------------------------------------------------------------

def pin(doc, ids, line, u, sheet=None, opts=None):
    """Pin a selection, keeping the words' spacing relative to each other."""
    o = dict(DEFAULTS)
    o.update(doc.get("opts") or {})
    o.update(opts or {})
    sel = [w for w in sorted(doc["words"], key=lambda w: w["ord"]) if w["id"] in set(ids)]
    if not sel:
        return 0
    group = new_id() if len(sel) > 1 else None
    cur = float(u)
    for w in sel:
        w["pin"] = {"line": int(line), "u": cur, "group": group}
        cur += word_width(w, o) + o["space_em"] * o["scale"]
    return len(sel)


def unpin(doc, ids):
    n = 0
    wanted = set(ids)
    for w in doc["words"]:
        if w["id"] in wanted and w.get("pin"):
            w["pin"] = None
            n += 1
    return n


def move_pins(doc, ids, dline, du):
    """Nudge already-pinned words; used by drag and by the arrow keys."""
    wanted = set(ids)
    for w in doc["words"]:
        if w["id"] in wanted and w.get("pin"):
            w["pin"]["line"] = int(w["pin"]["line"] + dline)
            w["pin"]["u"] = float(w["pin"]["u"] + du)


# ---------------------------------------------------------------------------
# Word images
# ---------------------------------------------------------------------------

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATE_PY = os.path.join(PROJECT, "scripts", "generate.py")
VENV_PYTHON = os.path.join(PROJECT, ".venv", "bin", "python3")


def word_seed(w):
    """A word's own seed, from its id and how many times it has been re-rolled.

    Derived rather than stored so it cannot drift out of step with the variant
    counter, and independent of neighbours so adding or deleting a word never
    changes how another word looks.
    """
    h = hashlib.blake2b(f"{w['id']}:{w.get('variant', 0)}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2 ** 31)


def _cache_key(w, style, ckpt):
    sig = f"{w['text']}|{word_seed(w)}|{style or ''}|{ckpt or ''}"
    return hashlib.blake2b(sig.encode(), digest_size=10).hexdigest()


def reroll(doc, ids):
    """Ask the generator for a different rendering of these words."""
    wanted = set(ids)
    n = 0
    for w in doc["words"]:
        if w["id"] in wanted:
            w["variant"] = int(w.get("variant", 0)) + 1
            w["png"] = None
            n += 1
    return n


def ensure_words(doc, cache_dir, style=None, ckpt=None, device="mps", log=None):
    """Generate images only for words that do not have one cached.

    The cache is keyed on the word's text, its own seed, the style crop and the
    checkpoint -- so editing one word regenerates exactly that word. Generating
    the whole page instead would re-roll every other word too, because the model
    is stochastic, and the page would visibly change under the user on every
    keystroke.
    """
    import subprocess
    from glob import glob as _glob
    from PIL import Image as _Image

    os.makedirs(cache_dir, exist_ok=True)
    need = []
    for w in doc["words"]:
        key = _cache_key(w, style, ckpt)
        path = os.path.join(cache_dir, key + ".png")
        if os.path.exists(path):
            w["png"] = path
            if not w.get("aspect"):
                iw, ih = _Image.open(path).size
                w["aspect"] = iw / ih
        else:
            need.append((w, path))

    if not need:
        return {"generated": 0, "cached": len(doc["words"])}

    if log:
        log(f"generating {len(need)} new word image(s); "
            f"{len(doc['words']) - len(need)} already cached")

    tmp = os.path.join(cache_dir, f"_tmp_{uuid.uuid4().hex[:8]}")
    os.makedirs(tmp, exist_ok=True)
    cmd = [VENV_PYTHON, GENERATE_PY,
           "--text", " ".join(w["text"] for w, _ in need),
           "--seeds", ",".join(str(word_seed(w)) for w, _ in need),
           "--device", device, "--output-dir", tmp]
    if style:
        cmd += ["--style", style]
    if ckpt:
        cmd += ["--ckpt", ckpt]
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"generate.py failed:\n{res.stderr[-2000:]}")

    pngs = [f for f in sorted(_glob(os.path.join(tmp, "*.png")))
            if not os.path.basename(f).startswith("_")]
    if len(pngs) != len(need):
        raise RuntimeError(f"asked for {len(need)} word images, got {len(pngs)} -- "
                           "a word was probably dropped by alphabet filtering")

    for (w, dest), src in zip(need, pngs):
        os.replace(src, dest)
        w["png"] = dest
        iw, ih = _Image.open(dest).size
        w["aspect"] = iw / ih
        w["gen"] = {"style": style, "ckpt": ckpt, "seed": word_seed(w)}
    for junk in _glob(os.path.join(tmp, "*")):
        os.remove(junk)
    os.rmdir(tmp)
    return {"generated": len(need), "cached": len(doc["words"]) - len(need)}
