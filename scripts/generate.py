#!/usr/bin/env python3
"""
generate.py -- CLI inference driver for the FW-GAN handwriting synthesis model.

Given an arbitrary text string and one or more style-reference crops (black ink
on white background), this loads the pretrained FW-GAN checkpoint and generates
one PNG image per word, plus a concatenated single-line preview image.

Using the pretrained (non-fine-tuned) checkpoint is EXPECTED to produce generic
IAM/English-style handwriting, not any particular person's handwriting -- the
goal here is just to prove the generation mechanism works end-to-end.
"""
import os
import sys
from glob import glob

# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT = "/Users/blakey5aces/Handwriting Analysis"
FW = os.path.join(PROJECT, "models", "FW_GAN")
sys.path.insert(0, FW)

import argparse
import math

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lib.utils import yaml2config
from networks import get_model


def default_style_refs(n=3):
    """Pick style references that actually exist right now.

    This used to be a hardcoded filename ('sample_synth_L01_W04.png') from the
    old crop naming scheme. Crops are now named '{page}__{box_id}.png', so the
    old file no longer exists and every run that did not pass --style died with
    "no valid style references remained after filtering" -- a confusing error
    for something the caller never set. Resolving against the manifest means
    the default cannot go stale again when naming changes.

    Widest crops are preferred: a longer word carries more of the writer's
    letterforms, which is what the style encoder is reading.
    """
    seg = os.path.join(PROJECT, "segmented")
    man = os.path.join(seg, "manifest.csv")
    cands = []
    if os.path.exists(man):
        import csv as _csv
        with open(man, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                pth = r.get("crop_path", "")
                if pth and os.path.exists(pth):
                    cands.append(pth)
    if not cands:
        cands = sorted(glob(os.path.join(seg, "*.png")))
    if not cands:
        return []
    try:
        from PIL import Image as _Im
        cands.sort(key=lambda q: -_Im.open(q).width)
    except Exception:
        pass
    return cands[:n]


# The model's alphabet is plain ASCII, so Spanish/French text used to come out
# with its accented letters silently deleted ("más" -> "ms", "año" -> "ao").
# Instead each such letter is generated as its base letter and the mark is drawn
# on afterwards. The generator lays every character in its own char_width-wide
# slot, so the letter's column is known exactly and the mark lands over it.
_COMBINING = {"́": "acute", "̀": "grave", "̃": "tilde",
              "̈": "diaeresis", "̂": "circumflex"}
_INVERTED = {"¿": "?", "¡": "!"}
# The model's own full stop and comma come out as a stray letter ("clase." read
# as "clases"), so a word's trailing ones are drawn as a plain pen dot instead.
_TRAILING = ".,"


def split_marks(word, alphabet):
    """(base word the model can draw, [(char index, mark kind), ...])."""
    import unicodedata
    base, marks = [], []
    tail = len(word.rstrip(_TRAILING))
    if tail == 0:
        tail = len(word)                     # a lone "." is still a word
    for ch in word[tail:]:
        marks.append((None, "append:" + ch))
    word = word[:tail]
    for ch in word:
        if ch in _INVERTED:
            # Not part of the base word: drawn as its own glyph, turned over,
            # and spliced in. Rotating it inside the word mangled its
            # neighbour, since the model lets strokes cross slot boundaries.
            marks.append((len(base), "invert:" + _INVERTED[ch]))
            continue
        if ch in alphabet:
            base.append(ch)
            continue
        parts = unicodedata.normalize("NFD", ch)
        if parts[0] in alphabet and all(p in _COMBINING for p in parts[1:]):
            for p in parts[1:]:
                marks.append((len(base), _COMBINING[p]))
            base.append(parts[0])
    return "".join(base), marks


def letter_band(img):
    """(top, base) rows of the word's letter bodies: x-height top and baseline.

    Taken from the sharpest rise and fall of the row-by-row ink profile, which
    ignores thin ascenders and descenders. A plain threshold on the profile was
    thrown by any word with a lot of tall or descending letters.
    """
    h = img.shape[0]
    prof = np.clip((200.0 - img.astype(np.float32)) / 150.0, 0, 1).sum(axis=1)
    if prof.max() <= 0:
        return h / 3.0, 2 * h / 3.0
    sm = np.convolve(prof, [1, 2, 3, 2, 1], "same") / 9.0
    d = np.diff(sm)
    # Same rule as doc.ink_baseline: the first hard fall after which the
    # letter bodies have mostly run out of ink, not merely the hardest fall.
    falls = [r for r in range(1, len(d) - 1)
             if d[r] < d[r - 1] and d[r] <= d[r + 1] and d[r] <= 0.6 * d.min()]
    ends = [r for r in falls if sm[r + 1] <= 0.65 * sm.max()]
    base = (ends[0] if ends else int(np.argmin(d))) + 1
    top = int(np.argmax(d[:max(1, base - 3)])) + 1
    return float(top), float(base)


def _drop_dot(img, x0, x1, top):
    """Erase an i/j's own dot so an accent can take its place.

    Left in, the dot under the accent reads as a diaeresis: "días" came out
    looking like "dïas".
    """
    from scipy import ndimage
    lab, n = ndimage.label(img < 215)     # faint dots count too
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        ys, xs = sl
        if (ys.stop <= top + 1 and ys.stop - ys.start <= 7
                and xs.start >= x0 - 3 and xs.stop <= x1 + 3):
            img[sl][lab[sl] == k] = 255


# First letters the model draws as if they began a sentence when they start a
# word: m as "M"/"H", n as "h", s as "S", e as "E", a lone a as a hook. Other
# letters were measured to come out the same with or without the lead space,
# and leading with it costs a stray mark that has to be cut away again.
LEAD_LETTERS = "mnsea"


def lead_cut(img, char_width):
    """Column where the throwaway leading space ends and the word begins.

    The model leaves a stray mark in the space's slot, so the cut goes through
    the emptiest column between that mark and the first letter.
    """
    prof = (img < 200).sum(axis=0).astype(np.float32)
    prof = np.convolve(prof, [1, 1, 1], "same")
    lo, hi = 4, min(len(prof) - 1, char_width + 8)
    if hi <= lo:
        return 0
    seg = prof[lo:hi + 1]
    # Rightmost of the emptiest columns: the stray mark sits at the far left.
    return lo + int(np.flatnonzero(seg == seg.min())[-1])


def drop_stray(img, first_letter):
    """Erase what is left of the stray mark at the word's left edge.

    The space's mark is a single thin stroke, drawn apart from the first
    letter. It goes only when it is clearly narrower than that letter could be.
    """
    from scipy import ndimage
    mask = img < 215
    lab, _ = ndimage.label(mask)
    objs = ndimage.find_objects(lab)
    if len(objs) < 2:
        return img
    k, sl = min(enumerate(objs, 1), key=lambda o: o[1][1].start)
    ys, xs = sl
    limit = 4 if first_letter in "se" else 6
    if xs.start <= 3 and xs.stop - xs.start <= limit and ys.stop - ys.start >= 7:
        img = img.copy()
        img[sl][lab[sl] == k] = 255
    return img


def letter_span(G, enc_z, word, idx, device, base_img):
    """(left, right, centre) columns where the generator drew word[idx].

    The generator does not keep letters in their nominal char_width slots -- a
    narrow "l" pulls everything after it left -- so an accent placed by slot
    landed on the "t" of "lástima" and vanished into it. (The model's own
    recogniser was no help: its alignment is just as slot-bound.) Instead the
    word is drawn again with the same noise and only that letter swapped; the
    columns that change are the letter's own.
    """
    swap = "l" if word[idx] != "l" else "o"
    alt = word[:idx] + swap + word[idx + 1:]
    import torch
    lbs = torch.LongTensor(_ENCODE(alt)).unsqueeze(0).to(device)
    other = G(enc_z, lbs, torch.IntTensor([len(alt)]).to(device))[0, 0, :, :base_img.shape[1]]
    other = (255 * ((other + 1) / 2)).clamp(0, 255).cpu().numpy()
    diff = np.abs(base_img.astype(np.float32) - other).sum(axis=0)
    diff = np.convolve(diff, np.ones(5) / 5, "same")
    if diff.max() <= 0:
        return None
    peak = int(np.argmax(diff))
    lo, hi = peak, peak
    while lo > 0 and diff[lo - 1] > 0.35 * diff[peak]:
        lo -= 1
    while hi < len(diff) - 1 and diff[hi + 1] > 0.35 * diff[peak]:
        hi += 1
    xs = np.arange(lo, hi + 1)
    return lo, hi, float((xs * diff[lo:hi + 1]).sum() / diff[lo:hi + 1].sum())


def letter_centre(G, enc_z, word, idx, device, base_img):
    span = letter_span(G, enc_z, word, idx, device, base_img)
    return span[2] if span else None


_ENCODE = None     # set in main() to the model's label encoder


def draw_marks(img, marks, char_width, seed=0, word="", centres=None):
    """Draw diacritics onto a generated word image (uint8, white ground)."""
    if not marks:
        return img
    from PIL import ImageDraw
    img = img.copy()
    h, w = img.shape
    band_top, _ = letter_band(img)
    ink_px = img[img < 128]
    ink = int(np.percentile(ink_px, 30)) if ink_px.size else 40
    rng = np.random.default_rng(seed)
    S = 4                                    # draw supersampled, then shrink
    for idx, kind in marks:
        if kind.startswith(("invert", "append")):
            continue
        if centres and centres.get(idx) is not None:
            half = char_width * 0.4
            x0, x1 = max(0, int(centres[idx] - half)), min(w, int(centres[idx] + half) + 1)
        else:
            x0, x1 = idx * char_width, min(w, (idx + 1) * char_width)
        if idx < len(word) and word[idx] in "ij":
            _drop_dot(img, x0, x1, band_top)
        col = img[:, x0:x1] < 128
        rows = np.where(col.any(axis=1))[0]
        cols = np.where(col.any(axis=0))[0]
        top = int(rows[0]) if rows.size else h // 3
        if idx < len(word) and word[idx].islower():
            # A lowercase letter's accent sits on its x-height. The slot's
            # first ink row is often a neighbour's ascender reaching in -- the
            # d of "días" -- which put the accent above the image top, gone.
            top = band_top
        if centres and centres.get(idx) is not None:
            cx = float(centres[idx])
        else:
            cx = x0 + (float(cols.mean()) if cols.size else char_width / 2)
        cx += rng.uniform(-1, 1)
        y = max(6.0, top - 2.5)              # baseline of the mark
        layer = Image.new("L", (w * S, h * S), 0)
        d = ImageDraw.Draw(layer)
        lw = 2 * S
        if kind == "acute":
            d.line([((cx - 1.5) * S, y * S), ((cx + 2.5) * S, (y - 5) * S)], fill=255, width=lw)
        elif kind == "grave":
            d.line([((cx + 1.5) * S, y * S), ((cx - 2.5) * S, (y - 5) * S)], fill=255, width=lw)
        elif kind == "tilde":
            pts = [((cx - 4 + t) * S, (y - 2 - 1.6 * math.sin(t / 8 * 2 * math.pi)) * S)
                   for t in np.linspace(0, 8, 12)]
            d.line(pts, fill=255, width=lw, joint="curve")
        elif kind == "diaeresis":
            for dx in (-2.5, 2.5):
                d.ellipse([((cx + dx) - 1.1) * S, (y - 3.2) * S, ((cx + dx) + 1.1) * S, (y - 1) * S], fill=255)
        elif kind == "circumflex":
            d.line([((cx - 3) * S, (y - 1) * S), (cx * S, (y - 5) * S), ((cx + 3) * S, (y - 1) * S)],
                   fill=255, width=lw, joint="curve")
        a = np.asarray(layer.resize((w, h), Image.LANCZOS), np.float32) / 255.0
        img = np.minimum(img, (255 * (1 - a) + ink * a)).astype(np.uint8)
    return img


def inverted_glyph(word_img, glyph):
    """Pen-drawn ¿ or ¡ sized to the word's own letter band.

    Asking the model for a lone "?" and turning it over came out as an
    unreadable squiggle, so these two are drawn as strokes instead.
    """
    from PIL import ImageDraw
    h = word_img.shape[0]
    ink_mask = word_img < 128
    top, base = letter_band(word_img)
    ink_px = word_img[ink_mask]
    ink = int(np.percentile(ink_px, 30)) if ink_px.size else 40
    S, gw = 4, 13
    layer = Image.new("L", (gw * S, h * S), 0)
    d = ImageDraw.Draw(layer)
    cx = 7.5
    d.ellipse([(cx - 1.2) * S, (top - 1.2) * S, (cx + 1.2) * S, (top + 1.2) * S], fill=255)
    if glyph == "?":
        stem_end = (top + base) / 2
        d.line([(cx * S, (top + 4) * S), (cx * S, stem_end * S)], fill=255, width=2 * S)
        r = min(4.0, max(2.5, (base - top) / 3.0))
        cy = stem_end + r - 0.5
        pts = [((cx + r * math.cos(t)) * S, (cy + r * math.sin(t)) * S)
               for t in np.linspace(-math.pi / 2, -math.pi / 2 - 1.6 * math.pi, 18)]
        d.line(pts, fill=255, width=2 * S, joint="curve")
    else:
        d.line([(cx * S, (top + 4) * S), ((cx - 0.4) * S, min(h - 2, base + 5) * S)], fill=255, width=2 * S)
    a = np.asarray(layer.resize((gw, h), Image.LANCZOS), np.float32) / 255.0
    return (255 * (1 - a) + ink * a).astype(np.uint8)


def trailing_glyph(word_img, glyph, seed=0, last="", char_width=16):
    """A pen full stop or comma, sitting where the last letter stands.

    The last letter's own foot is the surest baseline for a mark that follows
    it; the whole-word profile drifts on short words. A letter that descends
    gives no foot to go by, so those fall back to the word's band.
    """
    from PIL import ImageDraw
    h = word_img.shape[0]
    _, base = letter_band(word_img)
    if last and last not in "gjpqy,;":
        # The model often finishes a word short of its last slot, so look at
        # the last letter's worth of actual ink rather than the slot.
        # Faint strokes count: this model draws plenty of ink lighter than 128.
        inked = np.where((word_img < 175).any(axis=0))[0]
        right = int(inked[-1]) + 1 if inked.size else word_img.shape[1]
        foot = word_img[:, max(0, right - char_width):right] < 175
        rows = np.where(foot.sum(axis=1) >= 1)[0]
        if rows.size:
            base = float(rows[-1])
    ink_px = word_img[word_img < 128]
    ink = int(np.percentile(ink_px, 30)) if ink_px.size else 40
    rng = np.random.default_rng(seed)
    S, gw = 4, 7
    layer = Image.new("L", (gw * S, h * S), 0)
    d = ImageDraw.Draw(layer)
    cx, cy, r = 2.5 + rng.uniform(-0.5, 0.5), base - 1.2, 1.25
    d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=255)
    if glyph == ",":
        d.line([((cx + 0.6) * S, cy * S), ((cx - 1.2) * S, (cy + 4) * S)], fill=255, width=int(1.8 * S))
    a = np.asarray(layer.resize((gw, h), Image.LANCZOS), np.float32) / 255.0
    return (255 * (1 - a) + ink * a).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description="Generate handwriting-style word images with FW-GAN.")
    p.add_argument("--text", required=True, help="Arbitrary text string; words separated by spaces.")
    p.add_argument(
        "--style",
        nargs="+",
        default=None,   # resolved by default_style_refs() -- see why below
        help="One or more style-reference image paths (segmented/*.png crops). "
             "Defaults to real crops picked from the current manifest.",
    )
    p.add_argument(
        "--ckpt",
        default=os.path.join(PROJECT, "models", "weights", "FW-GAN.pth"),
        help="Path to the FW-GAN checkpoint.",
    )
    p.add_argument(
        "--config",
        default=os.path.join(FW, "configs", "fw_gan_iam.yml"),
        help="Path to the FW-GAN yaml config.",
    )
    p.add_argument("--output-dir", default=os.path.join(PROJECT, "output"), help="Directory to write PNG outputs.")
    p.add_argument("--device", default="mps", help="Torch device to run on (mps/cpu/cuda).")
    p.add_argument(
        "--seeds",
        help="Comma-separated seed per word. Each word's noise is drawn from its own "
             "seed, so a word's appearance depends only on itself -- adding or removing "
             "a word cannot change how any other word looks, and re-rolling one word is "
             "reproducible. Omit for the previous behaviour (fresh randomness each run).",
    )
    args = p.parse_args()
    if not args.style:
        args.style = default_style_refs()
        if not args.style:
            print("ERROR: no style reference given and no crops found in "
                  "segmented/. Run detection first, or pass --style.",
                  file=sys.stderr)
            sys.exit(1)
    return args


def resolve(path):
    """Resolve a possibly-relative path against the project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT, path)


def main():
    args = parse_args()

    ckpt_path = resolve(args.ckpt)
    config_path = resolve(args.config)
    output_dir = resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    device = args.device

    # --- Build config -----------------------------------------------------
    cfg = yaml2config(config_path)
    cfg.device = device
    cfg.ckpt = ckpt_path
    # Make the lexicon path absolute so get_lexicon() (called during
    # AdversarialModel.__init__) can find it regardless of cwd.
    cfg.training.lexicon = os.path.join(FW, "data", "english_words.txt")

    # --- Build + load model -------------------------------------------------
    model = get_model(cfg.model)(cfg, log_root=output_dir)
    model.set_mode("eval")
    epoch = model.load(cfg.ckpt, map_location="cpu")
    print(f"Loaded checkpoint '{ckpt_path}' (epoch {epoch})")

    # These mirror lib/datasets.py collect_fn + lib/path_config.py -- read
    # from config rather than hardcoding.
    img_height = cfg.img_height  # 32
    char_width = cfg.char_width  # 16
    gen_style_dim = cfg.GenModel.style_dim  # 128
    enc_style_dim = cfg.EncModel.style_dim  # 96
    noise_dim = gen_style_dim - enc_style_dim  # 32
    len_scale = img_height // 2  # 16 -- StyleEncoder divides img_len by 16

    style_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # --- Style refs -> style_mu -------------------------------------------
    style_mus = []
    for style_path in args.style:
        style_path = resolve(style_path)
        try:
            img = Image.open(style_path).convert("L")
        except Exception as e:
            print(f"WARNING: failed to load style ref '{style_path}': {e}", file=sys.stderr)
            continue

        w, h = img.size
        new_w = max(1, round(w * img_height / h))
        img = img.resize((new_w, img_height))

        if new_w < len_scale:
            print(
                f"WARNING: style ref '{style_path}' too narrow (<{len_scale}px after resize), skipping",
                file=sys.stderr,
            )
            continue

        norm = style_tf(img)  # (1, 32, new_w), white bg -> +1, ink -> -1

        pad_w = math.ceil(new_w / len_scale) * len_scale
        padded = torch.full((1, 1, img_height, pad_w), -1.0)  # pad fill value per spec (collect_fn convention)
        padded[..., :new_w] = norm.unsqueeze(0)

        padded = padded.to(device)
        with torch.no_grad():
            mu = model.models.E(padded, torch.tensor([pad_w], dtype=torch.int, device=device), model.models.S)
        style_mus.append(mu)
        print(f"Encoded style ref '{style_path}' -> mu shape {tuple(mu.shape)}")

    if len(style_mus) == 0:
        print("ERROR: no valid style references remained after filtering.", file=sys.stderr)
        sys.exit(1)

    style_mu = torch.stack(style_mus, dim=0).mean(dim=0)
    if not torch.isfinite(style_mu).all():
        print("ERROR: style_mu contains non-finite values.", file=sys.stderr)
        sys.exit(1)

    # --- Text -> words, with OOV filtering ---------------------------------
    alphabet_dict = model.label_converter.dict
    global _ENCODE
    _ENCODE = model.label_converter.encode
    max_word_len = cfg.training.max_word_len

    raw_words = args.text.split()
    words = []
    word_marks = []
    for raw_word in raw_words:
        raw_word, marks = split_marks(raw_word, alphabet_dict)
        filtered_chars = [c for c in raw_word if c in alphabet_dict]
        dropped = [c for c in raw_word if c not in alphabet_dict]
        if dropped:
            print(f"WARNING: word '{raw_word}' contains unsupported chars {dropped!r}, dropping them", file=sys.stderr)
        word = "".join(filtered_chars)
        if not word:
            print(f"WARNING: word '{raw_word}' became empty after filtering, skipping", file=sys.stderr)
            continue
        if len(word) > max_word_len:
            print(f"WARNING: word '{word}' exceeds max_word_len={max_word_len}, quality may degrade", file=sys.stderr)
        words.append(word)
        word_marks.append(marks)

    if not words:
        print("ERROR: no words survived alphabet filtering.", file=sys.stderr)
        sys.exit(1)

    # --- Generate per word --------------------------------------------------
    seeds = None
    if getattr(args, "seeds", None):
        seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
        if len(seeds) != len(words):
            print(f"WARNING: {len(seeds)} seed(s) for {len(words)} word(s); cycling",
                  file=sys.stderr)

    word_images = []
    with torch.no_grad():
        for idx, word in enumerate(words):
            # A word starting lowercase is drawn with a space in front and the
            # space cut off afterwards. On its own the model draws a word's
            # first letter as if it began a sentence -- "me" came out "Mc",
            # "no" as "ho", "es" as "Es" -- and a leading space is enough to
            # make it write the letter as it would mid-line.
            lead = word[0] in LEAD_LETTERS
            gword = " " + word if lead else word
            marks = [(i + 1 if (lead and i is not None) else i, k) for i, k in word_marks[idx]]
            enc = model.label_converter.encode(gword)  # plain string -> flat index list
            word_lbs = torch.LongTensor(enc).unsqueeze(0).to(device)  # (1, L)
            word_lb_lens = torch.IntTensor([len(gword)]).to(device)  # (1,)

            if seeds is not None:
                # Per word, not once per run: seeding once would make every word's
                # noise depend on how many words came before it.
                gen = torch.Generator(device="cpu").manual_seed(int(seeds[idx % len(seeds)]))
                noise = torch.randn((1, noise_dim), generator=gen).to(device)
            else:
                noise = torch.randn((1, noise_dim), device=device)  # (1, 32)
            enc_z = torch.cat([noise, style_mu], dim=1)  # (1, 128)

            fake = model.models.G(enc_z, word_lbs, word_lb_lens)  # (1,1,32, L*char_width)
            fake = fake[:, :, :, : len(gword) * char_width]

            assert torch.isfinite(fake).all(), f"non-finite output for word {word!r}"

            img = fake[0, 0]
            img = (255 * ((img + 1) / 2)).clamp(0, 255).to(torch.uint8).cpu().numpy()
            seed_i = seeds[idx % len(seeds)] if seeds else idx
            centres = None
            accented = [i for i, k in marks if not k.startswith(("append", "invert"))]
            if accented:
                centres = {i: letter_centre(model.models.G, enc_z, gword, i, device, img)
                           for i in accented}
            img = draw_marks(img, marks, char_width, seed=seed_i, word=gword,
                             centres=centres)
            if lead:
                img = drop_stray(img[:, lead_cut(img, char_width):], word[0])
            for pos, kind in reversed(word_marks[idx]):
                if kind.startswith("invert:"):
                    g = inverted_glyph(img, kind.split(":", 1)[1])
                    if pos == 0:
                        cut = 0
                    else:
                        cut = pos * char_width
                    img = np.concatenate([img[:, :cut], g, img[:, cut:]], axis=1)
            for _, kind in word_marks[idx]:
                if kind.startswith("append:"):
                    img = np.concatenate([img, trailing_glyph(img, kind[-1], seed_i, word[-1:], char_width)], axis=1)

            img_min, img_max, img_mean = int(img.min()), int(img.max()), float(img.mean())
            if img_max - img_min < 5:
                print(f"WARNING: possible blank/garbage output for word '{word}' (min={img_min}, max={img_max})", file=sys.stderr)

            safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in word)[:40]
            out_path = os.path.join(output_dir, f"{idx:03d}_{safe}.png")
            Image.fromarray(img, mode="L").save(out_path)
            print(f"word='{word}' min={img_min} max={img_max} mean={img_mean:.1f} -> saved {out_path}")

            word_images.append(img)

    # --- Concatenate into one line strip for visual QA ----------------------
    if word_images:
        sep = np.full((img_height, 16), 255, dtype=np.uint8)
        pieces = []
        for i, wi in enumerate(word_images):
            if i > 0:
                pieces.append(sep)
            pieces.append(wi)
        line = np.concatenate(pieces, axis=1)
        line_path = os.path.join(output_dir, "_line.png")
        Image.fromarray(line, mode="L").save(line_path)
        print(f"Saved combined line preview -> {line_path}")

    print(f"\nDone: generated {len(word_images)} word image(s) using {len(style_mus)} style ref(s). Output dir: {output_dir}")


if __name__ == "__main__":
    main()
