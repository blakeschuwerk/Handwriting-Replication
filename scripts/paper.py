"""Write generated handwriting onto a photo of real ruled paper.

photo -> find the ruling -> lay text out on the rules -> composite.

Three decisions worth knowing about, because none of them are the obvious
choice and all three were arrived at by measuring:

* Rules are found in the blue-minus-red channel, not luminance. They are blue
  ink on white paper, so B-R isolates them; being a channel *difference* it
  also cancels the lighting gradient that wrecks a luminance threshold.

* The page is located from the ruling itself, never from the paper outline. On
  a photo of an open notebook, thresholding brightness merges both pages into
  one blob, but only the page we want carries blue rules.

* Nothing is ever rectified. Words are placed directly in photo space using
  only the local geometry of the rule they sit on. A global homography assumes
  the page is planar, which a real notebook near the spine is not; local
  placement follows the curve for free, and the output keeps the photo's own
  perspective so it reads as a photograph rather than a scan.
"""

import math
import os
import subprocess
import sys
import tempfile
import time
from glob import glob

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_page import ink_alpha  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATE_PY = os.path.join(PROJECT, "scripts", "generate.py")
VENV_PYTHON = os.path.join(PROJECT, ".venv", "bin", "python3")
TEST_PAGE = os.path.join(PROJECT, "assets", "test_page.jpg")

BLUE_SIGMA = 51       # flat-field blur: >> rule thickness, << page size
N_SLABS = 16          # vertical slabs used to follow rules through perspective

# The generator centres each word in a fixed 32px canvas. Ink bounding boxes
# vary wildly between words -- one with no ascender starts a third of the way
# down -- so words MUST be aligned by a fixed fraction of the canvas and never
# by their own ink bbox, or the text visibly bounces.
#
# 0.907 measured over 282 generated words as the row where a word carrying no
# descender runs out of ink. Words with descenders end only ~0.03 lower, i.e.
# the model barely reserves a descender zone, so this doubles as the bottom.
# Eyeballing this put it at 0.766, which floats the text a visible ~13px above
# the rule; trust the measurement.
BASELINE_FRAC = 0.907


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_photo(path):
    """Read as BGR. HEIC goes via macOS `sips`; neither PIL nor cv2 decode it."""
    if path.lower().endswith((".heic", ".heif")):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        subprocess.run(["sips", "-s", "format", "png", path, "--out", tmp],
                       check=True, capture_output=True)
        path = tmp
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img


# ---------------------------------------------------------------------------
# Finding the ruling
# ---------------------------------------------------------------------------

def _flat_response(bgr):
    b, _, r = cv2.split(bgr.astype(np.float32))
    resp = b - r
    return resp - cv2.GaussianBlur(resp, (0, 0), BLUE_SIGMA)


def _estimate_spacing(profile):
    x = profile - profile.mean()
    if x.std() < 1e-6:
        return 0
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    lo, hi = 20, min(400, len(ac) - 1)
    if hi <= lo:
        return 0
    pk, _ = find_peaks(ac[lo:hi])
    return int(pk[np.argmax(ac[lo:hi][pk])] + lo) if len(pk) else 0


def _track_rules(flat, spacing):
    """Seed on the centre slab, walk outwards snapping to the nearest peak.

    Tracking rather than one global projection profile because perspective
    fans the rules apart -- they are not parallel in the photo, so no single
    rotation makes them all horizontal.
    """
    H, W = flat.shape
    edges = np.linspace(0, W, N_SLABS + 1).astype(int)
    cx = (edges[:-1] + edges[1:]) // 2
    peaks = []
    for i in range(N_SLABS):
        p = flat[:, edges[i]:edges[i + 1]].mean(axis=1)
        p = p - p.min()
        q, _ = find_peaks(p, distance=spacing * 0.55, height=p.max() * 0.30)
        peaks.append(q)

    def walk(seed, order):
        pts = [(cx[order[0]], seed)]
        for k in range(1, len(order)):
            s = order[k]
            if len(pts) < 2:
                pred = pts[-1][1]
            else:
                dx = cx[order[k - 1]] - cx[order[k - 2]]
                slope = (pts[-1][1] - pts[-2][1]) / (dx or 1)
                pred = pts[-1][1] + slope * (cx[s] - cx[order[k - 1]])
            q = peaks[s]
            if not len(q):
                continue
            j = q[np.argmin(abs(q - pred))]
            if abs(j - pred) < spacing * 0.40:
                pts.append((cx[s], j))
        return pts

    c = N_SLABS // 2
    out = []
    for seed in peaks[c]:
        pts = walk(seed, list(range(c, -1, -1)))[::-1] + walk(seed, list(range(c, N_SLABS)))[1:]
        if len(pts) < N_SLABS * 0.5:
            continue
        P = np.array(pts, float)
        # Quadratic, not linear: a notebook page bows near the spine, and a
        # straight fit drifts up to ~26px off at the ends -- a third of a line.
        poly = np.polyfit(P[:, 0], P[:, 1], 2 if len(P) >= 6 else 1)
        out.append({"poly": poly.tolist(), "yc": float(np.polyval(poly, W / 2))})
    out.sort(key=lambda d: d["yc"])
    return out


def _consistent_run(rules):
    """Longest evenly-spaced run. Drops desk clutter that also reads as blue."""
    if len(rules) < 3:
        return rules
    gaps = np.diff([d["yc"] for d in rules])
    med = np.median(gaps)
    best, run = [], [0]
    for i, g in enumerate(gaps):
        if abs(g - med) < med * 0.35:
            run.append(i + 1)
        else:
            if len(run) > len(best):
                best = run
            run = [i + 1]
    return [rules[i] for i in (run if len(run) > len(best) else best)]


def _extent(flat, rule, step=6):
    """Longest stretch of x where this rule's blue response holds up.

    Threshold is a fraction of the rule's own median response rather than a
    constant, so it follows the lighting falloff into the page curl instead of
    truncating every line at the shadow.
    """
    H, W = flat.shape
    xs = np.arange(0, W, step)
    ys = np.rint(np.polyval(rule["poly"], xs)).astype(int)
    ok = (ys >= 3) & (ys < H - 3)
    val = np.full(xs.size, -1e9, np.float32)
    for k in np.nonzero(ok)[0]:
        val[k] = flat[ys[k] - 3:ys[k] + 4, xs[k]].max()
    seen = val[val > -1e8]
    if seen.size < 10:
        return None
    thr = max(1.2, np.median(seen[seen > 0]) * 0.35) if (seen > 0).any() else 1.2
    good = binary_closing(val > thr, np.ones(9))  # bridge the spine crease
    best = cur = None
    for k, g in enumerate(good):
        if g:
            cur = k if cur is None else cur
            if best is None or k - cur > best[1] - best[0]:
                best = (cur, k)
        else:
            cur = None
    return (float(xs[best[0]]), float(xs[best[1]])) if best else None


def _robust_line(i, v):
    a, b = np.polyfit(i, v, 1)
    for _ in range(5):
        r = np.abs(v - (a * i + b))
        keep = r < max(np.median(r) * 2.0, 20)
        if keep.sum() < 4:
            break
        a, b = np.polyfit(i[keep], v[keep], 1)
    return float(a), float(b)


def detect_ruling(bgr):
    """Locate the ruled writing area in photo space."""
    H, W = bgr.shape[:2]
    flat = _flat_response(bgr)
    spacing = _estimate_spacing(flat[:, W // 2 - 100:W // 2 + 100].mean(axis=1))
    if not spacing:
        raise ValueError("no ruling detected -- is the paper ruled in blue?")
    rules = _consistent_run(_track_rules(flat, spacing))
    if len(rules) < 4:
        raise ValueError(f"only {len(rules)} rules found")

    ex = [_extent(flat, r) for r in rules]
    ok = [i for i, e in enumerate(ex) if e]
    if len(ok) < 4:
        raise ValueError("could not measure the width of the ruled area")
    idx = np.array(ok, float)
    la, lb = _robust_line(idx, np.array([ex[i][0] for i in ok]))
    ra, rb = _robust_line(idx, np.array([ex[i][1] for i in ok]))

    return {
        "rules": rules,
        "spacing": float(np.median(np.diff([r["yc"] for r in rules]))),
        "left": (la, lb),
        "right": (ra, rb),
        "size": (W, H),
    }


def line_geometry(ruling, i):
    """Left/right x, baseline y at a given x, tilt, and local line spacing."""
    r = ruling["rules"][i]
    la, lb = ruling["left"]
    ra, rb = ruling["right"]
    x0, x1 = la * i + lb, ra * i + rb
    nb = ruling["rules"][i + 1] if i + 1 < len(ruling["rules"]) else None
    pv = ruling["rules"][i - 1] if i > 0 else None
    local = (nb["yc"] - r["yc"]) if nb else (r["yc"] - pv["yc"])
    poly = np.asarray(r["poly"])
    slope = np.polyder(poly)
    return {
        "x0": x0, "x1": x1,
        "y": lambda x: float(np.polyval(poly, x)),
        "deg_at": lambda x: math.degrees(math.atan(float(np.polyval(slope, x)))),
        "spacing": float(local),
    }


# ---------------------------------------------------------------------------
# Generating the words
# ---------------------------------------------------------------------------

def generate_words(words, out_dir=None, style=None, ckpt=None, device="mps"):
    """One image per word *instance*. The model re-rolls each call, so a word
    appearing twice gets two different renderings -- no copy-paste artefacts."""
    if out_dir is None:
        out_dir = os.path.join(PROJECT, "output",
                               f"_paper_words_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [VENV_PYTHON, GENERATE_PY, "--text", " ".join(words),
           "--device", device, "--output-dir", out_dir]
    if style:
        cmd.append("--style")
        cmd.extend(style)
    if ckpt:
        cmd.extend(["--ckpt", ckpt])
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"generate.py failed:\n{res.stderr}")
    pngs = [f for f in sorted(glob(os.path.join(out_dir, "*.png")))
            if not os.path.basename(f).startswith("_")]
    if len(pngs) != len(words):
        raise RuntimeError(f"expected {len(words)} word images, got {len(pngs)}")
    return pngs


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def parse_text(text):
    """Text -> [(word, newlines_before, tabs_before)], preserving structure."""
    out = []
    pending_nl = 0
    pending_tab = 0
    for raw_line in text.split("\n"):
        stripped = raw_line.lstrip("\t")
        tabs = len(raw_line) - len(stripped)
        toks = stripped.split()
        if not toks:
            pending_nl += 1
            continue
        for k, w in enumerate(toks):
            out.append((w, pending_nl if k == 0 else 0,
                        (tabs + pending_tab) if k == 0 else 0))
            pending_nl = 1  # subsequent source lines start a new line
            pending_tab = 0
        pending_nl = 1
    return out


def layout(tokens, widths_at, ruling, start_line=0, scale=1.05,
           space_em=0.28, tab_em=0.6):
    """Greedy line-breaking onto the detected rules.

    widths_at(i, height) -> pixel width of word i drawn at that canvas height.
    Word size follows each rule's *local* spacing, so text further up the page
    is drawn smaller and the perspective stays honest.
    """
    placements = []
    n_rules = len(ruling["rules"])
    li = start_line
    g = line_geometry(ruling, li)
    h = g["spacing"] * scale
    x = g["x0"]
    first = True

    for wi, (word, nl, tabs) in enumerate(tokens):
        if nl and not first:
            li += nl
            if li >= n_rules:
                break
            g = line_geometry(ruling, li)
            h = g["spacing"] * scale
            x = g["x0"]
        if tabs:
            x += tabs * tab_em * h
        w = widths_at(wi, h)
        if not first and nl == 0 and x + w > g["x1"]:
            li += 1
            if li >= n_rules:
                break
            g = line_geometry(ruling, li)
            h = g["spacing"] * scale
            x = g["x0"]
            w = widths_at(wi, h)
        placements.append({"wi": wi, "line": li, "x": float(x),
                           "h": float(h), "w": float(w)})
        x += w + h * space_em
        first = False
    return placements


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def compose(bgr, ruling, placements, word_paths, darkness=0.22, jitter=0.05, seed=0):
    """Alpha-composite each word onto its rule.

    Darkening rather than painting a flat colour: the paper keeps its own
    shading and shadows, so the writing reads as being *on* the page instead of
    pasted over it.
    """
    rng = np.random.default_rng(seed)
    out = bgr.astype(np.float32)
    H, W = out.shape[:2]

    for p in placements:
        g = line_geometry(ruling, p["line"])
        alpha = ink_alpha(Image.open(word_paths[p["wi"]]))
        h = max(4, int(round(p["h"])))
        w = max(2, int(round(alpha.width * h / alpha.height)))
        a = np.asarray(alpha.resize((w, h), Image.LANCZOS), np.float32) / 255.0

        cx = p["x"] + p["w"] / 2
        ang = g["deg_at"](cx)  # local tangent, so each word follows the page bow
        if abs(ang) > 0.05:
            d = math.radians(ang)
            nw = int(abs(w * math.cos(d)) + abs(h * math.sin(d))) + 2
            nh = int(abs(w * math.sin(d)) + abs(h * math.cos(d))) + 2
            M = cv2.getRotationMatrix2D((w / 2, h / 2), -ang, 1.0)
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            a = cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR,
                               borderValue=0.0)
            w, h = nw, nh

        base = g["y"](cx) + rng.uniform(-jitter, jitter) * p["h"]
        x0 = int(round(p["x"]))
        y0 = int(round(base - BASELINE_FRAC * p["h"] - (h - p["h"]) / 2))

        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x0 + w), min(H, y0 + h)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        sub = a[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0][..., None]
        out[sy0:sy1, sx0:sx1] *= (1.0 - sub * (1.0 - darkness))

    return np.clip(out, 0, 255).astype(np.uint8)


def ruling_json(ruling, n_points=24):
    """Rule polylines for the UI overlay, so detection is checkable by eye."""
    lines = []
    for i in range(len(ruling["rules"])):
        g = line_geometry(ruling, i)
        xs = np.linspace(g["x0"], g["x1"], n_points)
        lines.append({"i": i, "pts": [[round(float(x), 1), round(g["y"](x), 1)] for x in xs]})
    return {"lines": lines, "spacing": round(ruling["spacing"], 1),
            "count": len(lines), "size": list(ruling["size"])}


def write_on_paper(photo_path, text, out_path, style=None, ckpt=None,
                   device="mps", scale=1.05, start_line=0, darkness=0.22, seed=0):
    """Full pipeline: photo + text -> composited image."""
    print(f"[paper] reading {os.path.basename(photo_path)}", flush=True)
    img = load_photo(photo_path)
    print(f"[paper] {img.shape[1]}x{img.shape[0]}, finding the ruling...", flush=True)
    ruling = detect_ruling(img)
    print(f"[paper] {len(ruling['rules'])} ruled lines, spacing {ruling['spacing']:.1f}px",
          flush=True)
    tokens = parse_text(text)
    if not tokens:
        raise ValueError("no text to write")
    print(f"[paper] generating {len(tokens)} words...", flush=True)
    paths = generate_words([t[0] for t in tokens], style=style, ckpt=ckpt, device=device)
    natural = [Image.open(p).size for p in paths]

    def widths_at(i, h):
        w, hh = natural[i]
        return w * h / hh

    pl = layout(tokens, widths_at, ruling, start_line=start_line, scale=scale)
    if len(pl) < len(tokens):
        print(f"[paper] WARNING: ran out of ruled lines -- "
              f"{len(tokens) - len(pl)} word(s) did not fit", flush=True)
    print(f"[paper] compositing {len(pl)} words", flush=True)
    out = compose(img, ruling, pl, paths, darkness=darkness, seed=seed)
    ext = os.path.splitext(out_path)[1].lower()
    cv2.imwrite(out_path, out,
                [cv2.IMWRITE_JPEG_QUALITY, 92] if ext in (".jpg", ".jpeg") else [])
    print(f"[paper] wrote {out_path}", flush=True)
    return {"out": out_path, "placed": len(pl), "of": len(tokens),
            "rules": len(ruling["rules"]), "spacing": ruling["spacing"]}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Write generated handwriting onto ruled paper.")
    ap.add_argument("--photo", default=TEST_PAGE)
    ap.add_argument("--text")
    ap.add_argument("--output")
    ap.add_argument("--style", nargs="*", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--scale", type=float, default=1.05,
                    help="word height as a multiple of the line spacing")
    ap.add_argument("--start-line", type=int, default=0)
    ap.add_argument("--darkness", type=float, default=0.22,
                    help="0 = black ink, 1 = invisible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detect-only", action="store_true",
                    help="print the ruling geometry as JSON and exit")
    a = ap.parse_args()

    if a.detect_only:
        import json
        print(json.dumps(ruling_json(detect_ruling(load_photo(a.photo)))))
        return
    if not a.text or not a.output:
        ap.error("--text and --output are required unless --detect-only")

    r = write_on_paper(a.photo, a.text, a.output, style=a.style, ckpt=a.ckpt,
                       device=a.device, scale=a.scale, start_line=a.start_line,
                       darkness=a.darkness, seed=a.seed)
    print(f"[paper] done: placed {r['placed']}/{r['of']} words on {r['rules']} lines")


if __name__ == "__main__":
    main()
