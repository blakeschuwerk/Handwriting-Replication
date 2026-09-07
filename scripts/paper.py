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

import sheet as _sheet

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


def detect_margin_x(bgr, y_lo=None, y_hi=None):
    """The printed red margin, as an x in pixels, or None.

    Found in red-minus-blue for the same reason rules are found in
    blue-minus-red: it isolates one ink colour and, being a channel
    difference, cancels the lighting gradient. The margin matters beyond
    cosmetics -- it anchors page-space u=0, so word positions stay meaningful
    across a re-fit.
    """
    b, _, r = cv2.split(bgr.astype(np.float32))
    resp = r - b
    resp = resp - cv2.GaussianBlur(resp, (0, 0), BLUE_SIGMA)
    H, W = resp.shape
    y_lo = int(y_lo if y_lo is not None else H * 0.3)
    y_hi = int(y_hi if y_hi is not None else H * 0.85)
    prof = resp[y_lo:y_hi, :].mean(axis=0)
    prof = prof - np.median(prof)
    pk, props = find_peaks(prof, prominence=prof.std() * 1.5, distance=60)
    if not len(pk):
        return None
    return pk, props["prominences"]


def _pick_margin(cands, x_min, x_max):
    """Choose the margin from red-line candidates: the strongest one sitting
    near the left edge of the ruling. A notebook often prints a red line down
    both sides, so 'strongest overall' picks the wrong one."""
    if cands is None:
        return None
    pk, prom = cands
    span = x_max - x_min
    ok = [(x, p) for x, p in zip(pk, prom)
          if x_min - 0.10 * span <= x <= x_min + 0.35 * span]
    if not ok:
        return None
    return float(max(ok, key=lambda t: t[1])[0])


def _rule_observations(flat, rules, spacing, step=50, half=25):
    """Dense (x, y, line_index) samples of the real ink.

    The slab tracker gives one point per rule per slab -- enough to seed, too
    few to fit well. This resamples finely and assigns each peak to the nearest
    tracked rule, which is what turns ~450 coarse points into ~1100 good ones.
    """
    H, W = flat.shape
    # Only sample where the tracked rules actually carry ink. Sampling the full
    # width pulls in the facing page and the desk, and a stray peak there can
    # be assigned to a rule and survive as a plausible-looking inlier.
    ex = [e for e in (_extent(flat, r) for r in rules) if e]
    if ex:
        x_lo = int(np.median([e[0] for e in ex]))
        x_hi = int(np.median([e[1] for e in ex]))
        pad = int(0.12 * max(x_hi - x_lo, 1))
        x_lo, x_hi = max(step, x_lo - pad), min(W - step, x_hi + pad)
    else:
        x_lo, x_hi = step, W - step
    obs = []
    for x in range(x_lo, x_hi, step):
        pred = np.array([np.polyval(r["poly"], x) for r in rules])
        lo = max(int(pred.min()) - int(spacing), 0)
        hi = min(int(pred.max()) + int(spacing), H)
        if hi - lo < 10:
            continue
        col = flat[lo:hi, max(0, x - half):x + half].mean(axis=1)
        col = col - col.min()
        if col.max() <= 0:
            continue
        pk, _ = find_peaks(col, distance=max(8, spacing * 0.55),
                           prominence=col.std() * 0.8)
        for q in pk + lo:
            j = int(np.argmin(np.abs(pred - q)))
            if abs(pred[j] - q) < spacing * 0.35:
                obs.append((x, float(q), j))
    return np.array(obs, float) if obs else np.zeros((0, 3))


def _observations_from_sheet(flat, sheet, spacing, step=50, half=25):
    """Re-collect ink observations using a fitted sheet as the predictor.

    Run after the grid has been extended, so the rules added at the edges get
    real observations behind them instead of pure extrapolation.
    """
    H, W = flat.shape
    lines = np.arange(sheet.i_first, sheet.i_first + sheet.n_lines, dtype=float)
    obs = []
    for x in range(step, W - step, step):
        _, ys = sheet.xy(lines, np.full(lines.size, 0.0))
        ys = sheet.xy(lines, np.full(lines.size, float(sheet.uv(x, H / 2)[0][0])))[1]
        lo = max(int(ys.min()) - int(spacing), 0)
        hi = min(int(ys.max()) + int(spacing), H)
        if hi - lo < 10:
            continue
        col = flat[lo:hi, max(0, x - half):x + half].mean(axis=1)
        col = col - col.min()
        if col.max() <= 0:
            continue
        pk, _ = find_peaks(col, distance=max(8, spacing * 0.55),
                           prominence=col.std() * 0.8)
        for q in pk + lo:
            j = int(np.argmin(np.abs(ys - q)))
            if abs(ys[j] - q) < spacing * 0.35:
                obs.append((x, float(q), float(lines[j])))
    return np.array(obs, float) if obs else np.zeros((0, 3))


def _anchor(sheet, bgr, flat):
    """Put u=0 on the printed margin and set the writing area from the ink.

    Must be redone after every fit: each fit chooses its own u origin, so
    carrying old u values onto a refitted sheet silently shifts the margin and
    the right edge across the page.
    """
    b, _, r = cv2.split(bgr.astype(np.float32))
    red = (r - b) - cv2.GaussianBlur(r - b, (0, 0), BLUE_SIGMA)
    lo, hi = _sheet.find_extent_u(sheet, flat)
    sheet.u_left, sheet.u_right = lo, hi
    mu = _sheet.find_margin_u(sheet, red)
    if mu is not None:
        sheet.shift_u(-mu)
        sheet.u_left = 0.0        # writing starts at the margin
        sheet.margin_x = float(sheet.point(sheet.i_first + sheet.n_lines // 2, 0.0)[0])
    return sheet


def detect_sheet(bgr):
    """Photo -> (Sheet, observations, info). The replacement for detect_ruling."""
    H, W = bgr.shape[:2]
    flat = _flat_response(bgr)
    spacing = _estimate_spacing(flat[:, W // 2 - 100:W // 2 + 100].mean(axis=1))
    if not spacing:
        raise ValueError("no ruling detected -- is the paper ruled in blue?")
    rules = _consistent_run(_track_rules(flat, spacing))
    if len(rules) < 4:
        raise ValueError(f"only {len(rules)} ruled lines found")

    obs = _rule_observations(flat, rules, spacing)
    if len(obs) < 30:
        raise ValueError(f"only {len(obs)} rule observations")

    sheet, info = _sheet.fit_sheet(obs, (W, H), spacing)
    sheet = _anchor(sheet, bgr, flat)

    # Grow onto rules the tracker missed, then re-observe and refit so the new
    # rules rest on measured ink rather than on extrapolation.
    sheet, grown = _sheet.extend_lines(sheet, flat)
    if grown != (0, 0):
        obs2 = _observations_from_sheet(flat, sheet, spacing)
        if len(obs2) >= 30:
            i0, n = sheet.i_first, sheet.n_lines
            sheet, info = _sheet.fit_sheet(obs2, (W, H), spacing)
            sheet.i_first, sheet.n_lines = i0, n
            sheet = _anchor(sheet, bgr, flat)
            obs = obs2
    info["grown"] = grown
    return sheet, obs, info


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


def layout(tokens, aspects, sheet, start_line=0, scale=1.05,
           space_em=0.28, tab_em=0.6):
    """Greedy line breaking, done entirely in page space.

    Because one unit of u equals one line height, a word's page-space width is
    just ``aspect * scale`` -- it does not depend on which line the word lands
    on. That is what lets breaks be decided before knowing the answer, and it
    is why the u scale was fixed the way it was.

    Words past the last rule are kept and marked, never dropped: losing text
    silently is worse than showing it as overflow.
    """
    placements = []
    first_line = sheet.i_first + start_line
    last_line = sheet.i_first + sheet.n_lines - 1
    li = first_line
    u = sheet.u_left
    first = True

    for wi, (word, nl, tabs) in enumerate(tokens):
        if nl and not first:
            li += nl
            u = sheet.u_left
        if tabs:
            u += tabs * tab_em * scale
        w = aspects[wi] * scale
        if not first and nl == 0 and u + w > sheet.u_right:
            li += 1
            u = sheet.u_left
        first = False
        if li > last_line:
            placements.append({"wi": wi, "line": li, "u": float(u),
                               "w": float(w), "overflow": True})
            u += w + space_em * scale
            continue
        placements.append({"wi": wi, "line": li, "u": float(u),
                           "w": float(w), "overflow": False})
        u += w + space_em * scale
    return placements


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def compose(bgr, sheet, placements, word_paths, scale=1.05,
            darkness=0.22, jitter=0.05, seed=0):
    """Alpha-composite each word onto its rule.

    Darkening rather than painting a flat colour: the paper keeps its own
    shading and shadows, so the writing reads as being *on* the page instead of
    pasted over it.
    """
    rng = np.random.default_rng(seed)
    out = bgr.astype(np.float32)
    H, W = out.shape[:2]

    for p in placements:
        if p.get("overflow"):
            continue
        line, u = p["line"], p["u"]
        h_px = sheet.spacing_px(line, u) * scale
        alpha = ink_alpha(Image.open(word_paths[p["wi"]]))
        h = max(4, int(round(h_px)))
        w = max(2, int(round(alpha.width * h / alpha.height)))
        a = np.asarray(alpha.resize((w, h), Image.LANCZOS), np.float32) / 255.0

        ang = sheet.tangent_deg(line, u + p["w"] / 2)
        if abs(ang) > 0.05:
            d = math.radians(ang)
            nw = int(abs(w * math.cos(d)) + abs(h * math.sin(d))) + 2
            nh = int(abs(w * math.sin(d)) + abs(h * math.cos(d))) + 2
            M = cv2.getRotationMatrix2D((w / 2, h / 2), -ang, 1.0)
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            a = cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0.0)
            w, h = nw, nh

        bx, by = sheet.point(line, u)
        by += rng.uniform(-jitter, jitter) * h_px
        x0 = int(round(bx))
        y0 = int(round(by - BASELINE_FRAC * h_px - (h - h_px) / 2))

        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x0 + w), min(H, y0 + h)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        sub = a[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0][..., None]
        out[sy0:sy1, sx0:sx1] *= (1.0 - sub * (1.0 - darkness))

    return np.clip(out, 0, 255).astype(np.uint8)


def sheet_json(sheet, response=None, n_points=24):
    """Everything the UI needs to draw and manipulate the sheet."""
    lines = []
    for k in range(sheet.n_lines):
        i = sheet.i_first + k
        pts = sheet.polyline(i, n_points)
        lines.append({"i": int(i),
                      "pts": [[round(float(x), 1), round(float(y), 1)] for x, y in pts]})

    def guide(u):
        ii = np.linspace(sheet.i_first, sheet.i_first + sheet.n_lines - 1, n_points)
        x, y = sheet.xy(ii, np.full(n_points, float(u)))
        return [[round(float(a), 1), round(float(b), 1)] for a, b in zip(x, y)]

    i0, i1 = sheet.i_first, sheet.i_first + sheet.n_lines - 1
    corners = [sheet.point(i0, sheet.u_left), sheet.point(i0, sheet.u_right),
               sheet.point(i1, sheet.u_right), sheet.point(i1, sheet.u_left)]
    out = {
        "sheet": sheet.to_json(),
        "lines": lines,
        "margin": guide(0.0),
        "right": guide(sheet.u_right),
        "quad": [[round(x, 1), round(y, 1)] for x, y in corners],
        "count": sheet.n_lines,
        "i_first": sheet.i_first,
        "spacing": round(sheet.spacing_px(i0 + sheet.n_lines // 2,
                                          (sheet.u_left + sheet.u_right) / 2), 1),
        "size": list(sheet.size),
    }
    if response is not None:
        out["confidence"] = [round(c, 3) for c in _sheet.line_confidence(sheet, response)]
    return out


def write_on_paper(photo_path, text, out_path, style=None, ckpt=None,
                   device="mps", scale=1.05, start_line=0, darkness=0.22, seed=0):
    """Full pipeline: photo + text -> composited image."""
    print(f"[paper] reading {os.path.basename(photo_path)}", flush=True)
    img = load_photo(photo_path)
    print(f"[paper] {img.shape[1]}x{img.shape[0]}, finding the ruling...", flush=True)
    sheet, _obs, info = detect_sheet(img)
    print(f"[paper] {sheet.n_lines} ruled lines, spacing "
          f"{sheet.spacing_px(sheet.i_first + sheet.n_lines // 2, sheet.u_right / 2):.1f}px"
          + (f", grew {info['grown']}" if info.get("grown") else ""), flush=True)
    tokens = parse_text(text)
    if not tokens:
        raise ValueError("no text to write")
    print(f"[paper] generating {len(tokens)} words...", flush=True)
    paths = generate_words([t[0] for t in tokens], style=style, ckpt=ckpt, device=device)
    aspects = [w / h for w, h in (Image.open(p).size for p in paths)]

    pl = layout(tokens, aspects, sheet, start_line=start_line, scale=scale)
    over = sum(1 for p in pl if p["overflow"])
    if over:
        print(f"[paper] WARNING: {over} word(s) ran past the last ruled line",
              flush=True)
    print(f"[paper] compositing {len(pl) - over} words", flush=True)
    out = compose(img, sheet, pl, paths, scale=scale, darkness=darkness, seed=seed)
    ext = os.path.splitext(out_path)[1].lower()
    cv2.imwrite(out_path, out,
                [cv2.IMWRITE_JPEG_QUALITY, 92] if ext in (".jpg", ".jpeg") else [])
    print(f"[paper] wrote {out_path}", flush=True)
    return {"out": out_path, "placed": len(pl) - over, "of": len(tokens),
            "rules": sheet.n_lines}


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
        img = load_photo(a.photo)
        sh, _o, _i = detect_sheet(img)
        print(json.dumps(sheet_json(sh, _flat_response(img))))
        return
    if not a.text or not a.output:
        ap.error("--text and --output are required unless --detect-only")

    r = write_on_paper(a.photo, a.text, a.output, style=a.style, ckpt=a.ckpt,
                       device=a.device, scale=a.scale, start_line=a.start_line,
                       darkness=a.darkness, seed=a.seed)
    print(f"[paper] done: placed {r['placed']}/{r['of']} words on {r['rules']} lines")


if __name__ == "__main__":
    main()
