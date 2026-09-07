"""A parametric model of the ruling on a photographed sheet of paper.

The whole point of this module is a parameter count. Fitting each ruled line
independently takes 3 numbers per line -- 84 for a 28-line page -- and measures
4.40px mean error against the real ink. One global sheet takes 7 and measures
2.74px. Fewer parameters, better fit.

That matters for more than tidiness. When detection is wrong it is now wrong
about a *frame*, not about 28 unrelated curves, so a user can correct it by
dragging four corners and letting the model re-solve against the ink. There is
no way to offer that on 84 independent parameters.

Coordinates
-----------
Page space is (u, v). ``v`` counts ruled lines, so v=3.0 is exactly on the
fourth rule. ``u`` runs along the lines, anchored at the red margin.

The scale of ``u`` is fixed so that one unit of u equals one unit of v -- the
map is a similarity at the page centre. This is load-bearing rather than
cosmetic: it makes a word's page-space width ``aspect * scale`` regardless of
which line the word sits on, which is what lets layout decide line breaks
before it knows what line anything landed on.

Model
-----
A homography carries image space to page space, and page-space curvature is
added on top::

    (u, v_persp) = H . (x, y)          # projective, 8 dof
    v            = v_persp + B(t)      # page bow, 2 dof
    B(t)         = c1 t(1-t) + c2 t(1-t)(2t-1)

``t`` is the normalised position across the writing area. The bow lives in
page-u rather than image-x so that both directions stay closed form, and the
two bubble functions are used because they vanish at both ends and are
therefore orthogonal to what the homography already expresses -- a plain cubic
in x is degenerate with H and the fit wanders.
"""

import json

import cv2
import numpy as np
from scipy.signal import find_peaks

VERSION = 1


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class Norm:
    """Pixels to a centred, unit-ish frame.

    Raw pixel coordinates reach ~4000 and the fit contains products like
    ``i * x``, which pushes design-matrix entries past 1e5 and destroys the
    conditioning. Everything is fitted in this frame instead.
    """

    __slots__ = ("w", "h", "s", "cx", "cy")

    def __init__(self, w, h):
        self.w, self.h = float(w), float(h)
        self.s = 2.0 / max(self.w, self.h)
        self.cx, self.cy = self.w / 2.0, self.h / 2.0

    def to(self, x, y):
        return (np.asarray(x, float) - self.cx) * self.s, (np.asarray(y, float) - self.cy) * self.s

    def back(self, xn, yn):
        return np.asarray(xn, float) / self.s + self.cx, np.asarray(yn, float) / self.s + self.cy


# ---------------------------------------------------------------------------
# The sheet
# ---------------------------------------------------------------------------

class Sheet:
    """Image <-> page mapping for one photographed sheet."""

    def __init__(self, M, c1, c2, u_left, u_right, i_first, n_lines,
                 size, spacing_hint=0.0, margin_x=None):
        self.M = np.asarray(M, float).reshape(3, 3)   # normalised image -> page
        self.Minv = np.linalg.inv(self.M)
        self.c1, self.c2 = float(c1), float(c2)
        self.u_left, self.u_right = float(u_left), float(u_right)
        self.i_first, self.n_lines = int(i_first), int(n_lines)
        self.size = (int(size[0]), int(size[1]))
        self.spacing_hint = float(spacing_hint)
        self.margin_x = None if margin_x is None else float(margin_x)
        self.norm = Norm(*self.size)

    # -- bow ---------------------------------------------------------------

    def _t(self, u):
        span = self.u_right - self.u_left
        if abs(span) < 1e-9:
            return np.zeros_like(np.asarray(u, float))
        return (np.asarray(u, float) - self.u_left) / span

    def _bow(self, u):
        t = self._t(u)
        return self.c1 * t * (1 - t) + self.c2 * t * (1 - t) * (2 * t - 1)

    # -- image -> page -----------------------------------------------------

    def uv(self, x, y):
        """Pixel -> (u, v). v is fractional line index."""
        xn, yn = self.norm.to(x, y)
        xn, yn = np.atleast_1d(xn), np.atleast_1d(yn)
        p = self.M @ np.stack([xn, yn, np.ones_like(xn)])
        w = np.where(np.abs(p[2]) < 1e-12, 1e-12, p[2])
        u, v = p[0] / w, p[1] / w
        return u, v + self._bow(u)

    # -- page -> image -----------------------------------------------------

    def xy(self, i, u):
        """(line index, u) -> pixel. Accepts scalars or arrays."""
        i = np.atleast_1d(np.asarray(i, float))
        u = np.atleast_1d(np.asarray(u, float))
        i, u = np.broadcast_arrays(i, u)
        v = i - self._bow(u)
        q = self.Minv @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
        w = np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])
        x, y = self.norm.back(q[0] / w, q[1] / w)
        return x.reshape(u.shape), y.reshape(u.shape)

    def point(self, i, u):
        x, y = self.xy(i, u)
        return float(x[0]), float(y[0])

    # -- local quantities (these replace line_geometry) ---------------------

    def spacing_px(self, i, u):
        """Pixel distance to the next rule: the actual height budget here."""
        x0, y0 = self.xy(i, u)
        x1, y1 = self.xy(np.asarray(i, float) + 1.0, u)
        return float(np.hypot(x1 - x0, y1 - y0)[0])

    def tangent_deg(self, i, u, d=0.05):
        """Direction of the rule at this point, in degrees. Includes the bow."""
        xa, ya = self.xy(i, np.asarray(u, float) - d)
        xb, yb = self.xy(i, np.asarray(u, float) + d)
        return float(np.degrees(np.arctan2(yb - ya, xb - xa))[0])

    def polyline(self, i, n=24):
        return self.polyline_at(i, n)

    def polyline_at(self, i, n=24):
        u = np.linspace(self.u_left, self.u_right, n)
        x, y = self.xy(np.full(n, float(i)), u)
        return np.stack([x, y], 1)

    def line_bounds(self, i):
        """Pixel endpoints of one rule across the writing area."""
        x, y = self.xy(np.array([float(i), float(i)]),
                       np.array([self.u_left, self.u_right]))
        return (float(x[0]), float(y[0])), (float(x[1]), float(y[1]))

    def shift_u(self, d):
        """Slide the u origin. Used to put u=0 on the margin once it is found."""
        T = np.array([[1.0, 0.0, float(d)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.M = T @ self.M
        self.Minv = np.linalg.inv(self.M)
        self.u_left += d
        self.u_right += d
        return self

    def sample(self, img, i_grid, u_grid):
        """Bilinear-sample an image on a page-space grid. Returns (len(i), len(u))."""
        II, UU = np.meshgrid(np.asarray(i_grid, float), np.asarray(u_grid, float),
                             indexing="ij")
        x, y = self.xy(II.ravel(), UU.ravel())
        mx = x.reshape(II.shape).astype(np.float32)
        my = y.reshape(II.shape).astype(np.float32)
        return cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # -- serialisation -----------------------------------------------------

    def to_json(self):
        return {
            "version": VERSION,
            "M": self.M.ravel().tolist(),
            "bow": [self.c1, self.c2],
            "u_left": self.u_left, "u_right": self.u_right,
            "i_first": self.i_first, "n_lines": self.n_lines,
            "size": list(self.size),
            "spacing_hint": self.spacing_hint,
            "margin_x": self.margin_x,
        }

    @staticmethod
    def from_json(d):
        return Sheet(d["M"], d["bow"][0], d["bow"][1], d["u_left"], d["u_right"],
                     d["i_first"], d["n_lines"], d["size"],
                     d.get("spacing_hint", 0.0), d.get("margin_x"))

    def __repr__(self):
        return (f"Sheet(lines={self.n_lines} from {self.i_first}, "
                f"u=[{self.u_left:.2f},{self.u_right:.2f}], "
                f"bow=({self.c1:+.3f},{self.c2:+.3f}))")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit_v_row(xn, yn, v, iters=6, spacing=1.0):
    """Robust fit of the two homography rows that produce v.

    v = (h0 x + h1 y + h2) / (h6 x + h7 y + 1), rearranged to be linear:
        h0 x + h1 y + h2 - v h6 x - v h7 y = v

    IRLS rather than plain least squares because spurious peaks (a shadow, the
    facing page) otherwise drag the whole sheet.
    """
    A = np.stack([xn, yn, np.ones_like(xn), -v * xn, -v * yn], 1)
    w = np.ones_like(v)
    p = None
    for _ in range(iters):
        Aw = A * w[:, None]
        p, *_ = np.linalg.lstsq(Aw, v * w, rcond=None)
        pred = (p[0] * xn + p[1] * yn + p[2]) / (p[3] * xn + p[4] * yn + 1.0)
        r = np.abs(pred - v)
        s = max(np.median(r) * 1.4826, 1e-4)
        w = 1.0 / (1.0 + (r / (3.0 * s)) ** 2)   # Cauchy
    resid = np.abs((p[0] * xn + p[1] * yn + p[2]) /
                   (p[3] * xn + p[4] * yn + 1.0) - v)
    inl = resid < 3.0 * max(np.median(resid) * 1.4826, 1e-4)
    return p, resid, inl


def _u_row(p_v, c):
    """Complete the homography with a u row.

    Only the v row is determined by the ruling; u has three degrees of freedom
    (offset, scale, shear) that no amount of looking at parallel lines can pin
    down. They are fixed by requiring the map to be a similarity at the page
    centre -- u perpendicular to v and at the same scale -- which is what makes
    one unit of u equal one line height. `c` slides the origin along u and is
    chosen so that u=0 lands on the margin.
    """
    h0, h1, h2, h6, h7 = p_v
    a = h0 - h2 * h6          # dv/dx at the page centre
    b = h1 - h2 * h7          # dv/dy
    # rotate grad(v) by -90 deg so u grows to the right
    h3 = b + c * h6
    h4 = -a + c * h7
    h5 = c
    return np.array([[h3, h4, h5],
                     [h0, h1, h2],
                     [h6, h7, 1.0]], float)


def _fit_bow(u, resid_v):
    """Least squares on the two bubble functions, in page-u."""
    lo, hi = np.percentile(u, [2, 98])
    span = max(hi - lo, 1e-9)
    t = (u - lo) / span
    B = np.stack([t * (1 - t), t * (1 - t) * (2 * t - 1)], 1)
    c, *_ = np.linalg.lstsq(B, resid_v, rcond=None)
    return float(c[0]), float(c[1]), float(lo), float(hi)


def fit_sheet(obs, size, spacing, margin_x=None, rounds=4):
    """Fit a Sheet to rule observations.

    obs: (N,3) array of (x, y, line_index) sampled from the real ink.

    Perspective and bow are alternated rather than fitted in one shot. The bow
    is defined over page-u, which does not exist until the perspective part is
    solved, so the first bow estimate is made against a slightly wrong u; two
    or three alternations converge and buy roughly 0.5px of mean error.
    """
    obs = np.asarray(obs, float)
    if len(obs) < 30:
        raise ValueError(f"need at least 30 rule observations, got {len(obs)}")
    nrm = Norm(*size)
    xn, yn = nrm.to(obs[:, 0], obs[:, 1])
    v = obs[:, 2]

    c1 = c2 = 0.0
    u_lo, u_hi = 0.0, 1.0
    p_v = resid = inl = None
    u0 = np.zeros_like(v)

    for _ in range(rounds):
        # perspective explains what the bow does not
        span = max(u_hi - u_lo, 1e-9)
        t = (u0 - u_lo) / span
        bow = c1 * t * (1 - t) + c2 * t * (1 - t) * (2 * t - 1)
        p_v, resid, inl = _fit_v_row(xn, yn, v - bow, spacing=spacing)

        M0 = _u_row(p_v, 0.0)
        q = M0 @ np.stack([xn, yn, np.ones_like(xn)])
        u0 = q[0] / q[2]
        v_persp = q[1] / q[2]
        c1, c2, u_lo, u_hi = _fit_bow(u0[inl], (v - v_persp)[inl])

    # anchor u=0 on the red margin when we have one, else on the left extent
    if margin_x is not None:
        mxn, myn = nrm.to(np.array([float(margin_x)]), np.array([nrm.cy]))
        qm = M0 @ np.stack([mxn, myn, np.ones_like(mxn)])
        shift = -float((qm[0] / qm[2])[0])
    else:
        shift = -u_lo

    # Writing starts at the margin when there is one -- that is what a margin
    # is for. The ink extent only tells us where the *rules* start, and on this
    # paper they run past the margin to the page edge.
    left = 0.0 if margin_x is not None else (u_lo + shift)
    M = _u_row(p_v, shift)
    sheet = Sheet(M, c1, c2, left, u_hi + shift,
                  int(np.floor(v[inl].min() + 0.5)),
                  int(round(v[inl].max() - v[inl].min())) + 1,
                  size, spacing_hint=spacing, margin_x=margin_x)
    return sheet, {"resid_v": resid, "inliers": inl}


def refit(sheet, obs, spacing=None):
    """Re-solve after the user has moved something. Keeps the writing area."""
    sp = spacing or sheet.spacing_hint
    new, info = fit_sheet(obs, sheet.size, sp, margin_x=sheet.margin_x)
    new.u_left, new.u_right = sheet.u_left, sheet.u_right
    new.i_first, new.n_lines = sheet.i_first, sheet.n_lines
    return new, info


def residuals_px(sheet, obs):
    """Per-observation error in pixels -- the number that actually matters."""
    obs = np.asarray(obs, float)
    _, v = sheet.uv(obs[:, 0], obs[:, 1])
    dv = v - obs[:, 2]
    mid = sheet.i_first + sheet.n_lines / 2.0
    sp = sheet.spacing_px(mid, (sheet.u_left + sheet.u_right) / 2.0)
    return np.abs(dv) * sp


def find_margin_u(sheet, red, search=(-8.0, 0.45), step=0.02):
    """Locate the printed margin in page space and return its u.

    Must be done in page space, not image space: perspective tilts the margin,
    so averaging fixed-x columns smears one sharp line across a hundred pixels
    and the peak both weakens and moves. Under the sheet the margin is straight
    and vertical, so a profile over u shows it cleanly.
    """
    lo = sheet.u_left + search[0]
    hi = sheet.u_left + search[1] * (sheet.u_right - sheet.u_left)
    us = np.arange(lo, hi, step)
    if len(us) < 20:
        return None
    lines = np.arange(sheet.i_first, sheet.i_first + sheet.n_lines, 0.5)
    grid = sheet.sample(red, lines, us)
    prof = np.median(grid, axis=0)          # median: a gap on a few lines can't kill it
    prof = prof - np.median(prof)
    if prof.std() < 1e-6:
        return None
    pk, props = find_peaks(prof, prominence=prof.std() * 2.0,
                           distance=max(3, int(0.25 / step)))
    if not len(pk):
        return None
    # Continuity: a real margin shows up on most lines, a smudge does not.
    # Measured against each line's own background rather than a global
    # threshold, which would be set by unrelated bright regions of the page.
    bg = np.median(grid, axis=1, keepdims=True)
    best, best_score = None, 0.0
    for k, x in enumerate(pk):
        frac = float(((grid[:, x] - bg[:, 0]) > 0.4 * prof[x]).mean())
        score = props["prominences"][k] * frac
        if frac > 0.55 and score > best_score:
            best, best_score = us[x], score
    return None if best is None else float(best)


def find_extent_u(sheet, response, pad=0.3, keep=0.25):
    """Where the ruling starts and ends, in page space.

    One measurement for the whole sheet instead of one per rule. The per-rule
    version this replaces produced left endpoints spanning 924px on a page
    whose ruling starts at a single place -- that spread was measurement noise,
    and it is what made the drawn lines look ragged.
    """
    span = sheet.u_right - sheet.u_left
    us = np.arange(sheet.u_left - pad * span, sheet.u_right + pad * span, 0.05)
    lines = np.arange(sheet.i_first, sheet.i_first + sheet.n_lines, 1.0)
    prof = np.median(sheet.sample(response, lines, us), axis=0)
    if prof.max() <= 0:
        return sheet.u_left, sheet.u_right
    good = prof > prof.max() * keep
    idx = np.flatnonzero(good)
    if not len(idx):
        return sheet.u_left, sheet.u_right
    return float(us[idx[0]]), float(us[idx[-1]])


def line_response(sheet, response, i, n=48):
    """Mean ink response along one rule, including rules outside n_lines."""
    H, W = response.shape
    pts = sheet.polyline_at(i, n)
    x = pts[:, 0]
    y = pts[:, 1]
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if ok.sum() < n * 0.6:
        return None
    return float(response[y[ok].astype(int), x[ok].astype(int)].mean())


def snap_response(sheet, response, i, n=48, span=0.45, steps=19):
    """Best ink response near rule i, searching a fraction of a line either way.

    Extrapolating the fit past the observed rules drifts by a few pixels per
    line, which is enough to sample a candidate rule in the gap beside it and
    score it as blank. Searching a window finds the rule that is really there.
    """
    best_v, best_d = None, 0.0
    for d in np.linspace(-span, span, steps):
        v = line_response(sheet, response, i + d, n)
        if v is not None and (best_v is None or v > best_v):
            best_v, best_d = v, float(d)
    return best_v, best_d


def extend_lines(sheet, response, max_add=8, keep=0.35):
    """Grow the sheet onto rules the tracker missed.

    Detection loses rules where the page curls into shadow, so the fitted grid
    stops short of the real ruling. The model is parametric, so candidate rules
    above and below cost nothing to evaluate -- accept them while they still
    carry ink comparable to the rules we already trust.
    """
    base = [line_response(sheet, response, sheet.i_first + k)
            for k in range(sheet.n_lines)]
    base = [b for b in base if b is not None]
    if not base:
        return sheet, (0, 0)
    thr = float(np.median(base)) * keep

    up = 0
    while up < max_add:
        v, _ = snap_response(sheet, response, sheet.i_first - (up + 1))
        if v is None or v < thr:
            break
        up += 1
    down = 0
    while down < max_add:
        v, _ = snap_response(sheet, response, sheet.i_first + sheet.n_lines + down)
        if v is None or v < thr:
            break
        down += 1

    sheet.i_first -= up
    sheet.n_lines += up + down
    return sheet, (up, down)


def line_confidence(sheet, response, n=48):
    """How much more ink sits ON each rule than in the gap beside it.

    Contrast against the immediate neighbourhood, not the raw response. The
    flat-fielded blue channel goes negative where the page falls into shadow,
    so a raw reading scores a perfectly placed rule at the top of the page as
    zero -- which is exactly the false alarm that teaches a user to ignore the
    warning. Contrast is immune to that offset, and it measures the thing we
    actually care about: is there a rule here rather than half a line away.

    Normalised so 1.0 is a typical rule on this page.
    """
    H, W = response.shape

    def sample(i):
        pts = sheet.polyline(i, n)
        xs = np.clip(pts[:, 0], 0, W - 1).astype(int)
        ys = np.clip(pts[:, 1], 0, H - 1).astype(int)
        return response[ys, xs].mean()

    out = []
    for k in range(sheet.n_lines):
        i = sheet.i_first + k
        on = sample(i)
        gap = 0.5 * (sample(i - 0.5) + sample(i + 0.5))
        out.append(float(on - gap))
    a = np.array(out)
    pos = a[a > 0]
    med = np.median(pos) if pos.size else 1.0
    if med <= 0:
        med = 1.0
    return np.clip(a / med, 0.0, 1.5).tolist()
