"""
Handwriting Synthesis — local dashboard backend.

A tiny single-user FastAPI server that wraps the existing pipeline scripts
(segment.py, package_dataset.py, finetune.py, generate.py, render_page.py) so the
whole workflow can be driven from a browser UI instead of the terminal.

Design notes:
  * The heavy lifting stays in the pinned scripts under scripts/. This server only
    uploads files, writes labels.csv, launches those scripts as subprocesses, and
    streams their logs back to the page. No ML logic lives here.
  * Everything runs against the project's own venv python (sys.executable, because
    this server is started BY that venv python) with PYTORCH_ENABLE_MPS_FALLBACK=1
    and cwd = project root, exactly like the reviewer-verified manual invocations.
  * Long-running training is a detached-ish subprocess writing to train_log.txt
    (the script already does this); the UI tails that file + polls samples/.
"""

import csv
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    FileResponse,
    PlainTextResponse,
)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT = Path("/Users/blakey5aces/Handwriting Analysis")
SCRIPTS = PROJECT / "scripts"
RAW = PROJECT / "raw_scans"
SEGMENTED = PROJECT / "segmented"
DATASET = PROJECT / "dataset"
CHECKPOINTS = PROJECT / "checkpoints"
SAMPLES = PROJECT / "samples"
OUTPUT = PROJECT / "output"
WEIGHTS = PROJECT / "models" / "weights" / "FW-GAN.pth"
LABELS = PROJECT / "labels.csv"
MANIFEST = SEGMENTED / "manifest.csv"
TRAIN_LOG = PROJECT / "train_log.txt"
DASHBOARD_DIR = PROJECT / "dashboard"
STATIC_INDEX = DASHBOARD_DIR / "index.html"

PAGES = PROJECT / "pages"
BOXES = PROJECT / "boxes"
ROTATION_JSON = PROJECT / "page_rotation.json"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".webp", ".bmp", ".tif", ".tiff"}

for d in (RAW, SEGMENTED, DATASET, CHECKPOINTS, SAMPLES, OUTPUT, PAGES, BOXES):
    d.mkdir(parents=True, exist_ok=True)

# Pipeline modules are imported directly (not shelled out to) where the server
# only needs a pure helper -- dictionary lookup, box geometry -- so a filter
# request doesn't pay subprocess startup.
sys.path.insert(0, str(SCRIPTS))
import boxstore as _boxstore  # noqa: E402
import curate_labels as _curate  # noqa: E402
import wordcheck as _wordcheck  # noqa: E402

_VOCAB = None
_LEX = None


def _lex():
    """Spell/plausibility judge, built once (it reads the whole manifest)."""
    global _LEX
    if _LEX is None:
        _LEX = _wordcheck.Lexicon.build(str(PROJECT))
    return _LEX

# ----------------------------------------------------------------------------
# Subprocess helpers
# ----------------------------------------------------------------------------
PY = sys.executable  # the venv python that launched this server


def _env():
    e = os.environ.copy()
    e["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Scripts self-insert models/FW_GAN, but set it too for belt-and-suspenders.
    fw = str(PROJECT / "models" / "FW_GAN")
    e["PYTHONPATH"] = fw + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    return e


def stream_script(args):
    """Run `python scripts/<...>` and yield combined stdout/stderr lines as SSE."""
    cmd = [PY] + args
    yield _sse(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT),
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:  # pragma: no cover - defensive
        yield _sse(f"[launch error] {exc}\n")
        yield _sse("__DONE__ 1")
        return
    for line in iter(proc.stdout.readline, ""):
        yield _sse(line.rstrip("\n"))
    proc.stdout.close()
    rc = proc.wait()
    yield _sse(f"__DONE__ {rc}")


def _sse(data: str) -> str:
    # Server-Sent Events: one "data:" per line; blank line terminates the event.
    return "data: " + data.replace("\r", "") + "\n\n"


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
app = FastAPI(title="Handwriting Synthesis Dashboard")


@app.get("/", response_class=HTMLResponse)
def index():
    if STATIC_INDEX.is_file():
        return STATIC_INDEX.read_text(encoding="utf-8")
    return HTMLResponse("<h1>index.html missing</h1>", status_code=500)


# ---- image serving ---------------------------------------------------------
def _safe_under(base: Path, name: str) -> Path:
    p = (base / name).resolve()
    if base.resolve() not in p.parents and p != base.resolve():
        raise HTTPException(400, "path escapes base directory")
    if not p.is_file():
        raise HTTPException(404, "not found")
    return p


@app.get("/img/raw/{name}")
def img_raw(name: str):
    return FileResponse(_safe_under(RAW, name))


@app.get("/img/segmented/{name}")
def img_segmented(name: str):
    return FileResponse(_safe_under(SEGMENTED, name))


@app.get("/img/segmented/debug/{name}")
def img_segmented_debug(name: str):
    return FileResponse(_safe_under(SEGMENTED / "debug", name))


@app.get("/img/samples/{name}")
def img_samples(name: str):
    return FileResponse(_safe_under(SAMPLES, name))


@app.get("/img/output/{name}")
def img_output(name: str):
    return FileResponse(_safe_under(OUTPUT, name))


@app.get("/file/output/{name}")
def file_output(name: str):
    p = _safe_under(OUTPUT, name)
    return FileResponse(p, filename=name)


# ---- status ----------------------------------------------------------------
@app.get("/api/status")
def status():
    raw = sorted(p.name for p in RAW.iterdir() if p.suffix.lower() in IMG_EXTS) if RAW.exists() else []
    crops = sorted(p.name for p in SEGMENTED.glob("*.png")) if SEGMENTED.exists() else []
    n_labeled = 0
    if LABELS.is_file():
        with LABELS.open(newline="", encoding="utf-8") as f:
            n_labeled = sum(1 for r in csv.DictReader(f) if r.get("text", "").strip())
    ckpts = sorted(p.name for p in CHECKPOINTS.glob("*.pth")) if CHECKPOINTS.exists() else []
    samples = sorted(p.name for p in SAMPLES.glob("*.png")) if SAMPLES.exists() else []
    return {
        "raw_scans": raw,
        "n_raw": len(raw),
        "n_crops": len(crops),
        "n_labeled": n_labeled,
        "dataset_exists": (DATASET / "dataset.h5").is_file(),
        "checkpoints": ckpts,
        "has_finetuned": (CHECKPOINTS / "latest.pth").is_file(),
        "n_samples": len(samples),
        "training": _train.running(),
        "pretrained_exists": WEIGHTS.is_file(),
    }


# ---- upload ----------------------------------------------------------------
@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    saved, rejected = [], []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in IMG_EXTS:
            rejected.append(f.filename)
            continue
        dest = RAW / Path(f.filename).name
        data = await f.read()
        dest.write_bytes(data)
        saved.append(dest.name)
    return {"saved": saved, "rejected": rejected}


@app.post("/api/delete_raw")
def delete_raw(name: str = Form(...)):
    p = _safe_under(RAW, name)
    p.unlink()
    return {"deleted": name}


# ---- segment ---------------------------------------------------------------
@app.get("/api/segment")
def segment(granularity: str = "word"):
    gran = granularity if granularity in ("word", "line") else "word"
    return StreamingResponse(
        stream_script([str(SCRIPTS / "segment.py"), "--granularity", gran]),
        media_type="text/event-stream",
    )


def _vocab():
    """Dictionary used for the label-plausibility flag, loaded once."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = _curate.load_dictionary()
    return _VOCAB


def _read_manifest():
    if not MANIFEST.is_file():
        return []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _flag_reason(text, conf, min_conf):
    """(why this label needs human eyes, what it probably should say).

    Confidence alone is a weak test -- Vision reports 1.00 for plenty of wrong
    readings ("value"->"valve") -- so spelling plausibility carries most of the
    weight and confidence is a second, independent input.
    """
    flag, sug, _src = _wordcheck.judge(text, _lex())
    if flag:
        return flag, sug
    if conf < min_conf:
        return "low_conf", None
    return None, None


@app.get("/api/crops")
def crops(filter: str = "all", min_conf: float = 0.5,
          offset: int = 0, limit: int = 200):
    """Crops + labels, filterable and paginated.

    Paginated because the page previously rendered every crop at once; at ~3k
    crops that is thousands of DOM nodes and image requests in one shot, which
    is what made the Label tab crawl.
    """
    labels = {}
    if LABELS.is_file():
        with LABELS.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                labels[os.path.basename(r["image_path"])] = r.get("text", "")

    # Boxes a human signed off on must not be re-flagged here either, or the
    # overview contradicts the page you accepted them on.
    ok_ids = set()
    for sp in (sorted(RAW.iterdir()) if RAW.exists() else []):
        if sp.suffix.lower() not in IMG_EXTS:
            continue
        for b in _boxstore.active(_boxstore.load(str(PROJECT), sp.name) or {}):
            if b.get("accepted"):
                ok_ids.add(b["id"])

    rows = []
    for r in _read_manifest():
        name = os.path.basename(r["crop_path"])
        if not (SEGMENTED / name).is_file():
            continue
        auto = r.get("auto_text", "")
        # A label counts as present only once CURATED into labels.csv. The raw
        # Vision guess stays separate as `auto` (offered as a suggestion), so
        # "unlabeled" means "nothing trustworthy committed yet" rather than
        # "the detector had no opinion" -- which is the set worth reviewing.
        text = labels.get(name, "")
        try:
            conf = float(r.get("line_conf") or 0)
        except ValueError:
            conf = 0.0
        if r.get("box_id") in ok_ids:
            reason, suggest = None, None
        else:
            reason, suggest = _flag_reason(text or auto, conf, min_conf)
        rows.append({
            "name": name, "text": text, "auto": auto, "conf": round(conf, 3),
            "flag": reason, "suggest": suggest,
            "scan": os.path.basename(r.get("source_scan", "")),
            "key": _boxstore.page_key(r.get("source_scan", "")),
            "box_id": r.get("box_id", ""), "edited": r.get("edited") == "1",
        })

    if filter == "unlabeled":
        rows = [r for r in rows if not r["text"].strip()]
    elif filter == "flagged":
        rows = [r for r in rows if r["flag"]]
    elif filter == "edited":
        rows = [r for r in rows if r["edited"]]

    total = len(rows)
    counts = {
        "all": total,
        "flagged": sum(1 for r in rows if r["flag"]),
    }
    return {"crops": rows[offset:offset + limit], "total": total,
            "offset": offset, "limit": limit, "counts": counts}


# ---- page / box editing ----------------------------------------------------
_FLAG_CACHE = {}
_LEX_GEN = 0


_MANIFEST_IDS = {"stamp": None, "ids": set()}


def _manifest_ids():
    """Box ids that actually produced a crop file.

    A box can exist in the box store yet never reach the dataset -- build_page
    skips anything below its minimum width/height. Counting those would make
    the running word total on the Build Dataset tab disagree with the number of
    pairs the build actually writes, which is the one number that tab exists to
    get right.
    """
    stamp = MANIFEST.stat().st_mtime_ns if MANIFEST.is_file() else 0
    if _MANIFEST_IDS["stamp"] != stamp:
        ids = set()
        for r in _read_manifest():
            if r.get("box_id"):
                ids.add(r["box_id"])
        _MANIFEST_IDS.update(stamp=stamp, ids=ids)
    return _MANIFEST_IDS["ids"]


def _page_review(scan_name, key, doc):
    """(flagged, trainable, characters) for one page.

    `trainable` is the number of boxes that would actually reach labels.csv --
    it applies the same rule build_labels.py does, so the count shown next to a
    page is the count that page contributes, not just its box total.
    `characters` is the alphabet those words cover, which matters more than raw
    volume: the model cannot learn to draw a letter it has never been shown.

    Cached per page: recomputing means decoding the page image and running
    connected components over every box (~0.7s for the whole set), and the page
    picker asks for all of them at once. Keyed on the box file's mtime and the
    lexicon generation, so an edit or a re-detect recomputes and nothing else
    does.
    """
    if not doc:
        return 0, 0, ""
    bp = Path(_boxstore.path_for(str(PROJECT), scan_name))
    stamp = bp.stat().st_mtime_ns if bp.is_file() else 0
    hit = _FLAG_CACHE.get(key)
    if hit and hit[0] == stamp and hit[1] == _LEX_GEN:
        return hit[2], hit[3], hit[4]
    lex, gray, cropped = _lex(), _page_gray(key), _manifest_ids()
    flagged = trainable = 0
    chars = set()
    for b in _boxstore.active(doc):
        text = (b.get("text") or "").strip()
        if b.get("accepted"):
            flag = None
        else:
            flag, _s, _src = _wordcheck.judge(text, lex, _ink(gray, b))
        if flag:
            flagged += 1
        elif text and b["id"] in cropped:
            trainable += 1
            chars.update(text)
    joined = "".join(sorted(chars))
    _FLAG_CACHE[key] = (stamp, _LEX_GEN, flagged, trainable, joined)
    return flagged, trainable, joined


@app.get("/api/pages")
def list_pages():
    """Scans with their box counts, for the page picker in the editor."""
    out = []
    for p in sorted(RAW.iterdir()) if RAW.exists() else []:
        if p.suffix.lower() not in IMG_EXTS:
            continue
        doc = _boxstore.load(str(PROJECT), p.name)
        key = _boxstore.page_key(p.name)
        flagged, trainable, chars = _page_review(p.name, key, doc)
        out.append({
            "scan": p.name,
            "key": key,
            "boxes": len(_boxstore.active(doc)) if doc else 0,
            # The picker showed only a green box count, which reads as "this
            # page is done" -- on a page where a third of the boxes are wrong.
            "flagged": flagged,
            "trainable": trainable,
            "chars": chars,
            "rotation": (doc or {}).get("rotation", 0),
            "ready": (PAGES / f"{key}.png").is_file(),
        })
    return {"pages": out}


@app.get("/page_img/{key}")
def page_img(key: str):
    return FileResponse(_safe_under(PAGES, key + ".png"))


@app.get("/api/checkword")
def checkword(text: str = ""):
    """Judge free-typed text against the document vocabulary, live.

    The browser's own spellcheck (turned on via the `spellcheck` attribute on
    the inputs) already underlines ordinary typos as you type -- that needs no
    endpoint. This exists for the words it gets backwards: "operations",
    "forecast", the domain jargon in these notes, which a generic dictionary
    flags as wrong but which is exactly right here. No ink check (there is no
    box yet, just typed text).
    """
    flag, sug, _src = _wordcheck.judge(text, _lex())
    return {"flag": flag, "suggest": sug}


_PAGE_CACHE = {}


def _page_gray(key):
    """The processed page image the editor overlays boxes on, cached by mtime."""
    import cv2
    p = PAGES / f"{key}.png"
    if not p.is_file():
        return None
    stamp = p.stat().st_mtime_ns
    hit = _PAGE_CACHE.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    _PAGE_CACHE[key] = (stamp, img)
    return img


def _ink(gray, b):
    """Fraction of a box covered by letter-shaped ink.

    Plain dark-pixel counting does not work here: the pages are ruled, and the
    scan carries the shadow of a spiral binding, so blank paper is not white.
    Binarising and then discarding components that are too short to be a letter
    or too long-and-flat to be anything but a rule removes most of that.

    It is still only a weak signal on a page this dense -- an empty box in the
    gutter between two lines can score as high as a legitimately sparse crop --
    so it flags the clear-cut cases only. The crop preview in the editor is what
    actually proves a box contains what you think it does.
    """
    import cv2
    if gray is None:
        return None
    y0, x0 = max(0, int(b["y"])), max(0, int(b["x"]))
    r = gray[y0:y0 + int(b["h"]), x0:x0 + int(b["w"])]
    if r.size < 100:
        return 0.0
    bw = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 12)
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    H = r.shape[0]
    keep = 0
    for i in range(1, n):
        _, _, w, h, a = st[i]
        if h < max(3, 0.14 * H) or a < 12:
            continue
        if w > 6 * h and h < 0.30 * H:          # a ruled line, not a letter
            continue
        keep += a
    return round(keep / float(r.size), 4)


@app.get("/img/box/{key}/{box_id}")
def img_box(key: str, box_id: str, x: int = -1, y: int = -1, w: int = 0, h: int = 0):
    """One box as a picture: the region, outlined, with a little context around it.

    Geometry can be passed explicitly, and the editor always does. That matters:
    a box the user has dragged but not yet saved is still at its old coordinates
    on disk, so looking the box up would draw the wrong region -- exactly the
    case (a box moved off its word) this preview exists to make visible.
    """
    import cv2
    box_id = box_id[:-4] if box_id.endswith(".png") else box_id
    gray = _page_gray(key)
    if gray is None:
        raise HTTPException(404, "page not built")

    box = None
    if x >= 0 and y >= 0 and w > 0 and h > 0:
        box = {"x": x, "y": y, "w": w, "h": h}
    else:
        for p in RAW.iterdir():
            if _boxstore.page_key(p.name) != key:
                continue
            for b in _boxstore.active(_boxstore.load(str(PROJECT), p.name) or {}):
                if b["id"] == box_id:
                    box = b
                    break
    if box is None:
        raise HTTPException(404, "unknown box")

    H, W = gray.shape[:2]
    # Wide but shallow context: enough of the neighbouring words to judge the
    # reading, without so much vertical padding that the word itself becomes an
    # unreadable sliver once the thumbnail is scaled to fit.
    px, py = max(12, int(0.60 * box["h"])), max(4, int(0.16 * box["h"]))
    x0, y0 = max(0, box["x"] - px), max(0, box["y"] - py)
    x1, y1 = min(W, box["x"] + box["w"] + px), min(H, box["y"] + box["h"] + py)
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(404, "box outside the page")
    crop = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    cv2.rectangle(crop, (box["x"] - x0, box["y"] - y0),
                  (box["x"] + box["w"] - x0, box["y"] + box["h"] - y0),
                  (60, 60, 220), 2)
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(content=buf.tobytes(), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/page/{key}/boxes")
def get_boxes(key: str):
    for p in RAW.iterdir():
        if _boxstore.page_key(p.name) == key:
            doc = _boxstore.load(str(PROJECT), p.name)
            if not doc:
                raise HTTPException(404, "no boxes yet — run detect first")
            boxes = _boxstore.active(doc)
            lex, gray = _lex(), _page_gray(key)
            for b in boxes:
                ink = _ink(gray, b)
                if b.get("accepted"):
                    # Human said this really is what the page says (these notes
                    # contain genuine misspellings). Stop second-guessing it.
                    flag = sug = src = None
                else:
                    flag, sug, src = _wordcheck.judge(b.get("text", ""), lex, ink)
                b["ink"] = ink
                b["flag"] = flag
                b["suggest"] = sug
                b["sug_src"] = src
            doc["boxes"] = boxes
            doc["flagged"] = sum(1 for b in boxes if b["flag"])
            return doc
    raise HTTPException(404, "unknown page")


@app.put("/api/page/{key}/boxes")
async def put_boxes(key: str, payload: dict):
    """Persist edited boxes.

    Anything arriving here has been through the editor, so each box is marked
    edited unless it is byte-identical to what detection produced. That flag is
    what protects it from being overwritten by the next detect pass.
    """
    for p in RAW.iterdir():
        if _boxstore.page_key(p.name) != key:
            continue
        doc = _boxstore.load(str(PROJECT), p.name)
        if not doc:
            raise HTTPException(404, "no boxes yet")
        prior = {b["id"]: b for b in doc.get("boxes", [])}
        incoming = payload.get("boxes", [])
        merged = []
        seen = set()
        for b in incoming:
            bid = b.get("id") or _boxstore.new_id()
            seen.add(bid)
            old = prior.get(bid)
            moved = (not old) or any(
                int(b.get(k, 0)) != int(old.get(k, 0)) for k in ("x", "y", "w", "h")
            ) or (b.get("text", "") != old.get("text", ""))
            rec = dict(old or _boxstore.make_box(0, 0, 0, 0))
            rec.update({
                "id": bid,
                "x": int(b.get("x", 0)), "y": int(b.get("y", 0)),
                "w": int(b.get("w", 0)), "h": int(b.get("h", 0)),
                "text": b.get("text", rec.get("text", "")),
                "source": b.get("source", rec.get("source", "manual")),
                "accepted": bool(b.get("accepted", rec.get("accepted", False))),
                "deleted": False,
            })
            if moved:
                rec["edited"] = True
            merged.append(rec)
        # boxes the editor dropped are soft-deleted, never lost
        for bid, old in prior.items():
            if bid not in seen:
                old["deleted"] = True
                merged.append(old)
        doc["boxes"] = merged
        _boxstore.assign_reading_order(doc)
        _boxstore.save(str(PROJECT), p.name, doc)
        return {"saved": len([b for b in merged if not b.get("deleted")]),
                "deleted": len([b for b in merged if b.get("deleted")])}
    raise HTTPException(404, "unknown page")


@app.post("/api/page/{key}/rotate")
def rotate_page(key: str, payload: dict):
    """Record a page's rotation. Vision reads upside-down text fine, so this
    cannot be auto-detected — see boxstore/vision_segment notes."""
    deg = int(payload.get("degrees", 180)) % 360
    rot = {}
    if ROTATION_JSON.is_file():
        rot = json.loads(ROTATION_JSON.read_text())
    for p in RAW.iterdir():
        if _boxstore.page_key(p.name) == key:
            if deg:
                rot[p.name] = deg
            else:
                rot.pop(p.name, None)
            ROTATION_JSON.write_text(json.dumps(rot, indent=2, ensure_ascii=False))
            return {"scan": p.name, "degrees": deg}
    raise HTTPException(404, "unknown page")


def _invalidate():
    """The document vocabulary is derived from the manifest, so any pass that
    rewrites the manifest also invalidates the judge and the page cache."""
    global _LEX, _LEX_GEN
    _LEX = None
    _LEX_GEN += 1          # retires every cached per-page flag count
    _PAGE_CACHE.clear()


@app.get("/api/detect")
def detect(only: str = ""):
    _invalidate()
    args = [str(SCRIPTS / "detect_and_build.py"), "all"]
    if only:
        args += ["--only", only]
    return StreamingResponse(stream_script(args), media_type="text/event-stream")


@app.get("/api/rebuild")
def rebuild():
    """Regenerate crops from the (possibly edited) box store."""
    _invalidate()
    return StreamingResponse(
        stream_script([str(SCRIPTS / "detect_and_build.py"), "build"]),
        media_type="text/event-stream")


@app.get("/api/buildlabels")
def buildlabels(pages: str = ""):
    """Box store -> labels.csv, honouring human overrides (see build_labels.py).

    `pages` is an optional comma-separated list of page keys, so the Build
    Dataset tab can train on a chosen subset rather than everything on disk.
    """
    _invalidate()
    args = [str(SCRIPTS / "build_labels.py")]
    if pages:
        args += ["--pages", pages]
    return StreamingResponse(stream_script(args), media_type="text/event-stream")


@app.get("/api/curate")
def curate(min_conf: float = 0.3):
    return StreamingResponse(
        stream_script([str(SCRIPTS / "curate_labels.py"), "--min-conf", str(min_conf)]),
        media_type="text/event-stream")


@app.post("/api/labels")
async def save_labels(payload: dict):
    """payload = {"labels": {crop_name: text, ...}}  -> MERGES into labels.csv.

    Merge, not overwrite. The Label tab paginates at 200 crops, and it posts
    the inputs currently on screen -- so a rewrite-from-scratch silently
    deleted every label not on the visible page (it took labels.csv from 1912
    rows down to 200 exactly once before this was caught). Only the keys
    actually sent are touched.
    """
    items = payload.get("labels", {})
    rows = {}
    if LABELS.is_file():
        with LABELS.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[os.path.basename(r["image_path"])] = r.get("text", "")
    before = len(rows)
    rows.update({os.path.basename(k): v for k, v in items.items()})

    tmp = LABELS.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "text", "writer_id"])
        for name, text in sorted(rows.items()):
            w.writerow([str(SEGMENTED / name), text, "me"])
    os.replace(tmp, LABELS)
    return {"written": len(rows), "submitted": len(items), "was": before,
            "non_empty": sum(1 for v in rows.values() if str(v).strip())}


@app.get("/api/dataset_info")
def dataset_info():
    """Read the built dataset back and report what is actually in it.

    Deliberately reads dataset.h5 rather than echoing what the builder claimed:
    the point of the confirmation panel is to tell you what landed on disk, so
    it has to come from the file, not from the numbers the UI predicted.
    """
    f = DATASET / "dataset.h5"
    if not f.is_file():
        return {"exists": False}
    import h5py
    with h5py.File(f, "r") as h:
        n = int(len(h["wids"]))
        height = int(h["imgs"].shape[0])
        px = int(h["imgs"].shape[1])
        chars = sorted({chr(c) for c in h["lbs"][:]})
        avg_len = float(h["lb_lens"][:].mean()) if n else 0.0
    words = []
    if LABELS.is_file():
        with LABELS.open(newline="", encoding="utf-8") as fh:
            words = [r["text"] for r in csv.DictReader(fh) if r.get("text", "").strip()]
    # Which requested labels the packager cannot represent. It validates every
    # crop against the model's fixed 81-character alphabet, so a word containing
    # anything else (an arrow, an equals sign) is dropped. Parsed straight out
    # of the model's own alphabet.py so this can't drift from the real rule.
    unsupported = {}
    try:
        alpha_src = (PROJECT / "models" / "FW_GAN" / "lib" / "alphabet.py").read_text(encoding="utf-8")
        m = re.search(r"'all':\s*'((?:[^'\\]|\\.)*)'", alpha_src)
        allowed = set(m.group(1).encode().decode("unicode_escape")) if m else None
    except Exception:
        allowed = None
    if allowed:
        for w in words:
            for ch in w:
                if ch not in allowed:
                    unsupported[ch] = unsupported.get(ch, 0) + 1

    # which pages are actually represented, derived from the label paths
    pages = set()
    label_names = set()
    if LABELS.is_file():
        with LABELS.open(newline="", encoding="utf-8") as fh:
            label_names = {os.path.basename(r["image_path"]) for r in csv.DictReader(fh)}
    for r in _read_manifest():
        if os.path.basename(r["crop_path"]) in label_names:
            pages.add(_boxstore.page_key(r.get("source_scan", "")))
    return {
        "exists": True,
        "samples": n,
        "pages": len(pages),
        "unique_words": len(set(w.lower() for w in words)),
        # labels.csv is what we asked for; `samples` is what survived the
        # packager's image validation. A gap means crops it refused.
        "requested": len(words),
        "unsupported": sorted(unsupported.items(), key=lambda kv: -kv[1]),
        "chars": "".join(chars),
        "n_chars": len(chars),
        "avg_word_len": round(avg_len, 1),
        "height": height,
        "total_px": px,
        "bytes": f.stat().st_size,
        "built": int(f.stat().st_mtime),
    }


@app.get("/api/package")
def package():
    return StreamingResponse(
        stream_script([str(SCRIPTS / "package_dataset.py")]),
        media_type="text/event-stream",
    )


# ---- training --------------------------------------------------------------
class TrainManager:
    def __init__(self):
        self.proc = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, params):
        if self.running():
            raise HTTPException(409, "training already running")
        args = [
            PY,
            str(SCRIPTS / "finetune.py"),
            "--epochs", str(params.get("epochs", 40)),
            "--batch-size", str(params.get("batch_size", 8)),
            "--lr", str(params.get("lr", 1e-4)),
            "--sample-every", str(params.get("sample_every", 25)),
            "--ckpt-every", str(params.get("ckpt_every", 25)),
            "--keep", str(params.get("keep", 3)),
            "--sample-text", str(params.get("sample_text", "the quick brown fox")),
        ]
        if params.get("max_steps"):
            args += ["--max-steps", str(params["max_steps"])]
        if params.get("style_ref"):
            args += ["--style-ref", str(SEGMENTED / params["style_ref"])]
        if params.get("resume"):
            args += ["--resume"]
        # Truncate the log so the UI shows only this run.
        TRAIN_LOG.write_text("", encoding="utf-8")
        self.proc = subprocess.Popen(
            args, cwd=str(PROJECT), env=_env(),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        return {"pid": self.proc.pid, "cmd": " ".join(shlex.quote(a) for a in args)}

    def stop(self):
        if not self.running():
            return {"stopped": False, "reason": "not running"}
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return {"stopped": True}


_train = TrainManager()


@app.post("/api/train/start")
def train_start(params: dict):
    if not (DATASET / "dataset.h5").is_file():
        raise HTTPException(400, "dataset/dataset.h5 not found — build the dataset first")
    return _train.start(params)


@app.post("/api/train/stop")
def train_stop():
    return _train.stop()


@app.get("/api/train/status")
def train_status():
    latest = None
    latest_mtime = -1
    if SAMPLES.exists():
        for p in SAMPLES.glob("step_*.png"):
            m = p.stat().st_mtime
            if m > latest_mtime:
                latest_mtime, latest = m, p.name
    tail = ""
    if TRAIN_LOG.is_file():
        tail = TRAIN_LOG.read_text(encoding="utf-8", errors="replace")
    return {
        "running": _train.running(),
        "latest_sample": latest,
        "log": tail[-8000:],  # last chunk only
    }


# ---- generate --------------------------------------------------------------
@app.get("/api/generate")
def generate(text: str, style: str = "", ckpt: str = "pretrained"):
    # Clear prior per-word PNGs so the gallery shows only this run's words.
    for p in OUTPUT.glob("*.png"):
        if p.name[:3].isdigit():
            try:
                p.unlink()
            except OSError:
                pass
    args = [str(SCRIPTS / "generate.py"), "--text", text, "--device", "mps"]
    if style:
        args += ["--style", str(SEGMENTED / style)]
    if ckpt and ckpt != "pretrained":
        args += ["--ckpt", str(CHECKPOINTS / ckpt)]
    return StreamingResponse(stream_script(args), media_type="text/event-stream")


@app.get("/api/output_words")
def output_words():
    """Individual generated word PNGs (exclude _line and page composites)."""
    names = []
    for p in sorted(OUTPUT.glob("*.png")):
        if p.name.startswith("_") or p.name.startswith("page") or p.name.startswith("rev"):
            continue
        # keep NNN_word.png pattern
        if p.name[:3].isdigit():
            names.append(p.name)
    return {"words": names, "line": (OUTPUT / "_line.png").is_file()}


# ---- render ----------------------------------------------------------------
@app.get("/api/render")
def render(text: str, style: str = "", ckpt: str = "pretrained", fmt: str = "png"):
    out_name = "dashboard_page." + ("pdf" if fmt == "pdf" else "png")
    args = [
        str(SCRIPTS / "render_page.py"),
        "--text", text,
        "--device", "mps",
        "--output", str(OUTPUT / out_name),
    ]
    if style:
        args += ["--style", str(SEGMENTED / style)]
    if ckpt and ckpt != "pretrained":
        args += ["--ckpt", str(CHECKPOINTS / ckpt)]
    return StreamingResponse(stream_script(args), media_type="text/event-stream")


# ---- write on paper --------------------------------------------------------
# Paper photos live in output/paper (uploads) plus the bundled test page in
# assets. The test page is deliberately a hard case -- perspective, page bow,
# uneven lighting, a second page and desk clutter in frame -- so detection
# regressions show up immediately.
PAPER = OUTPUT / "paper"
ASSETS = PROJECT / "assets"
BUILTIN_PAPER = "test_page.jpg"
PAPER_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
_RULING_CACHE = {}


def _paper_path(name: str) -> Path:
    if name == BUILTIN_PAPER:
        return ASSETS / BUILTIN_PAPER
    return _safe_under(PAPER, name)


@app.get("/api/paper/list")
def paper_list():
    PAPER.mkdir(parents=True, exist_ok=True)
    names = [BUILTIN_PAPER] if (ASSETS / BUILTIN_PAPER).is_file() else []
    names += sorted(p.name for p in PAPER.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    and not p.name.startswith("out_"))
    return {"papers": names, "builtin": BUILTIN_PAPER}


@app.post("/api/paper/upload")
async def paper_upload(files: list[UploadFile] = File(...)):
    PAPER.mkdir(parents=True, exist_ok=True)
    saved, rejected = [], []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in PAPER_EXTS:
            rejected.append(f.filename)
            continue
        stem = Path(f.filename).stem
        dest = PAPER / (stem + ext)
        dest.write_bytes(await f.read())
        # Browsers can't display HEIC, so normalise on the way in.
        if ext in {".heic", ".heif"}:
            jpg = PAPER / (stem + ".jpg")
            subprocess.run(["sips", "-s", "format", "jpeg", str(dest),
                            "--out", str(jpg)], capture_output=True)
            dest.unlink(missing_ok=True)
            dest = jpg
        saved.append(dest.name)
    return {"saved": saved, "rejected": rejected}


@app.get("/img/paper/{name}")
def img_paper(name: str):
    return FileResponse(_paper_path(name))


# Detection runs in-process, not as a subprocess: the editor re-derives the
# sheet on every handle drag and a process launch alone would blow the budget.
sys.path.insert(0, str(SCRIPTS))
import paper as _paper          # noqa: E402
import render_page as _render_page  # noqa: E402
import sheet as _sheetmod       # noqa: E402

_PAPER = {}   # name -> {sheet, obs, flat, mtime}


def _paper_state(name, rebuild=False):
    path = _paper_path(name)
    mt = path.stat().st_mtime
    st = _PAPER.get(name)
    if st is None or st["mtime"] != mt or rebuild:
        img = _paper.load_photo(str(path))
        sheet, obs, info = _paper.detect_sheet(img)
        st = {"sheet": sheet, "obs": obs, "flat": _paper._flat_response(img),
              "mtime": mt, "grown": info.get("grown")}
        _PAPER.clear()        # only ever one page open at a time
        _PAPER[name] = st
    return st


@app.get("/api/paper/detect")
def paper_detect(name: str, rebuild: bool = False):
    """Ruled-line geometry for the overlay."""
    try:
        st = _paper_state(name, rebuild)
    except Exception as exc:
        return {"error": str(exc)[:400]}
    return _paper.sheet_json(st["sheet"], st["flat"], obs=st.get("obs"))


@app.post("/api/paper/adjust")
async def paper_adjust(payload: dict):
    """Apply the user's corrections to the sheet.

    Most edits are pure rectangle changes -- dragging a corner along the rules
    only moves u_left/u_right, and dragging across them only picks a different
    first/last rule. Those need no optimisation at all, which matters: refitting
    on every drag would make a 3px nudge jump the whole page. The nonlinear
    refit is reserved for an explicit "snap to lines".
    """
    name = payload.get("name")
    if not name:
        return {"error": "no paper named"}
    try:
        st = _paper_state(name)
    except Exception as exc:
        return {"error": str(exc)[:400]}

    sh = _sheetmod.Sheet.from_json(payload["sheet"]) if payload.get("sheet") else st["sheet"]
    sh = _sheetmod.Sheet.from_json(sh.to_json())      # work on a copy

    # Free shape: the four corners are kept exactly where they were put, and
    # the writing area becomes that quadrilateral rather than a page-space
    # rectangle. Nothing is re-derived, so a hand-placed margin stays where it
    # was placed instead of snapping back onto a constant-u line.
    if "crop" in payload:
        c = payload.get("crop")
        sh.crop = [[float(x), float(y)] for x, y in c] if c else None
        if sh.crop:
            # Only lines that fit inside the shape in their entirety. This used
            # to round the crop's v range outward by up to half a line each end,
            # which is why rules appeared above the top corners and below the
            # bottom ones. Moving any corner re-derives from scratch, so the
            # corners stay the one source of truth for which lines exist.
            got = sh.lines_in_crop()
            if got:
                sh.i_first, sh.n_lines = got

    quad = payload.get("quad")
    if quad and len(quad) == 4 and sh.crop is None:
        us, vs = [], []
        for x, y in quad:
            u, v = sh.uv(float(x), float(y))
            us.append(float(u[0]))
            vs.append(float(v[0]))
        # corners are TL, TR, BR, BL
        corner = payload.get("corner")
        i0 = int(round(min(vs[0], vs[1])))
        i1 = int(round(max(vs[2], vs[3])))
        if corner is None:
            sh.u_left = min(us[0], us[3])
            sh.u_right = max(us[1], us[2])
        else:
            # One handle moved, so that handle decides the two edges it lies on.
            # Taking min/max across a pair instead meant the corner you were not
            # dragging pinned the value: you could haul a handle right across the
            # page and nothing moved, because its neighbour still held the old u.
            c = int(corner) % 4
            if c in (0, 3):
                sh.u_left = us[c]
            else:
                sh.u_right = us[c]
            if c in (0, 1):
                i0 = int(round(vs[c]))
            else:
                i1 = int(round(vs[c]))
        if i1 > i0:
            sh.i_first, sh.n_lines = i0, i1 - i0 + 1
        if sh.u_right - sh.u_left < 1.0:      # never let the area collapse
            sh.u_left, sh.u_right = min(us[0], us[3]), max(us[1], us[2])

    if payload.get("margin_x") is not None:
        mx = float(payload["margin_x"])
        my = sh.point(sh.i_first + sh.n_lines // 2, sh.u_right / 2)[1]
        u, _ = sh.uv(mx, my)
        sh.shift_u(-float(u[0]))
        sh.u_left = 0.0
        sh.margin_x = mx

    if payload.get("bow") is not None:
        sh.c1 = float(payload["bow"])
    if payload.get("add_top"):
        n = int(payload["add_top"])
        sh.i_first -= n
        sh.n_lines += n
    if payload.get("add_bottom"):
        sh.n_lines += int(payload["add_bottom"])
    sh.n_lines = max(1, sh.n_lines)

    if payload.get("resnap"):
        keep = (sh.u_left, sh.u_right, sh.i_first, sh.n_lines, sh.margin_x, sh.crop)
        obs = st["obs"]
        if sh.crop:
            # Refit against the rules the user pointed at, not the whole photo.
            # This is what makes the free shape useful beyond drawing: it tells
            # the detector which ink is the page.
            import cv2 as _cv2, numpy as _np
            poly = _np.array(sh.crop, _np.float32)
            keep_rows = [k for k in range(len(obs))
                         if _cv2.pointPolygonTest(
                             poly, (float(obs[k][0]), float(obs[k][1])), False) >= 0]
            if len(keep_rows) >= 30:
                obs = obs[keep_rows]
        sh, _info = _sheetmod.refit(sh, obs)
        sh.u_left, sh.u_right, sh.i_first, sh.n_lines, sh.margin_x, sh.crop = keep
        # refit moves M, and the crop is stored in image space, so the corners
        # now map to different page-space rows. Restoring the old line range
        # unchanged left the grid describing a v field that no longer existed --
        # the rules drifted off the quad. Re-derive against the new fit.
        if sh.crop:
            got = sh.lines_in_crop()
            if got:
                sh.i_first, sh.n_lines = got

    st["sheet"] = sh
    return _paper.sheet_json(sh, st["flat"], obs=st.get("obs"))


# ---- editable document -----------------------------------------------------
# All page geometry stays on this side. The browser receives each word already
# placed in photo pixels and sends back pixel positions; it never reimplements
# the sheet maths, so there is only one definition of where a word goes.
import doc as _doc          # noqa: E402

# Undo lives here rather than in the browser because the document does. The box
# editor's pattern is the same -- whole-state snapshots, capped depth -- just
# kept on the side that owns the state, so a reload cannot desynchronise it.
_UNDO: dict = {}
_REDO: dict = {}
_UNDO_MAX = 60


def _snapshot(name, document):
    _UNDO.setdefault(name, []).append(json.dumps(document))
    if len(_UNDO[name]) > _UNDO_MAX:
        _UNDO[name].pop(0)
    _REDO[name] = []


DOCS = OUTPUT / "paper" / "docs"
WORDS = OUTPUT / "paper" / "words"


def _doc_path(name):
    DOCS.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return DOCS / f"{safe}.json"


def _doc_style(document):
    st = document.get("style") or {}
    crop = st.get("crop")
    ckpt = st.get("ckpt")
    return (str(SEGMENTED / crop) if crop else None,
            str(CHECKPOINTS / ckpt) if ckpt and ckpt != "pretrained" else None)


def _load_doc(name, sheet):
    path = _doc_path(name)
    if path.exists():
        try:
            return _doc.load(str(path))
        except Exception:
            pass
    return _doc.new_doc(name, sheet)


def doc_view(document, sheet):
    """The document as the browser needs it: words placed in photo pixels."""
    o = dict(_doc.DEFAULTS)
    o.update(document.get("opts") or {})
    words = []
    for w in sorted(document["words"], key=lambda d: d["ord"]):
        p = w.get("placed")
        item = {"id": w["id"], "text": w["text"], "pinned": bool(w.get("pin")),
                "overflow": bool(w.get("overflow")), "clipped": bool(w.get("clipped")),
                "png": os.path.basename(w["png"]) if w.get("png") else None}
        if p and not w.get("overflow"):
            quad = _doc.word_quad(document, sheet, w, o)
            xs = [c[0] for c in quad]; ys = [c[1] for c in quad]
            item.update({
                "line": p["line"], "u": round(p["u"], 4),
                # The four corners the word actually maps to. The browser draws
                # from these rather than deriving geometry of its own, so the
                # preview and the export cannot disagree about where a word is.
                "quad": [[round(c[0], 1), round(c[1], 1)] for c in quad],
                # axis-aligned bounds, for hit-testing only
                "x": round(min(xs), 1), "y": round(min(ys), 1),
                "w": round(max(xs) - min(xs), 1), "h": round(max(ys) - min(ys), 1),
            })
        words.append(item)
    return {"words": words, "opts": o, "text": document.get("text", ""),
            "style": document.get("style"), "seed": document.get("seed"),
            "overflow": sum(1 for w in words if w["overflow"])}


def _harden_visible_breaks(document, ids):
    """Turn the wrapped line breaks inside a selection into recorded ones.

    A visible line break comes either from a stored newline or from reflow
    simply running out of room, and only the first kind is data. So inserting a
    break in front of a selection that straddles a wrap let every word in it
    flow onto one fresh, empty, full-width line -- Blake's "five words and five
    words get mashed into a single line". Nothing was clobbered; the separator
    he could see had never existed as anything.

    Recording the breaks that are actually on screen, for the span being
    edited, makes what he sees and what the document holds the same thing.
    Deliberately scoped to the selection: hardening every wrap on the page
    would stop the text re-wrapping when the size or the writing area changes.
    """
    ws = sorted(document["words"], key=lambda d: d["ord"])
    chosen = {w["ord"] for w in ws if w["id"] in set(ids)}
    if not chosen:
        return
    lo, hi = min(chosen), max(chosen)
    prev = None
    for w in ws:
        p = w.get("placed")
        cur = int(p["line"]) if p and p.get("line") is not None else None
        if (cur is not None and prev is not None and cur != prev
                and lo <= w["ord"] <= hi and not int(w.get("nl", 0) or 0)):
            w["nl"] = 1
        if cur is not None:
            prev = cur


def _rebuild_text(document):
    """Reconstruct the source text from the words, including their structure."""
    out = []
    for k, w in enumerate(sorted(document["words"], key=lambda d: d["ord"])):
        if k == 0:
            out.append("\t" * int(w.get("tabs", 0)))
        elif w.get("nl"):
            out.append("\n" * int(w["nl"]) + "\t" * int(w.get("tabs", 0)))
        else:
            # A tab on a word that does not start a line is still an indent.
            # Emitting a bare space here dropped it, and the next "Write it"
            # re-parsed the text over the records and made the loss permanent.
            out.append(" " + "\t" * int(w.get("tabs", 0)))
        out.append(w["text"])
    return "".join(out)


def _sync_and_reflow(name, document, sheet, generate=True, log=None):
    style, ckpt = _doc_style(document)
    if generate:
        _doc.ensure_words(document, str(WORDS), style=style, ckpt=ckpt, log=log)
    stats = _doc.reflow(document, sheet)
    _doc.save(str(_doc_path(name)), document)
    return stats


@app.get("/api/doc")
def doc_get(name: str):
    try:
        st = _paper_state(name)
    except Exception as exc:
        return {"error": str(exc)[:400]}
    document = _load_doc(name, st["sheet"])
    document["sheet"] = st["sheet"].to_json()
    # Only lay out a document that has never been laid out. Reflowing on every
    # read looked harmless but made reads and writes disagree: the view showed a
    # freshly reflowed page while the stored positions were the old ones, so the
    # next edit appeared to move everything back.
    if any(w.get("placed") is None and not w.get("overflow") for w in document["words"]):
        _doc.reflow(document, st["sheet"])
        _doc.save(str(_doc_path(name)), document)
    return doc_view(document, st["sheet"])


@app.post("/api/doc/text")
async def doc_text(payload: dict):
    name = payload.get("name")
    try:
        st = _paper_state(name)
    except Exception as exc:
        return {"error": str(exc)[:400]}
    document = _load_doc(name, st["sheet"])
    if payload.get("style"):
        document["style"] = payload["style"]
    if payload.get("opts"):
        document.setdefault("opts", {}).update(payload["opts"])
    _doc.sync_text(document, payload.get("text", ""))
    try:
        _sync_and_reflow(name, document, st["sheet"])
    except Exception as exc:
        return {"error": str(exc)[:600]}
    return doc_view(document, st["sheet"])


@app.post("/api/doc/history")
async def doc_history(payload: dict):
    """Step the document back or forward one edit."""
    name = payload.get("name")
    if not name:
        return {"error": "no paper named"}
    try:
        st = _paper_state(name)
    except Exception as exc:
        return {"error": str(exc)[:400]}
    sheet = st["sheet"]
    document = _load_doc(name, sheet)

    back = payload.get("dir", "undo") == "undo"
    src, dst = (_UNDO, _REDO) if back else (_REDO, _UNDO)
    stack = src.get(name) or []
    if not stack:
        return {"error": "nothing to " + ("undo" if back else "redo"), **doc_view(document, sheet)}
    dst.setdefault(name, []).append(json.dumps(document))
    document = json.loads(stack.pop())
    _doc.save(str(_doc_path(name)), document)
    return doc_view(document, sheet)


@app.post("/api/doc/edit")
async def doc_edit(payload: dict):
    name = payload.get("name")
    op = payload.get("op")
    ids = payload.get("ids") or []
    try:
        st = _paper_state(name)
    except Exception as exc:
        return {"error": str(exc)[:400]}
    sheet = st["sheet"]
    document = _load_doc(name, sheet)
    regen = False
    # Dragging moves what you grabbed and nothing else; only operations that
    # change the text reflow the page, the way a word processor does.
    reflow_ops = {"unpin", "delete", "retext", "opts", "indent", "linebreak"}
    _snapshot(name, document)

    if op == "pin":
        u, v = sheet.uv(float(payload["x"]), float(payload["y"]))
        _doc.pin(document, ids, int(round(float(v[0]))), float(u[0]))
    elif op == "nudge":
        # a pixel delta means nothing to the model; convert it on this side
        mid = sheet.i_first + sheet.n_lines // 2
        x0, y0 = sheet.point(mid, sheet.u_right / 2)
        u0, v0 = sheet.uv(x0, y0)
        u1, v1 = sheet.uv(x0 + float(payload.get("dx", 0)),
                          y0 + float(payload.get("dy", 0)))
        for w in document["words"]:
            if w["id"] in set(ids) and not w.get("pin") and w.get("placed"):
                w["pin"] = {"line": w["placed"]["line"], "u": w["placed"]["u"], "group": None}
        _doc.move_pins(document, ids, 0, float(u1[0] - u0[0]))
        if payload.get("dline"):
            _doc.move_pins(document, ids, int(payload["dline"]), 0.0)
    elif op == "unpin":
        _doc.unpin(document, ids)
    elif op == "reroll":
        _doc.reroll(document, ids)
        regen = True
    elif op == "delete":
        _harden_visible_breaks(document, ids)
        wanted = set(ids)
        # Carry a removed word's line break onto the next survivor. Deleting the
        # word that happened to begin a line used to take the break with it and
        # merge that line into the one above.
        ws = sorted(document["words"], key=lambda d: d["ord"])
        carried = 0
        for w in ws:
            if w["id"] in wanted:
                carried += int(w.get("nl", 0) or 0)
            elif carried:
                w["nl"] = int(w.get("nl", 0) or 0) + carried
                carried = 0
        document["words"] = [w for w in document["words"] if w["id"] not in wanted]
        for k, w in enumerate(sorted(document["words"], key=lambda d: d["ord"])):
            w["ord"] = k
        document["text"] = _rebuild_text(document)
    elif op in ("indent", "linebreak"):
        _harden_visible_breaks(document, ids)
        # Applied to the first word of the selection: that word and everything
        # after it move, which is what pressing Tab or Enter in front of a
        # selection does in a word processor.
        d = int(payload.get("delta", 1))
        sel = [w for w in sorted(document["words"], key=lambda x: x["ord"])
               if w["id"] in set(ids)]
        if sel:
            key = "tabs" if op == "indent" else "nl"
            first = sel[0]
            first[key] = max(0, int(first.get(key, 0)) + d)
            document["text"] = _rebuild_text(document)
    elif op == "opts":
        document.setdefault("opts", {}).update(payload.get("opts") or {})
    elif op == "retext":
        new = (payload.get("text") or "").strip()
        for w in document["words"]:
            if w["id"] in set(ids) and new:
                w["text"] = new.split()[0]
                w["png"] = None
                w["aspect"] = None
        document["text"] = _rebuild_text(document)
        regen = True
    else:
        return {"error": f"unknown op {op!r}"}

    try:
        if op in reflow_ops:
            _sync_and_reflow(name, document, sheet, generate=regen)
        else:
            style, ckpt = _doc_style(document)
            if regen:
                _doc.ensure_words(document, str(WORDS), style=style, ckpt=ckpt)
            _doc.place_pins(document)
            _doc.save(str(_doc_path(name)), document)
    except Exception as exc:
        return {"error": str(exc)[:600]}
    return doc_view(document, sheet)


@app.get("/img/word/{fname}")
def img_word(fname: str):
    """A word as a real alpha image, for the browser to composite live.

    First endpoint in this project to serve an alpha channel -- everything else
    is opaque. ink_alpha() already computes exactly this mask; it was previously
    consumed in-process and thrown away.
    """
    from PIL import Image
    src = _safe_under(WORDS, fname)
    im = Image.open(src)
    buf = io.BytesIO()
    Image.merge("LA", (im.convert("L"), _render_page.ink_alpha(im))).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.post("/api/doc/render")
async def doc_render(payload: dict):
    name = payload.get("name")
    try:
        st = _paper_state(name)
    except Exception as exc:
        return {"error": str(exc)[:400]}
    document = _load_doc(name, st["sheet"])
    try:
        _sync_and_reflow(name, document, st["sheet"])
    except Exception as exc:
        return {"error": str(exc)[:600]}
    import cv2
    img = _paper.load_photo(str(_paper_path(name)))
    out, drawn = _paper.compose_doc(img, st["sheet"], document)
    PAPER.mkdir(parents=True, exist_ok=True)
    token = re.sub(r"[^A-Za-z0-9_-]", "", str(payload.get("token") or "run"))[:40]
    dest = PAPER / f"out_{token}.jpg"
    cv2.imwrite(str(dest), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"url": f"/img/paper/{dest.name}", "drawn": drawn,
            "overflow": sum(1 for w in document["words"] if w.get("overflow"))}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("HW_DASH_PORT", "8765"))
    print(f"\n  Handwriting dashboard →  http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
