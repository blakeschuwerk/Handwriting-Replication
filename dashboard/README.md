# Handwriting Studio — dashboard

A local web app that runs the whole handwriting-synthesis pipeline from your browser.
No terminal commands, no dropping files into folders by hand.

## How to launch

**Easiest — double-click:**
Open the `dashboard` folder in Finder and double-click **`start.command`**.
Your browser opens at `http://127.0.0.1:8765`. (First time only: if macOS blocks it,
right-click → Open → Open.)

**Or one command in a terminal:**

```bash
"/Users/blakey5aces/Handwriting Analysis/.venv/bin/python3" "/Users/blakey5aces/Handwriting Analysis/dashboard/server.py"
```

To stop it: close the Terminal window, or press Ctrl-C.

## What each tab does

1. **Upload & Segment** — drag in photos/scans of your handwriting, then split them into word crops.
2. **Label** — type what each crop says (the ground truth). Includes a 10% QA re-check.
3. **Build Dataset** — packs your labeled crops into `dataset/dataset.h5` for training.
4. **Train** — fine-tune on your handwriting. Watch the live sample image; stop when it looks like your hand.
5. **Generate & Render** — type any text and render it in your handwriting, as words or a full page (PNG/PDF).

## The one thing to remember while training

On Apple Silicon, **closing the lid sleeps the Mac and pauses training** — even with
`caffeinate`. Leave the lid **open** and stay plugged in for the whole run. Display
sleep is fine; physical lid-close is not. (See `../C6_RUN_INSTRUCTIONS.md`.)

## Notes

- The dashboard is a thin wrapper — all the real work is still done by the pinned scripts
  in `../scripts/`. Nothing about the model pipeline changed.
- It listens only on `127.0.0.1` (your machine only), single user, no auth — by design.
- Change the port by setting `HW_DASH_PORT` before launching.
