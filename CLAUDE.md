# Project rules — Handwriting Replication

These rules are standing instructions for any Claude Code session working in this
repository. They apply automatically; no need to re-state them.

## Push to GitHub after every change

**Push to `origin main` every time a feature is added, a bug is fixed, or any
working change lands** — not batched at the end of a session. Each push should
be one coherent change with a commit message that says what changed and why,
in the same style as the code comments in this repo (explain the reasoning,
not just the diff).

Sequence for every change:
1. Make the change, verify it actually works (run it, don't just read it).
2. `git add` the specific files that changed — never a blind `git add -A`
   without checking `git status` first, since this repo intentionally excludes
   personal data and large binaries (see below).
3. Commit with a message explaining what changed and why.
4. Push immediately: `git push origin main`.

Do not wait for the user to ask "can you push that" — treat every completed
fix or feature as incomplete until it's pushed.

## What never gets committed

This repository is **public**. The working directory contains personal data
that must never reach it:

- `raw_scans/`, `pages/`, `segmented/`, `boxes/` — photos and crops of the
  user's real handwritten coursework, with his real name visible on some pages
- `labels.csv`, `labels_auto.csv`, `labels_review.csv` — transcriptions of
  that same personal handwriting
- `dataset/*.h5` — the packaged training set, which embeds the actual crop
  images
- `_archive/` — contains at least one photo with the user's real name and
  personal assignment details visible
- `checkpoints/*.pth`, `models/weights/*.pth` — 141MB–405MB each; GitHub
  hard-rejects anything over 100MB regardless of content
- `models/FW_GAN/` — third-party code with its own repo and MIT license;
  documented as a setup dependency, never vendored in
- `.env` — the GitHub token and any other credentials

All of the above are in `.gitignore`. Before ever changing that file, re-read
this section — the exclusions are deliberate privacy decisions, not oversights
to "clean up."

## Environment

The GitHub token lives in `.env` (`GITHUB_TOKEN`). Load it rather than asking
the user to paste it again:

    source .env && git remote set-url origin "https://${GITHUB_TOKEN}@github.com/blakeschuwerk/Handwriting-Replication.git"

Never print the token to chat, logs, or a committed file.
