# C6 — Overnight Fine-Tuning Run Instructions

`scripts/finetune.py` fine-tunes the pretrained FW-GAN checkpoint on
`dataset/dataset.h5`, saving a checkpoint **and** an independent rendered sample
image every N steps so you can catch silent GAN failures (mode collapse, style
non-transfer) that loss curves alone would hide.

---

## ⚠️ READ FIRST: the current dataset is PLACEHOLDER-labeled

`dataset/dataset.h5` was packaged (C5) from segmented crops with
**deterministic placeholder dictionary words**, NOT real transcriptions of your
handwriting. Fine-tuning on it **proves the training loop / checkpointing /
sampling / resume plumbing works** — it does **NOT** teach the model your
personal handwriting.

**To do a real personalization run, first regenerate the dataset from real data:**

1. Put real scans of your handwriting in `raw_scans/`.
2. `./.venv/bin/python3 scripts/segment.py ...` → word crops in `segmented/`.
3. `./.venv/bin/python3 scripts/label_tool.py` → type the **true** text for each
   crop → `labels.csv` (use `--qa` to spot-check).
4. `./.venv/bin/python3 scripts/package_dataset.py` → rebuild `dataset/dataset.h5`.
5. `./.venv/bin/python3 scripts/validate_dataset.py` → confirm it loads clean.

Only then does an overnight `finetune.py` run produce handwriting that resembles
yours.

> Note: real IAM-style training data is width-normalized so each word image's
> width equals `label_len * char_width`. The placeholder crops are not, so
> `finetune.py` bilinearly resizes the reconstruction to the real image width
> before the FDL loss (documented in-code). A properly width-normalized real
> dataset makes that resize a no-op.

---

## The overnight command

```bash
cd "/Users/blakey5aces/Handwriting Analysis"
caffeinate -i -s ./.venv/bin/python3 scripts/finetune.py \
    --epochs 300 \
    --batch-size 8 \
    --lr 1e-4 \
    --sample-every 200 \
    --ckpt-every 200 \
    --keep 5
```

- **`caffeinate -i -s`** — `-i` prevents idle sleep, `-s` prevents system sleep
  on AC power. The assertion lasts **exactly** as long as the python process.
- **`--lr 1e-4`** (default) is deliberately **lower** than the repo's `2e-4` to
  reduce catastrophic forgetting of the pretrained general-handwriting knowledge
  when fine-tuning on a small single-writer set. **Watch the samples** — if they
  degrade into garbage/noise, stop and lower the LR further.
- Resume an interrupted run with `--resume` (restores weights, both optimizer
  states, epoch, and global step from `checkpoints/latest.pth`).

### ‼️ caffeinate does NOT defeat physically closing the lid

On Apple Silicon, **closing the lid (clamshell sleep) still suspends the
machine** regardless of `caffeinate`. **Run with the LID OPEN.** The display may
sleep/dim — that's fine — but do not close the laptop.

### Disable auto-restart for macOS updates during the run window

A macOS auto-update reboot mid-run kills training. Disable it:

1.  Apple menu → **System Settings**.
2.  **General → Software Update**.
3.  Click the ⓘ next to **Automatic Updates**.
4.  Turn **OFF** "Install macOS updates" (and "Install Security Responses and
    system files" if you want to be extra safe) for the run window.
5.  Also make sure no separate Software Update download is already staged and
    waiting to install on restart.

`finetune.py` does a pre-flight free-disk check and warns if space is tight.

---

## Outputs

| Path | What |
|------|------|
| `checkpoints/latest.pth` | Atomically-overwritten resume anchor (weights + both optimizer states + epoch + step). |
| `checkpoints/step_{N}.pth` | Last `--keep` rotating snapshots (older pruned) to bound disk use (~0.5 GB each). |
| `samples/step_{N}.png` | Fixed-seed / fixed-style / fixed-text sample rendered from the CURRENT in-memory weights (never re-read from disk) — line up these PNGs in order to watch progression. Never pruned. |
| `train_log.txt` | Appended plaintext step/loss log + every sample/checkpoint path. |

Each checkpoint is a resumable snapshot; use any `step_{N}.pth` with
`scripts/generate.py --ckpt checkpoints/step_{N}.pth` to generate with that
snapshot's weights.

---

## Observed timing (this M4 MacBook Air, MPS)

Measured on a short verification run (batch 8, 43 placeholder samples):

- **~10 s / training step** (each step = one D-step + one G-step;
  `aten::_ctc_loss` falls back to CPU, which is the main tax — the FFT-based FDL
  loss runs natively on MPS).
- **~25 s / epoch** (5 steps/epoch at batch 8 with `drop_last=True`).

So a 300-epoch overnight run (~1500 steps) is on the order of **~4 hours**. Scale
`--epochs` to the wall-clock you have. A real width-normalized dataset changes
per-step cost somewhat (different image widths), so re-check timing on the first
few logged steps of a real run.

> ⚠️ A short verification run proves the loop is correct; it does **not** prove
> multi-hour stability. Watch the first hour of any real overnight run before
> trusting it unattended.
