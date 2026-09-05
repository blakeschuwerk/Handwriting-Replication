# C2: FW-GAN MPS Smoke Test — Notes

## How to run

```
cd "/Users/blakey5aces/Handwriting Analysis"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
PYTHONPATH="/Users/blakey5aces/Handwriting Analysis/models/FW_GAN" \
"/Users/blakey5aces/Handwriting Analysis/.venv/bin/python3" \
"/Users/blakey5aces/Handwriting Analysis/scripts/smoke_test.py"
```

Exit codes:
- `0` → PASS
- `2` → genuine MPS hard failure (op crashes with no CPU fallback)
- `1` → FAIL (NaN/Inf canary tripped, or unexpected exception)

## Config used

`scripts/c1_test_config.yml`, loaded via `yaml.safe_load` + `munchify`. Two
in-memory patches applied before model construction (config file itself is
NOT modified):

- `opt.training.lexicon` overwritten from the relative `./data/english_words.txt`
  to the absolute path `models/FW_GAN/data/english_words.txt` (constructor reads
  the lexicon immediately, before any chdir would help).
- `opt.training.batch_size` overwritten from `8` to `4` for a faster smoke test.

Device: `mps`. `img_height: 32`, `char_width: 16`, `n_class: 81`,
`GenModel.style_dim: 128`, `EncModel.style_dim: 96`, `WidModel.n_writer: 339`.

## Synthetic batch shape

Mirrors `lib/datasets.py::Hdf5Dataset.collect_fn` output exactly:
`(imgs, img_lens, lbs, lb_lens, wids)`.

- `imgs`: `(4, 1, 32, 96)` float32, values in `[-1, 1]` via `torch.rand(...)*2-1`.
- `img_lens`: `IntTensor([96,96,96,96])` — multiple of `img_height//2=16`.
- `lbs`: 4 six-letter words (`handle`, `models`, `tensor`, `object`) encoded via
  `label_converter.encode(list_of_words)` → `(4, 6)` int32.
- `lb_lens`: `IntTensor([6,6,6,6])`.
- `wids`: `LongTensor([0,1,2,3])`.
- CTC input length = `img_len // 8 = 12`, which is `>= lb_len = 6` ✓, and the
  Recognizer's actual output time dimension was verified as `T=12 == 12`, so no
  clamping was needed. No harness sizing adjustment required.

## Checkpoint

`model.load('/Users/blakey5aces/Handwriting Analysis/models/weights/FW-GAN.pth', map_location='cpu')`
succeeded cleanly on every run — all 7 submodules (G, D, HF_D, R, E, W, S)
loaded their `state_dict()` with no key/shape mismatch. `epoch` field in the
checkpoint was `83`. The model was then run on `mps` (construction already
moves every submodule to `opt.device` via `.to(device)`).

## Outcome: PASS

Ran twice for reproducibility (2 separate full processes), 3 D-step + G-step
iterations each time, exit code `0` both times. No NaN/Inf anywhere: not in
any of the 15 named losses, not in the 4 `torch.autograd.grad(..., create_graph=True)`
intermediate tensors (`grad_fake_adv`, `grad_fake_OCR`, `grad_fake_info`,
`grad_fake_wid`), and not in any optimizer parameter gradient (D-step: 205
grad tensors checked, total L2 norm 120–197; G-step: 113 grad tensors
checked, total L2 norm 373–987 across the two runs). Vanishing-gradient
canary (`total_norm > 0`) passed every iteration.

Representative loss values (run 2, iteration 2 / last):
```
real_ctc_loss=3.554553  real_wid_loss=12.975534  real_disc_loss=0.000000
fake_disc_loss=0.000000 hf_real_disc_loss=1.308537 hf_fake_disc_loss=0.381064
disc_loss=1.689601      adv_loss=5.527707        adv_loss_hf=1.161549
fake_ctc_loss=0.063185  info_loss=0.723382       fake_wid_loss=9.692046
fdl_loss=22.985458      kl_loss=909.467529       g_loss=55.903194
```

Note on `real_disc_loss` / `fake_disc_loss` reading `0.000000`: this is the
hinge loss (`relu(1 - real_disc)` / `relu(1 + fake_disc)`) saturating because
our "real" images are literally `torch.rand(...)*2-1` noise scored confidently
by the pretrained discriminator — not a CTC zero-infinity artifact (that
canary is checked separately and explicitly asserts `real_ctc_loss` /
`fake_ctc_loss` are non-zero, which they were, every iteration, both runs).

## CPU-fallback ops observed

Exactly one fallback warning appeared, on iteration 0 of every run:

```
UserWarning: The operator 'aten::_ctc_loss' is not currently supported on the
MPS backend and will fall back to run on the CPU. (Triggered internally at
.../ATen/mps/MPSFallback.mm:34.)
```

- **`CTCLoss` (both `real_ctc_loss` and `fake_ctc_loss`)**: falls back to CPU.
  Expect a slowdown proportional to how often CTC loss is computed during real
  training (every D-step and every G-step iteration where the critic runs).
  This is a **known PyTorch/MPS gap**, not a bug in this code — `aten::_ctc_loss`
  has no native MPS kernel as of torch 2.13.0.
- **No other fallback warnings were observed** across either run (3 iterations
  each, full D-step + G-step, including the double-backward `create_graph=True`
  calls).

### FFT path (the prime suspect per the task brief)

`FDL_loss.forward` (`networks/loss.py`) uses `F.interpolate(..., mode='bicubic')`,
`torch.fft.fftn`, `torch.angle`, and `torch.sort` — all in the G-step, and all
put through a double-backward via `g_loss.backward()` (since `fdl_loss` feeds
into `g_loss` and the graph is retained by the earlier `create_graph=True`
grad calls on other loss terms in the same graph).

**Result: all of these ran natively on MPS with no CPU fallback and no hard
error**, forward and backward, across all 3 iterations of both runs. No
fallback warning was printed for `fftn`, `angle`, `sort`, or bicubic
`interpolate` — only `aten::_ctc_loss` triggered a fallback warning. This is
a genuine finding: on this torch build (2.13.0), the FFT-based FDL loss is
NOT the MPS blocker that was anticipated; CTC loss is the only op that falls
back.

## Harness sizing adjustments

None needed. `img_len=96` with `ctc_len_scale=8` gave `real_ctc_lens=12`,
which matched the Recognizer's actual output time dimension `T=12` exactly
(verified explicitly in step 5 of the script before running the loop), and
comfortably covered `lb_len=6`. No clamping of `real_ctc_lens` was required.

## Approx runtime

- Run 1 (fresh MPS/Metal shader compilation cold-start): ~52.3s for construction
  + checkpoint load + 3 iterations.
  Second run showed cached iteration times: ~9.2s wall time for construction
  + checkpoint load + 3 iterations, since Metal shaders were likely already
  compiled by run 1.
- Practically: expect roughly 5–15s of one-time overhead the first time a
  fresh process runs this against the real training loop, plus per-iteration
  cost dominated by the CPU-fallback CTC loss and the large discriminator/
  generator forward+backward passes.

## Reviewer summary

PASS. Real `AdversarialModel` from `models/FW_GAN/networks/model.py`,
constructed from `scripts/c1_test_config.yml`, pretrained weights loaded
cleanly, ran 3 full D-step + G-step iterations (loss math copied verbatim
from `train()` lines 234–393, including the `create_graph=True` double-backward
grad-balance block) on device `mps`, twice, with zero NaN/Inf anywhere and
nonzero gradients throughout. Only CPU fallback observed: `aten::_ctc_loss`
(expected, well-known gap, not fatal — just slower). The FFT-heavy FDL loss,
the component most suspected of an MPS hard-failure, ran natively on MPS with
no fallback and no error in every iteration of every run.
