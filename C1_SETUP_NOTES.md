# C1 Environment Setup — Notes

## Environment manager: venv (not conda)
Chosen because:
- No conda installation dependency was assumed/required by the plan.
- venv + pip gives a simple, fully pip-freezable environment, which matches the
  deliverable requirement of a resolved, pinned `requirements.txt`.
- Location: `/Users/blakey5aces/Handwriting Analysis/.venv`

## Python version
- `/Users/blakey5aces/Handwriting Analysis/.venv/bin/python3` -> Python 3.11.9
  (this venv pre-existed from a related component in the same orchestration and
  was already built against `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11`).
- Explicitly avoided the system default `python3` (Homebrew, resolves to Python 3.14.6)
  because no torch wheels exist for 3.14 on PyPI at time of setup.

## torch / torchvision installation source
- Installed via plain `pip install torch torchvision` against the **default PyPI index**
  (no `--index-url` override). This resolved to:
  - `torch==2.13.0` (macOS arm64 wheel, `macosx_14_0_arm64`)
  - `torchvision==0.28.0` (macOS arm64 wheel, `macosx_14_0_arm64`)
- Deliberately did NOT use the upstream repo's CUDA index
  (`--index-url https://download.pytorch.org/whl/cu126`) — that index has no
  macOS/MPS wheels and would fail or silently give a CPU-only/incompatible build.
- Verified MPS backend:
  ```
  torch version: 2.13.0
  mps available: True built: True
  ```

## Config change: `device: 'cuda'` -> `device: 'mps'`
- Upstream configs (`configs/fw_gan_iam.yml`, `configs/fw_gan_vn.yml`) ship with
  `device: 'cuda'`.
- `networks/model.py` (`BaseModel.__init__`) does
  `self.device = torch.device(opt.device)` and all `.to(device)` calls are
  driven from that single field — no hardcoded `.cuda()` calls found anywhere
  in `networks/model.py`.
- For MPS testing, the upstream config was **copied**, not edited in place, to:
  `/Users/blakey5aces/Handwriting Analysis/scripts/c1_test_config.yml`
  with the single line changed to `device: 'mps'`. The original
  `models/FW_GAN/configs/fw_gan_iam.yml` is untouched (still says `cuda`),
  preserving upstream fidelity for anyone diffing against the pinned commit.

## `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Set as an environment variable when running the verification script (and
  should be set for all subsequent MPS runs of this repo, e.g. C2 smoke-test,
  C6 fine-tuning, C7 inference).
- Reason: MPS does not implement every ATen op that CUDA/CPU do. Setting this
  flag lets PyTorch silently fall back to CPU for any operation not yet
  implemented on MPS, rather than hard-crashing with
  `NotImplementedError: The operator 'aten::...' is not currently implemented
  for the MPS device`. FW-GAN uses fairly standard conv/BN/attention ops so
  most of the graph should run on MPS natively, but BigGAN-style spectral norm
  layers and any custom ops in `networks/BigGAN_networks.py` /
  `networks/BigGAN_layers.py` are unverified end-to-end at this stage (C1 only
  verifies import + checkpoint load + basic tensor placement, not a full
  forward/backward pass — that is C2's job).
- No specific op was observed requiring CPU fallback during C1's checks (only
  module import, `torch.load`, and a bare `torch.zeros(4).to('mps')` plus
  moving one real checkpoint tensor were exercised). A full-graph fallback
  audit is deferred to C2 (smoke-test), which is explicitly scoped for
  "MPS + FFT crash/NaN check before real data."

## Dependency deltas vs. upstream `requirements.txt`
- Upstream `requirements.txt` lists: tensorboard, munch, opencv-python, h5py,
  scikit-learn, numpy, matplotlib, PyYAML (all unpinned, and torch/torchvision
  are absent — installed separately per upstream's own README/CUDA-index
  instructions).
- **`timm` was required but is NOT listed in upstream `requirements.txt`.**
  `networks/BigGAN_networks.py` does `from timm.layers import DropPath`, which
  raised `ModuleNotFoundError: No module named 'timm'` on first import attempt.
  Installed `timm==1.0.28` (pulled in `safetensors==0.8.0` as a sub-dependency).
  This is a genuine upstream requirements-file omission, not a macOS/MPS-specific
  substitution — flagging it here so later components don't re-discover it as
  a fresh bug.
- `huggingface_hub` was added (not in upstream requirements.txt; not part of
  the model's runtime — only used by this setup script to fetch the pinned
  checkpoint by revision hash).
- All other packages (tensorboard, munch, opencv-python, h5py, scikit-learn,
  numpy, matplotlib, PyYAML) installed cleanly from default PyPI with no
  substitutions needed.
- Full resolved/pinned dependency graph: see
  `/Users/blakey5aces/Handwriting Analysis/requirements.txt` (`pip freeze`
  output, 50 packages).

## Deferred to later components
- HDF5 datasets (`train.hdf5`, `test.hdf5`, `*_vn.h5`, ~270MB total) were
  intentionally NOT downloaded as part of C1 — out of scope per the plan, and
  belong to dataset-packaging (C5) in this orchestration.
- Full model instantiation via `AdversarialModel(opt)` (constructing all
  submodules, losses, optimizers) and a real forward/backward pass on MPS
  were NOT exercised in C1 — the acceptance bar for C1 was import + checkpoint
  load + MPS tensor allocation, which all passed. End-to-end graph execution
  and any resulting CPU-fallback ops are C2's responsibility.
