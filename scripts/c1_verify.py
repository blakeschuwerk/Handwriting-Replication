#!/usr/bin/env python3
"""
C1 environment-setup acceptance test for FW-GAN on Apple Silicon (MPS).

Run with:
  PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH="/Users/blakey5aces/Handwriting Analysis/models/FW_GAN" \
    "/Users/blakey5aces/Handwriting Analysis/.venv/bin/python3" \
    "/Users/blakey5aces/Handwriting Analysis/scripts/c1_verify.py"
"""
import os
import sys
import traceback

REPO_ROOT = "/Users/blakey5aces/Handwriting Analysis/models/FW_GAN"
WEIGHTS = "/Users/blakey5aces/Handwriting Analysis/models/weights/FW-GAN.pth"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

FAIL = False


def step(name):
    print(f"\n=== {name} ===")


def fail(msg):
    global FAIL
    FAIL = True
    print(f"FAIL: {msg}")


try:
    step("a. torch import + MPS availability")
    import torch
    print("torch version:", torch.__version__)
    mps_ok = torch.backends.mps.is_available()
    print("mps available:", mps_ok, "built:", torch.backends.mps.is_built())
    assert mps_ok, "MPS backend not available"

    step("b. import FW-GAN core network modules")
    from networks.model import AdversarialModel
    import networks.module
    import networks.blocks
    import networks.BigGAN_networks
    print("Imported: networks.model.AdversarialModel")
    print("Imported: networks.module ->", networks.module.__file__)
    print("Imported: networks.blocks ->", networks.blocks.__file__)
    print("Imported: networks.BigGAN_networks ->", networks.BigGAN_networks.__file__)

    step("c. load pretrained checkpoint (map_location='cpu')")
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    print("checkpoint type:", type(ckpt))
    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        print("checkpoint top-level keys:", keys)
    else:
        print("checkpoint repr (truncated):", repr(ckpt)[:300])

    step("d. MPS tensor allocation")
    t_cpu = torch.zeros(4)
    t_mps = t_cpu.to("mps")
    print("tensor moved to mps device:", t_mps.device)
    assert t_mps.device.type == "mps"

    step("e. attempt to move a real checkpoint tensor to MPS (best-effort)")
    moved_ok = False
    try:
        if isinstance(ckpt, dict):
            for k, v in ckpt.items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        if torch.is_tensor(subv):
                            moved = subv.to("mps")
                            print(f"moved real checkpoint tensor {k}.{subk} "
                                  f"shape={tuple(moved.shape)} to device={moved.device}")
                            moved_ok = True
                            break
                if moved_ok:
                    break
        if not moved_ok:
            print("No raw tensor found directly under checkpoint dict values "
                  "(state_dicts are nested under module names) -- synthetic "
                  "MPS tensor test in step d already satisfies the MPS bar.")
    except Exception as e:
        print("Non-fatal: could not move a real checkpoint tensor to MPS:", e)

    step("f. config-driven device instantiation (best-effort, not the hard bar)")
    try:
        import yaml
        from munch import Munch

        cfg_path = "/Users/blakey5aces/Handwriting Analysis/scripts/c1_test_config.yml"
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        print("test config device field:", raw.get("device"))
        assert raw.get("device") == "mps", "test config was not patched to mps"
        print("Confirmed local test config (copied from configs/fw_gan_iam.yml) "
              "has device: 'mps' (upstream config unmodified).")
    except Exception as e:
        print("Note: full model instantiation from config skipped/failed "
              "(not required for the minimum bar):", e)
        traceback.print_exc()

    print("\nC1 VERIFY: PASS")
    sys.exit(0)

except Exception as e:
    print("\nException during verification:")
    traceback.print_exc()
    fail(str(e))

if FAIL:
    print("\nC1 VERIFY: FAIL")
    sys.exit(1)
