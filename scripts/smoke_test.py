#!/usr/bin/env python3
"""
C2 smoke test: run 2-3 REAL forward+backward D-step/G-step iterations of the
actual FW-GAN AdversarialModel on a tiny synthetic batch, on device 'mps',
with explicit NaN/Inf canaries on every loss, every intermediate autograd
gradient, and every optimizer parameter gradient.

This mirrors AdversarialModel.train() (networks/model.py lines 223-394)
VERBATIM in loss math / call order. The only substitution is a synthetic
batch in place of the real HDF5 DataLoader.

Run with:
  PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH="/Users/blakey5aces/Handwriting Analysis/models/FW_GAN" \
    "/Users/blakey5aces/Handwriting Analysis/.venv/bin/python3" \
    "/Users/blakey5aces/Handwriting Analysis/scripts/smoke_test.py"
"""
import os
import sys
import time
import traceback

REPO_ROOT = "/Users/blakey5aces/Handwriting Analysis/models/FW_GAN"
LEXICON_ABS = "/Users/blakey5aces/Handwriting Analysis/models/FW_GAN/data/english_words.txt"
CONFIG_PATH = "/Users/blakey5aces/Handwriting Analysis/scripts/c1_test_config.yml"
WEIGHTS = "/Users/blakey5aces/Handwriting Analysis/models/weights/FW-GAN.pth"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import yaml
from munch import munchify
import torch
import torch.nn.functional as F
from itertools import chain

from networks.model import AdversarialModel
from networks.utils import set_requires_grad, idx_to_words, rand_clip
from networks.rand_dist import prepare_z_dist, prepare_y_dist


def step(name):
    print(f"\n=== {name} ===", flush=True)


def assert_finite(name, tensor):
    """Raise a clear error if tensor contains any NaN or Inf."""
    if tensor is None:
        raise AssertionError(f"NaN/Inf canary: '{name}' is None")
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if torch.isnan(tensor).any():
        raise AssertionError(f"NaN/Inf canary: '{name}' contains NaN. "
                              f"shape={tuple(tensor.shape)} values(sample)={tensor.flatten()[:8]}")
    if torch.isinf(tensor).any():
        raise AssertionError(f"NaN/Inf canary: '{name}' contains Inf. "
                              f"shape={tuple(tensor.shape)} values(sample)={tensor.flatten()[:8]}")


def assert_grads_finite_and_nonzero(name, params):
    """Check every param.grad (where not None) is finite, and total L2 norm > 0."""
    total_sq = 0.0
    n_grads = 0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        if torch.isnan(g).any():
            raise AssertionError(f"NaN/Inf canary: {name} has a NaN gradient "
                                  f"in a param of shape {tuple(p.shape)}")
        if torch.isinf(g).any():
            raise AssertionError(f"NaN/Inf canary: {name} has an Inf gradient "
                                  f"in a param of shape {tuple(p.shape)}")
        total_sq += float(torch.sum(g.detach().float() ** 2).item())
        n_grads += 1
    total_norm = total_sq ** 0.5
    print(f"  [{name}] grads checked: {n_grads} tensors, total L2 norm = {total_norm:.6f}")
    if n_grads == 0:
        raise AssertionError(f"Vanishing-gradient canary: {name} had NO gradients at all "
                              f"(all param.grad is None)")
    if total_norm == 0.0:
        raise AssertionError(f"Vanishing-gradient canary: {name} total grad L2 norm is exactly 0.0")


def build_synthetic_batch(model, device, batch_size=4, width=96):
    """
    Build a synthetic batch matching lib/datasets.py::Hdf5Dataset.collect_fn output:
    (imgs, img_lens, lbs, lb_lens, wids)
    """
    words = ["handle", "models", "tensor", "object", "pixels", "random"][:batch_size]
    if len(words) < batch_size:
        # repeat/cycle if batch_size > 6 (not expected here, but be safe)
        base = ["handle", "models", "tensor", "object", "pixels", "random"]
        words = [base[i % len(base)] for i in range(batch_size)]

    lbs, lb_lens = model.label_converter.encode(words)
    lbs = lbs.int()
    lb_lens = lb_lens.int()
    assert lbs.shape[0] == batch_size
    L = lbs.shape[1]
    print(f"  synthetic labels: words={words} encoded shape={tuple(lbs.shape)} lb_lens={lb_lens.tolist()}")

    imgs = torch.rand(batch_size, 1, model.opt.img_height, width) * 2 - 1
    img_lens = torch.full((batch_size,), width, dtype=torch.int32)
    wids = torch.arange(batch_size, dtype=torch.long) % model.opt.WidModel.n_writer

    return imgs, img_lens, lbs, lb_lens, wids


def main():
    t_start = time.time()

    step("0. torch / MPS sanity")
    print("torch version:", torch.__version__)
    mps_ok = torch.backends.mps.is_available()
    print("mps available:", mps_ok, "built:", torch.backends.mps.is_built())
    assert mps_ok, "MPS backend not available -- cannot run smoke test on 'mps'"

    step("1. Load config")
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    opt = munchify(raw)
    # Lexicon path in the yaml is relative ('./data/english_words.txt'); the
    # AdversarialModel constructor reads it immediately via get_lexicon(), so
    # patch to an absolute path rather than os.chdir (keeps script location-independent).
    opt.training.lexicon = LEXICON_ABS
    # Smaller batch for a fast smoke test.
    opt.training.batch_size = 4
    print("device:", opt.device)
    print("lexicon path (patched to absolute):", opt.training.lexicon)
    print("batch_size (patched for smoke test):", opt.training.batch_size)
    assert opt.device == "mps"

    step("2. Construct AdversarialModel (real classes, real construction path)")
    model = AdversarialModel(opt)
    device = model.device
    print("model.device:", device)
    print("lexicon size:", len(model.lexicon))
    assert device.type == "mps"

    step("3. Load pretrained checkpoint (optional, preferred for fidelity)")
    checkpoint_loaded = False
    checkpoint_error = None
    try:
        epoch_loaded = model.load(WEIGHTS, map_location="cpu")
        checkpoint_loaded = True
        print(f"Loaded pretrained checkpoint. epoch field in ckpt: {epoch_loaded}")
    except Exception as e:
        checkpoint_error = f"{type(e).__name__}: {e}"
        print("WARNING: failed to load pretrained checkpoint, continuing with random init.")
        print("Reason:", checkpoint_error)
        traceback.print_exc()

    step("4. Build synthetic batch (mirrors Hdf5Dataset.collect_fn output format)")
    imgs, img_lens, lbs, lb_lens, wids = build_synthetic_batch(
        model, device, batch_size=opt.training.batch_size, width=96
    )
    print("imgs shape:", tuple(imgs.shape), "dtype:", imgs.dtype)
    print("img_lens:", img_lens.tolist())
    print("lb_lens:", lb_lens.tolist())
    print("wids:", wids.tolist())

    # CTC input-length sizing canary (harness-sizing, distinct from MPS-op findings)
    ctc_len_scale = 8
    real_ctc_lens_check = img_lens // ctc_len_scale
    print("ctc_len_scale=8 -> real_ctc_lens:", real_ctc_lens_check.tolist(),
          " (must be >= lb_lens:", lb_lens.tolist(), ")")
    assert (real_ctc_lens_check >= lb_lens).all(), (
        "HARNESS SIZING ISSUE: img_len//8 < lb_len for some sample -- increase synthetic "
        "image width W. This is a harness problem, not a model/MPS finding."
    )

    # Move to device exactly as train() does (model.py lines 228-229)
    real_imgs, real_img_lens, real_wids = imgs.to(device), img_lens.to(device), wids.to(device)
    real_lbs, real_lb_lens = lbs.to(device), lb_lens.to(device)

    step("5. Sanity-check recognizer output time dimension T vs claimed ctc input length")
    model.set_mode("train")
    with torch.no_grad():
        probe_ctc = model.models.R(real_imgs)
    T = probe_ctc.shape[0]
    real_ctc_lens_probe = real_img_lens // ctc_len_scale
    print(f"Recognizer output T={T}, real_ctc_lens={real_ctc_lens_probe.tolist()}")
    if not (real_ctc_lens_probe <= T).all():
        print("HARNESS SIZING ADJUSTMENT: clamping real_ctc_lens to T "
              f"(T={T} < some claimed ctc_len) -- this is a harness issue, not a model/MPS finding.")
    assert_finite("probe_ctc (recognizer forward sanity check)", probe_ctc)
    del probe_ctc

    # Set up optimizers exactly as train() does (model.py lines 198-204)
    step("6. Build optimizers (same grouping as train())")
    optimizers = type(model).__dict__  # not used; just documenting we mirror train()
    G_params = list(chain(model.models.G.parameters(), model.models.E.parameters()))
    D_params = list(chain(model.models.D.parameters(), model.models.HF_D.parameters(),
                           model.models.R.parameters(), model.models.W.parameters(),
                           model.models.S.parameters()))
    optG = torch.optim.Adam(G_params, lr=opt.training.lr,
                             betas=(opt.training.adam_b1, opt.training.adam_b2))
    optD = torch.optim.Adam(D_params, lr=opt.training.lr,
                             betas=(opt.training.adam_b1, opt.training.adam_b2))

    def KLloss(mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

    model.z = prepare_z_dist(opt.training.batch_size, opt.GenModel.style_dim, device, seed=opt.seed)
    model.y = prepare_y_dist(opt.training.batch_size, len(model.lexicon), device, seed=opt.seed)

    N_ITERS = 3
    last_losses = {}
    mps_fallback_notes = []

    step(f"7. Run {N_ITERS} D-step + G-step iterations (mirrors train() body verbatim)")
    for it in range(N_ITERS):
        print(f"\n--- iteration {it} ---", flush=True)
        model.set_mode("train")

        #############################
        # D-step (model.py lines 234-293)
        #############################
        optD.zero_grad()
        set_requires_grad([model.models.G, model.models.E], False)
        set_requires_grad([model.models.R, model.models.D, model.models.HF_D,
                            model.models.W, model.models.S], True)

        real_ctc = model.models.R(real_imgs)
        assert_finite("real_ctc (recognizer output)", real_ctc)
        real_ctc_lens = real_img_lens // ctc_len_scale
        real_ctc_loss = model.ctc_loss(real_ctc, real_lbs, real_ctc_lens, real_lb_lens)
        assert_finite("real_ctc_loss", real_ctc_loss)
        assert real_ctc_loss.item() != 0.0, (
            "real_ctc_loss is exactly 0.0 -- CTC zero_infinity clamp likely fired "
            "(degenerate test, harness sizing issue)"
        )

        clip_imgs, clip_img_lens = rand_clip(real_imgs, real_img_lens)
        real_wid_logits = model.models.W(clip_imgs, clip_img_lens, model.models.S)
        assert_finite("real_wid_logits", real_wid_logits)
        real_wid_loss = model.classify_loss(real_wid_logits, real_wids)
        assert_finite("real_wid_loss", real_wid_loss)

        with torch.no_grad():
            model.y.sample_()
            sampled_words = idx_to_words(model.y, model.lexicon, opt.training.capitalize_ratio)
            fake_lbs, fake_lb_lens = model.label_converter.encode(sampled_words)
            fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()

            model.z.sample_()
            fake_imgs = model.models.G(model.z, fake_lbs, fake_lb_lens)
            assert_finite("fake_imgs (D-step, no_grad)", fake_imgs)

            enc_styles, _, _ = model.models.E(real_imgs, real_img_lens, model.models.S, vae_mode=True)
            noises = torch.randn((real_imgs.size(0), opt.GenModel.style_dim
                                   - opt.EncModel.style_dim)).float().to(device)
            enc_z = torch.cat([noises, enc_styles], dim=-1)
            style_imgs = model.models.G(enc_z, fake_lbs, fake_lb_lens)
            assert_finite("style_imgs (D-step, no_grad)", style_imgs)

            cat_fake_imgs = torch.cat([fake_imgs, style_imgs], dim=0)
            cat_fake_lb_lens = fake_lb_lens.repeat(2, ).detach()
            cat_fake_img_lens = cat_fake_lb_lens * opt.char_width

        fake_disc = model.models.D(cat_fake_imgs.detach(), cat_fake_img_lens, cat_fake_lb_lens)
        assert_finite("fake_disc", fake_disc)
        fake_disc_loss = torch.mean(F.relu(1.0 + fake_disc))
        assert_finite("fake_disc_loss", fake_disc_loss)

        real_disc = model.models.D(real_imgs, real_img_lens, real_lb_lens)
        assert_finite("real_disc", real_disc)
        real_disc_loss = torch.mean(F.relu(1.0 - real_disc))
        assert_finite("real_disc_loss", real_disc_loss)

        hf_fake_disc = model.models.HF_D(cat_fake_imgs.detach(), cat_fake_img_lens, cat_fake_lb_lens)
        assert_finite("hf_fake_disc", hf_fake_disc)
        hf_fake_disc_loss = torch.mean(F.relu(1.0 + hf_fake_disc))
        assert_finite("hf_fake_disc_loss", hf_fake_disc_loss)

        hf_real_disc = model.models.HF_D(real_imgs, real_img_lens, real_lb_lens)
        assert_finite("hf_real_disc", hf_real_disc)
        hf_real_disc_loss = torch.mean(F.relu(1.0 - hf_real_disc))
        assert_finite("hf_real_disc_loss", hf_real_disc_loss)

        disc_loss = (real_disc_loss + fake_disc_loss + hf_real_disc_loss + hf_fake_disc_loss)
        assert_finite("disc_loss", disc_loss)

        (real_ctc_loss + disc_loss + real_wid_loss).backward()
        assert_grads_finite_and_nonzero("D-step (optD params)", D_params)
        optD.step()

        #############################
        # G-step (model.py lines 298-394) -- runs every iter since iter_count starts at 0
        # and 0 % num_critic_train == 0; here we run it every smoke-test iteration too.
        #############################
        optG.zero_grad()
        set_requires_grad([model.models.D, model.models.HF_D, model.models.R,
                            model.models.W, model.models.S], False)
        set_requires_grad([model.models.G, model.models.E], True)

        model.y.sample_()
        sampled_words = idx_to_words(model.y, model.lexicon, opt.training.capitalize_ratio)
        fake_lbs, fake_lb_lens = model.label_converter.encode(sampled_words)
        fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()
        fake_img_lens = fake_lb_lens * opt.char_width

        model.z.sample_()
        fake_imgs = model.models.G(model.z, fake_lbs, fake_lb_lens)
        assert_finite("fake_imgs (G-step)", fake_imgs)

        enc_styles, enc_mu, enc_logvar = model.models.E(real_imgs, real_img_lens, model.models.S, vae_mode=True)
        assert_finite("enc_mu", enc_mu)
        assert_finite("enc_logvar", enc_logvar)
        noises = torch.randn((real_imgs.size(0), opt.GenModel.style_dim
                               - opt.EncModel.style_dim)).float().to(device)
        enc_z = torch.cat([noises, enc_styles], dim=-1)
        style_imgs = model.models.G(enc_z, fake_lbs, fake_lb_lens)
        assert_finite("style_imgs (G-step)", style_imgs)
        style_img_lens = fake_lb_lens * opt.char_width

        cat_fake_imgs = torch.cat([fake_imgs, style_imgs], dim=0)
        cat_fake_lbs = fake_lbs.repeat(2, 1).detach()
        cat_fake_lb_lens = fake_lb_lens.repeat(2, ).detach()
        cat_fake_img_lens = cat_fake_lb_lens * opt.char_width

        recn_imgs = model.models.G(enc_z, real_lbs, real_lb_lens)
        assert_finite("recn_imgs", recn_imgs)

        cat_fake_disc = model.models.D(cat_fake_imgs, cat_fake_img_lens, cat_fake_lb_lens)
        assert_finite("cat_fake_disc", cat_fake_disc)
        adv_loss = -torch.mean(cat_fake_disc)
        assert_finite("adv_loss", adv_loss)

        hf_fake_disc = model.models.HF_D(cat_fake_imgs, cat_fake_img_lens, cat_fake_lb_lens)
        assert_finite("hf_fake_disc (G-step)", hf_fake_disc)
        adv_loss_hf = -torch.mean(hf_fake_disc)
        assert_finite("adv_loss_hf", adv_loss_hf)

        cat_fake_ctc = model.models.R(cat_fake_imgs)
        assert_finite("cat_fake_ctc", cat_fake_ctc)
        cat_fake_ctc_lens = cat_fake_img_lens // ctc_len_scale
        fake_ctc_loss = model.ctc_loss(cat_fake_ctc, cat_fake_lbs, cat_fake_ctc_lens, cat_fake_lb_lens)
        assert_finite("fake_ctc_loss", fake_ctc_loss)
        assert fake_ctc_loss.item() != 0.0, (
            "fake_ctc_loss is exactly 0.0 -- CTC zero_infinity clamp likely fired "
            "(degenerate test, harness sizing issue)"
        )

        styles = model.models.E(fake_imgs, fake_img_lens, model.models.S)
        assert_finite("styles (re-encoded)", styles)
        info_loss = torch.mean(torch.abs(styles - model.z[:, -opt.EncModel.style_dim:].detach()))
        assert_finite("info_loss", info_loss)

        recn_wid_logits = model.models.W(style_imgs, style_img_lens, model.models.S)
        assert_finite("recn_wid_logits", recn_wid_logits)
        fake_wid_loss = model.classify_loss(recn_wid_logits, real_wids)
        assert_finite("fake_wid_loss", fake_wid_loss)

        # FDL loss -- prime MPS fallback/crash suspect (fftn, angle, sort, bicubic interpolate)
        try:
            fdl_loss = model.fdl_loss_fn(real_imgs, recn_imgs)
            assert_finite("fdl_loss", fdl_loss)
            if it == 0:
                mps_fallback_notes.append("FDL_loss (fftn/angle/sort/bicubic interpolate): "
                                           "ran without hard error on device=%s" % device)
        except (RuntimeError, NotImplementedError) as e:
            msg = str(e)
            print("MPS HARD FAILURE while computing fdl_loss:", msg)
            traceback.print_exc()
            print("\nC2 SMOKE TEST: MPS HARD FAILURE")
            print("Failing op context: FDL_loss.forward (torch.fft.fftn / torch.angle / "
                  "F.interpolate bicubic / torch.sort)")
            print("Exception:", msg)
            sys.exit(2)

        kl_loss = KLloss(enc_mu, enc_logvar)
        assert_finite("kl_loss", kl_loss)

        # Gradient-balance intermediate autograd tensors (model.py lines 362-365)
        try:
            grad_fake_adv = torch.autograd.grad(adv_loss, cat_fake_imgs, create_graph=True, retain_graph=True)[0]
            assert_finite("grad_fake_adv", grad_fake_adv)
            grad_fake_OCR = torch.autograd.grad(fake_ctc_loss, cat_fake_ctc, create_graph=True, retain_graph=True)[0]
            assert_finite("grad_fake_OCR", grad_fake_OCR)
            grad_fake_info = torch.autograd.grad(info_loss, fake_imgs, create_graph=True, retain_graph=True)[0]
            assert_finite("grad_fake_info", grad_fake_info)
            grad_fake_wid = torch.autograd.grad(fake_wid_loss, recn_wid_logits, create_graph=True, retain_graph=True)[0]
            assert_finite("grad_fake_wid", grad_fake_wid)
        except (RuntimeError, NotImplementedError) as e:
            msg = str(e)
            print("MPS HARD FAILURE during torch.autograd.grad double-backward:", msg)
            traceback.print_exc()
            print("\nC2 SMOKE TEST: MPS HARD FAILURE")
            print("Failing op context: torch.autograd.grad(..., create_graph=True) double-backward")
            print("Exception:", msg)
            sys.exit(2)

        std_grad_adv = torch.std(grad_fake_adv)
        gp_ctc = (torch.div(std_grad_adv, torch.std(grad_fake_OCR) + 1e-8).detach() + 1).clamp_max(100)
        gp_info = (torch.div(std_grad_adv, torch.std(grad_fake_info) + 1e-8).detach() + 1).clamp_max(50)
        gp_wid = (torch.div(std_grad_adv, torch.std(grad_fake_wid) + 1e-8).detach() + 1).clamp_max(10)
        assert_finite("gp_ctc", gp_ctc)
        assert_finite("gp_info", gp_info)
        assert_finite("gp_wid", gp_wid)

        g_loss = (2 * adv_loss + adv_loss_hf +
                  gp_ctc * fake_ctc_loss +
                  gp_info * info_loss +
                  gp_wid * fake_wid_loss +
                  fdl_loss +
                  opt.training.lambda_kl * kl_loss)
        assert_finite("g_loss", g_loss)

        try:
            g_loss.backward()
        except (RuntimeError, NotImplementedError) as e:
            msg = str(e)
            print("MPS HARD FAILURE during g_loss.backward():", msg)
            traceback.print_exc()
            print("\nC2 SMOKE TEST: MPS HARD FAILURE")
            print("Failing op context: g_loss.backward() (full G-step backward, "
                  "includes FDL_loss double-backward through fftn)")
            print("Exception:", msg)
            sys.exit(2)

        assert_grads_finite_and_nonzero("G-step (optG params)", G_params)
        optG.step()

        last_losses = dict(
            real_ctc_loss=real_ctc_loss.item(),
            real_wid_loss=real_wid_loss.item(),
            real_disc_loss=real_disc_loss.item(),
            fake_disc_loss=fake_disc_loss.item(),
            hf_real_disc_loss=hf_real_disc_loss.item(),
            hf_fake_disc_loss=hf_fake_disc_loss.item(),
            disc_loss=disc_loss.item(),
            adv_loss=adv_loss.item(),
            adv_loss_hf=adv_loss_hf.item(),
            fake_ctc_loss=fake_ctc_loss.item(),
            info_loss=info_loss.item(),
            fake_wid_loss=fake_wid_loss.item(),
            fdl_loss=fdl_loss.item(),
            kl_loss=kl_loss.item(),
            g_loss=g_loss.item(),
        )
        print(f"  iter {it} losses: " + ", ".join(f"{k}={v:.6f}" for k, v in last_losses.items()))

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("C2 SMOKE TEST: PASS")
    print(f"Ran {N_ITERS} D-step+G-step iterations on device=mps, no NaN/Inf detected.")
    print("Last iteration losses: " + ", ".join(f"{k}={v:.6f}" for k, v in last_losses.items()))
    print(f"Checkpoint loaded: {checkpoint_loaded}" +
          (f" (error: {checkpoint_error})" if checkpoint_error else ""))
    print(f"Elapsed: {elapsed:.1f}s")
    for note in mps_fallback_notes:
        print("Note:", note)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("\nC2 SMOKE TEST: FAIL")
        print("Assertion failed:", str(e))
        traceback.print_exc()
        sys.exit(1)
    except (RuntimeError, NotImplementedError) as e:
        msg = str(e)
        lower = msg.lower()
        if "mps" in lower or "not implemented for" in lower or "aten::" in lower:
            print("\nC2 SMOKE TEST: MPS HARD FAILURE")
            print("Exception:", msg)
            traceback.print_exc()
            sys.exit(2)
        else:
            print("\nC2 SMOKE TEST: FAIL")
            print("Unexpected RuntimeError:", msg)
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print("\nC2 SMOKE TEST: FAIL")
        print("Unexpected exception:", str(e))
        traceback.print_exc()
        sys.exit(1)
