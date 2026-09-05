#!/usr/bin/env python3
"""
C6 fine-tuning driver for FW-GAN handwriting synthesis.

This is a THIN driver around the repo's REAL AdversarialModel. It does NOT call
the shipped model.train() (which is structurally incompatible with our
dataset/dataset.h5 -- it builds its DataLoader through lib/path_config.py which
does not know about our file -- and only checkpoints once per epoch, skipping
epoch 0, with no step-based sampling). Instead it:

  1. Constructs the real AdversarialModel from scripts/c1_test_config.yml.
  2. Loads the pretrained FW-GAN.pth via model.load(map_location='cpu').
  3. Builds DataLoader(Hdf5Dataset('dataset','dataset.h5', ...),
     collate_fn=Hdf5Dataset.collect_fn, num_workers=0) -- num_workers=0 because
     MPS + fork workers on macOS is a known hang/crash risk.
  4. Runs the VERBATIM D-step / G-step loss body copied from scripts/smoke_test.py
     (which mirrors networks/model.py train() lines 234-394 and was reviewer-
     verified on MPS in C2). No loss math is reinvented.
  5. Wraps that in our own epoch/step loop with:
       - step-based dual logging: every --ckpt-every steps save a checkpoint AND
         every --sample-every steps render a sample image from CURRENT in-memory
         weights (fixed seed + fixed style ref + fixed text) so visual progression
         is attributable to training, not noise variance;
       - bounded disk use: one atomically-overwritten latest.pth resume anchor
         plus the last --keep rotating step_{N}.pth files (older pruned);
       - resume support (--resume) restoring module weights, both optimizer
         states, epoch and global step;
       - a plaintext train_log.txt of step/loss.

DATA PROVENANCE: dataset/dataset.h5 now holds REAL transcriptions -- 1,975 word
crops cut from the user's own 18 scans, every one either detected-and-unobjected
or explicitly confirmed by hand in the box editor, all under a single writer id.
No other handwriting corpus is mixed in.

What IS foreign is the STARTING POINT: models/weights/FW-GAN.pth is the authors'
pretrained generator, trained on public handwriting. Fine-tuning adapts those
weights toward this writer, which is why the very first sample at step 0 already
looks like clean handwriting -- that is the pretrained model, not the user's
hand. Progress means the samples drifting toward the user's style from there.

Overnight invocation (see C6_RUN_INSTRUCTIONS.md):
  caffeinate -i -s ./.venv/bin/python3 scripts/finetune.py --epochs 40 ...
caffeinate does NOT defeat physical lid-close/clamshell sleep on Apple Silicon;
run with the LID OPEN (display sleep is fine).
"""
import argparse
import os
import shutil
import sys
import time
import traceback
from itertools import chain

# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT = "/Users/blakey5aces/Handwriting Analysis"
REPO_ROOT = os.path.join(PROJECT, "models", "FW_GAN")
LEXICON_ABS = os.path.join(REPO_ROOT, "data", "english_words.txt")
CONFIG_PATH = os.path.join(PROJECT, "scripts", "c1_test_config.yml")
WEIGHTS = os.path.join(PROJECT, "models", "weights", "FW-GAN.pth")
DEFAULT_STYLE_REF = os.path.join(PROJECT, "segmented", "sample_synth_L01_W04.png")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from munch import munchify
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor

from lib.datasets import Hdf5Dataset
from networks.model import AdversarialModel
from networks.rand_dist import prepare_y_dist, prepare_z_dist
from networks.utils import idx_to_words, rand_clip, set_requires_grad


# --------------------------------------------------------------------------- #
# Canaries (copied from smoke_test.py so training shares the same guards)
# --------------------------------------------------------------------------- #
def assert_finite(name, tensor):
    if tensor is None:
        raise AssertionError(f"NaN/Inf canary: '{name}' is None")
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if torch.isnan(tensor).any():
        raise AssertionError(f"NaN/Inf canary: '{name}' contains NaN. shape={tuple(tensor.shape)}")
    if torch.isinf(tensor).any():
        raise AssertionError(f"NaN/Inf canary: '{name}' contains Inf. shape={tuple(tensor.shape)}")


def grad_norm(params):
    total_sq = 0.0
    n = 0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        if torch.isnan(g).any() or torch.isinf(g).any():
            raise AssertionError("NaN/Inf gradient detected")
        total_sq += float(torch.sum(g.detach().float() ** 2).item())
        n += 1
    if n == 0:
        raise AssertionError("Vanishing-gradient canary: no gradients at all")
    return total_sq ** 0.5


def KLloss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)


# --------------------------------------------------------------------------- #
# Checkpointing (bounded disk use, atomic writes)
# --------------------------------------------------------------------------- #
def build_ckpt(model, optG, optD, epoch, global_step):
    ckpt = {}
    for m in model.models.values():
        ckpt[type(m).__name__] = m.state_dict()
    ckpt["optG"] = optG.state_dict()
    ckpt["optD"] = optD.state_dict()
    ckpt["Epoch"] = epoch
    ckpt["global_step"] = global_step
    return ckpt


def atomic_save(ckpt, path):
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_resume(model, optG, optD, path, device):
    print(f"RESUME: loading {path}", flush=True)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for m in model.models.values():
        m.load_state_dict(ckpt[type(m).__name__])
        m.to(device)
    optG.load_state_dict(ckpt["optG"])
    optD.load_state_dict(ckpt["optD"])
    return int(ckpt.get("Epoch", 0)), int(ckpt.get("global_step", 0))


def prune_rotating(ckpt_dir, keep):
    steps = []
    for fn in os.listdir(ckpt_dir):
        if fn.startswith("step_") and fn.endswith(".pth"):
            try:
                steps.append((int(fn[len("step_"):-len(".pth")]), fn))
            except ValueError:
                pass
    steps.sort()
    for _, fn in steps[:-keep] if keep > 0 else steps:
        try:
            os.remove(os.path.join(ckpt_dir, fn))
            print(f"  pruned old checkpoint {fn}", flush=True)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Sampling (mirrors generate.py, reads CURRENT in-memory weights)
# --------------------------------------------------------------------------- #
def encode_style_ref(model, style_path, img_height, len_scale, device):
    img = Image.open(style_path).convert("L")
    w, h = img.size
    new_w = max(1, round(w * img_height / h))
    img = img.resize((new_w, img_height))
    if new_w < len_scale:
        raise ValueError(f"style ref '{style_path}' too narrow (<{len_scale}px after resize)")
    tf = transforms.Compose([ToTensor(), Normalize([0.5], [0.5])])
    norm = tf(img)
    import math
    pad_w = math.ceil(new_w / len_scale) * len_scale
    padded = torch.full((1, 1, img_height, pad_w), -1.0)
    padded[..., :new_w] = norm.unsqueeze(0)
    padded = padded.to(device)
    with torch.no_grad():
        mu = model.models.E(padded, torch.tensor([pad_w], dtype=torch.int, device=device), model.models.S)
    return mu


def render_sample(model, words, fixed_noises, style_mu, char_width, out_path, device, log):
    """Render a fixed-text sample from current in-memory weights. Returns (min,max,mean)."""
    was_training = model.models.G.training
    model.set_mode("eval")
    pieces = []
    sep = np.full((32, 16), 255, dtype=np.uint8)
    try:
        with torch.no_grad():
            for idx, word in enumerate(words):
                enc = model.label_converter.encode(word)
                word_lbs = torch.LongTensor(enc).unsqueeze(0).to(device)
                word_lb_lens = torch.IntTensor([len(word)]).to(device)
                enc_z = torch.cat([fixed_noises[idx], style_mu], dim=1)
                fake = model.models.G(enc_z, word_lbs, word_lb_lens)
                fake = fake[:, :, :, : len(word) * char_width]
                arr = fake[0, 0]
                arr = (255 * ((arr + 1) / 2)).clamp(0, 255).to(torch.uint8).cpu().numpy()
                if idx > 0:
                    pieces.append(sep)
                pieces.append(arr)
    finally:
        if was_training:
            model.set_mode("train")
    line = np.concatenate(pieces, axis=1)
    Image.fromarray(line, mode="L").save(out_path)
    mn, mx, mean = int(line.min()), int(line.max()), float(line.mean())
    tag = ""
    if mx - mn < 5:
        tag = "  *** PIXEL CANARY: near-constant output (possible blank/degenerate) ***"
    log(f"  sample -> {out_path}  pixel min={mn} max={mx} mean={mean:.1f}{tag}")
    return mn, mx, mean


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune FW-GAN on dataset/dataset.h5 (step-based dual logging).")
    p.add_argument("--epochs", type=int, default=40, help="Number of epochs.")
    p.add_argument("--max-steps", type=int, default=0, help="Stop after this many global steps (0 = no cap).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam LR (default 1e-4, LOWER than repo 2e-4 to reduce catastrophic forgetting).")
    p.add_argument("--sample-every", type=int, default=25, help="Render a sample every N global steps.")
    p.add_argument("--ckpt-every", type=int, default=25, help="Save a checkpoint every N global steps.")
    p.add_argument("--keep", type=int, default=3, help="Number of rotating step_{N}.pth files to retain.")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoints/latest.pth if present.")
    p.add_argument("--device", default="mps")
    p.add_argument("--style-ref", default=DEFAULT_STYLE_REF)
    p.add_argument("--sample-text", default="the quick brown fox")
    p.add_argument("--ckpt-dir", default=os.path.join(PROJECT, "checkpoints"))
    p.add_argument("--sample-dir", default=os.path.join(PROJECT, "samples"))
    p.add_argument("--log-file", default=os.path.join(PROJECT, "train_log.txt"))
    p.add_argument("--seed", type=int, default=123456)
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    logf = open(args.log_file, "a", buffering=1)

    def log(msg):
        line = msg if msg.startswith(" ") else msg
        print(line, flush=True)
        logf.write(line + "\n")

    log("=" * 70)
    log(f"C6 finetune start {time.strftime('%Y-%m-%d %H:%M:%S')}  args={vars(args)}")

    # --- disk preflight ---
    free = shutil.disk_usage(args.ckpt_dir).free
    free_gb = free / 1e9
    est_gb = (args.keep + 2) * 0.5  # each resumable ckpt ~= weights + Adam state
    log(f"Disk free: {free_gb:.1f} GB; estimated max checkpoint footprint ~{est_gb:.1f} GB "
        f"(latest.pth + {args.keep} rotating).")
    if free_gb < est_gb + 2:
        log(f"  WARNING: low free disk ({free_gb:.1f} GB) relative to checkpoint footprint.")

    # --- torch / MPS sanity ---
    log(f"torch {torch.__version__}  mps_available={torch.backends.mps.is_available()}")

    # --- config ---
    with open(CONFIG_PATH) as f:
        opt = munchify(yaml.safe_load(f))
    opt.device = args.device
    opt.training.lexicon = LEXICON_ABS
    opt.training.batch_size = args.batch_size
    opt.training.lr = args.lr

    # --- model ---
    model = AdversarialModel(opt)
    device = model.device
    log(f"Constructed AdversarialModel on {device}; lexicon size {len(model.lexicon)}")

    # --- dataset / loader with batch-size safety (H2) ---
    ds = Hdf5Dataset(
        root=os.path.join(PROJECT, "dataset"),
        split="dataset.h5",
        transforms=Compose([ToTensor(), Normalize([0.5], [0.5])]),
        alphabet_key="all",
    )
    n = len(ds)
    log(f"Dataset dataset.h5 loaded: {n} samples")
    if n < 1:
        log("FATAL: empty dataset."); sys.exit(1)
    batch_size = args.batch_size
    if n < batch_size:
        log(f"  batch_size {batch_size} > dataset size {n}; capping batch_size to {n}")
        batch_size = n
    train_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, collate_fn=Hdf5Dataset.collect_fn,
    )
    steps_per_epoch = len(train_loader)
    log(f"batch_size={batch_size}  steps/epoch={steps_per_epoch} (drop_last=True)")
    if steps_per_epoch == 0:
        log("FATAL: 0 batches per epoch (drop_last dropped everything). Reduce batch size.")
        sys.exit(1)

    # --- optimizers (same grouping as model.py 198-204) ---
    G_params = list(chain(model.models.G.parameters(), model.models.E.parameters()))
    D_params = list(chain(model.models.D.parameters(), model.models.HF_D.parameters(),
                          model.models.R.parameters(), model.models.W.parameters(),
                          model.models.S.parameters()))
    optG = torch.optim.Adam(G_params, lr=opt.training.lr, betas=(opt.training.adam_b1, opt.training.adam_b2))
    optD = torch.optim.Adam(D_params, lr=opt.training.lr, betas=(opt.training.adam_b1, opt.training.adam_b2))

    # --- init weights: resume or pretrained ---
    start_epoch, global_step = 0, 0
    latest_path = os.path.join(args.ckpt_dir, "latest.pth")
    if args.resume and os.path.isfile(latest_path):
        start_epoch, global_step = load_resume(model, optG, optD, latest_path, device)
        log(f"Resumed at epoch {start_epoch}, global_step {global_step}")
    else:
        ep = model.load(WEIGHTS, map_location="cpu")
        log(f"Loaded pretrained FW-GAN.pth (epoch field {ep})")

    # --- sampling distributions ---
    model.z = prepare_z_dist(batch_size, opt.GenModel.style_dim, device, seed=opt.seed)
    model.y = prepare_y_dist(batch_size, len(model.lexicon), device, seed=opt.seed)
    ctc_len_scale = 8

    # --- fixed sampling setup (H4): fixed noise + fixed style ref + fixed text ---
    img_height = opt.img_height
    char_width = opt.char_width
    noise_dim = opt.GenModel.style_dim - opt.EncModel.style_dim
    len_scale = img_height // 2
    sample_words = [w for w in args.sample_text.split() if w]
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    fixed_noises = [torch.randn((1, noise_dim), generator=g).to(device) for _ in sample_words]
    log(f"Sample text={sample_words!r}  style_ref={args.style_ref}")

    def do_sample():
        style_mu = encode_style_ref(model, args.style_ref, img_height, len_scale, device)
        out_path = os.path.join(args.sample_dir, f"step_{global_step:06d}.png")
        render_sample(model, sample_words, fixed_noises, style_mu, char_width, out_path, device, log)

    def do_ckpt():
        ckpt = build_ckpt(model, optG, optD, epoch, global_step)
        atomic_save(ckpt, latest_path)
        step_path = os.path.join(args.ckpt_dir, f"step_{global_step:06d}.pth")
        atomic_save(ckpt, step_path)
        prune_rotating(args.ckpt_dir, args.keep)
        log(f"  checkpoint -> {step_path} (+ latest.pth)")

    # --- initial sample from the pretrained/resumed weights (baseline for progression) ---
    epoch = start_epoch
    log(f"--- initial baseline sample+ckpt at global_step {global_step} ---")
    do_sample()
    do_ckpt()

    # --------------------------------------------------------------------- #
    # Training loop (verbatim D-step / G-step body from smoke_test.py)
    # --------------------------------------------------------------------- #
    stop = False
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        for imgs, img_lens, lbs, lb_lens, wids in train_loader:
            model.set_mode("train")
            real_imgs, real_img_lens, real_wids = imgs.to(device), img_lens.to(device), wids.to(device)
            real_lbs, real_lb_lens = lbs.to(device), lb_lens.to(device)

            #########################  D-step  #########################
            optD.zero_grad()
            set_requires_grad([model.models.G, model.models.E], False)
            set_requires_grad([model.models.R, model.models.D, model.models.HF_D,
                               model.models.W, model.models.S], True)

            real_ctc = model.models.R(real_imgs)
            real_ctc_lens = real_img_lens // ctc_len_scale
            real_ctc_loss = model.ctc_loss(real_ctc, real_lbs, real_ctc_lens, real_lb_lens)
            assert_finite("real_ctc_loss", real_ctc_loss)

            clip_imgs, clip_img_lens = rand_clip(real_imgs, real_img_lens)
            real_wid_logits = model.models.W(clip_imgs, clip_img_lens, model.models.S)
            real_wid_loss = model.classify_loss(real_wid_logits, real_wids)
            assert_finite("real_wid_loss", real_wid_loss)

            with torch.no_grad():
                model.y.sample_()
                sampled_words = idx_to_words(model.y, model.lexicon, opt.training.capitalize_ratio)
                fake_lbs, fake_lb_lens = model.label_converter.encode(sampled_words)
                fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()
                model.z.sample_()
                fake_imgs = model.models.G(model.z, fake_lbs, fake_lb_lens)
                enc_styles, _, _ = model.models.E(real_imgs, real_img_lens, model.models.S, vae_mode=True)
                noises = torch.randn((real_imgs.size(0), opt.GenModel.style_dim
                                      - opt.EncModel.style_dim)).float().to(device)
                enc_z = torch.cat([noises, enc_styles], dim=-1)
                style_imgs = model.models.G(enc_z, fake_lbs, fake_lb_lens)
                cat_fake_imgs = torch.cat([fake_imgs, style_imgs], dim=0)
                cat_fake_lb_lens = fake_lb_lens.repeat(2, ).detach()
                cat_fake_img_lens = cat_fake_lb_lens * opt.char_width

            fake_disc = model.models.D(cat_fake_imgs.detach(), cat_fake_img_lens, cat_fake_lb_lens)
            fake_disc_loss = torch.mean(F.relu(1.0 + fake_disc))
            real_disc = model.models.D(real_imgs, real_img_lens, real_lb_lens)
            real_disc_loss = torch.mean(F.relu(1.0 - real_disc))
            hf_fake_disc = model.models.HF_D(cat_fake_imgs.detach(), cat_fake_img_lens, cat_fake_lb_lens)
            hf_fake_disc_loss = torch.mean(F.relu(1.0 + hf_fake_disc))
            hf_real_disc = model.models.HF_D(real_imgs, real_img_lens, real_lb_lens)
            hf_real_disc_loss = torch.mean(F.relu(1.0 - hf_real_disc))
            disc_loss = (real_disc_loss + fake_disc_loss + hf_real_disc_loss + hf_fake_disc_loss)
            assert_finite("disc_loss", disc_loss)

            (real_ctc_loss + disc_loss + real_wid_loss).backward()
            dnorm = grad_norm(D_params)
            optD.step()

            #########################  G-step  #########################
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
            enc_styles, enc_mu, enc_logvar = model.models.E(real_imgs, real_img_lens, model.models.S, vae_mode=True)
            noises = torch.randn((real_imgs.size(0), opt.GenModel.style_dim
                                  - opt.EncModel.style_dim)).float().to(device)
            enc_z = torch.cat([noises, enc_styles], dim=-1)
            style_imgs = model.models.G(enc_z, fake_lbs, fake_lb_lens)
            style_img_lens = fake_lb_lens * opt.char_width

            cat_fake_imgs = torch.cat([fake_imgs, style_imgs], dim=0)
            cat_fake_lbs = fake_lbs.repeat(2, 1).detach()
            cat_fake_lb_lens = fake_lb_lens.repeat(2, ).detach()
            cat_fake_img_lens = cat_fake_lb_lens * opt.char_width

            recn_imgs = model.models.G(enc_z, real_lbs, real_lb_lens)

            cat_fake_disc = model.models.D(cat_fake_imgs, cat_fake_img_lens, cat_fake_lb_lens)
            adv_loss = -torch.mean(cat_fake_disc)
            hf_fake_disc = model.models.HF_D(cat_fake_imgs, cat_fake_img_lens, cat_fake_lb_lens)
            adv_loss_hf = -torch.mean(hf_fake_disc)

            cat_fake_ctc = model.models.R(cat_fake_imgs)
            cat_fake_ctc_lens = cat_fake_img_lens // ctc_len_scale
            fake_ctc_loss = model.ctc_loss(cat_fake_ctc, cat_fake_lbs, cat_fake_ctc_lens, cat_fake_lb_lens)

            styles = model.models.E(fake_imgs, fake_img_lens, model.models.S)
            info_loss = torch.mean(torch.abs(styles - model.z[:, -opt.EncModel.style_dim:].detach()))

            recn_wid_logits = model.models.W(style_imgs, style_img_lens, model.models.S)
            fake_wid_loss = model.classify_loss(recn_wid_logits, real_wids)

            # FDL compares real vs its reconstruction and requires matching
            # spatial width. The repo's IAM dataset width-normalizes each word
            # image to label_len*char_width so recn_imgs (width Lmax*char_width)
            # matches by construction. Our C5 PLACEHOLDER dataset kept each
            # crop's natural aspect width, so widths differ -> resize recn_imgs
            # (which feeds ONLY FDL) to real width. This deviation exists solely
            # because the placeholder crops are not width-normalized; a real
            # width-normalized personalization dataset would not need it.
            if recn_imgs.shape[-1] != real_imgs.shape[-1]:
                recn_for_fdl = F.interpolate(
                    recn_imgs, size=(real_imgs.shape[-2], real_imgs.shape[-1]),
                    mode="bilinear", align_corners=False)
            else:
                recn_for_fdl = recn_imgs
            fdl_loss = model.fdl_loss_fn(real_imgs, recn_for_fdl)
            kl_loss = KLloss(enc_mu, enc_logvar)

            grad_fake_adv = torch.autograd.grad(adv_loss, cat_fake_imgs, create_graph=True, retain_graph=True)[0]
            grad_fake_OCR = torch.autograd.grad(fake_ctc_loss, cat_fake_ctc, create_graph=True, retain_graph=True)[0]
            grad_fake_info = torch.autograd.grad(info_loss, fake_imgs, create_graph=True, retain_graph=True)[0]
            grad_fake_wid = torch.autograd.grad(fake_wid_loss, recn_wid_logits, create_graph=True, retain_graph=True)[0]

            std_grad_adv = torch.std(grad_fake_adv)
            gp_ctc = (torch.div(std_grad_adv, torch.std(grad_fake_OCR) + 1e-8).detach() + 1).clamp_max(100)
            gp_info = (torch.div(std_grad_adv, torch.std(grad_fake_info) + 1e-8).detach() + 1).clamp_max(50)
            gp_wid = (torch.div(std_grad_adv, torch.std(grad_fake_wid) + 1e-8).detach() + 1).clamp_max(10)

            g_loss = (2 * adv_loss + adv_loss_hf +
                      gp_ctc * fake_ctc_loss +
                      gp_info * info_loss +
                      gp_wid * fake_wid_loss +
                      fdl_loss +
                      opt.training.lambda_kl * kl_loss)
            assert_finite("g_loss", g_loss)
            g_loss.backward()
            gnorm = grad_norm(G_params)
            optG.step()

            global_step += 1

            if global_step % opt.training.print_iter_val == 0 or global_step == 1:
                log(f"[e{epoch} s{global_step}] "
                    f"d_loss={disc_loss.item():.3f} g_loss={g_loss.item():.3f} "
                    f"real_ctc={real_ctc_loss.item():.3f} fake_ctc={fake_ctc_loss.item():.3f} "
                    f"fdl={fdl_loss.item():.3f} adv={adv_loss.item():.3f} "
                    f"|gD|={dnorm:.1f} |gG|={gnorm:.1f}")

            if global_step % args.sample_every == 0:
                do_sample()
            if global_step % args.ckpt_every == 0:
                do_ckpt()

            if args.max_steps and global_step >= args.max_steps:
                log(f"Reached max-steps {args.max_steps}; stopping.")
                stop = True
                break

    # --- final sample + ckpt ---
    log(f"--- final sample+ckpt at global_step {global_step} ---")
    do_sample()
    do_ckpt()

    elapsed = time.time() - t_start
    epochs_done = max(1, (epoch - start_epoch + 1))
    log(f"DONE. global_step={global_step}, wall={elapsed:.1f}s, "
        f"~{elapsed / max(1, global_step):.2f}s/step, "
        f"~{elapsed / epochs_done:.1f}s/epoch (over {epochs_done} epoch(s)).")
    logf.close()


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("\nFINETUNE FAIL (canary):", str(e))
        traceback.print_exc()
        sys.exit(1)
    except (RuntimeError, NotImplementedError) as e:
        print("\nFINETUNE RUNTIME ERROR:", str(e))
        traceback.print_exc()
        sys.exit(2)
