#!/usr/bin/env python3
"""
generate.py -- CLI inference driver for the FW-GAN handwriting synthesis model.

Given an arbitrary text string and one or more style-reference crops (black ink
on white background), this loads the pretrained FW-GAN checkpoint and generates
one PNG image per word, plus a concatenated single-line preview image.

Using the pretrained (non-fine-tuned) checkpoint is EXPECTED to produce generic
IAM/English-style handwriting, not any particular person's handwriting -- the
goal here is just to prove the generation mechanism works end-to-end.
"""
import os
import sys
from glob import glob

# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT = "/Users/blakey5aces/Handwriting Analysis"
FW = os.path.join(PROJECT, "models", "FW_GAN")
sys.path.insert(0, FW)

import argparse
import math

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lib.utils import yaml2config
from networks import get_model


def default_style_refs(n=3):
    """Pick style references that actually exist right now.

    This used to be a hardcoded filename ('sample_synth_L01_W04.png') from the
    old crop naming scheme. Crops are now named '{page}__{box_id}.png', so the
    old file no longer exists and every run that did not pass --style died with
    "no valid style references remained after filtering" -- a confusing error
    for something the caller never set. Resolving against the manifest means
    the default cannot go stale again when naming changes.

    Widest crops are preferred: a longer word carries more of the writer's
    letterforms, which is what the style encoder is reading.
    """
    seg = os.path.join(PROJECT, "segmented")
    man = os.path.join(seg, "manifest.csv")
    cands = []
    if os.path.exists(man):
        import csv as _csv
        with open(man, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                pth = r.get("crop_path", "")
                if pth and os.path.exists(pth):
                    cands.append(pth)
    if not cands:
        cands = sorted(glob(os.path.join(seg, "*.png")))
    if not cands:
        return []
    try:
        from PIL import Image as _Im
        cands.sort(key=lambda q: -_Im.open(q).width)
    except Exception:
        pass
    return cands[:n]


def parse_args():
    p = argparse.ArgumentParser(description="Generate handwriting-style word images with FW-GAN.")
    p.add_argument("--text", required=True, help="Arbitrary text string; words separated by spaces.")
    p.add_argument(
        "--style",
        nargs="+",
        default=None,   # resolved by default_style_refs() -- see why below
        help="One or more style-reference image paths (segmented/*.png crops). "
             "Defaults to real crops picked from the current manifest.",
    )
    p.add_argument(
        "--ckpt",
        default=os.path.join(PROJECT, "models", "weights", "FW-GAN.pth"),
        help="Path to the FW-GAN checkpoint.",
    )
    p.add_argument(
        "--config",
        default=os.path.join(FW, "configs", "fw_gan_iam.yml"),
        help="Path to the FW-GAN yaml config.",
    )
    p.add_argument("--output-dir", default=os.path.join(PROJECT, "output"), help="Directory to write PNG outputs.")
    p.add_argument("--device", default="mps", help="Torch device to run on (mps/cpu/cuda).")
    args = p.parse_args()
    if not args.style:
        args.style = default_style_refs()
        if not args.style:
            print("ERROR: no style reference given and no crops found in "
                  "segmented/. Run detection first, or pass --style.",
                  file=sys.stderr)
            sys.exit(1)
    return args


def resolve(path):
    """Resolve a possibly-relative path against the project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT, path)


def main():
    args = parse_args()

    ckpt_path = resolve(args.ckpt)
    config_path = resolve(args.config)
    output_dir = resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    device = args.device

    # --- Build config -----------------------------------------------------
    cfg = yaml2config(config_path)
    cfg.device = device
    cfg.ckpt = ckpt_path
    # Make the lexicon path absolute so get_lexicon() (called during
    # AdversarialModel.__init__) can find it regardless of cwd.
    cfg.training.lexicon = os.path.join(FW, "data", "english_words.txt")

    # --- Build + load model -------------------------------------------------
    model = get_model(cfg.model)(cfg, log_root=output_dir)
    model.set_mode("eval")
    epoch = model.load(cfg.ckpt, map_location="cpu")
    print(f"Loaded checkpoint '{ckpt_path}' (epoch {epoch})")

    # These mirror lib/datasets.py collect_fn + lib/path_config.py -- read
    # from config rather than hardcoding.
    img_height = cfg.img_height  # 32
    char_width = cfg.char_width  # 16
    gen_style_dim = cfg.GenModel.style_dim  # 128
    enc_style_dim = cfg.EncModel.style_dim  # 96
    noise_dim = gen_style_dim - enc_style_dim  # 32
    len_scale = img_height // 2  # 16 -- StyleEncoder divides img_len by 16

    style_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # --- Style refs -> style_mu -------------------------------------------
    style_mus = []
    for style_path in args.style:
        style_path = resolve(style_path)
        try:
            img = Image.open(style_path).convert("L")
        except Exception as e:
            print(f"WARNING: failed to load style ref '{style_path}': {e}", file=sys.stderr)
            continue

        w, h = img.size
        new_w = max(1, round(w * img_height / h))
        img = img.resize((new_w, img_height))

        if new_w < len_scale:
            print(
                f"WARNING: style ref '{style_path}' too narrow (<{len_scale}px after resize), skipping",
                file=sys.stderr,
            )
            continue

        norm = style_tf(img)  # (1, 32, new_w), white bg -> +1, ink -> -1

        pad_w = math.ceil(new_w / len_scale) * len_scale
        padded = torch.full((1, 1, img_height, pad_w), -1.0)  # pad fill value per spec (collect_fn convention)
        padded[..., :new_w] = norm.unsqueeze(0)

        padded = padded.to(device)
        with torch.no_grad():
            mu = model.models.E(padded, torch.tensor([pad_w], dtype=torch.int, device=device), model.models.S)
        style_mus.append(mu)
        print(f"Encoded style ref '{style_path}' -> mu shape {tuple(mu.shape)}")

    if len(style_mus) == 0:
        print("ERROR: no valid style references remained after filtering.", file=sys.stderr)
        sys.exit(1)

    style_mu = torch.stack(style_mus, dim=0).mean(dim=0)
    if not torch.isfinite(style_mu).all():
        print("ERROR: style_mu contains non-finite values.", file=sys.stderr)
        sys.exit(1)

    # --- Text -> words, with OOV filtering ---------------------------------
    alphabet_dict = model.label_converter.dict
    max_word_len = cfg.training.max_word_len

    raw_words = args.text.split()
    words = []
    for raw_word in raw_words:
        filtered_chars = [c for c in raw_word if c in alphabet_dict]
        dropped = [c for c in raw_word if c not in alphabet_dict]
        if dropped:
            print(f"WARNING: word '{raw_word}' contains unsupported chars {dropped!r}, dropping them", file=sys.stderr)
        word = "".join(filtered_chars)
        if not word:
            print(f"WARNING: word '{raw_word}' became empty after filtering, skipping", file=sys.stderr)
            continue
        if len(word) > max_word_len:
            print(f"WARNING: word '{word}' exceeds max_word_len={max_word_len}, quality may degrade", file=sys.stderr)
        words.append(word)

    if not words:
        print("ERROR: no words survived alphabet filtering.", file=sys.stderr)
        sys.exit(1)

    # --- Generate per word --------------------------------------------------
    word_images = []
    with torch.no_grad():
        for idx, word in enumerate(words):
            enc = model.label_converter.encode(word)  # plain string -> flat index list
            word_lbs = torch.LongTensor(enc).unsqueeze(0).to(device)  # (1, L)
            word_lb_lens = torch.IntTensor([len(word)]).to(device)  # (1,)

            noise = torch.randn((1, noise_dim), device=device)  # (1, 32)
            enc_z = torch.cat([noise, style_mu], dim=1)  # (1, 128)

            fake = model.models.G(enc_z, word_lbs, word_lb_lens)  # (1,1,32, L*char_width)
            fake = fake[:, :, :, : len(word) * char_width]

            assert torch.isfinite(fake).all(), f"non-finite output for word {word!r}"

            img = fake[0, 0]
            img = (255 * ((img + 1) / 2)).clamp(0, 255).to(torch.uint8).cpu().numpy()

            img_min, img_max, img_mean = int(img.min()), int(img.max()), float(img.mean())
            if img_max - img_min < 5:
                print(f"WARNING: possible blank/garbage output for word '{word}' (min={img_min}, max={img_max})", file=sys.stderr)

            safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in word)[:40]
            out_path = os.path.join(output_dir, f"{idx:03d}_{safe}.png")
            Image.fromarray(img, mode="L").save(out_path)
            print(f"word='{word}' min={img_min} max={img_max} mean={img_mean:.1f} -> saved {out_path}")

            word_images.append(img)

    # --- Concatenate into one line strip for visual QA ----------------------
    if word_images:
        sep = np.full((img_height, 16), 255, dtype=np.uint8)
        pieces = []
        for i, wi in enumerate(word_images):
            if i > 0:
                pieces.append(sep)
            pieces.append(wi)
        line = np.concatenate(pieces, axis=1)
        line_path = os.path.join(output_dir, "_line.png")
        Image.fromarray(line, mode="L").save(line_path)
        print(f"Saved combined line preview -> {line_path}")

    print(f"\nDone: generated {len(word_images)} word image(s) using {len(style_mus)} style ref(s). Output dir: {output_dir}")


if __name__ == "__main__":
    main()
