"""
Step 3 — StyleGAN2 Generator Wrapper
======================================
Loads the pretrained StyleGAN2-FFHQ generator and provides a single function
to decode a W+ latent code into a PIL image.

This module is imported by invert.py, edit.py, and extract_hair_direction.py.
It can also be run standalone as a quick smoke test:

    python src/generate.py --latent results/photo/w.npy --output results/photo/test.png
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_DIR  = os.path.join(PROJECT_ROOT, "weights")
SG2_PT       = os.path.join(WEIGHTS_DIR, "stylegan2-ffhq-config-f.pt")
E4E_REPO     = os.path.join(PROJECT_ROOT, "vendor", "encoder4editing")


# ─────────────────────────────────────────────────────────────────────────────
# Generator cache (avoids re-loading in loops)
# ─────────────────────────────────────────────────────────────────────────────

_GENERATOR_CACHE: dict = {}


def _ensure_e4e_on_path() -> None:
    """Add the encoder4editing repo to sys.path (idempotent)."""
    if not os.path.isdir(E4E_REPO):
        raise FileNotFoundError(
            f"encoder4editing not found at: {E4E_REPO}\n"
            "Run 'python setup.py' to clone and set it up."
        )
    if E4E_REPO not in sys.path:
        sys.path.insert(0, E4E_REPO)


def get_generator(pt_path: str = SG2_PT, device_str: str = "cuda") -> torch.nn.Module:
    """
    Load and cache the StyleGAN2-FFHQ generator.

    Parameters
    ----------
    pt_path    : path to stylegan2-ffhq-config-f.pt
    device_str : "cuda" or "cpu"

    Returns
    -------
    Generator in eval mode on the specified device.
    """
    cache_key = (pt_path, device_str)
    if cache_key in _GENERATOR_CACHE:
        return _GENERATOR_CACHE[cache_key]

    if not os.path.isfile(pt_path):
        raise FileNotFoundError(
            f"StyleGAN2 checkpoint not found: {pt_path}\n"
            "Run 'python setup.py' to download it."
        )

    _ensure_e4e_on_path()
    from models.stylegan2.model import Generator

    device = torch.device(device_str)
    G = Generator(1024, 512, 8, channel_multiplier=2).to(device)

    ckpt = torch.load(pt_path, map_location=device)
    G.load_state_dict(ckpt["g_ema"], strict=False)
    G.eval()

    print(f"[generate] Generator loaded from {pt_path} → device={device_str}")
    _GENERATOR_CACHE[cache_key] = G
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_from_latent(
    w:          np.ndarray,
    pt_path:    str = SG2_PT,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Image.Image:
    """
    Decode a W+ latent code into a PIL image.

    Parameters
    ----------
    w          : np.ndarray, shape (18, 512) — W+ latent code
    pt_path    : path to the StyleGAN2 .pt checkpoint
    device_str : "cuda" or "cpu"

    Returns
    -------
    PIL.Image.Image  (RGB, 1024×1024)
    """
    if w.ndim == 1:
        w = np.tile(w[None], (18, 1))  # (512,) → (18, 512)
    if w.ndim != 2 or w.shape[1] != 512:
        raise ValueError(f"Expected w shape (18, 512), got {w.shape}")

    device   = torch.device(device_str)
    G        = get_generator(pt_path, device_str)
    w_tensor = torch.from_numpy(w).float().unsqueeze(0).to(device)  # (1, 18, 512)

    with torch.no_grad():
        img, _ = G(
            [w_tensor],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=False,
        )  # (1, 3, 1024, 1024), range [-1, 1]

    # Convert to uint8 RGB
    img = (img.clamp(-1, 1) + 1) / 2
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).round().astype(np.uint8)

    return Image.fromarray(img)


# ─────────────────────────────────────────────────────────────────────────────
# CLI (smoke test)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3 — Decode a .npy latent code to an image (smoke test)"
    )
    parser.add_argument("--latent", required=True, help="Path to w.npy (shape: 18×512)")
    parser.add_argument("--output", required=True, help="Output image path (.png)")
    parser.add_argument("--sg2_pt", default=SG2_PT, help="StyleGAN2 checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    w = np.load(args.latent)
    img = generate_from_latent(w, pt_path=args.sg2_pt, device_str=args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    img.save(args.output)
    print(f"[generate] Saved → {args.output}")
