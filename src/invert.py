"""
Step 2 — Latent Space Inversion
=================================
Uses the pretrained e4e (encoder4editing) encoder to map an aligned face
photo into a W+ latent code that the StyleGAN2 generator can decode.

The encoder runs in a single forward pass — there is no per-image optimisation,
so inversion takes a couple of seconds. e4e is tuned for *editability*: it keeps
the latent in the well-behaved region of W+ where linear attribute directions
work properly, at the cost of a slightly soft reconstruction.

Usage:
    python src/invert.py --input inputs/photo_aligned.jpg --output_dir results/photo/

Outputs:
    w.npy             — latent code, shape (18, 512)
    reconstructed.png — decoded w through StyleGAN2
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# Reduces CUDA allocator fragmentation on small-VRAM GPUs. `expandable_segments`
# is only recognised by torch>=2.1's allocator (older versions raise at first
# CUDA call), so gate it by version. Safe as long as it's set before the first
# CUDA allocation — importing torch does not itself initialise a CUDA context.
if torch.cuda.is_available() and tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 1):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_DIR  = os.path.join(PROJECT_ROOT, "weights")
E4E_CKPT     = os.path.join(WEIGHTS_DIR, "e4e_ffhq_encode.pt")
SG2_PT       = os.path.join(WEIGHTS_DIR, "stylegan2-ffhq-config-f.pt")
E4E_REPO     = os.path.join(PROJECT_ROOT, "vendor", "encoder4editing")

# Add src/ to path for cross-imports
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

IMG_RESOLUTION = 256  # e4e was trained on 256-px aligned faces


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(path: str) -> torch.Tensor:
    """Load a PIL image → normalised tensor (1, 3, 256, 256) in [-1, 1]."""
    transform = T.Compose([
        T.Resize((IMG_RESOLUTION, IMG_RESOLUTION)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_e4e_on_path() -> None:
    """Add the encoder4editing repo to sys.path (idempotent)."""
    if not os.path.isdir(E4E_REPO):
        raise FileNotFoundError(
            f"encoder4editing not found at: {E4E_REPO}\n"
            "Run 'python setup.py' to clone and set it up."
        )
    if E4E_REPO not in sys.path:
        sys.path.insert(0, E4E_REPO)


def _load_e4e(ckpt_path: str, device: torch.device):
    """Load the pretrained e4e encoder.  Returns the model in eval mode."""
    _ensure_e4e_on_path()

    from argparse import Namespace
    from models.psp import pSp

    ckpt = torch.load(ckpt_path, map_location="cpu")
    opts = ckpt["opts"]
    opts["checkpoint_path"] = ckpt_path
    opts["device"]          = str(device)

    net = pSp(Namespace(**opts))
    net.eval()
    net.load_state_dict(ckpt["state_dict"], strict=False)
    net.to(device)

    print(f"[invert] Loaded e4e encoder from {ckpt_path}")
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def invert(
    input_path:   str,
    output_dir:   str,
    e4e_ckpt:     str = E4E_CKPT,
    sg2_pt:       str = SG2_PT,
    device_str:   str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """
    Invert an aligned face photo into a W+ latent code.

    Parameters
    ----------
    input_path   : path to the aligned face image (1024×1024)
    output_dir   : directory to save w.npy and reconstructed.png
    e4e_ckpt     : path to the e4e encoder checkpoint
    sg2_pt       : path to the StyleGAN2 generator checkpoint
    device_str   : "cuda" or "cpu"

    Returns
    -------
    w : np.ndarray, shape (18, 512)
    """
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

    # ── Encode with e4e (single forward pass, no optimisation) ──
    net   = _load_e4e(e4e_ckpt, device)
    img_t = _load_image(input_path).to(device)  # (1,3,256,256) in [-1,1]

    with torch.no_grad():
        _, w_tensor = net(img_t, randomize_noise=False, return_latents=True)

    w = w_tensor.squeeze(0).cpu().numpy()  # (18, 512)

    # Free the encoder (~1.1 GB) before loading the generator
    del net, w_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from generate import get_generator
    G = get_generator(sg2_pt, device_str)

    # ── Save latent code ──
    w_path = os.path.join(output_dir, "w.npy")
    np.save(w_path, w)
    print(f"[invert] Latent code saved → {w_path}  shape={w.shape}")

    # ── Reconstruct image from the final latent ──
    w_t = torch.from_numpy(w).float().unsqueeze(0).to(device)

    with torch.no_grad():
        img_recon, _ = G(
            [w_t],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=False,
        )

    img_recon = (img_recon.clamp(-1, 1) + 1) / 2
    img_recon = img_recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_recon = (img_recon * 255).round().astype(np.uint8)

    recon_path = os.path.join(output_dir, "reconstructed.png")
    Image.fromarray(img_recon).save(recon_path)
    print(f"[invert] Reconstruction saved → {recon_path}")

    return w


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2 — e4e inversion: aligned photo → W+ latent code"
    )
    parser.add_argument("--input",      required=True,    help="Aligned face image (1024×1024)")
    parser.add_argument("--output_dir", required=True,    help="Directory to save w.npy + reconstructed.png")
    parser.add_argument("--e4e_ckpt",   default=E4E_CKPT, help="Path to e4e checkpoint")
    parser.add_argument("--sg2_pt",     default=SG2_PT,   help="Path to StyleGAN2 checkpoint")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    invert(
        input_path=args.input,
        output_dir=args.output_dir,
        e4e_ckpt=args.e4e_ckpt,
        sg2_pt=args.sg2_pt,
        device_str=args.device,
    )
