"""
Step 4 — Attribute Editing
============================
Edits ONE facial attribute at a time by adding alpha * direction to the
W+ latent code and generating the result via StyleGAN2.

Supported attributes and their boundary sources:
    age, pose, smile          →  encoder4editing/editings/interfacegan_directions/
                                 (downloaded by setup.py)
    hair_black, hair_length   →  weights/boundaries/ (committed to this repo;
                                 see extract_hair_direction.py to regenerate)

All five work out of the box after `python setup.py`.

Mechanism:
    w_edited = w + alpha * direction
    (direction is broadcast across all 18 W+ layers)

Usage:
    python src/edit.py --latent results/photo/w.npy --attribute age --alpha 3.0 --output_dir results/photo/
    python src/edit.py --latent results/photo/w.npy --attribute hair_blond --alphas -5 -2.5 0 2.5 5 --output_dir results/photo/
"""

import argparse
import os
import sys

# Bypass OpenMP conflicting-DLL error on Windows (PyTorch vs NumPy)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_DIR  = os.path.join(PROJECT_ROOT, "weights")
SG2_PT       = os.path.join(WEIGHTS_DIR, "stylegan2-ffhq-config-f.pt")
E4E_DIRS     = os.path.join(
    PROJECT_ROOT, "vendor", "encoder4editing", "editings", "interfacegan_directions"
)

# Add src/ to path for cross-imports
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Attribute boundary registry
# ─────────────────────────────────────────────────────────────────────────────

BOUNDARIES = {
    # Bundled with the encoder4editing repo (.pt format), downloaded by setup.py
    "age":         os.path.join(E4E_DIRS, "age.pt"),
    "pose":        os.path.join(E4E_DIRS, "pose.pt"),
    "smile":       os.path.join(E4E_DIRS, "smile.pt"),
    # Committed to this repo (~4 KB each) — see extract_hair_direction.py to
    # regenerate them or to add further hair attributes.
    "hair_black":  os.path.join(WEIGHTS_DIR, "boundaries", "hair_black.npy"),
    "hair_length": os.path.join(WEIGHTS_DIR, "boundaries", "hair_length.npy"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-attribute W+ layer ranges
# ─────────────────────────────────────────────────────────────────────────────
#
# StyleGAN2's 18 W+ layers are not interchangeable. Coarse layers (0-3, 4x4-8x8)
# carry pose and overall head/hair shape; middle layers (4-7, 16x16-32x32) carry
# facial features and hairstyle; fine layers (8-17, 64x64+) carry colour and
# micro-texture. The vendored GANSpace implementation restricts its edits the
# same way (see vendor/encoder4editing/editings/ganspace.py).
#
# Applying a direction to all 18 layers therefore bleeds an edit into places it
# does not belong — a hair *colour* direction applied to coarse layers nudges
# face geometry, and a hair *shape* direction applied to fine layers muddies
# skin tone and colour. Restricting each edit to the layers that actually own
# the attribute keeps the rest of the face still.
#
# None = apply to all layers (correct for the bundled InterFaceGAN directions,
# which were fitted against the full W+ code).
LAYER_RANGES = {
    # Hair colour lives in the fine layers.
    "hair_black":  (8, 18),
    # Hair shape/volume lives in the coarse+middle layers.
    "hair_length": (0, 8),
}


# ─────────────────────────────────────────────────────────────────────────────
# Load boundary direction
# ─────────────────────────────────────────────────────────────────────────────

def load_direction(attribute: str, boundaries: dict = BOUNDARIES) -> np.ndarray:
    """
    Load the InterFaceGAN boundary vector for the given attribute.

    Returns
    -------
    direction : np.ndarray, shape (512,) or (18, 512)
        Unit normal vector to the SVM decision boundary in W-space.
    """
    if attribute not in boundaries:
        raise ValueError(
            f"Unknown attribute '{attribute}'. "
            f"Choose from: {list(boundaries.keys())}"
        )

    path = boundaries[attribute]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Boundary file not found: {path}\n"
            "For age/pose/smile, run 'python setup.py' first.\n"
            "hair_black/hair_length ship with the repo — if missing, regenerate\n"
            "with 'python src/extract_hair_direction.py --all'."
        )

    # Load .pt (PyTorch tensor) or .npy (NumPy array)
    if path.endswith(".pt"):
        import torch
        direction = torch.load(path, map_location="cpu").numpy()
    else:
        direction = np.load(path)

    direction = direction.squeeze()
    return direction / (_l2_norm(direction) + 1e-8)


def _l2_norm(x: np.ndarray) -> float:
    """
    L2 (Frobenius) norm without going through BLAS.

    `np.linalg.norm` ravels its input and calls `x.dot(x)`. On this Windows
    build that dispatches to threaded MKL once the array is large enough, which
    aborts the process with a native OpenMP fault (0xC06D007F) and no Python
    traceback. It bit `pose`, whose direction is (18, 512) = 9216 elements,
    while the (512,) directions stayed under the threading threshold and worked
    — so the failure looked attribute-specific and produced no error at all.

    An explicit square-and-reduce is elementwise only, so it never reaches BLAS.
    float64 accumulation keeps the result accurate for float32 input.
    """
    return float(np.sqrt(np.square(x, dtype=np.float64).sum()))


# ─────────────────────────────────────────────────────────────────────────────
# Core: single latent edit
# ─────────────────────────────────────────────────────────────────────────────

def edit_latent(
    w:           np.ndarray,
    direction:   np.ndarray,
    alpha:       float,
    layer_range: tuple = None,
) -> np.ndarray:
    """
    Apply a single attribute edit to a W+ latent code.

    Parameters
    ----------
    w           : (18, 512) — original W+ latent code
    direction   : (512,) or (18, 512) — boundary direction vector
    alpha       : edit strength (negative reverses direction)
    layer_range : optional (start, end) half-open W+ layer range to edit.
                  None edits all 18 layers. See LAYER_RANGES for why this
                  matters — restricting an edit to the layers that own the
                  attribute keeps it from bleeding into the rest of the face.

    Returns
    -------
    w_edited : (18, 512)
    """
    if w.ndim != 2 or w.shape != (18, 512):
        raise ValueError(f"Expected w shape (18, 512), got {w.shape}")

    # Broadcast (512,) → (18, 512) if needed
    if direction.shape == (512,):
        direction = np.tile(direction, (18, 1))
    elif direction.shape != (18, 512):
        raise ValueError(f"Expected direction shape (512,) or (18, 512), got {direction.shape}")

    delta = alpha * direction
    if layer_range is not None:
        start, end = layer_range
        mask = np.zeros((18, 1), dtype=delta.dtype)
        mask[start:end] = 1.0
        delta = delta * mask

    return w + delta


# ─────────────────────────────────────────────────────────────────────────────
# High-level: edit one attribute with an alpha sweep
# ─────────────────────────────────────────────────────────────────────────────

def edit_attribute(
    latent_path: str,
    attribute:   str,
    alphas:      list,
    output_dir:  str,
    sg2_pt:      str = SG2_PT,
    device_str:  str = None,
    all_layers:  bool = False,
) -> list:
    """
    Edit a single attribute across a list of alpha values.

    Parameters
    ----------
    latent_path : path to w.npy (shape 18×512)
    attribute   : one of the keys in BOUNDARIES
    alphas      : list of float alpha values
    output_dir  : directory to write output images
    sg2_pt      : StyleGAN2 checkpoint path
    device_str  : "cuda" or "cpu" (auto-detected if None)
    all_layers  : ignore the attribute's LAYER_RANGES entry and edit all 18
                  W+ layers (the old behaviour — useful for comparison)

    Returns
    -------
    output_paths : list of str — paths to saved images
    """
    import torch
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    from generate import generate_from_latent

    w         = np.load(latent_path)       # (18, 512)
    direction = load_direction(attribute)
    layer_range = None if all_layers else LAYER_RANGES.get(attribute)
    if layer_range is not None:
        print(f"[edit] {attribute}: editing W+ layers {layer_range[0]}-{layer_range[1] - 1}")

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    for alpha in alphas:
        w_edited = (
            w.copy() if alpha == 0.0
            else edit_latent(w, direction, alpha, layer_range=layer_range)
        )
        img = generate_from_latent(w_edited, pt_path=sg2_pt, device_str=device_str)

        sign  = "+" if alpha >= 0 else ""
        fname = f"{attribute}_{sign}{alpha:.1f}.png"
        fpath = os.path.join(output_dir, fname)
        img.save(fpath)
        output_paths.append(fpath)
        print(f"[edit] {attribute:>10s}  alpha={alpha:+.1f}  → {fpath}")

    return output_paths


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 4 — Edit ONE facial attribute at a time."
    )
    parser.add_argument(
        "--latent", required=True,
        help="Path to w.npy (output of invert.py)",
    )
    parser.add_argument(
        "--attribute", required=True, choices=list(BOUNDARIES.keys()),
        help="Which attribute to edit",
    )

    alpha_group = parser.add_mutually_exclusive_group(required=True)
    alpha_group.add_argument(
        "--alpha", type=float,
        help="Single alpha value (edit strength)",
    )
    alpha_group.add_argument(
        "--alphas", nargs="+", type=float,
        help="Space-separated list of alpha values for a sweep",
    )

    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--sg2_pt",     default=SG2_PT, help="StyleGAN2 checkpoint")
    parser.add_argument("--device",     default=None,    help="cuda or cpu (auto if omitted)")
    parser.add_argument(
        "--all_layers", action="store_true",
        help="Edit all 18 W+ layers instead of only the layers that own this "
             "attribute (see LAYER_RANGES). Mostly useful for comparison.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    alphas = [args.alpha] if args.alpha is not None else args.alphas

    edit_attribute(
        latent_path=args.latent,
        attribute=args.attribute,
        alphas=alphas,
        output_dir=args.output_dir,
        sg2_pt=args.sg2_pt,
        device_str=args.device,
        all_layers=args.all_layers,
    )
