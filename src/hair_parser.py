"""
Hair Segmentation & Measurement (SegFormer face-parsing)
==========================================================
Wraps a SegFormer model fine-tuned on CelebAMask-HQ (19 facial classes) to
segment the hair region of a face and turn it into objective scalar metrics.

Why this exists
---------------
CLIP zero-shot scoring works acceptably for hair *colour* (colour is a strong,
global image statistic), but it is a poor signal for hair *shape* attributes
like length / volume / baldness: ranking faces by absolute CLIP similarity to
"a person with long hair" is dominated by confounds (gender, image quality,
background) rather than by the actual amount of hair.

A face-parsing network measures hair directly — it labels every pixel — so
"how much hair" becomes a geometric fact instead of a text-similarity guess.
That gives both (a) a much cleaner training signal for boundary extraction and
(b) a quantitative metric for evaluating edits.

Model:  jonathandinu/face-parsing  (SegFormer-B5, CelebAMask-HQ, 19 classes)
        Downloaded automatically by `transformers` on first use (~350 MB).
Hair is class id 13.

Usage (standalone sanity check):
    python src/hair_parser.py --image results/photo/reconstructed.png
"""

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Fix the Windows SSL cert-store crash before the HuggingFace download
import _ssl_patch  # noqa: F401,E402

MODEL_ID = "jonathandinu/face-parsing"
HAIR_CLASS = 13
FACE_CLASSES = (1, 2, 4, 5, 6, 7, 10, 11, 12)  # skin, nose, eyes, brows, mouth, lips

# SegFormer's own normalisation (ImageNet stats)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_PARSER_CACHE = {}


def get_parser(device_str: str = "cuda"):
    """Load and cache the SegFormer face-parsing model (eval mode)."""
    if device_str in _PARSER_CACHE:
        return _PARSER_CACHE[device_str]

    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError as exc:
        raise ImportError(
            "transformers is required for hair segmentation.\n"
            "Install with: pip install transformers"
        ) from exc

    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model = model.eval().to(torch.device(device_str))
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[hair_parser] SegFormer face-parsing loaded on {device_str}")
    _PARSER_CACHE[device_str] = model
    return model


def parse_logits(model, imgs: torch.Tensor, device, size: int = 512) -> torch.Tensor:
    """
    Run face parsing on a batch of StyleGAN images.

    Parameters
    ----------
    imgs : (B, 3, H, W) float tensor in [-1, 1] (StyleGAN2 output range)
    size : resolution fed to the segmenter

    Returns
    -------
    labels : (B, size, size) int64 tensor of class ids
    """
    x = (imgs.clamp(-1, 1) + 1) / 2                      # -> [0, 1]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = (x - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)

    out = model(pixel_values=x).logits                    # (B, 19, size/4, size/4)
    out = F.interpolate(out, size=(size, size), mode="bilinear", align_corners=False)
    return out.argmax(dim=1)


def hair_metrics(labels: torch.Tensor) -> dict:
    """
    Turn parsed label maps into hair metrics.

    Parameters
    ----------
    labels : (B, H, W) int64 class ids

    Returns
    -------
    dict with one (B,) float numpy array:
        hair_ratio : hair pixels / (hair + face) pixels

    `hair_ratio` is normalised by the head area rather than by the whole frame,
    so it does not drift when the head occupies a different fraction of the
    image. 0.0 means the parser found no hair at all (bald).
    """
    hair = (labels == HAIR_CLASS)
    face = torch.zeros_like(hair)
    for c in FACE_CLASSES:
        face |= (labels == c)

    hair_px = hair.flatten(1).sum(1).float()
    face_px = face.flatten(1).sum(1).float()

    return {"hair_ratio": (hair_px / (hair_px + face_px + 1e-6)).cpu().numpy()}


def measure_images(model, imgs: torch.Tensor, device, size: int = 512) -> dict:
    """Convenience: parse a batch of [-1,1] images and return hair metrics."""
    with torch.no_grad():
        labels = parse_logits(model, imgs, device, size=size)
    return hair_metrics(labels)


# ─────────────────────────────────────────────────────────────────────────────
# CLI (sanity check on a single image file)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from PIL import Image

    ap = argparse.ArgumentParser(description="Measure hair coverage in a face image")
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_mask", default=None, help="Optional path to save the hair mask PNG")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = get_parser(args.device)

    arr = np.array(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1

    with torch.no_grad():
        labels = parse_logits(model, t, device)
    m = hair_metrics(labels)

    print(f"  hair_ratio = {m['hair_ratio'][0]:.4f}  (hair / (hair + face) pixels)")

    if args.save_mask:
        mask = (labels[0] == HAIR_CLASS).cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask).save(args.save_mask)
        print(f"  mask saved -> {args.save_mask}")
