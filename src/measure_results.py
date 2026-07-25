"""
Quantitative Measurement of Edit Results
==========================================
Measures each edited image produced by edit.py and writes a CSV, so the
effect of an edit is reported as a number rather than an impression.

Two metrics per image, both relative to the unedited alpha=0 reconstruction:

  perceptual_change : LPIPS distance to the alpha=0 image.
      How much the picture changed overall. Should grow smoothly with |alpha|;
      a sudden jump means the edit has broken down.

  hair_ratio : measured amount of hair, from the face-parsing network.
      Recorded for EVERY attribute, not just the hair ones — seeing age/pose/
      smile leave it flat is the evidence that the edits are disentangled.

Note this reports no identity-similarity score. LPIPS measures overall visual
change, which is related to but not the same as "is this still the same
person".

Usage:
    python src/measure_results.py --dir results/abdo --output results/measurements.csv
"""

import argparse
import csv
import os
import re
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import _ssl_patch  # noqa: F401,E402
from hair_parser import get_parser, measure_images

_FNAME_RE = re.compile(
    r"^(age|pose|smile|hair_black|hair_length)_([+-]?\d+(?:\.\d+)?)\.png$"
)


def _load(path, device):
    """PNG -> (1, 3, H, W) tensor in [-1, 1]."""
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2 - 1


def main():
    ap = argparse.ArgumentParser(description="Measure edited images produced by edit.py")
    ap.add_argument("--dir", required=True, help="Directory of edited PNGs")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[-3.0, -1.5, 0.0, 1.5, 3.0],
                    help="Only measure these alpha values")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    keep = {f"{a:.1f}" for a in args.alphas}

    # Collect <attribute, alpha> -> path
    rows = []
    for fname in sorted(os.listdir(args.dir)):
        m = _FNAME_RE.match(fname)
        if not m:
            continue
        attr, alpha = m.group(1), float(m.group(2))
        if f"{alpha:.1f}" not in keep:
            continue
        rows.append((attr, alpha, os.path.join(args.dir, fname)))

    if not rows:
        print(f"[measure] No matching images in {args.dir}")
        raise SystemExit(1)

    import lpips
    percept = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    parser_model = get_parser(args.device)

    # Reference = the alpha=0 image of each attribute (identical across
    # attributes by construction, but read per-attribute so a missing one
    # fails loudly instead of silently comparing against the wrong picture).
    refs = {a: p for a, al, p in rows if al == 0.0}

    out = []
    for attr, alpha, path in rows:
        if attr not in refs:
            print(f"[measure] WARNING: no alpha=0 reference for '{attr}', skipping")
            continue

        img = _load(path, device)
        ref = _load(refs[attr], device)

        with torch.no_grad():
            d = float(percept(img, ref).mean())
        m = measure_images(parser_model, img, device)

        out.append({
            "attribute":         attr,
            "alpha":             f"{alpha:+.1f}",
            "perceptual_change": f"{d:.4f}",
            "hair_ratio":        f"{float(m['hair_ratio'][0]):.4f}",
        })
        print(f"[measure] {attr:<12} alpha={alpha:+.1f}  "
              f"perceptual={d:.4f}  hair_ratio={float(m['hair_ratio'][0]):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"\n[measure] Wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
