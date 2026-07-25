"""
Step 1 — Face Alignment (FFHQ Canonical)
==========================================
Detects 68 facial landmarks with dlib and aligns the face to the exact
FFHQ layout that StyleGAN2 and e4e were trained on.

Reference:
    https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

Algorithm:
    1. Detect 68-point landmarks via dlib.
    2. Compute eye/mouth centroids.
    3. Solve a similarity transform (rotation + scale + translation)
       mapping those centroids to fixed canonical pixel positions.
    4. Warp at 4× resolution for quality, then downsample to 1024×1024.

Usage:
    python src/align_face.py --input inputs/photo.jpg --output inputs/photo_aligned.jpg
"""

import argparse
import os
from typing import Optional

import numpy as np
from PIL import Image

try:
    import dlib
except ImportError as exc:
    raise ImportError(
        "dlib not found.\n"
        "Install with:  conda install -c conda-forge dlib\n"
        "  or:          pip install dlib"
    ) from exc

try:
    from scipy.ndimage import gaussian_filter
except ImportError as exc:
    raise ImportError("scipy not found. Install with: pip install scipy") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LANDMARKS_DAT  = os.path.join(PROJECT_ROOT, "weights", "shape_predictor_68_face_landmarks.dat")

OUTPUT_SIZE    = 1024   # FFHQ resolution
TRANSFORM_SIZE = 4096   # internal upsample for higher-quality warp
ENABLE_PADDING = True


# ─────────────────────────────────────────────────────────────────────────────
# 68-point iBUG landmark index groups
# ─────────────────────────────────────────────────────────────────────────────

LM_LEFT_EYE  = list(range(36, 42))
LM_RIGHT_EYE = list(range(42, 48))
LM_OUTER_LIP = list(range(48, 60))


# ─────────────────────────────────────────────────────────────────────────────
# Landmark detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_landmarks(detector, predictor, img_rgb: np.ndarray) -> Optional[np.ndarray]:
    """Run dlib detector + predictor.  Returns (68, 2) float64 array or None."""
    detections = detector(img_rgb,1)
    if len(detections) == 0:
        return None

    # Pick the largest bounding box
    best = max(detections, key=lambda d: d.width() * d.height())
    shape = predictor(img_rgb, best)
    return np.array(
        [[shape.part(i).x, shape.part(i).y] for i in range(68)],
        dtype=np.float64,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FFHQ alignment (faithful port of the official FFHQ preparation script)
# ─────────────────────────────────────────────────────────────────────────────

def _ffhq_align(img: Image.Image, lm: np.ndarray) -> Image.Image:
    """
    Align and crop a face to the FFHQ 1024×1024 canonical layout.

    Steps:
        1. Compute eye/mouth centroids from 68-pt landmarks.
        2. Build a quad in the original image coordinate space.
        3. Pad, shrink, crop, and warp via PIL.Image.transform (QUAD).
        4. Apply a light unsharp-mask, then downsample to 1024×1024.
    """
    # ── Step 1: landmark centroids ──
    lm_eye_left  = lm[LM_LEFT_EYE].mean(axis=0)
    lm_eye_right = lm[LM_RIGHT_EYE].mean(axis=0)
    lm_mouth     = lm[LM_OUTER_LIP].mean(axis=0)

    eye_avg      = (lm_eye_left + lm_eye_right) * 0.5
    eye_to_eye   = lm_eye_right - lm_eye_left
    eye_to_mouth = lm_mouth - eye_avg

    # "x" axis: perpendicular to eye-to-mouth, scaled like eye separation
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

    y = np.flipud(x) * [-1, 1]                  # y axis
    c = eye_avg + eye_to_mouth * 0.1             # face center

    # ── Step 2: source quad ──
    quad  = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # ── Step 3: pad image ──
    img_np = np.array(img)
    if ENABLE_PADDING:
        pad = int(np.round(qsize * 0.1))
        img_np = np.pad(img_np, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        img_work = Image.fromarray(img_np)
        quad_w = quad + pad
    else:
        img_work = img.copy()
        quad_w = quad.copy()

    # ── Step 4: shrink if quad is much larger than OUTPUT_SIZE ──
    shrink = max(int(np.floor(qsize / (OUTPUT_SIZE / 2))), 1)
    if shrink > 1:
        new_size = (
            int(np.round(img_work.width / shrink)),
            int(np.round(img_work.height / shrink)),
        )
        img_work = img_work.resize(new_size, Image.LANCZOS)
        quad_w  /= shrink
        qsize   /= shrink

    # ── Step 5: crop bounding box around the quad ──
    border = max(int(np.round(qsize * 0.1)), 3)
    x0 = max(int(np.floor(quad_w[:, 0].min())) - border, 0)
    y0 = max(int(np.floor(quad_w[:, 1].min())) - border, 0)
    x1 = min(int(np.ceil(quad_w[:, 0].max()))  + border, img_work.width)
    y1 = min(int(np.ceil(quad_w[:, 1].max()))  + border, img_work.height)

    img_work = img_work.crop((x0, y0, x1, y1))
    quad_w  -= np.array([x0, y0])

    # ── Step 6: warp to TRANSFORM_SIZE ──
    img_transform = img_work.transform(
        (TRANSFORM_SIZE, TRANSFORM_SIZE),
        Image.QUAD,
        (quad_w + 0.5).flatten(),
        Image.BICUBIC,
    )

    # Light unsharp-mask (matches FFHQ preprocessing)
    arr     = np.array(img_transform, dtype=np.float32)
    blurred = gaussian_filter(arr, sigma=TRANSFORM_SIZE * 0.005)
    arr     = np.clip(arr + (arr - blurred) * 1.0, 0, 255).astype(np.uint8)
    img_transform = Image.fromarray(arr)

    # ── Step 7: downsample to final resolution ──
    return img_transform.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def align_face(
    input_path:    str,
    output_path:   str,
    landmarks_dat: str = LANDMARKS_DAT,
) -> bool:
    """
    Align a face image to the FFHQ 1024×1024 canonical layout.

    Parameters
    ----------
    input_path    : path to the raw input photo
    output_path   : where to save the aligned image
    landmarks_dat : path to dlib's shape_predictor_68_face_landmarks.dat

    Returns
    -------
    True if a face was detected and the aligned image was saved, False otherwise.
    """
    if not os.path.isfile(landmarks_dat):
        raise FileNotFoundError(
            f"Landmark model not found: {landmarks_dat}\n"
            "Run 'python setup.py' to download it automatically."
        )

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmarks_dat)

    img    = Image.open(input_path).convert("RGB")
    img_np = np.array(img)

    lm = _detect_landmarks(detector, predictor, img_np)
    if lm is None:
        print(f"[align] No face detected in: {input_path}")
        return False

    img_aligned = _ffhq_align(img, lm)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    img_aligned.save(output_path)
    print(f"[align] Aligned image saved → {output_path}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1 — FFHQ face alignment (dlib 68-pt landmarks)"
    )
    parser.add_argument("--input",     required=True,         help="Input photo (JPG/PNG)")
    parser.add_argument("--output",    required=True,         help="Output aligned image path")
    parser.add_argument("--landmarks", default=LANDMARKS_DAT, help="Path to shape_predictor_68_face_landmarks.dat")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    success = align_face(args.input, args.output, landmarks_dat=args.landmarks)
    if not success:
        raise SystemExit(1)
