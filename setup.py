"""
setup.py — Automated Project Setup
====================================
Clones the encoder4editing repo, patches its CUDA ops with pure-PyTorch
fallbacks, and downloads all required model weights.

Usage:
    pip install -r requirements.txt
    python setup.py
"""

import os
import sys
import bz2
import shutil
import subprocess
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR   = os.path.join(PROJECT_ROOT, "vendor")
E4E_DIR      = os.path.join(VENDOR_DIR, "encoder4editing")
WEIGHTS_DIR  = os.path.join(PROJECT_ROOT, "weights")
PATCHES_DIR  = os.path.join(PROJECT_ROOT, "src", "patches")

# Fix the Windows SSL cert-store crash (ssl.SSLError: [ASN1: NOT_ENOUGH_DATA])
# before the dlib landmark file is downloaded via urllib below. gdown (used
# for the other two checkpoints) is unaffected since it already relies on
# certifi rather than the OS trust store.
_SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import _ssl_patch  # noqa: F401,E402

# Hair boundaries ship in the repo itself (~4 KB each), so a fresh clone can
# edit hair immediately with no extraction step and nothing to download.
# Regenerate with: python src/extract_hair_direction.py --all
BUNDLED_BOUNDARIES = ["hair_black.npy", "hair_length.npy"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Clone encoder4editing
# ─────────────────────────────────────────────────────────────────────────────

def clone_encoder4editing():
    """Clone the encoder4editing repository into vendor/."""
    if os.path.isdir(E4E_DIR):
        print("[setup] vendor/encoder4editing already exists — skipping clone.")
        return

    print("[setup] Cloning encoder4editing ...")
    os.makedirs(VENDOR_DIR, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/omertov/encoder4editing", E4E_DIR],
        check=True,
    )
    print("[setup] [OK] Cloned encoder4editing.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Patch CUDA ops with pure-PyTorch fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def patch_cuda_ops():
    """
    Replace the CUDA JIT-compiled ops in encoder4editing with pure-PyTorch
    implementations from src/patches/.

    This avoids the need for a C++ compiler or CUDA toolkit at import time.
    The patched files are functionally equivalent for inference.
    """
    target_dir = os.path.join(E4E_DIR, "models", "stylegan2", "op")

    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"Expected directory not found: {target_dir}\n"
            "Make sure encoder4editing was cloned successfully."
        )

    files_to_patch = ["upfirdn2d.py", "fused_act.py"]

    for filename in files_to_patch:
        src = os.path.join(PATCHES_DIR, filename)
        dst = os.path.join(target_dir, filename)

        if not os.path.isfile(src):
            raise FileNotFoundError(f"Patch file missing: {src}")

        shutil.copy2(src, dst)
        print(f"[setup] [OK] Patched {filename}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Download model weights
# ─────────────────────────────────────────────────────────────────────────────

def _download_and_decompress_bz2(url, output_path, description):
    """
    Download a .bz2 file and decompress it on the fly.
    Used for the dlib landmark model from dlib.net.
    """
    print(f"[setup] Downloading {description} ...")
    bz2_path = output_path + ".bz2"

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb  = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)
        else:
            mb = downloaded / (1024 * 1024)
            print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)

    urllib.request.urlretrieve(url, bz2_path, reporthook=_reporthook)
    print("")

    print("[setup] Decompressing ...")
    with bz2.open(bz2_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(bz2_path)
    print(f"[setup] [OK] Saved -> {output_path}\n")


def download_weights():
    """Download all required model weights."""
    os.makedirs(os.path.join(WEIGHTS_DIR, "boundaries"), exist_ok=True)

    # ── e4e encoder (Google Drive via gdown) ──
    e4e_path = os.path.join(WEIGHTS_DIR, "e4e_ffhq_encode.pt")
    if os.path.isfile(e4e_path):
        print(f"[setup] e4e encoder already exists — skipping.")
    else:
        try:
            import gdown
            print("[setup] Downloading e4e encoder (~1.1 GB) ...")
            gdown.download(
                id="1cUv_reLE6k3604or78EranS7XzuVMWeO",
                output=e4e_path,
                quiet=False,
            )
            print(f"[setup] [OK] Saved → {e4e_path}\n")
        except ImportError:
            print(
                "[setup] [ERROR] gdown not installed. Install requirements first:\n"
                "  pip install -r requirements.txt\n"
                "Then re-run: python setup.py"
            )
            sys.exit(1)

    # ── StyleGAN2 generator (Google Drive via gdown) ──
    sg2_path = os.path.join(WEIGHTS_DIR, "stylegan2-ffhq-config-f.pt")
    if os.path.isfile(sg2_path):
        print(f"[setup] StyleGAN2 generator already exists — skipping.")
    else:
        try:
            import gdown
            print("[setup] Downloading StyleGAN2-FFHQ generator (~365 MB) ...")
            gdown.download(
                id="1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT",
                output=sg2_path,
                quiet=False,
            )
            print(f"[setup] [OK] Saved → {sg2_path}\n")
        except ImportError:
            print(
                "[setup] [ERROR] gdown not installed. Install requirements first:\n"
                "  pip install -r requirements.txt\n"
                "Then re-run: python setup.py"
            )
            sys.exit(1)

    # ── dlib 68-point landmark model (official dlib.net source) ──
    dlib_path = os.path.join(WEIGHTS_DIR, "shape_predictor_68_face_landmarks.dat")
    if os.path.isfile(dlib_path):
        print(f"[setup] dlib landmark model already exists — skipping.")
    else:
        _download_and_decompress_bz2(
            url="http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            output_path=dlib_path,
            description="dlib landmark model (~95 MB compressed)",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b: Verify the committed hair boundaries
# ─────────────────────────────────────────────────────────────────────────────

def check_boundaries():
    """
    Confirm the hair boundaries that ship with the repo are present.

    Nothing is downloaded here — these files are committed. This only catches
    the case where they were deleted or the clone was incomplete, and points
    at the command that regenerates them.
    """
    boundaries_dir = os.path.join(WEIGHTS_DIR, "boundaries")
    os.makedirs(boundaries_dir, exist_ok=True)

    missing = [
        f for f in BUNDLED_BOUNDARIES
        if not os.path.isfile(os.path.join(boundaries_dir, f))
    ]
    if missing:
        print(
            "[setup] [WARN] Hair boundaries missing (they ship with the repo):\n"
            f"          {', '.join(missing)}\n"
            "        Regenerate with:\n"
            "          python src/extract_hair_direction.py --all\n"
        )
    else:
        print("[setup] [OK] Hair boundaries present "
              f"({', '.join(BUNDLED_BOUNDARIES)}).\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Create required directories
# ─────────────────────────────────────────────────────────────────────────────

def create_directories():
    """Ensure all working directories exist."""
    dirs = [
        os.path.join(PROJECT_ROOT, "inputs"),
        os.path.join(PROJECT_ROOT, "results"),
        os.path.join(WEIGHTS_DIR, "boundaries"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[setup] [OK] Directories ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  StyleGAN Face Edit — Project Setup")
    print("=" * 60)
    print()

    create_directories()
    clone_encoder4editing()
    patch_cuda_ops()
    download_weights()
    check_boundaries()

    print("=" * 60)
    print("  [OK] Setup complete!")
    print("=" * 60)
    print()
    print("Ready to use: age, pose, smile, hair_black, hair_length")
    print()
    print("Next steps:")
    print("  1. Place a face photo in inputs/")
    print("  2. Align:  python src/align_face.py --input inputs/photo.jpg --output inputs/photo_aligned.jpg")
    print("  3. Invert: python src/invert.py --input inputs/photo_aligned.jpg --output_dir results/photo/")
    print("  4. Edit:   python src/edit.py --latent results/photo/w.npy --attribute age --alpha 3.0 --output_dir results/photo/")
    print()


if __name__ == "__main__":
    main()
