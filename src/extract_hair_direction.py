"""
Extract Hair Attribute Directions
===================================
Finds per-attribute latent directions for StyleGAN2-FFHQ using the
InterFaceGAN SVM method, with the scoring signal chosen per attribute.

Currently extracts two attributes — `black` (colour) and `length` (shape) —
one per scoring backend. See HAIR_COLORS / HAIR_SHAPES below
to widen the set.

    1. Sample N random W+ latent vectors from the generator.
    2. ONE generation pass: render all N faces and cache, per image,
       CLIP image features and/or face-parsing hair metrics.
       (Generating 1024x1024 faces dominates runtime and the images are
       identical for every attribute, so this pass is shared.)
    3. Score every sample per attribute from the cached features:
         - colour/texture -> contrastive CLIP
         - shape          -> face-parsing measurement
    4. Label the top-k% as positive (+1) and bottom-k% as negative (-1).
    5. Train a Linear SVM on (latent, label) pairs.
    6. Save the SVM normal vector as weights/boundaries/hair_<attribute>.npy

See docs/extract_hair_direction.md for why each backend was chosen and the
measurements that motivated it.

Usage:
    # Extract ALL hair attributes at once (recommended — one shared pass)
    python src/extract_hair_direction.py --all --n_samples 5000

    # Extract a single attribute
    python src/extract_hair_direction.py --attribute blond --n_samples 5000
    python src/extract_hair_direction.py --attribute length --n_samples 5000
"""

import argparse
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import LinearSVC

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
SG2_PT       = os.path.join(WEIGHTS_DIR, "stylegan2-ffhq-config-f.pt")
E4E_REPO     = os.path.join(PROJECT_ROOT, "vendor", "encoder4editing")

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if E4E_REPO not in sys.path:
    sys.path.insert(0, E4E_REPO)

# Fix the Windows SSL cert-store crash before any HTTPS download (CLIP weights)
import _ssl_patch  # noqa: F401,E402


# ─────────────────────────────────────────────────────────────────────────────
# Hair attribute definitions
#
# Two scoring backends, chosen per attribute by what actually measures it:
#
#   CLIP (contrastive)  — for appearance attributes (colour, curliness).
#       Scored as sim(image, positive) - mean(sim(image, negatives)).
#       The contrast is essential: absolute CLIP similarity to a single prompt
#       is dominated by confounds (does the person have visible hair at all,
#       image quality, gender), so single-prompt scoring yields entangled,
#       near-parallel boundaries. Measured on the previous single-prompt
#       version: brown/red cosine 0.679, blond/brown 0.596 — i.e. "make blond"
#       also dragged toward brown and red. One-vs-rest contrast fixes that.
#
#   SEG (face parsing)  — for geometric attributes (how much hair, how long).
#       Scored by directly measuring the hair region with a CelebAMask-HQ
#       face-parsing network. Text similarity is a poor proxy for "amount of
#       hair"; pixel counting is not a proxy at all, it is the measurement.
#
# Alpha sign follows the attribute: +alpha moves toward it, -alpha away.
# ─────────────────────────────────────────────────────────────────────────────

# ── Colour prompt pool ───────────────────────────────────────────────────────
# Every colour listed here acts as part of the one-vs-rest CONTRAST SET, even
# if it is not itself extracted. That distinction matters: the negatives are
# what keep a colour boundary disentangled. Scoring "black" against only itself
# collapses back to absolute single-prompt similarity, which is what produced
# the 0.679 brown/red entanglement in the first version of this file.
# Extra entries here cost nothing at runtime — they are only text embeddings.
COLOR_PROMPTS = {
    "black": "a photo of a person with black hair",
    "blond": "a photo of a person with blond hair",
    "brown": "a photo of a person with brown hair",
    "red":   "a photo of a person with red hair",
    "gray":  "a photo of a person with gray hair",
}

# ── What actually gets extracted ─────────────────────────────────────────────
# Narrowed to the two attributes in use. To extract more, add the key back
# here (colours must also exist in COLOR_PROMPTS above) and add matching
# entries to BOUNDARIES + LAYER_RANGES in edit.py and to _FNAME_RE in
# measure_results.py.

# Colours, scored one-vs-rest against COLOR_PROMPTS.
HAIR_COLORS = {
    "black": COLOR_PROMPTS["black"],
}

# Geometric attributes, measured by face parsing rather than text.
# Value is a metric name returned by hair_parser.hair_metrics().
HAIR_SHAPES = {
    "length": "hair_ratio",   # amount of hair: bald <-> full head of hair
}

HAIR_ATTRIBUTES = list(HAIR_COLORS) + list(HAIR_SHAPES)

# CLIP normalisation constants (ImageNet / OpenAI CLIP)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_generator(device):
    """Load the StyleGAN2-FFHQ generator."""
    from models.stylegan2.model import Generator

    g = Generator(1024, 512, 8, channel_multiplier=2)
    ckpt = torch.load(SG2_PT, map_location="cpu")
    g.load_state_dict(ckpt["g_ema"], strict=False)
    g.eval().to(device)
    print(f"[extract] StyleGAN2 generator loaded on {device}")
    return g


def load_clip_model(device):
    """
    Load CLIP (ViT-B/32) for zero-shot hair color classification.

    Uses open_clip_torch — the model weights (~350 MB) are downloaded
    automatically on first run and cached locally.
    """
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip_torch is required for CLIP-based hair color extraction.\n"
            "Install with: pip install open_clip_torch"
        ) from exc

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    print(f"[extract] CLIP model (ViT-B/32) loaded on {device}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Latent sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_w_plus(generator, n, device, batch_size=16):
    """Sample n random W+ codes → Tensor of shape (n, 18, 512)."""
    all_w = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            bs = min(batch_size, n - start)
            z  = torch.randn(bs, 512, device=device)
            w  = generator.style(z).unsqueeze(1).repeat(1, 18, 1)
            all_w.append(w.cpu())
            if (start // batch_size) % 10 == 0:
                print(f"\r  sampling latents {start}/{n} ...", end="", flush=True)
    print(f"\r  sampling latents {n}/{n} — done.")
    return torch.cat(all_w, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP-based scoring
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_clip_input(imgs_tensor, device):
    """
    Prepare StyleGAN2 output for CLIP.

    Input:  (B, 3, 1024, 1024), range [-1, 1]
    Output: (B, 3, 224, 224), CLIP-normalised
    """
    # [-1, 1] → [0, 1]
    imgs = (imgs_tensor.clamp(-1, 1) + 1) / 2

    # Resize to CLIP input size (224×224)
    imgs = F.interpolate(imgs, size=(224, 224), mode="bicubic", align_corners=False)

    # Apply CLIP normalisation
    mean = CLIP_MEAN.to(device)
    std  = CLIP_STD.to(device)
    imgs = (imgs - mean) / std

    return imgs


def _encode_prompts(clip_model, tokenizer, prompts, device):
    """Encode a list of text prompts into L2-normalised CLIP features."""
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats                                   # (len(prompts), D)


def compute_features(
    generator, w_all, device, batch_size=4,
    clip_model=None, seg_parser=None, seg_metrics=(),
):
    """
    ONE generation pass over all latents, collecting whatever the requested
    attributes need. Every attribute is then scored from these cached features
    without regenerating anything — generating 1024x1024 faces is by far the
    dominant cost, and the images are identical for every attribute.

    Returns
    -------
    clip_feats : (n, D) float32 array of L2-normalised CLIP image features,
                 or None if clip_model was not supplied
    metrics    : dict metric_name -> (n,) float array, for the requested
                 segmentation metrics (empty if seg_parser was not supplied)
    """
    # NOTE: fp16 autocast was tried here and rejected — StyleGAN2's weight
    # demodulation divides by rsqrt(sum of squares), which underflows to
    # 0/NaN in fp16 and silently produced NaN scores end-to-end (confirmed
    # by testing). Full fp32 it is; memory is instead controlled via
    # --batch_size and periodic cache clearing below.
    n        = len(w_all)
    is_cuda  = device.type == "cuda"
    feats_acc = [] if clip_model is not None else None
    metric_acc = {k: [] for k in seg_metrics}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        w_batch = w_all[start:end].to(device)

        with torch.no_grad():
            imgs, _ = generator(
                [w_batch], input_is_latent=True, randomize_noise=False
            )

            if clip_model is not None:
                imgs_in = _prepare_clip_input(imgs, device)
                f = clip_model.encode_image(imgs_in)
                f = f / f.norm(dim=-1, keepdim=True)
                feats_acc.append(f.float().cpu())

            if metric_acc:
                from hair_parser import measure_images
                m = measure_images(seg_parser, imgs, device)
                for k in metric_acc:
                    metric_acc[k].append(m[k])

        del imgs, w_batch
        # Periodic (not per-batch, to avoid the cudaFree/cudaMalloc sync cost)
        # cache clear keeps fragmentation from causing OOMs on small-VRAM GPUs.
        if is_cuda and (start // batch_size) % 20 == 0:
            torch.cuda.empty_cache()

        print(f"\r  generating + measuring {end}/{n} ...", end="", flush=True)

    print(f"\r  generating + measuring {n}/{n} — done." + " " * 20)

    # Kept as a torch CPU tensor rather than numpy: doing the scoring matmul
    # through numpy's BLAS right after heavy torch CUDA work reliably crashed
    # this Windows/MKL stack with a native 0xc06d007f OpenMP fault.
    clip_feats = torch.cat(feats_acc) if feats_acc is not None else None
    metrics = {k: np.concatenate(v) for k, v in metric_acc.items()}
    return clip_feats, metrics


def score_from_features(
    plan, clip_feats, metrics, clip_model=None, tokenizer=None, device=None,
):
    """
    Score every sample for one attribute using the cached features.

    CLIP contrastive:  score = sim(img, pos) - mean_i sim(img, neg_i)
    Segmentation:      score = the measured metric itself
    """
    if "seg_metric" in plan:
        return metrics[plan["seg_metric"]]

    pos = _encode_prompts(clip_model, tokenizer, [plan["pos_prompt"]], device).float().cpu()
    neg = _encode_prompts(clip_model, tokenizer, list(plan["neg_prompts"]), device).float().cpu()

    # Contrastive: how much more does this look like the positive prompt than
    # like the competing alternatives? Done in torch (see compute_features).
    with torch.no_grad():
        s = (clip_feats @ pos[0]) - (clip_feats @ neg.T).mean(dim=1)
    return s.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# SVM boundary extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_direction(w_all, scores, threshold_pct=30):
    """
    Train a LinearSVC on the top/bottom percentile samples and return the
    boundary normal vector of shape (512,).
    """
    n     = len(scores)
    k     = int(n * threshold_pct / 100)
    order = np.argsort(scores)

    low_idx  = order[:k]        # lowest similarity → "does NOT have this color"
    high_idx = order[-k:]       # highest similarity → "HAS this color"

    # Use the first W layer as the feature vector
    W = w_all[:, 0, :].numpy()  # (n, 512)
    X = np.concatenate([W[low_idx], W[high_idx]], axis=0)
    y = np.array([-1] * k + [1] * k)

    print(f"  training LinearSVC on {len(X)} samples ...")
    clf = LinearSVC(C=0.1, max_iter=5000)
    clf.fit(X, y)

    direction = clf.coef_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _scoring_plan(attribute):
    """
    Resolve an attribute name to its scoring backend.

    Returns a kwargs dict for score_from_features().
    """
    if attribute in HAIR_COLORS:
        # One-vs-rest: contrast this colour against every OTHER colour in the
        # full prompt pool — not just the extracted ones. Drawing negatives
        # from COLOR_PROMPTS (rather than HAIR_COLORS) is what lets the set of
        # extracted colours be narrowed without weakening the contrast.
        negs = [p for c, p in COLOR_PROMPTS.items() if c != attribute]
        if not negs:
            raise ValueError(
                f"No contrast prompts for colour '{attribute}'. A colour needs "
                "at least one other entry in COLOR_PROMPTS to score against — "
                "without it this degrades to single-prompt scoring."
            )
        return {"pos_prompt": HAIR_COLORS[attribute], "neg_prompts": negs}

    if attribute in HAIR_SHAPES:
        return {"seg_metric": HAIR_SHAPES[attribute]}

    raise ValueError(f"Unknown hair attribute: {attribute}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract hair attribute boundaries (contrastive CLIP for "
                    "colour, face-parsing segmentation for shape)"
    )

    attr_group = parser.add_mutually_exclusive_group(required=True)
    attr_group.add_argument(
        "--attribute", choices=HAIR_ATTRIBUTES,
        help="Extract direction for a single hair attribute",
    )
    attr_group.add_argument(
        "--all", action="store_true",
        help="Extract directions for ALL hair attributes",
    )

    parser.add_argument("--n_samples",  type=int, default=5000,  help="Random faces to generate")
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size (default 4 fits ~6 GB GPUs; raise if you have more VRAM, "
             "lower to 1-2 if you still hit CUDA OOM)",
    )
    parser.add_argument("--threshold",  type=int, default=30,    help="Top/bottom %% for labels")
    parser.add_argument("--output_dir", default=os.path.join(WEIGHTS_DIR, "boundaries"))
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Determine which attributes to extract
    attributes = list(HAIR_ATTRIBUTES) if args.all else [args.attribute]
    plans = {a: _scoring_plan(a) for a in attributes}
    needs_clip = any("pos_prompt" in p for p in plans.values())
    needs_seg  = any("seg_metric" in p for p in plans.values())

    print("=" * 60)
    print("  Hair Attribute Direction Extraction")
    print("=" * 60)
    print(f"  device     = {device}")
    print(f"  n_samples  = {args.n_samples}")
    print(f"  batch_size = {args.batch_size}")
    print(f"  threshold  = {args.threshold}%")
    print(f"  attributes = {', '.join(attributes)}")
    print(f"  scorers    = "
          f"{'contrastive-CLIP ' if needs_clip else ''}"
          f"{'face-parsing-seg' if needs_seg else ''}".strip())
    print(f"  output_dir = {args.output_dir}")
    print()

    # ── Load models (only what this run actually needs) ──
    generator = load_generator(device)
    clip_model = tokenizer = None
    if needs_clip:
        clip_model, tokenizer = load_clip_model(device)
    seg_parser = None
    if needs_seg:
        from hair_parser import get_parser
        seg_parser = get_parser(args.device)
    print()

    # ── Sample latent codes (shared across all attributes) ──
    print("[extract] Step 1 — Sampling random W+ latents ...")
    w_all = sample_w_plus(generator, args.n_samples, device, args.batch_size)
    print()

    # ── ONE generation pass, shared by every attribute ──
    seg_metrics = sorted({p["seg_metric"] for p in plans.values() if "seg_metric" in p})
    print(f"[extract] Step 2 — Generating {args.n_samples} faces and extracting features ...")
    try:
        clip_feats, metrics = compute_features(
            generator, w_all, device,
            batch_size=args.batch_size,
            clip_model=clip_model if needs_clip else None,
            seg_parser=seg_parser,
            seg_metrics=seg_metrics,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA out of memory during hair-attribute extraction.\n"
            f"Current --batch_size is {args.batch_size}. Try:\n"
            "  1. Lower --batch_size (e.g. 2 or 1)\n"
            "  2. Lower --n_samples\n"
            "  3. Use --device cpu (much slower, but works on any amount of VRAM)"
        )
    print()

    # ── Fit a boundary per attribute from the cached features (cheap) ──
    for i, attribute in enumerate(attributes, 1):
        plan = plans[attribute]
        output_path = os.path.join(args.output_dir, f"hair_{attribute}.npy")
        backend = "CLIP-contrastive" if "pos_prompt" in plan else "face-parsing"

        print(f"[extract] Step 3.{i} — Fitting '{attribute}' via {backend} ...")

        scores = score_from_features(
            plan, clip_feats, metrics,
            clip_model=clip_model, tokenizer=tokenizer, device=device,
        )
        print(f"  scores: min={scores.min():.4f}  max={scores.max():.4f}  "
              f"mean={scores.mean():.4f}")

        direction = extract_direction(w_all, scores, threshold_pct=args.threshold)
        np.save(output_path, direction)
        print(f"  [OK] Saved hair_{attribute} direction {direction.shape} -> {output_path}")
        print()

    # ── Summary ──
    print("=" * 60)
    print("  [OK] All done!")
    print("=" * 60)
    print()
    print("Test with:")
    for attribute in attributes:
        print(f"  python src/edit.py --latent results/<name>/w.npy "
              f"--attribute hair_{attribute} --alpha 3.0 --output_dir results/<name>/")


if __name__ == "__main__":
    main()
