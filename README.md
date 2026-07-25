# StyleGAN Face Edit

Semantic facial attribute editing with StyleGAN2. Given a photograph of a face,
the pipeline produces edited versions with a modified **age**, **head pose**,
**facial expression**, **hair colour** or **hair volume**.

The system is inference-only. It composes pretrained models — a StyleGAN2
generator, the e4e encoder, and a set of linear attribute directions — and no
network is trained or fine-tuned.

| | |
|---|---|
| ▶️ **Edit an image** | [Workflow A](#a-editing-an-image) — align, invert, edit, measure |
| 🔬 **Reproduce the hair boundaries** | [Workflow B](#b-reproducing-the-hair_black-and-hair_length-boundaries) — optional; they ship with the repo |
| 📄 **Report** | [Report (PDF, Google Drive)](https://drive.google.com/file/d/1LoeipHRXscEqdQjSQS8RSYDUU6TT0vgM/view?usp=sharing) — method and results |
| 🖼️ **Generated images** | [`examples/`](examples/README.md) — committed outputs, viewable without running anything |

---

## How it works

StyleGAN2 synthesises an image from a latent code $w$ of shape `18 × 512`
(the space usually written $\mathcal{W}^+$). Many facial attributes are
approximately linearly separable in this space, so each attribute can be
represented by a direction vector $d$ and applied by translation:

```
w_edited = w + α · d
```

`α` sets the strength and the sign of the edit. Everything else in this
repository exists to obtain a good `w` for a real photograph, and a good `d`
for each attribute.

```
photo.jpg
    │
    ▼  align_face.py       dlib 68 landmarks → FFHQ canonical crop, 1024×1024
photo_aligned.jpg
    │
    ▼  invert.py           e4e encoder, single forward pass → W+ code
w.npy (18×512) + reconstructed.png
    │
    ▼  edit.py             w + α·d, restricted to the relevant layers
age_+3.0.png, hair_black_+3.0.png, …
    │
    ▼  measure_results.py  hair segmentation + LPIPS
measurements.csv
```

### Layer-restricted editing

The 18 layers of $\mathcal{W}^+$ act at different spatial scales:

| Layers | Resolution | Dominant factors |
|---|---|---|
| 0–3 | 4²–8² | head pose, global head and hair shape |
| 4–7 | 16²–32² | facial features, hairstyle |
| 8–17 | 64²–1024² | colour scheme, skin tone, micro-texture |

Each edit is therefore applied only to the layers that govern its attribute:
hair **colour** to layers 8–17, hair **shape** to layers 0–7. Applying an edit
to all 18 layers lets it leak into unrelated factors — an unrestricted colour
edit perturbs facial geometry, an unrestricted shape edit perturbs skin tone.

`age`, `pose` and `smile` are applied to all layers, as their published
boundaries were fitted against the full code. Pass `--all_layers` to disable
the restriction for comparison.

---

## Supported attributes

| Attribute | −α | +α | α range | Direction source |
|---|---|---|---|---|
| `age` | younger | older | ±3 | published (encoder4editing) |
| `pose` | yaw left | yaw right | ±10 | published (encoder4editing) |
| `smile` | neutral | smiling | ±3 | published (encoder4editing) |
| `hair_black` | lighter hair | blacker hair | ±3 | **computed here** |
| `hair_length` | less hair | more hair | ±3 | **computed here** |

All five work immediately after `python setup.py`; there is nothing to train
and nothing to extract. The three published directions are downloaded by the
setup script; the two computed directions are committed to this repository
(~4 KB each).

**On the α ranges.** All directions are unit vectors but are not equally
effective per unit of displacement: a step along `pose` produces a smaller
visible change than the same step along `age`. Start at half the listed range
and adjust.

---

## Setup

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.8+ | tested on 3.8 |
| CUDA GPU | strongly recommended; CPU works but is slow |
| Git | used to clone the encoder repository |
| Conda | recommended, mainly for installing dlib |

### Install

```bash
conda create -n stylegan-face python=3.8 -y
conda activate stylegan-face

# dlib via conda avoids a source build
conda install -c conda-forge dlib -y

pip install -r requirements.txt
```

### Run the setup script

```bash
python setup.py
```

This clones [encoder4editing](https://github.com/omertov/encoder4editing) into
`vendor/`, replaces its JIT-compiled CUDA operators with equivalent
pure-PyTorch implementations (removing the need for a C++ toolchain), and
downloads the model checkpoints (~1.6 GB).

---

## A. Editing an image

This is the main workflow. Everything needed is already in the repository —
**no training and no boundary extraction is required.**

### Step 1 — Align the photo

```bash
python src/align_face.py --input inputs/salah.jpg --output inputs/salah_aligned.jpg
```

Detects 68 landmarks and warps the face into the FFHQ canonical frame.
**This step is not optional.** The generator and the encoder were trained only
on aligned faces; skipping it degrades inversion quality markedly and is the
most common cause of poor results.

### Step 2 — Invert to a latent code

```bash
python src/invert.py --input inputs/salah_aligned.jpg --output_dir results/salah/
```

Writes two files:

| File | Contents |
|---|---|
| `results/salah/w.npy` | the latent code, shape `18 × 512` |
| `results/salah/reconstructed.png` | that code decoded back to an image — the unedited reference |

Takes about 35 s on a 6 GB GPU, nearly all of it loading the two checkpoints;
the inversion itself is a single forward pass.

### Step 3 — Apply an edit

One attribute at a time. A single value:

```bash
python src/edit.py --latent results/salah/w.npy --attribute age --alpha 3 --output_dir results/salah/
```

…or a sweep of several strengths at once:

```bash
python src/edit.py --latent results/salah/w.npy --attribute age --alphas -3 -1.5 0 1.5 3 --output_dir results/salah/
```

All five attributes, as used for the report — note the wider range for `pose`:

```bash
python src/edit.py --latent results/salah/w.npy --attribute age         --alphas -3 -1.5 0 1.5 3 --output_dir results/salah/
python src/edit.py --latent results/salah/w.npy --attribute pose        --alphas -10 -5 0 5 10   --output_dir results/salah/
python src/edit.py --latent results/salah/w.npy --attribute smile       --alphas -3 -1.5 0 1.5 3 --output_dir results/salah/
python src/edit.py --latent results/salah/w.npy --attribute hair_black  --alphas -3 -1.5 0 1.5 3 --output_dir results/salah/
python src/edit.py --latent results/salah/w.npy --attribute hair_length --alphas -3 -1.5 0 1.5 3 --output_dir results/salah/
```

Output filenames follow `<attribute>_<±alpha>.png`. The measurement script
parses this convention, so do not rename the files.

### Step 4 — Measure the results

```bash
python src/measure_results.py --dir results/salah \
       --alphas -10 -5 -3 -1.5 0 1.5 3 5 10 \
       --output results/measurements.csv
```

The `--alphas` list must cover every value generated in step 3, including
pose's ±10.

### Expected output

`hair_ratio` should respond sharply for `hair_length` while staying flat for
the other attributes:

| Strength | −max | −half | 0 | +half | +max |
|---|---|---|---|---|---|
| `hair_length` | **0.000** | 0.113 | 0.171 | 0.231 | **0.298** |
| `hair_black` | 0.172 | 0.172 | 0.171 | 0.171 | 0.172 |

`hair_length` reaches `0.000` at −3: the segmentation network detects no hair
pixels at all. `hair_black` varies by `0.001` across its full range while
visibly changing hair colour — the colour edit leaves hair geometry measurably
untouched.

Absolute values depend on the input photograph; the pattern is the result.

### Using your own photo

Put it in `inputs/` and substitute the paths:

```bash
python src/align_face.py --input inputs/myphoto.jpg --output inputs/myphoto_aligned.jpg
python src/invert.py     --input inputs/myphoto_aligned.jpg --output_dir results/myphoto/
python src/edit.py       --latent results/myphoto/w.npy --attribute smile --alphas -3 0 3 --output_dir results/myphoto/
```

---

## Evaluation

`measure_results.py` records two quantities per generated image, both relative
to the unedited α = 0 reconstruction:

| Column | Meaning |
|---|---|
| `hair_ratio` | measured quantity of hair, `hair / (hair + face)` pixels, from a CelebAMask-HQ face-parsing network |
| `perceptual_change` | LPIPS distance to the α = 0 image |

`hair_ratio` is recorded for **all** attributes, not only the hair ones: for
attributes that should not affect hair, a flat response is evidence of
selectivity. `perceptual_change` acts as a stability check — it should grow
smoothly with `|α|`.

**Scope.** These measure edit *strength* and *selectivity*. They are **not** an
identity metric: LPIPS responds to overall visual change, which is correlated
with but distinct from loss of identity. An embedding-based score such as
ArcFace cosine similarity would measure identity directly and is a natural
extension; it is not reported. Identity preservation is assessed qualitatively
against the α = 0 column of the results figure.

---

## B. Reproducing the `hair_black` and `hair_length` boundaries

> **You do not need to do this to edit images.** Both directions are already
> committed to this repository (`weights/boundaries/`, ~4 KB each), and
> workflow A uses them directly. Follow this section only to verify the
> directions, or as a starting point for adding a new attribute.

The `age`, `pose` and `smile` directions cannot be reproduced here — they are
published InterFaceGAN boundaries that arrive with the cloned encoder4editing
repository. The two hair directions were computed for this project, and this
section reproduces them.

### Run the extraction

```bash
# both directions in one pass (recommended)
python src/extract_hair_direction.py --all --n_samples 5000

# or one at a time
python src/extract_hair_direction.py --attribute black  --n_samples 5000
python src/extract_hair_direction.py --attribute length --n_samples 5000
```

This overwrites `weights/boundaries/hair_black.npy` and
`weights/boundaries/hair_length.npy`. Takes roughly 25 minutes on a 6 GB GPU.
Lower `--batch_size` to 2 or 1 if you hit CUDA out-of-memory.

Requires `open_clip_torch` and `scikit-learn` (both in `requirements.txt`).
CLIP weights (~350 MB) download automatically on first run.

### What it does

The procedure follows InterFaceGAN:

1. sample 5000 latent codes and synthesise the corresponding faces;
2. give each face a scalar score for the attribute;
3. label the top 30 % positive and the bottom 30 % negative, discarding the
   ambiguous middle band;
4. fit a linear SVM to the labelled latent codes;
5. take its normalised weight vector as the direction.

The 5000 faces are synthesised **once** and their features cached, so
extracting both attributes costs one synthesis pass, not two.

Step 2 is where the two attributes differ:

| Attribute | Scoring method | Score |
|---|---|---|
| `hair_black` | contrastive CLIP | `sim(image, "black hair") − mean sim(image, {blond, brown, red, grey})` |
| `hair_length` | face-parsing segmentation | `hair / (hair + face)` pixel ratio |

**Why contrastive for colour.** Similarity to a single prompt is dominated by
factors unrelated to the attribute — whether hair is visible at all, general
image quality. Those factors affect the competing prompts equally, so
subtracting a reference built from them cancels the shared part and leaves a
score that reflects the colour itself. The competing prompts stay in
`COLOR_PROMPTS` even though only `black` is extracted, because they are what
defines the contrast.

**Why segmentation for volume.** For a geometric quantity, text similarity is
only an indirect stand-in. Counting hair pixels measures it directly.

### Verifying the result

Re-run workflow A step 4 and compare against the expected-output table. Exact
values will differ slightly — the latent codes are sampled randomly and the
script sets no fixed seed — but the pattern should hold: `hair_length` swings
`hair_ratio` across a wide range while `hair_black` holds it nearly constant.

### Adding a new attribute

1. add an entry to `HAIR_COLORS` or `HAIR_SHAPES` in `extract_hair_direction.py`;
2. add matching entries to `BOUNDARIES` and `LAYER_RANGES` in `edit.py`;
3. add the attribute name to `_FNAME_RE` in `measure_results.py`;
4. re-run the extraction.

---

## Project structure

```
stylegan-face-edit/
├── README.md
├── requirements.txt
├── setup.py                    one-time setup: clone, patch, download
│
├── src/
│   ├── align_face.py           stage 1 — FFHQ alignment (dlib)
│   ├── invert.py               stage 2 — e4e inversion
│   ├── generate.py             stage 3 — generator wrapper (cached)
│   ├── edit.py                 stage 4 — attribute editing
│   ├── measure_results.py      stage 5 — quantitative measurement
│   ├── hair_parser.py          hair segmentation and measurement
│   ├── extract_hair_direction.py   offline — compute hair directions
│   ├── _ssl_patch.py           Windows certificate-store workaround
│   └── patches/                pure-PyTorch replacements for CUDA ops
│
├── report/                     figures used by the report (PDF linked above)
├── examples/                   generated images
│
├── weights/
│   └── boundaries/             hair_black.npy, hair_length.npy (committed)
├── inputs/                     input photographs
└── results/                    generated output (git-ignored)
```

---

## Models

| Model | Used by | Role | Source |
|---|---|---|---|
| dlib 68-point predictor | `align_face.py` | landmark detection | `setup.py` |
| e4e encoder | `invert.py` | image → $\mathcal{W}^+$ | `setup.py` |
| StyleGAN2 FFHQ config-f | `generate.py` | $\mathcal{W}^+$ → image | `setup.py` |
| SegFormer (CelebAMask-HQ) | `hair_parser.py` | hair segmentation | auto |
| LPIPS (AlexNet) | `measure_results.py` | perceptual metric | auto |
| CLIP ViT-B/32 | `extract_hair_direction.py` | colour scoring (extraction only) | auto |

The first three are downloaded by `setup.py`; the rest are fetched by their
libraries on first use.

---

## Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `No face detected` during alignment | Face too small, rotated or occluded; crop closer and retry |
| Reconstruction resembles a different person | Input was not aligned — run `align_face.py` first |
| `FileNotFoundError` on a checkpoint | Run `python setup.py` |
| Ninja or C++ build errors on import | The CUDA-operator patch was not applied; re-run `python setup.py` |
| CUDA out of memory | Close other GPU processes, or pass `--device cpu` |
| `ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]` | Windows certificate-store defect; the scripts fall back to `certifi` automatically — ensure `certifi` is installed |
| `dlib` fails to install via pip | Use `conda install -c conda-forge dlib` |

---

## Licences

This project uses the following pretrained models under their respective
licences:

- [encoder4editing](https://github.com/omertov/encoder4editing) — MIT
- [StyleGAN2 (PyTorch port)](https://github.com/rosinality/stylegan2-pytorch) — MIT
- [dlib](http://dlib.net/) — Boost Software License
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — MIT
- [face-parsing (SegFormer, CelebAMask-HQ)](https://huggingface.co/jonathandinu/face-parsing) — see model card
