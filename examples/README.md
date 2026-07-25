# Generated Images

Output of the pipeline on one input photo. These are committed so the results
can be inspected without running anything.

Resized to 768×768 JPEG (quality 95) to keep the repository small — the
pipeline itself outputs 1024×1024 PNG.

## Contents

| File | What it is |
|---|---|
| `00_input_aligned.jpg` | The input photo after FFHQ alignment (**Step 1**) |
| `01_reconstructed.jpg` | The face rebuilt from its latent code, no edit applied (**Step 2**). This is the α = 0 reference for every edit below. |
| `<attribute>_<±α>.jpg` | The edited results (**Step 3–4**) |

## Attributes

| Attribute | −α | +α | α range | Direction source |
|---|---|---|---|---|
| `age` | younger | older | ±3 | pre-made (encoder4editing) |
| `pose` | turn left | turn right | **±10** | pre-made (encoder4editing) |
| `smile` | neutral | smiling | ±3 | pre-made (encoder4editing) |
| `hair_black` | lighter hair | blacker hair | ±3 | **ours** (contrastive CLIP) |
| `hair_length` | less hair | more hair | ±3 | **ours** (face-parsing) |

`pose` uses a wider range because the directions are not equally strong — at ±3
the head rotation is too subtle to see.

The α = 0 image is omitted because it is identical to `01_reconstructed.jpg`
for every attribute.

## Side-by-side comparison

For all 25 images laid out as a single grid, see
[`../report/figures/fig_results.png`](../report/figures/fig_results.png).

## Reproducing these

See [Workflow A — Editing an image](../README.md#a-editing-an-image). The
measured numbers for these exact images are in the report
([`../report/report.tex`](../report/report.tex)).
