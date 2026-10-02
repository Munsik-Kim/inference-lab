# Case 012 — Allocating an Image-Generation Budget: Loops vs Steps

English | [한국어](README.ko.md)

Built a tool to compare how image-generation time is divided between **steps that update a noisy image** and **internal loops that repeat the model's core computation within each step**. It generates three images from the same starting noise and lets readers inspect time alongside count, color and left/right requirements.

**Identical initial noise · Per-setting time and memory records · Blinded annotation**

[Compare every image](publication/IMAGES.md) · [Adapter code](source/adapter.py) · [Full report](REPORT.md)

[Loops and steps](#loops) · [Current results](#results) · [Interpretation](#interpretation) · [Inspect directly](#run) · [Sources](#sources)

<a id="loops"></a>
## What are loops and steps?

- **Step:** one update of an image that starts as noise.
- **Loop:** another application of the same model's core blocks within a generation step.

Loops 4 / Steps 50 updates the image 50 times, repeating the core computation four times within each step. Changing both can change time and the final image. B/32 denotes patch size, not 32 loops.

<a id="results"></a>
## What has been measured?

The same **64 prompt–seed pairs**, from 16 prompts and four starting-noise seeds each, were generated at three settings. Generation and measurement are complete for 192 MAIN images and 24 separate SMOKE images.

| Internal loops | Generation steps | Median request time (seconds) |
|---:|---:|---:|
| 1 | 89 | 4.534 |
| 2 | 66 | 4.385 |
| 4 | 50 | 4.762 |

RTX 5080 · 512×512 · one Looped-DiT B/32 checkpoint · 64 requests per setting. Time includes tokenization, the T5 text encoder, sampling and CPU image conversion; it excludes model loading and PNG writes.

Loops 2 / Steps 66 had the shortest median. The ±5% time-matching target was missed, so these are not exactly equal-time configurations. The time ranking of Loops 1 and 4 changed in fresh-process repeats.

**Quality annotations are pending.** Constraint pass rates and quality-ranked presets have not been calculated. A generated image is not automatically a correct image.

<a id="interpretation"></a>
## How should these results be read?

This example requests three yellow cubes and one green sphere from the same starting noise.

![Three images from identical initial noise: Loops 1/Steps 89, Loops 2/Steps 66, and Loops 4/Steps 50](publication/figures/compound-cubes.png)

Visual inspection shows three cubes in the first two images and four in the last. The deeper-loop/fewer-step configuration loses a count condition met by the other two. Since loops and steps change together, this does not isolate a causal effect of loops alone.

This illustration was selected after a code agent inspected 48 images from the first frozen seed. It is **post-hoc, non-blind AI visual inspection**, not human annotation or an overall accuracy score. A separate example requesting four balloons shows five at all three settings; reallocating computation did not resolve every observed error.

The inspected examples do not support a simple expectation that deeper loops always give more correct images. Compare measured time with constraint gains and losses. Inspect [all 64 pairs](publication/IMAGES.md) and the [detailed analysis](REPORT.md#s7).

<a id="run"></a>
## Inspect directly

The [image comparison document](publication/IMAGES.md) is readable directly on GitHub. After downloading the repository, open `demo/viewer.en.html` in a browser to select results or `demo/annotation.en.html` to annotate without setting labels. GitHub displays HTML source rather than running the viewer.

For model-free checks, run from the Case 012 directory:

```bash
python3 -m venv ../../.venv-case012-cpu
source ../../.venv-case012-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
python analysis/audit.py
python publication/verify.py
```

These check saved PNGs, time, input pairing and original-file preservation without executing a model. [CPU reproduction and annotation](REPRODUCTION.md) · [Original PNG gallery](demo/gallery.en.html) · [Frozen settings](configs/main_settings.json)

<a id="sources"></a>
## Sources and scope

**Stack:** PyTorch · Transformers · NumPy · Pillow · HTML/CSS/JavaScript. OpenSenseNova provides Looped-DiT; DIOVA implements the paired-noise runner, measured-time selection, annotation and result inspection. [Attribution](NOTICE.md)

The same checkpoint trained at four loops runs at Loops 1/2/4. This is neither a comparison of three separately trained models nor a new image-generation algorithm. The 16 prompts cover count, color and left/right compositions on white backgrounds, separately from the full official GenEval benchmark. [Model identity](provenance/models.json) · [Original and public records](publication/README.md)
