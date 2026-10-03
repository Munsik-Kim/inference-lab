# Case 012 — Completed human review

The user completed the image checklist. One hundred images were directly checked; answers from those images were explicitly reused for 92 visually similar images of the same request. All 192 images have assigned answers.

## Observed results

| Setting | All constraints met | Rate | Median complete request |
|---|---:|---:|---:|
| L1 / S89 | 52/64 | 81.25% | 4.534 s |
| L2 / S66 | 51/64 | 79.69% | 4.385 s |
| L4 / S50 | 53/64 | 82.81% | 4.762 s |

L4 gained all-constraint passes in 4 inputs and lost them in 3, for a net difference of one image (1.5625 percentage points). Both settings passed 49 inputs; neither passed 8.

L2 had the shortest recorded median time. L4 had one more pass than L1 and two more than L2 in this grouped review. The measured times differ; this is not an exact equal-time comparison.

## Which requests were difficult?

| Requested condition | L1 | L2 | L4 |
|---|---:|---:|---:|
| Count | 13/16 | 12/16 | 13/16 |
| Object-color binding | 15/16 | 14/16 | 15/16 |
| Left/right | 16/16 | 16/16 | 16/16 |
| Combined constraints | 8/16 | 9/16 | 9/16 |

The combined requests were hardest: all constraints were met in roughly half of their images. This checklist scores count, requested colors and spatial relations. It does not separately score realism or the natural shape of identifiable objects.

## How these answers were counted

Of 480 constraint answers, 254 were direct and 226 were reused. Their image/group/source identities are preserved in annotations.json. Visual grouping is a workload convenience, not a claim that different PNGs are identical.
Uncertain answers count as unmet: three constraint answers were uncertain. If uncertain answers were instead accepted, passes would be 52/64, 53/64 and 54/64. This is a sensitivity calculation, not a replacement assessment.
The 192 assigned labels are not 192 independent human judgments. No confidence interval or significance claim is attached to reused judgments. Sixteen prompts, each with four noise seeds, were evaluated under three settings. The reviewer had access to the previously published site; independent full blinding is not claimed.

## Comparison with the preserved AI assessment

The original AI passes remain 51/64, 53/64 and 53/64. Human and AI pass judgments differ on 11/192 images and 19/480 constraint answers. These are disagreements, not a ground-truth test of either assessor.

## Check the records without models

```bash
python -B tools/showcase/case012_human.py --output ../case012-human-audit.json
```

Run from a repository clone with the existing CPU requirements installed. The output must be a new file outside the repository. The auditor reads recorded answers, image IDs, timing and origins; no generation or model download is performed.

[Annotations](annotations.json) · [Summary](summary.json) · [Manifest](manifest.json)
[Original study](../../../cases/012-looped-dit-inference-budget/README.md)

DIOVA added the assessment export, provenance checks and scalar comparison tools. OpenAI Codex assisted implementation and analysis. The ratings themselves are user-provided; no additional AI ratings were substituted.
