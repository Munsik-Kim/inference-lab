# Case021 — Why Can the Same INT4 Weights Produce Different Results?

[한국어](README.ko.md) · [Report](REPORT.md) · [CPU verification](REPRODUCE.md)

**The same stored four-bit values produced different outputs when reconstruction arithmetic and GPU execution changed.** Keeping Q/K norm–RoPE intermediates in FP32 did not consistently bring the two execution paths closer together.

Weights are the numbers used by a model to calculate its outputs. INT4 stores them as four-bit integers; scales convert those integers back to the magnitude used in computation. This study reuses the existing D16 artifact for Qwen3-0.6B. It performs no new quantization or training.

| Path | What it does |
|---|---|
| D0 | Reconstruct weights in the original arithmetic order, then run dense linear operations |
| D1 | Reconstruct using the BF16 combined scales supplied to Marlin, then run dense operations |
| P | Supply the same integers and scales to the actual packed Marlin operator |

D0–D1 measures a change in reconstruction arithmetic and scale representation. D1–P measures differences that remain after matching the scale format. D1 was already a diagnostic in Case013; it is not a newly invented quantization method.

Before attention compares tokens, Q/K norm–RoPE normalizes query/key vectors and incorporates position. **R0** follows the installed HF implementation. **R1** keeps only this region's intermediates in FP32. Both still supply BF16 Q/K to the same attention backend.

The main result uses 24 EVAL documents with 512-token inputs. Each document contributes 16 fixed positions, with the full vocabulary retained at each position. KL measures a difference between probability distributions. Lower KL here means **closer agreement between execution paths**, not better model quality.

| Observation | Result |
|---|---:|
| Mean D1–P KL, R0 → R1 | 0.00069525 → 0.00078153 |
| Documents with reduced / increased discrepancy | 9 / 15 |
| 95% interval for mean reduction E0−E1 | −0.00020682 to +0.00003909 |

The interval includes zero, so the decision is **INCONCLUSIVE**. More accurate intermediate arithmetic does not guarantee that two different execution paths will become more similar.

![Document-level R0/R1 comparison](figures/02_rounding_discrepancy.png)

Each line represents one document. The right panel uses fixed D0_R0 as its reference, which differs from the D1–P comparison on the left.

Replaying two selected layers with exactly the same input exposed both format and execution differences. Replacing the output of layer25, selected using DEV only, reduced EVAL discrepancy in 10 documents and increased it in 14. This does not establish an improvement method.

The practical lesson is to check **stored integers and scales, reconstruction arithmetic, the actual backend, and downstream computation separately**. Matching dtype names alone does not establish equivalent execution.

- [Report](REPORT.md): direct errors, same-input local replay, replacement sensitivity and measured execution cost.
- [Method contract](METHOD_CONTRACT.md): exact D0/D1/P and R0/R1 arithmetic.
- [Tables and decision](results/summary/): document, position, layer, timing and memory records with actual denominators.
- [Reproduction](REPRODUCE.md): model-free checks of the small public logit subset and full scalar tables; full KL evidence remains local.
- [Sources](SOURCE_MAP.md) and [reuse map](REUSE_MAP.md): HF/Marlin/Torch versus the new fixtures, adapter, measurement and audit code.

This is a **local publication candidate**. Remote publication and CI have not run. It is not a reproduction of RL results, a model-quality improvement, a new kernel, or a serving speedup.
