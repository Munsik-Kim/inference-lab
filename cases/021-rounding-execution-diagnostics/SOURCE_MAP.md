# Sources and evidence

| Source | Relationship to Case021 | Evidence |
|---|---|---|
| [Ke, Cai and Wang, arXiv2610.32951v1](https://arxiv.org/abs/2610.32951v1), §5.1 / Figure2 / §6 / Table1 | Motivation for Q/K norm–RoPE rounding boundaries and static discrepancy. Original source-reading scope is retained; this publication does not claim a new PDF rendering check. No RL reproduction. | [Original source-reading map](history/p0p3/SOURCE_MAP.md.txt) |
| [Case013 at fixed base](https://github.com/Munsik-Kim/inference-lab/tree/bb2ab39ba5e9f6cea7e4c7d122fa6b26e143e688/cases/013-dualquant-reference-quality) | D16 W4A16 artifact, original dense order, pre-existing matched-scale diagnostic and failed old 2% gate | [Original report snapshot](history/p0p3/REPORT.md.txt) |
| [Qwen3-0.6B revision](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | Base checkpoint; 196 projections, group64. No checkpoint/weights in this package. | [Environment](evidence/p0p3/environment.json), [execution map](evidence/p0p3/execution_map.json) |
| Installed Transformers5.17.0 / vLLM0.29.0 / Torch2.13.0+cu130 | Native RMSNorm/rotary/SDPA, actual Marlin repack/GEMM/kernel, tensor operations | [Execution source](execution/p0p3/source/), [P4 kernel evidence](evidence/p4/kernel_evidence.json) |
| NumPy / Python Fraction, math.fsum / Matplotlib | Distinct CPU arithmetic/reduction checks and figures; no claim of independent human review | [Public CPU checks](history/p5/source/public_cpu.py), [new aggregate derivation](publication/derive.py) |

P0–P3 GPU source bundle: `5876ea3c7a89ae77eda9f237059dc6101feeea834fe00df47ca9f715122841e7`. P4 [AB](evidence/p4/freeze_AB.json), [ABv2](evidence/p4/freeze_AB_v2.json), [C](evidence/p4/freeze_C_final.json) and P5 [facts/identities](history/p5/facts.json) retain separate freezes. Git base `bb2ab39ba5e9f6cea7e4c7d122fa6b26e143e688` did not contain this new publication tree. The publication commit and eventual merge are separate Git objects.

New public facts derive from saved per-position/document/local/operator aggregates. The full private logits/activation audit is evidenced by earlier receipts, not rerun or replaced with saved PASS flags. [Mapping](publication/preservation_map.json) records exact copies, path-redacted copies and synthetic integer-ID conversions. [Current facts](summary/facts.json) link numerical scopes; [history](HISTORY.md) explains corrections and omissions.
