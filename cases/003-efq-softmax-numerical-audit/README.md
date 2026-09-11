# EFQ-Softmax on real Qwen attention traces

Can the published EFQ-Softmax operating points replace exp-then-nearest probability generation without materially increasing local attention-output error?

**EFQ-Mean had 11.49–14.49% median attention-output error on these Qwen3-0.6B traces. The calibrated/MMLU setting was close to same-scale nearest rounding in the aggregate, while headroom-scale nearest rounding had lower median and p95 error at every length. This was an FP32 numerical simulation; packed FP4 kernels, speedup and downstream quality were not measured.**

Values are **median / p95 relative output Frobenius error**, not task accuracy. Each length has 192 repeated head units (16 documents × 3 layers × 4 heads), with 16 sampled query rows per unit and all causal-valid keys per query. The independent document count is 16. Lengths share prefixes of the same documents.

| Method | 512 tokens | 2048 tokens | 4096 tokens |
|---|---:|---:|---:|
| Nearest, headroom scale | 7.14% / 14.58% | 5.76% / 9.21% | 5.67% / 8.82% |
| Nearest, EFQ scale | 8.39% / 16.18% | 6.78% / 12.80% | 6.52% / 14.86% |
| EFQ-MMLU | 8.09% / 15.18% | 6.79% / 11.69% | 6.51% / 11.41% |
| EFQ-Mean | 14.49% / 23.25% | 12.13% / 22.28% | 11.49% / 22.06% |
| EFQ-Balance | 9.99% / 18.28% | 8.08% / 12.75% | 7.92% / 13.25% |
| EFQ-LUT | 8.82% / 18.16% | 6.88% / 13.63% | 6.75% / 15.63% |
| EFQ calibrated | 8.09% / 15.18% | 6.79% / 11.69% | 6.51% / 11.41% |

The development grid selected tau=-2.90, h=2.00, exactly the published EFQ-MMLU point. Both names remain in the table to separate the public setting from the development selection.

The prespecified screen uses **Nearest, EFQ scale** as its denominator. Under the same EFQ scale, the calibrated affine code generator stayed within the recorded aggregate error-ratio limits relative to exp-then-nearest rounding. That result does not compare EFQ with every MXFP4 implementation.

Against **Nearest, headroom scale**, calibrated EFQ's median error was 1.133×, 1.179× and 1.148× as large. The headroom reference uses an independently chosen scale rule; equivalence to the paper's MXFP4 baseline was not established.

A limited microkernel feasibility follow-up could test the cost of code generation while retaining both nearest controls. This audit does not recommend EFQ on fidelity alone. No kernel implementation or timing experiment was performed.

- [Analysis: mapping, scale, calibration and clustered uncertainty](ANALYSIS.md)
- [Methods, paper-to-code classification and reproduction commands](METHODS.md)
- [Independent row-derived calculations and all reporting differences](results/comparison_decomposition.json)
- [Raw per-query measurements](results/eval_rows.csv.gz) and [per-head records](results/eval_units.jsonl)
- [Fixed experiment specification](configs/experiment_spec.json) and [post-hoc audit plan](configs/publication_audit_plan.json)
- [Model](provenance/model.json), [environment](provenance/environment.json), [inputs](inputs/manifest.json), [small fixture](tests/fixtures/qwen_dev_small.npz)

The recorded setup was RTX 5080 SM120 / WSL2, Qwen3-0.6B BF16, Transformers 5.17.0 and PyTorch 2.13.0+cu130. Trace capture uses post-QK-normalization/post-RoPE Q/K with the corresponding V; QK, V, mask and FP32 accumulation are common to all methods. Tile size is 128 and the primary microscaling block is 32.

EFQ-Softmax and its published parameters are [Han et al.'s work](https://arxiv.org/html/2609.09721v1). This project contributes an independent implementation and numerical evidence. Codex assisted with synthetic inputs, code, execution, analysis and documentation. The independent audit refers to a separate calculation path within this project, not third-party replication. See [NOTICE](NOTICE.md).
