# Case021 — Why can the same INT4 artifact produce different results?

[한국어](README.ko.md) · [Detailed report](REPORT.md) · [CPU checks](REPRODUCE.md)

The same compressed numbers can produce different final values when rounding boundaries or execution paths change. Using an existing Qwen3-0.6B INT4 artifact—a saved bundle of integer codes and scales—we separated **reconstruction format, dense/Marlin execution and the precision of downstream Q/K calculations**.

The tested single terms agreed. Combining actual activations, the values passed between model calculations, left small differences. Higher intermediate precision did not consistently bring the two final output distributions closer.

| Path | Meaning |
| --- | --- |
| D0 | Dense artifact reference reconstructed in the original arithmetic order |
| D1 | Preconstruct BF16 weights from codes and BF16 scales, then use dense GEMM |
| P | Supply the same packed codes/scales to the Marlin GPU operation |
| R0 /R1 | Native shared Q/K norm–RoPE rounding /FP32 intermediates until final BF16 Q/K |

- **Full model:** the original24 EVAL documents at length512 did not establish that R1 reduces KL. KL measures distribution disagreement, not answer accuracy. The verdict remains **INCONCLUSIVE**.
- **Single terms:** D1/P agreed in two q_proj projections. Both matched weight-first rounding candidate A at406,626 discriminating first-block coordinates. These are fixture/row/output records, not input-output pairs deduplicated across repeated coefficients in different fixtures.
- **Multiple terms:** four fixed layer/document combinations agreed at K1/2. The first observed difference was at K8 for one and K64 for three. K is a retained-axis budget; input magnitude also changes, so this is not a universal threshold.

![Local execution and original full-model discrepancy](figures/01_overview.png)

Left: local projection mismatches as more terms are retained. Right: one line per original EVAL document, averaging16 positions before weighting documents equally. These panels have different error definitions and denominators.

The case contributes fixtures, adapters, measurement, exact binary-rounding/CPU audits and explanation. Marlin, Torch GEMM/allocation and the existing quantized artifact are reused. Internal accumulation order, partial sums and scale placement remain unisolated.

[Methods and terms](METHOD_CONTRACT.md) · [All key tables and counterexamples](REPORT.md) · [Public/private reproduction scopes](REPRODUCE.md) · [Sources and implementation boundaries](SOURCE_MAP.md)
