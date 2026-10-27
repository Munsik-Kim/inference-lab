# Reuse and direct work

| Reused | Direct work |
|---|---|
| Case013 D16 layout/arithmetic and matched-scale idea | Explicit CLI adapters; controlled D0/D1/P comparisons and block/local replay |
| HF Qwen native norm/rotary/attention, Torch operations | Scoped R1 intermediate-precision adapter, no-op checks and fixed-position logging |
| vLLM Marlin repack/scale permutation/GEMM | Backend identity checks, core input contracts and bounded projection fixtures |
| NumPy, math.fsum, Fraction, Matplotlib | FP64/reference formulas, BF16 rounding/ULP tests, energy decomposition, statistical/figure tools |
| Frozen P5 public CPU checker | New publication manifest/link/privacy checks, original/P4 bootstrap reaggregation and bilingual display |

No codec, kernel, allocator, quantizer or training solver was developed here. No performance or quality optimization is claimed. The existing matched-scale option is not a new method. Shared readers/serialization are listed rather than counted as independent algorithms. Exact scalar arithmetic and separately written reductions are numerical cross-checks, not external review or another-machine reproduction.

Codex assisted implementation, analysis, checks and bilingual documentation. Current wrappers operate only on public aggregates and synthetic inputs. Historical GPU source is retained for inspection, with declared private-path redactions where needed; it is not imported by public CPU entry points. [Mapping](publication/preservation_map.json) retains both source and target hashes. The original P5 public bundle and its historical manifest remain byte-exact under [history/p5](history/p5/).
