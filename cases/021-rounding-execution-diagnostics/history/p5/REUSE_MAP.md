# Implementation boundaries and retained history

| Reused | New P5 work |
|---|---|
| Existing D16 .rpi4 artifacts, original Case021 layout reader | Bounded coefficient selection, exact single-term arithmetic hypotheses and nested masks |
| Original Dense/Packed modules via P4 explicit-path runtime helper | Projection-only orchestration, identity/control checks, first-block retention and repeat hashes |
| Existing Marlin, Torch GEMM/allocator/profiler, NumPy | CPU energy/ULP/candidate/oracle audits and process accounting |
| P4 serialization/schema and supervisor design | Supervisor ending telemetry is included in the P5 debit; P4 historical wall is unchanged |
| Matplotlib | Three scientific figures and bilingual explanation |

Shared serialization/layout readers are not counted as independent algorithms. Fraction/int arithmetic and math.fsum reductions provide distinct numerical checks. Same-code replay verifies reproducibility, not independent authorship or external review. Codex assisted implementation, analysis and documentation; no new codec/kernel/solver was developed.

The first source versions and failed receipts remain in the local root. Public amendment records retain the observed errors:

- [Scalar uint32 shift](records/fixture_failure_amendment.json): NumPy1.26 promoted the scalar shift result; explicit uint32 fixed decoding before derived P5 results/GPU output.
- [Numeric signed zero](records/selection_zero_amendment.json): Fraction canonical zero differed in sign bit from negative zero. Numeric ±0 equality was already the intended contract; no nonzero difference or looser tolerance was accepted.
- [Outer command mapping](records/orchestration_attempt.json): dictionary unpacking failed before the supervisor; explicit keys fixed orchestration without scientific-source changes.
- A foreign compute PID initially blocked C before worker creation. One bounded, charged availability wait preceded a fresh successful gate. Original rejection and all telemetry remain private; sanitized accounting is public.

Frozen scientific sources are unchanged after execution. New public CPU/review/replay/document wrappers form the publication layer. Initial source files are preserved, but current commands use arithmetic_v2, supervise_v3 and test_contracts_v3. This local candidate is not a publication commit.
