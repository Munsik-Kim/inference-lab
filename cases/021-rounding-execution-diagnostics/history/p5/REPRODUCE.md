# Case021 P5 reproduction

The public path checks **aggregate tables→statistics/figures** and small synthetic formulas. Actual activations, codes, scales and single-term outputs are excluded. It cannot reproduce private raw calculations or GPU runs by itself. This is a local candidate; remote CI and another-machine reproduction were not run.

Tested on Linux/WSL: Python3.11.15, NumPy1.26.4 and Matplotlib3.10.9, using an existing read-only CPU environment. No installation or upgrade occurred. The [dependency list](requirements-cpu.txt) identifies versions. These commands neither install packages nor load Torch, models or CUDA. Native Windows was not tested.

Run from public_candidate. Every output must be a new external path. Quick mode validates the manifest, energy identities/sign classes, candidate agreement, sparse-condition observations and synthetic exact rounding. Full mode adds three PNG/SVG figures. The test commands exercise9 contract tests and6 public path/tamper cases; reruns do not increase the unique test count.

```bash
python -B source/public_cpu.py --root . --output ../p5-quick-new --quick
python -B source/public_cpu.py --root . --output ../p5-figures-new
python -B tests/test_contracts_v3.py --output ../p5-fixtures-new
python -B tests/test_public.py --root . --output ../p5-public-tests-new
```

Quick/full replay and the tests were actually run into fresh external paths. [Verification](verification.json) and [command/output mapping](records/reproduction.json) record scope and logical output names. Exact local paths and commands remain in the private P5 worker ledger.

This is local cross-checking, not an external person's independent review. The synthetic rounding path compares integer/Fraction nearest-even against vectorized arithmetic. Public aggregate checks do not claim to reconstruct hidden private outputs. Single-term coordinates may reveal weights and therefore remain private.

The local private-data command is `source/replay_cpu.py --root P5_ROOT --output NEW_EXTERNAL`. It requires the original Case021/P4 location manifests, two current artifacts, saved Z/outputs and the existing CPU environment. It was run without GPU calls and reproduced20 table/JSON files byte-for-byte. The GPU runner is preserved as execution history; this guide provides no new GPU launch command. Authorized P5 GPU diagnostics are closed.

Initial failed sources and receipts remain available. The recommended contract test is v3 with arithmetic_v2; see [reuse/history](REUSE_MAP.md). PUBLIC_MANIFEST excludes itself and identifies this public arrangement, separately from historical execution bundles or Git commits.
