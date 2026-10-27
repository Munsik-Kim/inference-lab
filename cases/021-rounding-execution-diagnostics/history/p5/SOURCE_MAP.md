# Sources and evidence

| Question | Evidence / implementation | Scope |
|---|---|---|
| Original D0/D1/P, R0/R1 and full-model verdict | Original Case021 METHOD_CONTRACT, runtime/modes, original decision; [hash references](records/historical_references.json) | Original evidence, not new P5 full-model computation |
| Existing local energy components and P4 actual output bits | Immutable P4 outputs/p4b_v1 and records/gpu-q{0,25,27}-b0 | P5A reaggregation, original posthoc DEV/EVAL labels retained |
| Candidate A/B and direct BF16 rounding | [arithmetic_v2](source/arithmetic_v2.py), independent Fraction path, [tests](tests/test_contracts_v3.py) | New diagnostic fixture and numerical audit |
| Same artifact / real Marlin core | Original Case021 runtime and P4 gpu_worker explicit runtime adapter; two q_proj artifacts | Codes/scales/D1 weight hashes and beta/layout checks; no quantization or model loading |
| Actual backend | [kernel evidence](records/kernel_evidence.json), installed vLLM0.29.0 and Torch2.13.0 | Existing CUDA Marlin symbols, no silent fallback |
| Timing and memory | [ledger](records/ledger.json), [memory](records/memory.json) | Parent accounting and bounded projection memory, not a speed benchmark |
| Freeze and public layout | [identity](records/identity.json), [GPU freeze index](records/freeze_GPU.json), [mapping](records/public_mapping.json) | Execution bundle, Git base and publication layout are separate identities |

The original paper connection is inherited from Case021 SOURCE_MAP: [arXiv2610.32951v1](https://arxiv.org/abs/2610.32951v1), §5.1 and §6, Figure2/Table1. P5 uses the existing rounding-boundary/static-discrepancy motivation and does not freshly reproduce or reinterpret its RL experiments. Original Case013's matched-scale probe predates this study; see [the fixed Case013 source](https://github.com/Munsik-Kim/inference-lab/tree/bb2ab39ba5e9f6cea7e4c7d122fa6b26e143e688/cases/013-dualquant-reference-quality). No original PDF/model/dataset is redistributed.

Local originals are connected through explicit references/locations.json in the private P5 root. Public historical references expose hashes, not private user paths. Hash-only references do not make private evidence downloadable. Original reports and failed attempts remain unchanged at those roots.
