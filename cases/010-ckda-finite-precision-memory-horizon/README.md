# Case 010 — Finite-Precision Recurrent Memory: Storage, Restart, and Readout Horizon

[한국어](README.ko.md) · [DIOVA home](../../README.md) · [Report](REPORT.md) · [CPU verification](REPRODUCTION.md)

Built tools that store recurrent state in real low-bit payloads and resume in a new process while preserving numerical failure and random-number state. The study compares how long a fixed model reads the correct symbolic state at a given storage cap, alongside arithmetic precision, state evolution and CPU cost.

## What you can inspect

| Capability | Implementation and checks |
| --- | --- |
| Pack state codes, scales and RNG into actual bytes | [PackedCodec](versions/v2/source/v1_reference/codec/packed.py) · [packing tests](versions/v1/tests/) |
| Transport online residuals and compare a mixed-precision budget | [OnlineAdapter](versions/v2/source/online_v2.py) · [CAL-based arm construction](versions/v2/source/arms.py) |
| Save cursor, terminal reason and first numerical-failure write; keep the last committed body and RNG | [Failure-aware adapter](versions/v2/source/online_v2.py) · [contract tests](versions/v2/tests/test_online_v2.py) |
| Resume the same suffix in a new process | [Evaluation checkpoint](versions/v2/source/checkpoint.py) · [fresh-process test child](versions/v2/tests/restart_online_v2_child.py) |
| Recalculate first error, mean consecutive-correct length and risk bounds | [Metrics](versions/v2/source/metrics.py) · [independent scalar auditor](versions/v2/analysis/audit_v2.py) |
| Diagnose fixed-coefficient FP32/FP64 arithmetic and time-varying state | [Precision paths](versions/v2/source/precision.py) · [retained diagnostic scalars](versions/v2/results/diagnostic-summary/) |

**Implementation:** Python, NumPy and CPU PyTorch; actual packed byte buffers, checksummed records and Matplotlib figures. The learned experiment uses one small S3 Complex KDA layer, 12 heads and the same three trained checkpoints throughout both stages.

## Recorded observations

- **Serialized state per stream: 12,305 → 3,137 bytes** from native FP32 to INT8, a **74.5% reduction**. The mean consecutive-correct length differed by less than one token in point estimates for each checkpoint. This is a cache-size and observed-lifetime comparison, not a whole-model RAM/VRAM measurement or equivalence test. [Storage and native comparison](REPORT.md#native-int8).
- **Failure survives restart.** The v2 implementation retains terminal state, cursor and RNG across call and process boundaries. Its historical evidence includes 19 comparison cells and four immediate post-failure continuations. [Restart contract and evidence](REPORT.md#restart).
- **Equal-budget results depend on the checkpoint.** Rank2 minus Mixed5/6 changes mean consecutive-correct length by +9.27 / −20.76 / +4.91 tokens. Their N=128 totals are 292,633 and 292,469 bytes. [All five settings, intervals and CPU costs](REPORT.md#fresh).

## One project, two research stages

**Stage A / v1 — explore storage methods and strengthen the comparison menu.** Real packing, transported residuals, uniform/mixed baselines and first-error analysis were evaluated on 512 inputs per checkpoint. A supplementary same-TEST budget audit adds four feasible mixed allocations.

**Stage B / v2 — preserve failure and test on fresh inputs.** Failure-aware serialization fixes a restart boundary defect. The same three checkpoints then evaluate five fixed settings on 1,024 new inputs each. The implementation repair and the memory-horizon comparison are separate results.

[v2 current execution source](versions/v2/source/online_v2.py) · [Stage A original report](versions/v1/REPORT.md) · [Stage B original report](versions/v2/REPORT.md) · [Version map](VERSION_MAP.md)

## Run a CPU check

From this case directory, use Python with NumPy already available:

```bash
python -B scripts/verify_unified.py
python -B scripts/check_cpu.py --output /PATH/TO/NEW_CASE010_CHECK
```

The second command restores both historical layouts outside the case, audits stored results and runs synthetic restart contracts. It does not load a trained checkpoint or run model inference. [Requirements, commands and audit boundaries](REPRODUCTION.md).

[Integrated report](REPORT.md) · [Derived summary](summary/project.json) · [Source inventory](provenance/snapshot_manifest.json) · [Portfolio](PORTFOLIO.md) · [Attribution and license](NOTICE.md)

Complex KDA supplies the model, task and original readout. DIOVA implements the storage adapters, restart contracts, controlled comparisons and audit tooling; Codex assisted implementation and documentation. Original notices remain with each snapshot.
