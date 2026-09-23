# Case010 v2 — Restarting a recurrent stream without losing its failure

[한국어](README.ko.md)

Implemented a recurrent-state codec that preserves numerical failure, cursor and RNG state across a fresh-process restart. The same three trained checkpoints were evaluated on new inputs to compare packed storage budgets with the length of continuously correct symbolic readout.

[Full report](REPORT.md) · [Audit and reproduce](REPRODUCTION.md) · [Frozen design](protocol_v2.json) · [State implementation](source/online_v2.py)

- Execution contract: finite, terminal and stochastic cache restarts are tested. A wrong symbolic label does not terminate a stream.
- Storage: every arm adds 9 B of failure metadata per stream. At N128, rank2 uses 292,633 B and mixed 5/6 uses 292,469 B.
- Fresh inputs: 1024 sequences × 2048 tokens per checkpoint, five frozen arms. Rank2 has a larger supported 5%-risk horizon than mixed in 1/3 checkpoints. Its paired mean consecutive-correct length differences are 9.27 / -20.76 / 4.91 tokens.
- Precision diagnostic: the four FP32/FP64 paths make identical predictions and first failures on 32 separate diagnostic sequences per checkpoint.

![Fresh first-error survival](figures/fresh_survival.png)

Persisting failure improves the execution contract. A memory-horizon improvement is a separate [equal-budget result](REPORT.md#results). No new training, GPU run or remote publication occurred.

Attribution and Codex assistance: [NOTICE](NOTICE.md).
