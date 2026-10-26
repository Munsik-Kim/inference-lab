# Case020 P4 — Compact memory, measured call and setup costs

**How much resident memory does request-time reconstruction save, and when does it cost latency?**

In the original LARGE/Z32 study, B retained548,928 tensor bytes versus A's268,435,456-byte reconstructed table. In this follow-up, C was faster than B in0/30 shared-context H blocks. Even Z644 with repeated IDs gave a1.142 C/B median ratio. Actual sessions had mixed A/B ordering at N128; A was faster in all three blocks at N1024 and4096.

PLE means the extra embedding a token uses at each layer. A **codebook** is a small vector dictionary; a **decoder** reconstructs embeddings with an affine calculation; **resident** tensors remain loaded; **workspace** is working memory used by an operator library.

Read the [integrated report](INTEGRATED_REPORT.md) for execution choices, the [P4 report](REPORT.md) for block evidence, and [CPU reproduction](REPRODUCE.md) for raw→tables→figures without a model or GPU. [한국어](README.ko.md).

- A: lookup in a BF16 table reconstructed from the same compressed representation.
- B: the unchanged official decoder, reconstructing every requested row.
- C: per-call unique→decode U rows→inverse gather, returning M rows.
- S: sort→decode M rows→restore order, only in one duplicate-free core condition; it is not deduplication.

P4 is **COMPLETE**:30/30 timing workers,96/96 method-blocks,27/27 lifecycle workers, plus3 preflights,1 smoke and3 memory workers. No worker failed, retried or remained missing. All747 original Case files and18 external originals were preserved. Remote CI and publishing were not run.

This directory is the entire new scope, identified by [`PUBLIC_MANIFEST_P4.json`](PUBLIC_MANIFEST_P4.json). The [original report](../../REPORT.md) and manifest remain historical records.
