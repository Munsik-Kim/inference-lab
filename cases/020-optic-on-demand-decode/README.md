# Case020 — Does Smaller Embedding Storage Also Run Faster?

[한국어](README.ko.md) · [Report](REPORT.md) · [CPU checks](REPRODUCE.md)

Some AI models keep an extra number vector for each token at each layer. A token is a small unit of text; an embedding is a vector representing it. This study asks whether to calculate the whole table ahead of time or reconstruct only the rows requested by their IDs.

We used **the official OPTIC decoder and synthetic embedding tables**, rather than loading an LLM. On an RTX 5080, all methods received the same compressed representation and IDs and returned the same number of rows across every layer. A's table is also reconstructed from that representation; it is not the original uncompressed embedding table.

| Method | What happens | Kept between calls |
|---|---|---|
| A · Precompute all | Reconstruct the same compressed representation once, then look up rows | A large reconstructed table |
| B · Compute requested rows | Call the official decoder for all M requested rows | Small codes, codebooks and decoder values |
| C · Compute distinct rows, then restore order | Find duplicates on every call, decode U rows and return M rows | The same compact state as B |


**Less memory did not automatically mean a cheaper call.**

- **Memory:** for LARGE/Z32/M2048/U512, A's roughly 304 MiB became roughly 81 MiB for B: about 73% less **PyTorch allocated peak**, not total device VRAM.
- **Individual calls:** P4 kept the methods in one process and alternated their order. Across7 conditions and 30 process-block comparisons, C was faster than B in **0/30**. A block is a fresh-process repetition, not an independent user workload.
- **Reuse:** measuring setup, N real calls and cleanup together, A beat B in all three blocks at N1024 and 4096. Session ordering was mixed at N128. The whole process, including Python startup and common CUDA initialization, favored B in 3/3 N128 blocks, but its median advantage was only about 17 ms.

![Memory costs for the same output](figures/01_memory.png)

The left panel shows tensors kept on the GPU between calls; the right shows the peak recorded allocation during calls. Black markers are three process blocks; identical values overlap. C needed a smaller allocation peak than B here, while its calls remained slower. Reserved memory, which includes space held for reuse, is reported separately.

## What did a call cost?

Each entry is the median of the within-block host medians, in µs (millionths of a second). Calls alternate within the same process, and the timer includes waiting for completion.

| Z / M / U | A | B | C |
|---|---|---|---|
| 32/1/1 | 26.50 | 98.25 | 230.02 |
| 32/2048/512 | 40.23 | 151.52 | 264.42 |
| 644/2048/512 | 54.04 | 330.87 | 431.95 |


The first two conditions have 6 blocks, the last3. Each block measures120 calls per method. The [direct-time CSV](summary/representative_direct_summary.csv) and [report](REPORT.md) retain p95, every block and ranges. A median of paired C/B block ratios can differ from dividing the C and B medians in this table.

## Which option should I consider?

**Consider B when keeping a large GPU table is impractical.** It retains compact state and pays reconstruction cost for each call. **Consider A when the same table will be reused long enough.** Its initial preparation can be spread over many lookups. The measured median session ordering changed between128 and 1024 calls; we did not invent an exact operational crossover.

**Evaluate C's memory and total time together.** Duplicate IDs reduce reconstructed rows but add duplicate finding and order restoration. With a wider intermediate vector Z644, repeated IDs still gave a1.142 median paired C/B ratio. The historical489-fold resident-tensor ratio applies only to Z32; at Z644 it was about 24.65-fold.

The work implemented here includes fixtures, comparison wrappers, measurement, independent NumPy checks and public reproduction tools. OPTIC decoding and Torch matrix multiplication/memory management are reused. Codex assisted implementation, analysis and documentation. The [reuse map](REUSE_MAP.md) links these boundaries to code.

## Run and inspect

Follow [clone → new CPU environment → quick/full checks](REPRODUCE.md), without a model or GPU. Original source, inputs, both P2 runs, P4 raw and historical documents are kept in the [exact-byte originals](publication/originals/pre-publication/README.md) and [preservation map](publication/preservation_map.json). This publication ran no new GPU work.

[Report and validation scope](REPORT.md) · [New summary CSVs](summary/) · [Four figures](figures/) · [History](publication/HISTORY.md) · [Sources](SOURCE_MAP.md)
