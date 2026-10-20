# Case016 — NVFP4 KV Cache on RTX5080: Storage, Correctness, and Serving Performance

Built tools integrating vLLM0.32.0 and FlashInfer native compressed KV to verify storage/read correctness, fixed-pool capacity and concurrent-request costs. Qwen3-4B weights remain BF16.

At the same approximately4GiB pool, B16 has29,120 slots and NVFP4 has103,552, a token-capacity ratio of 3.5560×. Serving gains were observed in every registered16K/C2 and C4 paired round.

**16K/C2: 1.21× [1.21, 1.21] · 16K/C4: 1.35× [1.35, 1.36]**

Ratios are successful HTTP requests/s for F4/B16, with three-round median and range. They are not total VRAM reductions, agent-success gains or100K single-context support.

[Report](REPORT.md) · [CPU / GPU reproduction](REPRODUCE.md) · [Runner](followups/nvfp4-technical-r1/src/nvfp4_technical/runner.py) · [Raw / CSV](followups/nvfp4-technical-r1/results/) · [Stage map](STAGE_MAP.md)

| KV | bytes/token | actual pool bytes | blocks ×16 | token slots |
|---|---|---|---|---|
| B16 | 147456 | 4293918720 | 1820 | 29120 |
| F4 | 41472 | 4294508544 | 6472 | 103552 |

| Input | C | B16 req/s | F4 req/s | F4/B16 median [3-round range] | success B16 / F4 |
|---|---|---|---|---|---|
| 8K | 1 | 0.396 | 0.397 | 1.001 [1.000, 1.003] | 24/24 / 24/24 |
| 8K | 2 | 0.563 | 0.561 | 0.996 [0.992, 1.000] | 24/24 / 24/24 |
| 8K | 4 | 0.630 | 0.712 | 1.125 [1.125, 1.131] | 24/24 / 24/24 |
| 16K | 1 | 0.251 | 0.247 | 0.986 [0.985, 0.991] | 24/24 / 24/24 |
| 16K | 2 | 0.251 | 0.303 | 1.209 [1.205, 1.211] | 24/24 / 24/24 |
| 16K | 4 | 0.251 | 0.340 | 1.351 [1.350, 1.357] | 24/24 / 24/24 |

Terms: a KV cache stores intermediate values for text already read by the model. BF16 is the16-bit baseline; NVFP4 stores four-bit values with block scales. C is concurrent client requests; a token is a text unit read by the model. Higher req/s means more successful requests per second.

The earlier agent DEV retains B16 0/48 and F4 6/48, repeating12 tasks. The NOT_MET baseline gate,96 original verdicts and CPU audit are preserved in the [stage map](STAGE_MAP.md). Agent P2/P3 were not run. Bilingual Pages build integration is prepared; manual deployment is outside this execution.
