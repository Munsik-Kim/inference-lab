# Case016 — Can a Smaller KV Cache Complete More Jobs Correctly?

A synthetic job reads several sources, chooses the required version, calculates an amount, updates a record and validates it.

Implemented real compressed-KV vLLM execution and a long-context workflow harness/auditor. On the frozen 96 DEV conditions, B16 completed 0/48 verified workflows and NVFP4 6/48. These reuse 12 fixtures under 4 menu/cap combinations. The baseline expansion gate is NOT_MET. P2/P3 are outside this run and are not executed.

A KV cache stores intermediate attention results for text already read. All model weights stay BF16; only KV storage changes. Money uses integer minor units: quantity × unit price with a stated integer discount rule.

Implemented: long evidence/tool menus → raw SSE/token capture → isolated tool execution → state/observation/final-answer scoring → CPU replay on fresh fixtures.

| KV | menu | cap | workflow | state | final | false done |
|---|---|---|---|---|---|---|
| B16 | 8 | 256 | 0/12 | 2/12 | 0/12 | 4 |
| B16 | 8 | 1024 | 0/12 | 2/12 | 0/12 | 4 |
| B16 | 32 | 256 | 0/12 | 4/12 | 0/12 | 2 |
| B16 | 32 | 1024 | 0/12 | 4/12 | 0/12 | 2 |
| F4 | 8 | 256 | 0/12 | 2/12 | 0/12 | 3 |
| F4 | 8 | 1024 | 0/12 | 2/12 | 0/12 | 3 |
| F4 | 32 | 256 | 3/12 | 5/12 | 5/12 | 1 |
| F4 | 32 | 1024 | 3/12 | 5/12 | 5/12 | 1 |

[Result explorer](demo/en.html) · [Full report](REPORT.md) · [CPU reproduction](REPRODUCE.md) · [Runner](src/kv_agent_bench/episode_runner.py) · [Raw records](results/p1_r1/)

Each cell contains12 observed fixtures. Correct file state and a complete workflow with current observed validation are separate endpoints. This is a local review, not a GitHub/Pages publication.
