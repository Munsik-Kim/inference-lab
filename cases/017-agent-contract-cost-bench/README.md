# Case017 — Agent Contract–Cost Bench

We tested an AI agent on synthetic records: calculating and saving order totals, transferring inventory, and updating document metadata. The benchmark checks the saved state, the validation results the agent observed, and its final answer. It compares output constraints and one retry after a contract error.

We implemented the strict action parser and runner, isolated synthetic records, pre-generation observation tracking, and an auditor that replays saved results.

Qwen3-4B weights stay BF16 in both configurations. The KV cache stores intermediate attention values for preceding text. BF16 KV (K0) is the 16-bit baseline; native NVFP4 KV (K1) stores packed four-bit values with scales.

Free output (C0) gives format instructions in the prompt and stops at the first contract error. Grammar (G) applies a public JSON schema to every response. Neutral retry (N) and error retry (E) each offer one retry after the first error, using a neutral message or the public error code, path and expected type. The schema supplies neither correct values nor the next action.

| KV storage | C0 | G | N | E |
| --- | --- | --- | --- | --- |
| BF16 KV (K0) | 16/24 | 16/24 | 16/24 | 16/24 |
| NVFP4 KV (K1) | 7/24 | 15/24 | 9/24 | 13/24 |

With BF16 weights and native NVFP4 KV, applying a public JSON schema to every response increased verified completion from 7/24 to 15/24 on newly generated synthetic tasks: 8 recoveries and 0 losses relative to free output. Correct saved states increased from 7/24 to 22/24. A correct state without observed validation of the updated state did not count as verified completion. The completion difference is 33.33 percentage points (pp), with an exploratory 95% interval of [16.67, 50.00]pp.

**Protocol deviation: BF16 KV with free output completed 5/8 DEV tasks, below the original 6/8 expansion criterion. The local plan and execution gate omitted this success threshold and proceeded to P2 after checking completion, freezing and sealing. No separate waiver approval was found. We disclose this deviation and present P2 as an exploratory measurement on new synthetic tasks.** [PROTOCOL_DEVIATION.md](PROTOCOL_DEVIATION.md)

More tasks completed, while total execution costs also increased. Dividing all 24 runs, including failures, by completions gives approximately 3.62% more input tokens, 12.36% more output tokens and 11.27% more episode-wall time per completion.

[REPORT.md](REPORT.md) · [REPRODUCE.md](REPRODUCE.md) · [Runner](src/agent_contract_cost/runner.py) · [Full ledger](publication/summary/episode_ledger.csv)

All three workflows use owned synthetic records. Product totals use integer minor units. No real orders, payments or company accounts are connected. Validation timing, SKU and document-status errors remain under G. BF16 completion counts were unchanged, while correct-state counts rose from 21 under C0 to 23 under G.

**Reproduction scope:** all 256 complete saved responses, SSE streams and tool events are retained losslessly. Python standard-library commands replay them into new fixtures and recalculate endpoints and paired tables. This is not a reproduction of model inference or a quality-equivalence test. P3 is NOT_RUN. The [original entry documents](publication/originals/pre-publication/README.md) remain preserved.
