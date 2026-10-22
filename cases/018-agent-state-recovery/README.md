# Case018 — Checking inventory recovery and completion reports

We asked an AI agent to undo wrong-target inventory changes, duplicate transfers and partial debits. We checked the recovered state separately from its completion report.

We built an inventory/operation-ledger environment and auditing tools that track saved values, protection of unrelated records, validation actually received after changes, and the final answer. After recovery, the evaluator runs two fixed follow-up programs and compares them with the same programs from a clean checkpoint.

**State and follow-up checks (Q)** require restored business values, protection throughout recovery and correct results from both follow-up programs. **Validation and correct report (W)** additionally requires an accepted, correct completion response after observing successful current validation.

| KV storage | Completion policy | State and follow-up checks (Q) | Validation and correct report (W) |
| --- | --- | --- | --- |
| BF16 | Basic acceptance (R0) | 24/24 | 0/24 |
| BF16 | Current-validation guard (R1) | 24/24 | 0/24 |
| NVFP4 | Basic acceptance (R0) | 19/24 | 5/24 |
| NVFP4 | Current-validation guard (R1) | 19/24 | 5/24 |

Each row uses the same 24 synthetic tasks, producing 96 episodes in total. Qwen weights stay BF16 in both configurations. KV stores the keys and values used for attention over previous tokens.

With BF16 KV, both policies passed state and follow-up checks in 24/24 tasks, but 0/24 also met validation and correct-report requirements. The NVFP4 KV counts were 19/24 and 5/24. Agents repeatedly restored inventory while reporting the quantities they had read at the start.

The current-validation guard blocked ineligible completion responses without increasing recovered or fully verified completions. NVFP4 output tokens increased by 18.69%.

[Recorded episodes](REPORT.md#recorded-episodes) · [Results and method](REPORT.md) · [CPU record replay](REPRODUCE.md)

## Replay the records

```bash
python3 -B cases/018-agent-state-recovery/publication/run_checks.py --output /tmp/case018-public-check-new --tests
```

Use Python 3.12+ and the standard library to replay records into a fresh output directory. No model download is needed.

[CPU record replay · main](https://github.com/Munsik-Kim/inference-lab/actions/workflows/case018-checks.yml?query=branch%3Amain)

Direct work comprises the inventory/operation ledger, compensation and release APIs, strict parser, completion guard, pre-generation observation tracking, separate state/continuation audits, posthoc quantity analysis and public tools. Qwen, vLLM, FlashInfer KV kernels and xgrammar come from upstream.

[Posthoc definition](publication/analysis/DEFINITION.json) · [120-row ledger](publication/summary/episode_ledger.csv) · [Sources and licensing](NOTICE.md)
