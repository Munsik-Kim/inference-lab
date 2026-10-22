# Case018 — State Recovery and Final-Report Consistency

We built a small tool environment for recovering wrong-target inventory changes, duplicate transfers and partial debits. The evaluator checks the terminal state, validation actually observed by the model, final quantities and two fixed CPU continuations. These continuations are evaluator programs, not autonomous follow-up work by the model.

On24 new synthetic HOLDOUT tasks, both policies scored Q24/W0 with BF16 KV and Q19/W5 with native NVFP4 KV. The freshness guard blocked15 ineligible final attempts across five NVFP4 episodes but recovered no Q/W successes; output cost increased18.69%. Weights were BF16 in both configurations.

| KV | Policy | Q /24 | W /24 | Requests | Input tokens | Output tokens | Guard rejections | Episode wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | R0 | 24 | 0 | 168 | 323375 | 10953 | 0 | 164.144812115 |
| BF16 | R1 | 24 | 0 | 168 | 323375 | 10953 | 0 | 164.157640174 |
| NVFP4 | R0 | 19 | 5 | 186 | 369881 | 11568 | 0 | 179.978763365 |
| NVFP4 | R1 | 19 | 5 | 206 | 439813 | 13730 | 15 | 213.922486411 |

Q checks business restoration, protection throughout recovery and the two fixed continuations. W additionally requires an accepted, correct final answer and observed current successful validation. This is exploratory measurement on a small synthetic sample; metric rankings differ. It does not establish KV quality equivalence, serving speed or internal memory mechanisms.

Posthoc analysis confirmed that all76 Q1/W0 episodes reported their first observed quantities. Among91 accepted finals, all13 with observed current record quantities matched the terminal quantity map; all78 without them differed. The13 include three accurate reports of incorrectly recovered states, so reporting consistency is not successful recovery. Five guard-limited episodes have no accepted-final comparison. This is an association, not a forced-reread intervention.

[Results and method](REPORT.md) · [Model-free CPU replay](REPRODUCE.md) · [Posthoc definition](publication/analysis/DEFINITION.json) · [120-row CSV](publication/summary/episode_ledger.csv) · [NOTICE](NOTICE.md)

Direct contributions: synthetic environment, strict parser, completion guard, observation tracking, independent state replay and diagnostic/publication tools. Qwen, vLLM/FlashInfer KV kernels and xgrammar are upstream. P0 is CPU contract testing, P1 is DEV24, P2 is HOLDOUT96. This publication added zero GPU/model requests.
