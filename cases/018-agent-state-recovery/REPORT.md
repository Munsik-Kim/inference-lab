# Case018 — Checking inventory recovery and completion reports

## Question and implementation

We asked an AI agent to undo wrong-target inventory changes, duplicate transfers and partial debits. We checked the recovered state separately from its completion report.

We built an inventory/operation-ledger environment and auditing tools that track saved values, protection of unrelated records, validation actually received after changes, and the final answer. After recovery, the evaluator runs two fixed follow-up programs and compares them with the same programs from a clean checkpoint.

## HOLDOUT results and pairing

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

## Recorded episodes

### A00: first read, restored inventory and report

In the BF16 episode, inventory returned to 43 / 28, but the report still said 37 / 34. The table shows the two affected records; the recovery target was 43 / 28.

| Stage | BF16 KV · basic acceptance | NVFP4 KV · basic acceptance |
| --- | --- | --- |
| First read | 37 / 34 | 37 / 34 |
| Saved after repair | 43 / 28 | 43 / 28 |
| Reported quantities | 37 / 34 | 43 / 28 |

`HOLDOUT-A00_K0_R0`: first read (e000) → compensation (e003) → release (e004) → current validation valid:true (e005) → completion response. The agent received validation, yet reported the first quantities: Q=1/Fresh=1/W=0.

`HOLDOUT-A00_K1_R0`: after the same initial input and repair/release, it reread current inventory (e005), validated (e006) and reported the current quantities: W=1. Selected original excerpts retain event IDs, raw model text and pre-generation available results.

The public validator checks nonnegative inventory, protected fields, conservation and ledger references. Its response does not include all current quantities. A separate evaluator compares the reported quantity map with saved inventory. Observing successful current validation (Fresh) and rereading current quantities are therefore different observations.

[BF16 A00 excerpt](publication/summary/selected/HOLDOUT-A00_K0_R0.json) · [NVFP4 A00 excerpt](publication/summary/selected/HOLDOUT-A00_K1_R0.json)

### What the guard checked and blocked

**Check:** the agent must have received successful validation matching the current scope/digest/version before generating its completion response.

**Block:** 15 ineligible attempts across 5 NVFP4 R1 episodes were not accepted.

**Remaining failure:** the guard neither repairs state nor compares final quantities. Those states remained unrecovered; R0 and R1 had identical Q and W counts in both KV settings. System acceptance and correctness are assessed separately.

### A06/C01: accurate reports of wrong recoveries

C01 NVFP4 should restore 62 / 16, but subtracting destination stock while undoing a source-only debit left 62 / 12. R0 accurately reported that inventory but scored Q=0/Fresh=0/W=0. R1 blocked three completion attempts while the same failure remained.

A06 NVFP4 left 80 / 0 instead of 42 / 28, increasing total inventory by 10 and violating protection. Accurately reporting the ending inventory does not undo damage during recovery.

[Selected excerpts and full source pointers](publication/summary/selected/) distinguish explanatory summaries from raw text.

## Posthoc quantity observation diagnostic

This posthoc analysis examines existing records, motivated by A00 and the already-seen Q=1/W=0 cases. [Definitions, code and input hashes](publication/analysis/ANALYSIS_FREEZE.json) were saved before the new full aggregate. The table compares 91 completion responses accepted by the system.

| Current quantities reread | Report matches terminal quantities | Mismatch |
| --- | --- | --- |
| Yes | 13 | 0 |
| No | 0 | 78 |

Of the 13 matching reports, 3 **accurately described incorrectly recovered states**. The 5 episodes without an accepted final have separate NA comparisons. All 76 episodes that passed Q but failed W reported their first observed quantities. These are episodes across repeated conditions, not distinct tasks.

Current-quantity observations and accurate reports occurred together in these saved trajectories. Actions and KV configurations also differed, so measuring the effect of requiring a reread would need a separate comparison experiment.

<details>
<summary>Tables by condition and exact observation definition</summary>


Current-quantity re-observation requires successful read_records results actually delivered for every required record, with no subsequent quantity change. Failed, undelivered and partial reads do not count as full observation. Change-and-restore expires the old read; a metadata-only revision does not. The separately stored after-all-state-mutations flag coincides in this sample. Neither final correctness nor checkpoint gold enters the flag. The anchor is the accepted final’s pre-generation available event IDs, or a last-generation diagnostic without accepted final. Of five guard-limited episodes, three observed current quantities and two did not.

| KV | Policy | Accepted / none | No read · mismatch | No read · match | Read · mismatch | Read · match |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 | R0 | 24 / 0 | 24 | 0 | 0 | 0 |
| BF16 | R1 | 24 / 0 | 24 | 0 | 0 | 0 |
| NVFP4 | R0 | 24 / 0 | 16 | 0 | 0 | 8 |
| NVFP4 | R1 | 19 / 5 | 14 | 0 | 0 | 5 |

DEV uses a separate six-task denominator. Each BF16 cell has two current-read/matching reports and four no-read/mismatching reports. NVFP4 R0 has six no-read/mismatching reports; R1 has five plus one episode without accepted final. The 18 DEV Q=1/W=0 episodes also reported first-read quantities. The Q=1 subgroup is an outcome-selected auxiliary analysis retained in the [full ledger](publication/summary/quantity_ledger.json).

[Exact definition](publication/analysis/DEFINITION.json) · [All 120 diagnostic rows](publication/summary/quantity_ledger.json)

</details>

## Method and original contract

The 24 synthetic tasks contain 8 examples each of wrong target (A), duplicate processing (B) and partial debit (C). A restores the wrong-SKU records; B preserves one valid transfer and compensates the extra one; C undoes only the source debit. Quantity compensation and active-operation release use separate APIs.

The evaluator’s first follow-up program performs the pending normal transfer for A/C, or the next one-unit transfer for B, then reads and validates. The second redelivers the original registered transfer and performs the next one-unit transfer; B’s completed transfer must be ignored as a duplicate. Each program starts from an independent state copy, reads the current revision publicly and compares business values, ledger state, duplicate effects and protection with the same program from a clean checkpoint.

All runs use Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554` with BF16 weights. K0=BF16 KV; K1=native NVFP4 KV. R0 accepts a well-formed final for subsequent scoring; R1 checks prior observation of current successful validation. The common system already requires validation after the last change in both policies. Both share grammar G, a strict parser, eight APIs and greedy decoding: 16 requests, 512 output tokens/request, 8,192 output tokens/episode, 24,576 context tokens and 180/900-second request/episode limits. Rejections, errors and later generations use the same budget. The original model-success expansion threshold is null.

B=business restoration, P=protection throughout, C1/C2=two fixed continuations and Q=B∧P∧C1∧C2. Fresh means prior observation of successful current validation; W=Q∧accepted correct final∧Fresh. Final-content checks cover task/recovered, the full quantity map and validation event ID. D counts recovered:true declarations, including rejected finals. Without accepted final, Q is assessed at termination and W=0; Fresh then describes observations before the last actual generation.

The public validator checks nonnegative inventory, protected fields, conservation and ledger references. Its response does not include all current quantities. A separate evaluator compares the reported quantity map with saved inventory. Observing successful current validation (Fresh) and rereading current quantities are therefore different observations.

<details>
<summary>Component metrics, separate DEV and paired intervals</summary>


| KV | Policy | D | B | P | Fresh | C1 | C2 | Q | W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | R0 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 0 |
| BF16 | R1 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 0 |
| NVFP4 | R0 | 24 | 19 | 23 | 19 | 19 | 19 | 19 | 5 |
| NVFP4 | R1 | 24 | 19 | 23 | 19 | 19 | 19 | 19 | 5 |

| KV | Policy | Q / 6 | W / 6 |
| --- | --- | --- | --- |
| BF16 | R0 | 6 | 2 |
| BF16 | R1 | 6 | 2 |
| NVFP4 | R0 | 5 | 0 |
| NVFP4 | R1 | 5 | 0 |

| Paired comparison | Metric | Difference (pp) | Exploratory 95% interval (pp) |
| --- | --- | --- | --- |
| K0_R1-R0 | Q | 0.00 | 0.00 / 0.00 |
| K0_R1-R0 | W | 0.00 | 0.00 / 0.00 |
| K1_R1-R0 | Q | 0.00 | 0.00 / 0.00 |
| K1_R1-R0 | W | 0.00 | 0.00 / 0.00 |
| R0_K1-K0 | Q | -20.83 | -33.33 / -8.33 |
| R0_K1-K0 | W | 20.83 | 8.33 / 37.50 |
| R1_K1-K0 | Q | -20.83 | -33.33 / -8.33 |
| R1_K1-K0 | W | 20.83 | 8.33 / 37.50 |

The family-stratified paired task bootstrap jointly resamples a task’s four conditions over 10,000 draws, seed 61022018. Intervals are exploratory pointwise percentile 95%. Zero R1−R0 Q/W differences and intervals here leave differences on other tasks unresolved. B/C1/C2 success sets coincide in the measured P1/P2 samples, so added continuation discrimination was not observed here. P0 synthetic counterexamples separately test discrimination of operational-state faults.

</details>

## Costs, KV storage and technical scope

| KV | Policy | Requests | Input tokens | Output tokens | Rejections | Episode wall s |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 | R0 | 168 | 323375 | 10953 | 0 | 164.144812115 |
| BF16 | R1 | 168 | 323375 | 10953 | 0 | 164.157640174 |
| NVFP4 | R0 | 186 | 369881 | 11568 | 0 | 179.978763365 |
| NVFP4 | R1 | 206 | 439813 | 13730 | 15 | 213.922486411 |

NVFP4 R0→R1 increases input by 18.91%, output by 18.69% and episode wall by 18.86%. The table includes all 24 successes and failures. Cost/Q and cost/W divide these total costs by Q/W successes; zero successes gives undefined. Input and output are shown separately because their computational costs differ.

P1 records 180 DEV requests and six TECH attempts (four complete/two cancelled); P2 records 728 HOLDOUT and three complete TECH responses. Totals are 917 attempts/915 complete responses and an observed output lower bound of 57987 tokens. Actual engine output for the two cancellations remains null. P2 HOLDOUT input/output is 1,456,444/47,204; TECH is 2,901/39. Worker wall 439.692244164 + 855.069382600 = 1294.761626764 seconds includes loading, compilation, failures and shutdown and overlaps episode wall. New editorial CPU verification cost is recorded separately.

The approximately 4 GiB KV pool holds 29,120 BF16 token slots or 103,552 NVFP4 slots (about 3.556×). Actual pools are 4,293,918,720/4,294,508,544 bytes, at 147,456/41,472 bytes per token. Weights and temporary execution memory are outside this slot calculation. Technical receipts retain packed uint8 data, FP8 block scales, storage/reads across 36 layers and finite attention outputs. This compares configurations including vLLM 0.32.0, FlashInfer 0.7.0.post1, xgrammar 0.2.8 and their actual kernel paths. Failed attempts and EngineCore force-kill/semaphore shutdown warnings remain in the evidence.

## Reproduction, sources and conclusion

```bash
python3 -B cases/018-agent-state-recovery/publication/run_checks.py --output /tmp/case018-public-check-new --tests
```

Use Python 3.12+ with the standard library and a fresh output directory to replay saved records without a model download.

CPU replay recalculates transitions, observations and verdicts for 120 model episodes from public evidence. The 26 P0 synthetic traces are a separate test set. Phase-specific frozen sources are used; ReferenceState, protection, continuations and SSE assembly are independently implemented, while strict parser/serialization/public builder/wire rendering are shared. Stored booleans are comparison targets for recalculated verdicts. Original tokenizer re-encoding uses retained receipts. A local public-only copy and GitHub-hosted Ubuntu execution have passed.

[CPU record replay · main](https://github.com/Munsik-Kim/inference-lab/actions/workflows/case018-checks.yml?query=branch%3Amain)

Direct work comprises the inventory/operation ledger, compensation and release APIs, strict parser, completion guard, pre-generation observation tracking, separate state/continuation audits, posthoc quantity analysis and public tools. Qwen, vLLM, FlashInfer KV kernels and xgrammar come from upstream.

The three error types and 24 synthetic tasks show failures in this environment. We independently adapt post-recovery evaluation to small tools; [SOURCE_MAP](SOURCE_MAP.md) explains the differences from the original GUI/on-policy setting. [REUSE_MAP](REUSE_MAP.md) and [NOTICE](NOTICE.md) document reuse, Codex assistance and licenses; [REPRODUCE](REPRODUCE.md) gives the replay details.
