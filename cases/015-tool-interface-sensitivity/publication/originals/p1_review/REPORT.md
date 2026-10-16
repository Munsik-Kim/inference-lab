# Case015 — Same Tools, Different Interfaces

**Tool-Interface Sensitivity of Quantized LLM Agents**

Implemented a local comparison tool that changes the names and ordering of the same tools, then checks whether the agent actually completes the task. It connects exact interface mappings, stepwise equivalence, real fixture execution and saved-response replay.

Completed 96 FREE episodes with Qwen3-4B BF16/RTN/D16 and four interfaces. Opaque names recovered D16's two original patch failures; lowerCamelCase caused one new update-task loss. The registered signed interaction is **+8.33 percentage points**, exploratory paired 95% CI **[-8.33,+25.00] pp**, including zero.

## Contents

[Background](#background) · [Questions](#questions) · [Interface contract](#contract) · [Method](#method) · [Experiment](#experiment) · [Results](#results) · [Failures and recovery](#failures) · [Conclusion](#conclusion) · [Reproduction and sources](#references)

<a id="background"></a>
## Background

Case013 observed D16 UpperCamelCase name failures. Later ENUM and enhanced feedback improved the same existing DEV tasks. Both mitigations are OFF here. This study evaluates interface sensitivity with an executable evidence pipeline, without new quantization or CIPR training.

RTN is simple round-to-nearest INT4; D16 is the independent DualQuant weight-INT4 reference. Functional success checks the requested actual actions and state together with the final JSON.

<a id="questions"></a>
## Questions and hypotheses

The registered primary asks whether D16's signed interface change is worse than BF16's:

`d(m,t)=mean_i=I1,I2,I3 Y(m,i,t)-Y(m,I0,t)`

`Delta=mean_t[d(D16,t)-d(BF16,t)]`

Negative means worse signed change for D16. Gains can cancel losses, so loss/gain pairs, absolute outcome-change rate S and all-interface success are also reported. Secondary questions locate observed differences in names, semantics, arguments, recovery and end state, and motivate later diagnostic tests. I2 was designed knowing past name failures; I3 uses one fixed assignment per task.

<a id="contract"></a>
## Interface and execution contract

| Interface | Change | Preserved |
|---|---|---|
| I0 ORIGINAL | None | Original names, order, schema, descriptions and messages |
| I1 ORDER_ONLY | Seeded task-specific nonidentity permutation | All content and names |
| I2 STYLE_RENAME | snake_case to lowerCamelCase | Arguments, descriptions, order and types |
| I3 OPAQUE_RENAME | Answer-independent tool_NN assignment | Arguments, descriptions, order and types |

The semantic identity is internal and never exposed to the model. Only typed declarations and verified TASK API spans change; arbitrary text, fixture values and generated responses do not. [Mappings](specs/interfaces.json) record field paths, reversals and exact before/after values. Original descriptions distinguish document facts, quantity/price, prior/current status, multiplication, configuration replacement and validation. Opaque allocation does not read gold or declaration position.

An exact exposed-name lookup precedes dispatch to the unchanged original callable. Old names and typos produce UNKNOWN_TOOL without side effects. Model-visible error strings remain the generic original strings; sidecar labels and allowed-action sets are not feedback hints. No enum or aliases apply.

Primary FUNCTIONAL_SUCCESS requires original-task reads/calculation in order and actual patch/validation. LEGACY_SUCCESS preserves Case013 final-answer/state scoring separately. Irrelevant extra reads can receive diagnostic WRONG_TOOL labels without automatically failing the episode; operand commutation is accepted. All 96 current cells happen to have identical functional and legacy scores; the contracts remain distinct and are tested with counterexamples.

<a id="method"></a>
## Method

Read-only existing WSL Qwen/Qwen3-4B-Instruct-2507 snapshot, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Twelve local model files including all three weight shards, and 504 saved RTN/D16 projection files, were SHA256-checked before and after execution. This establishes local identity, without a new official remote digest comparison.

| Method | Projection files | Stored bytes |
|---|---:|---:|
| RTN | 252 | 2,043,746,208 |
| D16 | 252 | 2,274,662,304 |


These are projection payloads, not standalone checkpoint sizes or equal-byte methods. Embeddings, head, norms, attention, RoPE and tying remain original. D16 retains FP32 input beta-inverse transport followed by BF16, and BF16 W_pre. No folded weight_hat substitution or double beta application. Missing artifacts fail; solve/encode/serialize/full-sweep paths are guarded.

Greedy, enable_thinking=False, maximum 192 new tokens, 8 model calls / 6 attempted tool calls, 900 seconds per episode, context limit 24,576, batch 1, SDPA, BF16, TF32=False, BF16/FP16 reduced precision reduction=False, within-request KV ON and across-turn cache OFF, logits_to_keep=1. Only six registered local fixture functions execute; generated shell/Python is data.

<a id="experiment"></a>
## Experiment

Eight previously observed tasks: two each document/combine/updates/fixture_patch, preserving original 4K/8K allocation. 96 primary episodes are 3 methods × 4 interfaces × 8 tasks, not 96 independent tasks. Two separate BF16/I0 AGENT-6/7 smoke episodes are excluded from primary results. Each method runs eight I0 first, then 24 seeded, shuffled cells. Model load order: BF16, D16, RTN.

Interface/execution/bootstrap seeds 6101601/6101602/6101603. 27 unique CPU tests and 32 oracle episodes / 48 step comparisons passed before source/input freeze. All 24 I0 initial input hashes, first output token-ID sequences and legacy outcomes exactly matched historical records. [Historical check](results/p1_r1/historical_I0_comparison.json).

| Measurement | Actual value |
|---|---:|
| Primary episodes / missing / unobserved | 96 / 0 / 0 |
| Primary generation requests / generated tokens / tool attempts | 228 / 5,765 / 160 |
| Separate smoke episodes / requests / generated tokens | 2 / 4 / 100 |
| Cumulative worker wall | 452.013 s |
| Peak Torch allocated / reserved | 10,359,526,400 / 12,769,558,528 B |
| New fitting/quantization/install/model download; retries | 0; 0 |

Worker wall includes load, decode, smoke, CPU, I/O and failures; it is not serving latency or GPU-busy time. Request-level CUDA free/total samples are stored; a continuous Windows-wide device peak was not measured. [Resources](results/p1_r1/resource_summary.json).

<a id="results"></a>
## Results

| Method | I0 original | I1 order | I2 lowerCamelCase | I3 opaque |
|---|---:|---:|---:|---:|
| BF16 | 8/8 | 8/8 | 8/8 | 7/8 |
| RTN | 8/8 | 8/8 | 8/8 | 8/8 |
| D16 | 6/8 | 6/8 | 5/8 | 8/8 |


Each cell uses the same eight DEV tasks. [All 96 task rows](results/p1_r1/task_outcomes.csv), [episode explorer](demo/en.html#episodes).

| Contrast | Signed interaction (pp) | Exploratory paired 95% CI (pp) | Role |
|---|---:|---|---|
| D16-minus-BF16 | +8.33 | [-8.33, +25.00] | primary |
| RTN-minus-BF16 | +4.17 | [+0.00, +12.50] | secondary |
| D16-minus-RTN | +4.17 | [-8.33, +16.67] | secondary |


The primary interval includes zero. The secondary RTN–BF16 interval touches zero and does not establish superiority/equivalence. No observations are missing: worst/best missingness bounds coincide with +0.083333.

| Method | Changed pairs /24 | S | All four interfaces successful tasks /8 |
|---|---:|---:|---:|
| BF16 | 1/24 | 4.17% | 7/8 |
| RTN | 0/24 | 0.00% | 8/8 |
| D16 | 3/24 | 12.50% | 5/8 |


D16 changed 3 of 24 original-to-perturbed outcomes: one I2 loss (AGENT-4), two I3 gains (AGENT-6/7). BF16 lost AGENT-3 under I3. RTN preserved outcomes on all 24 pairs; its action histories were not invariant.

| Method | Interface vs I0 | Both success | I0 only: loss | Changed only: gain | Both fail |
|---|---|---:|---:|---:|---:|
| BF16 | I1 | 8 | 0 | 0 | 0 |
| BF16 | I2 | 8 | 0 | 0 | 0 |
| BF16 | I3 | 7 | 1 | 0 | 0 |
| RTN | I1 | 8 | 0 | 0 | 0 |
| RTN | I2 | 8 | 0 | 0 | 0 |
| RTN | I3 | 8 | 0 | 0 | 0 |
| D16 | I1 | 6 | 0 | 0 | 2 |
| D16 | I2 | 5 | 1 | 0 | 2 |
| D16 | I3 | 6 | 0 | 2 | 0 |


| Family | Task IDs | Primary interaction (pp) | Leave this family out (pp) |
|---|---|---:|---:|
| combine | AGENT-2, AGENT-3 | +16.67 | +5.56 |
| document | AGENT-0, AGENT-1 | +0.00 | +11.11 |
| fixture_patch | AGENT-6, AGENT-7 | +33.33 | +0.00 |
| updates | AGENT-4, AGENT-5 | -16.67 | +16.67 |


Leave-one-family-out primary range:0 to +16.67 pp. The 10,000-draw bootstrap resamples each whole 12-cell task vector, retaining repeated I0 dependence. These exploratory eight-task/four-family intervals are not external-population guarantees or confirmatory p-values. Restricting to the six tasks both methods initially solved produces one new loss per method with equal signed size; that selected subset is not evidence of universal invariance.

![Functional success](figures/p1_r1/success_matrix.svg)

Higher is better; each cell is 8 DEV tasks, with no invented CI.

![Losses and gains](figures/p1_r1/paired_losses_gains.svg)

Below zero: original success becomes failure. Above zero: original failure becomes success.

![Interaction](figures/p1_r1/paired_interaction.svg)

Units are percentage points, with whole-task exploratory intervals.

<a id="failures"></a>
## Failure and recovery analysis

| Case | Observed behavior | Verdict/source |
|---|---|---|
| D16/I0 AGENT-6/7 | Unregistered ApplyFixturePatch/ValidateFixtureTest; repeats after errors; valid:true without patch | Failure; [AGENT-6raw](results/p1_r1/workers/D16/attempt01/cells/D16__I0_ORIGINAL__AGENT-6.json) |
| D16/I2 AGENT-4 | No call; returns status:latest_status | Missing required read and wrong answer; [raw](results/p1_r1/workers/D16/attempt01/cells/D16__I2_STYLE_RENAME__AGENT-4.json) |
| BF16/I3 AGENT-3 | Reads 11 and 3, but calls registered multiplication with 10 and 5.5; final 55 | Name valid, argument values and answer wrong; [raw](results/p1_r1/workers/BF16/attempt01/cells/BF16__I3_OPAQUE_RENAME__AGENT-3.json) |
| D16/I3 AGENT-6 | Calls tool_56, tool_89, actually patches and validates, then valid:true | Recovered success; [raw](results/p1_r1/workers/D16/attempt01/cells/D16__I3_OPAQUE_RENAME__AGENT-6.json) |

| Method | Interface | Valid name / evaluable call objects |
|---|---|---:|
| BF16 | I0 | 12/12 |
| BF16 | I1 | 12/12 |
| BF16 | I2 | 12/12 |
| BF16 | I3 | 13/13 |
| RTN | I0 | 12/12 |
| RTN | I1 | 12/12 |
| RTN | I2 | 12/12 |
| RTN | I3 | 14/14 |
| D16 | I0 | 8/16 |
| D16 | I1 | 8/16 |
| D16 | I2 | 12/16 |
| D16 | I3 | 13/13 |


Name-validity denominators include evaluable parsed call objects. No-call episodes are NOT_APPLICABLE; parsing failures are separate. Similar-looking unknown names never receive execution credit. Error labels can overlap, so event counts are not summed into failure-episode counts.

The registered broad continuation field includes sidecar-only errors and later tool events. A separate **post-hoc denominator refinement** considers actual error/negative-validation results followed by a later model turn, excluding same-turn calls. Among 6 eligible episodes, 2 later used a registered name, 0 recovered a correct required action, and 0 finished successfully. Raw/registered scores remain unchanged. [Explicit post-hoc analysis](results/p1_r1/feedback_recovery.posthoc.json).

[72 common-event-prefix comparisons](results/p1_r1/common_event_prefixes.json) align only the same initial state and actual shared event prefix, never arbitrary turn numbers. Free rollouts do not prove internal semantic ability or token-margin causality.

<a id="conclusion"></a>
## Conclusion

Q1's registered signed comparison is inconclusive. D16's observed outcome-change rate exceeds BF16's, with gains and losses in different tasks. Q2 reveals distinct unregistered-name, bad-value and missing-read starts. Q3 supports considering same-error-feedback diagnostics with prespecified successes as controls, then separate interface-specific comparisons of existing ENUM/FEEDBACK. Those follow-ups were not executed.

| Scope | Status |
|---|---|
| Interface transformation/equivalence/fixture execution/CPUaudit | Executed |
| Existing 8 DEV × 12 conditions | All96completed |
| New tasks/repeated opaque allocations/other models/D4 | Not run |
| P2 diagnostics / P3 fresh64 / CIPR training | Not run |
| Packed serving/business services/remote publication | Not run |

<a id="references"></a>
## Reproduction and sources

[CPUcommands](REPRODUCE.md), [reuse](REUSE_MAP.md), [preservation](results/p1_r1/preservation.json), [audit](results/p1_r1/independent_audit.json), [verification](results/p1_r1/verification.json).27 new unique CPU tests; 46 existing package tests reexecuted. Repeated runs and historical Case013 39 tests are not added to new counts.

The first P0 attempt had one exact-binary-equality assertion failure for thirds, fixed before freeze. The first package test attempt had three failures because relative PYTHONPATH broke in subprocess temporary cwd; absolute PYTHONPATH passed 46 without package changes. Failure logs are retained. The GPU execution source stayed frozen.

The auditor reparses 228 saved responses, reexecutes 160 fixture actions, checks 96 endpoints and independently recalculates the primary scalar/bootstrap. It does not reproduce excluded logits/weight computations. Review content includes code, synthetic inputs, light evidence and bilingual explorer; model / INT4 payloads / environments / large tensors are excluded.

<details><summary>Identities and attribution</summary>

- Git base:`e647b474bdc9fa558d851bd9c998b664b0e33638`; no new commit.
- Execution source SHA256:`5a2c18000b4623c8c011390f71c8f3eb5fc91cf231d9905320e546a0dcc85a2b`; distinct from Git SHA.
- [Context-dependent name BPE and TASK positions](provenance/tokenization.posthoc.json): post-hoc input metadata for32registeredprompts, re-encoded without model forward.
- [Source freeze](provenance/source_freeze.json), [local model identities](provenance/model_identity.json), [artifacts](provenance/artifact_identity.json), [inputs](provenance/input_manifest.json).
- [CIPR v1](https://arxiv.org/html/2610.20576v1), [pinned author README](https://github.com/ArtinTD/cross-interface-policy-regularization/tree/0465909ce219cdd1202affff75e41330697e4862), [actual source audit](provenance/source_audit.json). This evaluation does not train CIPR or claim official implementation/theory reproduction.
- Codex/ChatGPT assisted implementation, execution and separate CPU recalculation, without independent human review or external GPU replication. [NOTICE](NOTICE.md).

</details>
