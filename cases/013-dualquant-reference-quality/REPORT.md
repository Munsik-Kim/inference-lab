# Case 013 — Does Lower Weight Error Improve Real Answers?

English | [한국어](REPORT.ko.md)

Implemented INT4 conversion, packed storage/reload and matched-input quality tools. Applied a public-equation reference variant across two complete Qwen models, linking language scores, exact long-input answers and free tool use. Smaller weight error and task success did not consistently move together.

## Contents

1. [Background](#background)
2. [Hypotheses and questions](#questions)
3. [Theory and cost model](#theory)
4. [Methods](#methods)
5. [Experiments](#experiments)
6. [Results](#results)
7. [Analysis](#analysis)
8. [Conclusion](#conclusion)
9. [References and contributions](#references)

<a id="background"></a>
## 1. Background

Quantization stores weights with fewer bits. Weight reconstruction MSE, next-symbol probability scores and correct answers/tool actions address different questions. Acceleration needs an actual low-bit kernel and complete-request measurements.

DualQuant adjusts row/column scales using weights alone. This project independently implements public INT4 equations without the private block-format dependency. The authors' [v1 paper](https://arxiv.org/html/2610.13767v1) and this RTX 5080 reference study are separate evidence.

<a id="questions"></a>
## 2. Hypotheses and questions

Can the INT4 reference run across a complete model? Does lower weight MSE than RTN improve probability and answer scores? What changes during long-input and multi-tool use?

These are REFERENCE_DIAGNOSTIC/DEV results. Settings were frozen before outputs; no clipping search, best-iteration selection or calibration was chosen from scores. The 16K agent extension was a follow-up after base DEV: the first ID per family was selected structurally and its inputs frozen before that run. Formal 120-task TEST, paired GPTQ serving and noninferiority/GO_SPEED/GO_CAPACITY decisions were not performed.

<a id="theory"></a>
## 3. Theory and cost model

W has shape [out,in], X [tokens,in]. For each group64 block, positive inverse scales a,b define Q=clip(round(W/(a·b)),−8,7). A fixed-code least-squares update of a is followed by fresh codes and an update of b. Denominators include ε=1e−12. Every output row participates in the column reduction.

After 15 iterations, final group absmax scale γ gives W_tilde=Q_final·a·γ and W_effective=W_tilde·b. D16 computes linear(X·b,W_tilde,bias). Beta is not applied twice. D4 adds dynamic token/group64 INT4 quantization to X·b. BF16 rounding can make algebraically equivalent representations numerically different.

Signed codes span −8…7 with ties-to-even rounding. Two codes occupy one byte; the ledger includes codes, FP32 γ/a/b, shape and a 24-byte header. RTN omits identity a/b. Positive row scaling can cancel in final absmax RTN, so the row-only control need not improve final error. Inner iteration objectives are separate from final serialized-weight MSE.

The theoretical 4B BF16 KV payload is 2×36×8×128×2=147,456B/token/stream: 1.125/2.25/4.5GiB at 8K/16K/32K. Weights, workspace, block rounding and generated tokens are additional. This is a calculation, not measured VRAM.

<a id="methods"></a>
## 4. Methods

Reviewed public commit `d84b7e9dff653320e47c06a365eb8295c1825bc3` for element INT4, initialization, BCD and final group casting. The fixed setting is group64, one output-row split, 15 iterations, max_abs row/L1 column initialization, FP32 offline compute/scales, no scale projection and element-only inner quantization. It differs from paper BF16 scale projection and public default inner=false; structural zeros use an explicit unit-scale exception. The study is therefore **PAPER_INT4_REFERENCE_VARIANT**; D4 is **W4A4_REFERENCE_VARIANT**.

Compare BF16 / RTN W4A16 / D16 / D4. D16 and D4 reuse identical weight artifacts; D4 adds activation INT4. Neither RTN W4A4 nor existing GPTQ was evaluated in this study.

Convert q/k/v/o/gate/up/down in every block: 196 projections in 0.6B, 252 in 4B. Keep embeddings/lm_head/norm/attention/RoPE/KV and ties unchanged. Original weights alone enter the solver. [Solver/adapter](source/paper_int4.py) and [runner](source/paper_model_runner.py) have separate roles.

The quality path reconstructs W_tilde to BF16 for BF16 matrix operations; input beta multiplication is FP32. Actual code packing exists, but this quality path is not packed deployment acceleration. Weight artifacts remain private/local; the public manifest identifies them without publishing payloads.

<a id="experiments"></a>
## 5. Experiments

Models: Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`; Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Sequential loading on RTX5080 SM120, driver610.88, Torch2.13.0+cu130, Transformers5.17.0, NumPy2.3.5 and vLLM0.29.0. New training, calibration search, model downloads and existing-environment upgrades: zero.

WikiText-2 raw test: nonempty rows joined by two newlines, no BOS/EOS, first 32 nonoverlapping 2,048-token windows. Score 2,047 positions/window, 65,504/model/method. NLL covers all scored positions; full-vocabulary KL(BF16||candidate) uses 32 frozen positions/window; argmax flips cover all scored positions. Four methods reuse the same windows. Interpretation focuses on changes against each model's original baseline.

A real 512-token full forward ran once per condition. The CSV 512-prefix PPL column is a post-hoc CPU calculation from the first511 losses of saved2048 records. This is neither full official benchmark PPL nor a new formal TEST.

Long DEV: four retrieval, four document-join and four state-update tasks. Same-ID 8K/16K/32K versions are not independent samples. Greedy, batch1, output limit96, BF16 KV, SDPA, logits_to_keep=1, without offload/truncation/KV quantization.

Agent DEV: eight tasks, two per family. Gold-defined common tool replay is separate from free rollout. Tools access only owned documents/records/tables/calculations/JSON fixtures/validation. Initial contexts approximately4K/8K, output192, limits8 model calls/6 tool calls,900-second REFERENCE_PILOT deadline and24K context limit. Prefix caching isOFF between turns; within-request KV isON. This differs from the earlier serving180-second boundary.

Recorded GPU worker wall totals2,251.160 seconds of the14,400-second budget. It includes loading, CPU serialization and idle, not just GPU kernel busy time. Publication preparation only reads saved records on CPU.

<a id="results"></a>
## 6. Results

### A. Full-model probability scores

| Model | Method | NLL ↓ | PPL ↓ | KL ↓ | Argmax flip | Normalized weight MSE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| 06b | BF16 | 3.057517 | 21.274661 | 0.000000 | 0.000% | 0.00000000 |
| 06b | RTN | 3.316951 | 27.576153 | 0.398976 | 28.933% | 0.01228731 |
| 06b | D16 | 3.196423 | 24.444930 | 0.261293 | 22.070% | 0.01208061 |
| 06b | D4 | 3.547731 | 34.734400 | 0.675540 | 36.998% | 0.01208061 |
| 4b | BF16 | 2.315826 | 10.133291 | 0.000000 | 0.000% | 0.00000000 |
| 4b | RTN | 2.380479 | 10.810076 | 0.213157 | 14.340% | 0.01216967 |
| 4b | D16 | 2.388248 | 10.894392 | 0.195624 | 13.532% | 0.01201816 |
| 4b | D4 | 2.512728 | 12.338547 | 0.389818 | 22.573% | 0.01201816 |

D16 relative changes against RTN: 06b: MSE -1.6822%, PPL -11.3548%; 4b: MSE -1.2449%, PPL +0.7800%. Both measures improved in0.6B; in4B, MSE/KL/argmax correspondence improved while PPL slightly worsened. Adding activation INT4 inD4 increased PPL. [0.6B unrounded CSV](records/quality_06b.csv) · [4B CSV](records/quality_4b.csv)

### B. Exact long-input answers

| Method | 8K exact | 16K exact | 32K exact |
| --- | --- | --- | --- |
| BF16 | 12/12 | 12/12 | 10/12 |
| RTN | 11/12 | 10/12 | NOT_RUN_MEMORY_POLICY |
| D16 | 10/12 | 10/12 | NOT_RUN_MEMORY_POLICY |
| D4 | 8/12 | 7/12 | NOT_RUN_MEMORY_POLICY |

Only BF16 ran at32K. After observing16K workspace, the conservative candidate-memory gate skipped their32K cells. These are not OOM failures or maximum stable-context measurements. [Operational update](records/operational_update_v3.json)

Keep gains and losses on the same task IDs; equal totals can conceal different outcomes.

| Length | Baseline → candidate | Both pass | Baseline only | Candidate only | Both fail |
| --- | --- | --- | --- | --- | --- |
| 8192 | BF16 → RTN | 11 | 1 | 0 | 0 |
| 8192 | RTN → D16 | 9 | 2 | 1 | 0 |
| 8192 | RTN → D4 | 7 | 4 | 1 | 0 |
| 16384 | BF16 → RTN | 10 | 2 | 0 | 0 |
| 16384 | RTN → D16 | 10 | 0 | 0 | 2 |
| 16384 | RTN → D4 | 7 | 3 | 0 | 2 |

**POST_HOC_SAME_RECORDS:** primary scoring requires exact answer/evidence values and allows extraction of the first valid JSON object. Strict whole-document JSON validity is separate. BF16's two32K failures and threeRTN failures used a correct numeric expression such as `7 * 3 = 21` instead of `21`. D16 also lost READY-0's suffix; D4 mixed wrong key case, operand-as-answer, state suffix loss and output-budget exhaustion. Primary scores are unchanged. [All post-hoc categories](records/long_context_dev.csv)

### C. Tool replay and free action

| Method | Base replay | Base free | 16K replay | 16K free | Base tool/parse errors |
| --- | --- | --- | --- | --- | --- |
| BF16 | 8/8 | 8/8 | 4/4 | 4/4 | 0/0 |
| RTN | 8/8 | 8/8 | 4/4 | 4/4 | 0/0 |
| D16 | 8/8 | 6/8 | 4/4 | 3/4 | 8/0 |
| D4 | 8/8 | 0/8 | 4/4 | 0/4 | 2/2 |

Base has8 unique tasks;16K reuses4 of them as follow-up length variants. The96 episode conditions are not96 independent tasks. Replay already supplies a common tool result.

D16 AGENT-6/7 used CamelCase names for `apply_fixture_patch`/`validate_fixture_test`, repeated rejected calls and returned valid:true without changing the fixture. BF16/RTN completed the same contracts. No candidate-specific aliases were added. D4 examples include no tool action, treating background text as state, wrong names and an invalid final fixture. [Complete raw episodes](records/raw/4b/D16/agent_dev.jsonl)

### D. Storage, time and memory

| Model | Method | Projection artifact bytes | NLL loop wall (s) | Torch peak allocated/reserved (B) | Convert or reload (s) |
| --- | --- | --- | --- | --- | --- |
| 06b | BF16 | N/A | 2.8858 | 1338743808/1400897536 | 0.000 |
| 06b | RTN | 247,730,784 | 3.2063 | 1339890688/1472200704 | 520.493 |
| 06b | D16 | 276,402,784 | 2.9750 | 1339890688/1447034880 | 48.709 |
| 06b | D4 | 276,402,784 | 4.0918 | 1339890688/1501560832 | 17.521 |
| 4b | BF16 | N/A | 7.7518 | 8297826304/8380219392 | 0.000 |
| 4b | RTN | 2,043,746,208 | 8.6668 | 8381351936/8598323200 | 107.126 |
| 4b | D16 | 2,274,662,304 | 8.7787 | 8384497664/8619294720 | 185.284 |
| 4b | D4 | 2,274,662,304 | 10.8704 | 8462288896/8732540928 | 65.451 |

Artifact bytes count converted projections' codes/scales/header, not a complete checkpoint. Runtime weights are reconstructedBF16; file payload reductions are not runtimeVRAM reductions. NLL-loop wall includes metric transfer and some reference I/O, not servingTTFT/TPOT. D4's last column isD16 artifact reload.

0.6B RTN used the earlier Python byte-iteration serializer, while subsequentD16/4B used byte-identical bulk copy. That conversion-time difference is not an algorithm speedup comparison. System GPU peak, TTFT, TPOT and cached tokens were not measured.

### E. Algebra and packed correspondence

Twenty-eight rows on four actual projection matrices: FP64 transport max difference0, FP32 max1.43051e−6, BF16 folded-expression relativeRMSE max0.0040052, FP64 row-only cancellation max6.66134e−16. A tiny1e−15 fixture's zero beta was rejected explicitly; it is not a measured real-model failure rate.

ActualSM120 CUDA execution used vLLM Marlin packedINT4 weights with fused unpack to BF16 tensor-core MMA. It isW4A16, not nativeINT4 MMA/W4A4/the authors' Nunchaku route. Beta remains, while FP32 a·γ is rounded intoBF16 kernel scales in this probe variant.

GEMM relativeRMSE0.003078 and one block0.005673 passed the2% gates. Final32-token full-forward error0.026054 exceeded0.02. Matching four greedy output tokens did not override the gate. Earlier2.5533%/2.2598% and final2.6054% are retained. The first two handlers exited0 while semantic status remainedBLOCKED; the finalCLI exits1. [Final probe](records/diagnostics/packed_precision_aligned_03.json)

<a id="analysis"></a>
## 7. Analysis

Smaller weight MSE did not guarantee exact long-input answers or stable tool behavior. Nearly unchanged4B PPL coexisted with tool-name/action errors in this pilot. This is not a claim that every DualQuant setting or model fails: scope is this nondefaultINT4 variant, smallDEV and common cacheOFF policy.

Correct numbers and output-contract violations are distinguished. Relaxing the scorer afterwards would change the primary question. Replay success establishes neither free-action success nor complete information preservation.

Packed memory/time belong to a prototype that failed the correspondence gate. A validated deployment route and stockGPTQ comparison with matched requests/KV/output/cache/parser are needed for speed/capacity conclusions. The4B quality path keepsBF16 weights resident; its storedW4 artifacts alone do not demonstrate longer usable context.

The original39 unique CPU tests (29old+10new) remain in the [historical receipt](records/verification.json) and logs. Publication checks are separate. The public auditor recomputesNLL/PPL/flips and answer/action scores, but not full-logitKL, original-weightMSE or private weight-payload hashes. [Audit scope](analysis/audit.py) states those boundaries.

<a id="conclusion"></a>
## 8. Conclusion

Completed full-modelINT4 reference conversion, real storage/reload, long/agent evaluation and independent readback. D16 PPL improved againstRTN in0.6B, slightly worsened in4B, and reduced4B free-agent success. D4 showed larger quality loss.

This is evidence to separate weight error, language scores and task success under this variant/DEV budget, not a failure verdict on all author methods. Full packed-model correspondence and formal serving remain open. Inspect the [CPU reproduction](REPRODUCTION.md) or [complete records](records).

<a id="references"></a>
## 9. References and contributions

[NOTICE](NOTICE.md) identifies DualQuantv1/pinned code, Qwen revisions, WikiText, libraries and contributions. Evidence: [paper/code mapping](records/paper_code_mapping.txt), [protocol](records/experiment_manifest.json), [scale contract](records/reference_contract.json) and [original/public hashes](provenance/export_map.json).

Originalr1 failed input preparation on pyarrow/return_dict before anyGPU run. r2 reused an existingCPU environment and explicitreturn_dict=False. P0/r1/r2, failed probes and99Case009 files remain preserved; Cases001–012 were not modified. Private-path redaction affects publication copies only. This publication uses a feature branch and introduces no newGPU experiments.


## Detailed contracts and figures

[Paper/code/study contract](METHOD_CONTRACT.md) · [Evaluation protocol](EVALUATION_PROTOCOL.md) · [Failures and controls](FAILURE_ANALYSIS.md) · [Packed probe](PACKED_PROBE.md) · [Three figures](figures/README.md) · [Proposed roadmap](ROADMAP.md)

D16 has 11.2987% more 4B quantized-projection payload than RTN, including its extra a/b scales. This is not equal-byte comparison. BF16 N/A means no separate INT4 artifact.
