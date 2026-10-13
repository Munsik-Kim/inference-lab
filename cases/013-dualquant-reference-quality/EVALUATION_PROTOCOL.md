# Evaluation protocol / 평가 계약

문장 확률·장문 exact 답·직접 도구 실행·packed 수치 대응을 따로 판단합니다. 같은 task의 길이·방법·replay/free 변경본을 독립 표본으로 합치지 않습니다. 자세한 원형은 [experiment manifest](records/experiment_manifest.json), [reference contract](records/reference_contract.json), [16K follow-up registration](records/agent16k_extension/preregistration.json)입니다.

## Language scoring

WikiText-2 raw test, nonempty rows joined by two newlines, no added BOS/EOS. First32 nonoverlapping2048-token windows. Window position0 is unscored;2047×32=65,504 valid loss positions per model/method. This is REFERENCE_DIAGNOSTIC, not full benchmark PPL or hidden formal TEST. NLL is token-weighted; PPL=exp(NLL).

KL direction is BF16||candidate, full vocabulary at32 frozen positions/window (1,024 selected positions). Windows contribute equally because each has32 positions. Flip rate counts all65,504 argmax comparisons. The public auditor recomputes loss aggregates/flips and aggregates the recorded KL sums; it cannot recompute KL from excluded full logits. It also recomputes normalized weight SSE/energy and payload totals from per-layer metadata, not from excluded weights.

512-prefix PPL uses the first511 losses of saved2048 windows as POST_HOC_SAME_RECORDS. A real512-token full-forward/limited-head parity ran once per condition; these are different records. Source WikiText text and token arrays are excluded; their original hashes remain in the manifest.

## Long exact DEV

Twelve unique tasks: retrieval/join/state_update, four each. Greedy/batch1, max96outputtokens, BF16 KV, use_cache=True, SDPA, logits_to_keep=1. No automatic truncate, CPU offload or KV quantization. Methods share frozen task IDs, token bytes and gold.8K/16K are common paired cells;32K ran only forBF16.

Primary parser extracts the first valid JSON object and requires exact answer/evidence values. An expression ending in the right number can fail the exact string contract. Whole-document strict JSON validity and error categories are post-hoc and never replace primary scores. [Raw outputs](records/raw/4b/D16/long_dev.jsonl), [paired counts](records/paired_dev_summary.json), [post-hoc columns](records/long_context_dev.csv).

Candidate32K records are NOT_RUN_MEMORY_POLICY after the conservative workspace gate was updated from16K observation. These are neither OOM failures nor maximum context capacity. The denominator is not filled with invented successes/failures. [Operational update](records/operational_update_v3.json).

## Local tool pilot

Eight owned fixtures: document facts, combining records/calculation, latest updates, JSON fixture edit/validation; two each. Replay supplies a gold-defined common tool history. Free rollout chooses tools/actions. Tool schemas, chat templates and first input hashes are matched across methods. Request-local KV ON; turn-to-turn prefix cache OFF with full-history reinsertion.

Max192 outputtokens/response,8 model calls/6 tool calls,900-second reference deadline, context limit24576. Base initial inputs are4,085–4,096 or8,180–8,191tokens. Four structurally selected IDs in16K follow-up have16,379/16,374/16,383/16,381tokens; this registration occurred after baseDEV, before extension inference. These are four length variants among the same eight tasks.

Free fixture-edit success requires both the correct final answer and the expected actual JSON file plus successful validation event. Exact tool names/argument keys matter; no aliases, grammar enforcement or automatic corrections were added for a candidate. Every failed episode's time is retained. Automatic runtime retries remain separate from repeated model tool choices. This differs from Case009/formal planning's180-second/prefix-cache serving contracts and is not native serving goodput evidence.

Core64 and extension32 are episode-condition counts, not96 independent tasks. Pairing uses shared task_id within cohort/mode. Marginal totals alone cannot determine wins/losses; [CPU audit](analysis/audit.py) reads actual per-ID outcomes. No post-hoc superiority thresholds or statistical badges are assigned.

## Environment and resources

Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`; Qwen3-4B-Instruct-2507 `cdbee75f17c01a7cc42f958dc650907174af0554`. RTX5080 SM120, driver610.88, Torch2.13.0+cu130, Transformers5.17.0 and existing vLLM0.29.0 Marlin. [Observed environment](provenance/observed_environment.json) is separate from [new CPU-check requirements](publication/requirements-cpu.txt); it is not an installation recipe.

r1 had input-preparation pyarrow/return_dict failures and GPU0. r2 used an existing CPU data environment and explicit return_dict=False. A tiny1e−15 zero-beta fixture is a synthetic contract failure, not a real-model failure rate. Original39 CPU tests are historical receipts; new publication tests/CPU readback are counted separately.

Worker wall2,251.160s includes loading/CPU/serialization/idle. NLL-loop wall includes metrics transfer/I/O; it is not TTFT/TPOT/throughput. Torch peak allocated/reserved is not Windows/system-wide GPU usage. System GPU peak, TTFT, TPOT and prefixcachedtokens are NOT_MEASURED. 0.6B RTN used a slower Python byte-iteration serializer; later methods used byte-identical bulk copy. D4 conversion columns are D16-artifact reloads. These values must not become one algorithm-speedup plot.

## Final status axes

| Axis | Status |
|---|---|
| Independent INT4 variant / complete-model quality | ALGORITHM_REFERENCE_EXECUTED |
| Long and tool tasks | DEV / REFERENCE_PILOT completed |
| Packed projection GEMM / one block | Numeric gate PASS |
| Packed full model | BLOCKED_TOLERANCE |
| Stock GPTQ paired serving | NOT_RUN |
| Formal120task / noninferiority / GO_SPEED / GO_CAPACITY | NOT_ESTABLISHED |
