# Case 013 follow-up: tool-name generation and recovery

Implemented registry-aware error feedback and a tool-name enum decoder, then reran the original eight DEV tasks on RTX 5080. D16 changed from 6/8 baseline to 8/8 with both changes; the combined condition had 0 unregistered-name events. D4 combined success was 2/8. These are system mitigation results on observed DEV, not a new quantizer or held-out evaluation.

## Method

BASELINE retains free decoding and generic errors. FEEDBACK distinguishes unknown names from invalid arguments and lists actual registered names. ENUM masks only the top-level tool-call name string; tool use, arguments and final answers remain model-generated. BOTH combines the changes. No aliases, answer repair or gold-dependent decoding are used. ASCII names/keys and batch1 are supported; escaped name/key forms are rejected.

The same Qwen3-4B revision cdbee75f17c01a7cc42f958dc650907174af0554, saved RTN/D16 artifacts, 252 projections and dequantized BF16 reference arithmetic are unchanged. D4 retains the existing activation variant. Greedy generation: 192 output tokens per call, 8 model/6 tool calls per episode, 900-second deadline and 24,576 context limit. Within-request KV is on; across-turn prefix cache is off. Initial prompts are paired; later feedback differences are the registered intervention. Four families, two tasks each, are reused across all conditions.

## Results

Task success uses the original scorer. Patch tasks require actual expected config and successful validation, in addition to final JSON.

| Model condition | Original | Feedback | Name enum | Both |
|---|---:|---:|---:|---:|
| BF16 | 8/8 | NOT_RUN | NOT_RUN | NOT_RUN |
| RTN | 8/8 | NOT_RUN | NOT_RUN | NOT_RUN |
| D16 | 6/8 | 8/8 | 8/8 | 8/8 |
| D4 | 0/8 | 0/8 | 2/8 | 2/8 |


Event counts are not failed-episode counts:

| Method | Condition | Success / tasks | Unknown-name events | Tool errors | Parse errors | Model requests |
|---|---|---:|---:|---:|---:|---:|
| BF16 | BASELINE | 8/8 | 0 | 0 | 0 | 18 |
| RTN | BASELINE | 8/8 | 0 | 0 | 0 | 19 |
| D16 | BASELINE | 6/8 | 8 | 8 | 0 | 20 |
| D16 | FEEDBACK | 8/8 | 4 | 4 | 0 | 20 |
| D16 | ENUM | 8/8 | 0 | 0 | 0 | 18 |
| D16 | BOTH | 8/8 | 0 | 0 | 0 | 18 |
| D4 | BASELINE | 0/8 | 2 | 2 | 2 | 13 |
| D4 | FEEDBACK | 0/8 | 5 | 5 | 2 | 15 |
| D4 | ENUM | 2/8 | 0 | 0 | 2 | 12 |
| D4 | BOTH | 2/8 | 0 | 0 | 2 | 12 |


Paired counts on the same task IDs:

| Method | Candidate vs baseline | Both pass | Baseline only | Candidate only | Both fail |
|---|---|---:|---:|---:|---:|
| D16 | FEEDBACK | 6 | 0 | 2 | 0 |
| D16 | ENUM | 6 | 0 | 2 | 0 |
| D16 | BOTH | 6 | 0 | 2 | 0 |
| D4 | FEEDBACK | 0 | 0 | 0 | 8 |
| D4 | ENUM | 0 | 0 | 2 | 6 |
| D4 | BOTH | 0 | 0 | 2 | 6 |


Failed task IDs:

- BF16 / BASELINE: none
- RTN / BASELINE: none
- D16 / BASELINE: AGENT-6, AGENT-7
- D16 / FEEDBACK: none
- D16 / ENUM: none
- D16 / BOTH: none
- D4 / BASELINE: AGENT-0, AGENT-1, AGENT-2, AGENT-3, AGENT-4, AGENT-5, AGENT-6, AGENT-7
- D4 / FEEDBACK: AGENT-0, AGENT-1, AGENT-2, AGENT-3, AGENT-4, AGENT-5, AGENT-6, AGENT-7
- D4 / ENUM: AGENT-0, AGENT-1, AGENT-2, AGENT-3, AGENT-4, AGENT-5
- D4 / BOTH: AGENT-0, AGENT-1, AGENT-2, AGENT-3, AGENT-4, AGENT-5

## Interpretation

Feedback recovers D16's two failures while retaining four initial unknown-name events. Enum decoding recovers the same failures with zero unknown names. ENUM and BOTH have the same observed success/error counts; because no other tool errors occur under enum decoding, an additional benefit from feedback in the combined arm is not demonstrated. The existing diova-compare CLI independently agrees on two recovered tasks and no new losses. D16's extracted final answers are identical across baseline/BOTH; the difference is actual fixture execution. CLI intervals are POST_HOC_SAME_RECORDS descriptive outputs.

This tests whether decoding/feedback changes recover previously observed task failures. Feedback may repair behavior after the first erroneous call; enum decoding may prevent that name from being emitted. Neither establishes a numerical fix to DualQuant. Correct names do not guarantee correct tool choice, arguments, state tracking or final answers. Original Case 013 and the earlier unsuccessful single-projection restoration remain unchanged. No generalization, noninferiority, packed speed or context-capacity benefit is established.

## Verification and cost

4 GPU workers; 80 episodes / 8 original task IDs / 10 logical conditions; 165 generation requests. Conservative worker wall 517.614 s. Torch peak allocated 10,690,878,464 B (9.957 GiB), reserved 14,241,759,232 B (13.264 GiB).

Worker wall includes loading, artifact reads, CPU masking and serialization. It is not TTFT, TPOT or native INT4 kernel time. Torch allocated/reserved differ from device-wide and Windows memory, which were not sampled. Seven unique CPU tests and one real-tokenizer mask smoke passed. A stdlib audit independently replayed tool events/config, original success rules, pairs and denominators; historical test counts were not added. No full-logit/weight recomputation was claimed.

All 5,743 Case001–013 files and original r2/prior follow-up/source inventories are preserved. Large private artifact/array preservation uses size/mtime; used projection payload hashes were checked on load. No solver, training, download, existing-environment upgrade or remote publication. Model weights and large artifacts are excluded from the review bundle. The original reference and harness are reused; this work adds the decoder, feedback, bounded factorial runner and audit. Codex assisted implementation and reporting.

The closed Case 013 public verifier also passed for 519 files. Its first system-Python attempt lacked NumPy; the existing reference Python then passed the CPU-only check without installing packages. Both attempts are retained in validation_attempts.json.

## Reproduce

See [the execution guide](README.ko.md), [decision](decision.json), [independent verification](verification.json) and [actual GPU commands](gpu_commands.json). The bundle's CPU replay explicitly skips certification of absent external originals. The full GPU run requires the existing reference environment, original model and saved projection artifacts.
