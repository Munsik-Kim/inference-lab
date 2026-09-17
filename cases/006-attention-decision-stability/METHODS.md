# Methods

## Scope and executable boundary

The model is Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`. All arms use the same BF16 weights, output head, tokenizer, non-thinking chat template and exact token IDs. Batch size is one, with no padding. Only zero-based layer 13's square causal prompt prefill changes: 16 query heads, 8 KV heads, head dimension 128. All other layers and every single-token decode step use native BF16 SDPA. This is neither weight quantization nor an all-layer intervention.

| Arm | Explicit operator | Effective bundle |
|---|---|---|
| B | Installed Qwen SDPA | BF16 fused Flash path observed in a diagnostic profiler run |
| A_PUBLIC | `sageattn_qk_int8_pv_fp8_cuda` | INT8 QK, FP8 PV, `per_warp`, `smooth_k=True`, `smooth_v=False`, `pv_accum_dtype="fp32+fp16"`, V scale maximum 2.25 |
| V4 | `sageattn_qk_int8_pv_fp16_cuda` | INT8 QK, FP16 PV, `per_warp`, `smooth_k=True`, `smooth_v=False`, `pv_accum_dtype="fp32"`; required V conversion inside the call |

Both candidates use HND tensors and return BF16 attention output. These are implementation bundles, not a single ordered precision variable. SageAttention is pinned to `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`. The scoped adapter delegates all non-target calls and restores the original registry entry even on exceptions. Runtime function/operand probes and profiler kernel names support execution attribution; no new SASS-level accumulator claim is made.

## Data and answer interface

Facts and exact gold answers are computed before rendering. RETRIEVAL binds a value to a key; COMPARISON retrieves records and compares their values; CODE evaluates a restricted program through a bounded whitelist AST interpreter. No `eval`, arbitrary execution or model judge is used. Foils must be distinct and wrong. Gold, support IDs and latent facts are stored separately from model requests.

The seed is 606160. English instructions and restricted Python are used; this is not a bilingual benchmark. Six smoke and 96 development base scenarios precede 192 unfiltered standard scenarios. A separate 192-scenario BF16-only pool supplies the baseline-conditioned stress set. Stable lexical IDs select up to six non-tied scenarios per task per raw-logit winner-gap bin: (0,0.1), [0.1,0.5), [0.5,1), [1,infinity). Exact ties and empty bins remain recorded, without refill. This conditioning prevents treating stress rates as ordinary-workload rates.

Primary prompts contain exactly 4096 tokens including the chat template. A deterministic irrelevant archive-note field and punctuation supply filler without truncating question, facts or labels. The repeated filler and shared templates narrow the input distribution. Exact text, tokens and hashes are published. Distinct base IDs, fact/program identities and cross-split overlap checks are retained. Repeated lengths and label rotations are not new scenarios.

The allowed continuations are A/B/C/D, token IDs 32/33/34/35. The complete prefix-plus-label tokenization is checked for prefix stability and exactly one new token. The observed prompt includes no gold continuation. Native `logits_to_keep=1` returns only the final prompt position's logits. No unseen continuation enters whole-sequence quantization statistics.

The BF16-only development audit made zero presentation revisions. One scenario per task was cyclically relabelled in four dependent variants. Weak comparison/code utility and label-position sensitivity are retained rather than filtered. Length support uses the first 16 hash-ordered standard IDs per task at 512 and 2048 tokens. Generation and timing use respectively the first eight and four hash-ordered IDs per task.

## Local freezes and validity

[Design freeze](configs/design_freeze_v1.json) followed development. [Evaluation manifest freeze](configs/eval_manifest_freeze_v1.json) followed BF16-only pool selection and preceded candidate access to evaluation IDs. They record 19 critical source hashes; these remain unchanged. These are local execution records, not external preregistration. Some inherited draft strings still say “pending” or “planned”; the freeze's stage fields and the retained integration report resolve their actual status without rewriting the frozen object.

The first integration diagnostic rejected a bitwise comparison between native BF16 LM-head calls with different matrix shapes. Investigation found identical hidden states and exact agreement with the same-shape, final-position LM head; full-position versus last-position logits differed by at most 0.03125. Every experimental arm uses the same native last-position path. Both the initial failure and corrected diagnostic are retained. Native versus B passthrough and B repeat checks were bitwise equal.

Primary checks cover complete attention outputs, every decoder-block output, final normalized hidden states and the full returned vocabulary logits. Identical layer-13 Q/K/V hashes, input immutability, BF16 dtype, mask/scale/GQA/layout, routing and standalone-versus-integrated operator correspondence are checked. Absent or nonfinite evidence blocks validity. This replaces the unexecuted Case005 fresh-path shortcut that passed literal validity flags; Case005 itself is unchanged.

## Scores and independent units

CPU FP64 analysis operates on native BF16-rounded logits; it does not recover precision already lost. For the four option logits s, q=softmax(s), gold y:

- Choice NLL = logsumexp(s) - s[y]. Choice Brier = sum over four labels of `(q[j] - 1[j=y])²`, without division by four.
- Gold margin = s[y] minus the largest other option logit. Winner gap = largest minus second-largest option logit, irrespective of gold.
- Full-vocabulary gold NLL uses the full log-normalizer. Allowed-label mass M is also retained; `NLL_full = NLL_choice - log(M)` is checked.
- Full-vocabulary KL is computed from complete final-token vectors, not a top-k surrogate. Four-logit evidence alone cannot independently reconstruct it. Full vectors remain private.

The primary estimand is the equally task-weighted mean paired change in gold choice NLL on standard L4096, separately for both candidates versus B. Positive means a worse gold log score. Brier, margins, accuracy and full-label mass accompany it. Forced-choice accuracy is not free-generation compliance.

Correctness cells are both-correct, regression, gain and both-wrong. Wrong-to-wrong label changes are included in flips. Conditional regression/gain rates retain B-correct/B-wrong denominators; an empty denominator is undefined. B is a comparator, not the oracle.

Bootstrap seed 606901 uses 5000 paired base-scenario cluster draws within task, with equal task weights and linear percentile intervals. Heads are dependent diagnostics. Per-length tables are separate; repeated lengths do not increase the independent sample size. Pointwise intervals do not establish equivalence or non-inferiority. Zero observed events do not establish zero population risk. An optional exact zero-event bound explicitly assumes independent Bernoulli sampling, which does not describe the stratified stress set.

`R = (max(delta_logits)-min(delta_logits))/B_winner_gap` is undefined at ties. A gap greater than this oscillation guarantees the unique winner is unchanged; R at least one does not guarantee a flip. It requires both outputs and is an after-the-fact identity check, not a learned risk predictor or router.

## Local and hidden-state diagnostics

A shared FP32 reference uses identical BF16 Q/K/V, FP32 score/softmax/PV, TF32 disabled, and the 32 query positions `floor(i*(N-1)/31)`. Each query uses every causal-valid key; all 16 heads are included. Head/query chunks avoid a full multi-layer attention matrix. The FP32 output is never injected into the model.

For a unit, relative error is the square root of squared-error sum divided by reference squared-norm sum. Reference RMS at or below 1e-6 produces a null relative value, not an epsilon-adjusted pass. Item pooling sums norms/errors across heads before taking the ratio; last-query error is separate. Candidate-versus-native-BF16 error uses a different reference and is labelled separately.

Retained last-position boundaries are attention output before layer-13 o_proj, post-attention residual before post_attention_layernorm, block outputs 13/14/18/22/27, final norm, and final logits. Absolute RMS and reference RMS accompany normalized differences. Percentage ratios across different denominators are not an absolute amplification factor. Public scalar norms permit calculation checks, not regeneration of omitted hidden tensors.

## Bounded generation and timing

The 24 preselected scenarios receive a separate frozen JSON-output prompt requesting choice and supporting record IDs. Greedy decoding is an experimental control, capped at 64 new tokens. Each arm keeps its own prompt-derived cache. Decode is BF16.

Common-prefix diagnostics score each position before feeding the next B token; no future token is visible. Free-running diagnostics consume each arm's own tokens. A later same-index token or eight-token alignment does not imply cache/state recovery. EOS, schema validity, answer/evidence correctness and censoring are separate. A separately executed uninstrumented request records actual length and elapsed time; it is not a service latency benchmark.

Model-prefill timing uses 12 fixed scenarios, three separate local processes, five warmups per arm/input, five paired blocks of five calls. Input IDs are GPU-resident; request caches reset each call. The measured call includes model forward, native last-position LM head, intervention wrapper and required conversions. It excludes tokenizer, reference, diagnostic finite scans, hooks, profiler and file I/O. Wall time synchronizes before and after each block. CUDA events and local first-token argmax use separate blocks. Block-mean p95 is not request p95. No tokenizer, network/server TTFT or pure decode-speed claim is made.

Speedup is the median of paired B/candidate wall ratios. Seed 606902 uses 5000 resamples of base scenarios and shared global process IDs, then paired blocks inside cells. The same process draw is shared across scenarios. These are same-device pointwise intervals. Torch allocated/reserved peaks and sampled whole-device readings have distinct scopes. All arms retain BF16 weights, so no model-weight memory savings are claimed.

## Reanalysis limits

Production analysis and the independent scalar checker use distinct score/norm/aggregation paths. Timing has its own independent scalar/bootstrap checker. This establishes calculation consistency, not third-party kernel replication. Reproducing GPU outputs requires the pinned local snapshot, recorded environment and capture/intervention code. The public package excludes full activations, full logit vectors, model weights and environments.
