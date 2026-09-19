# Scope and remaining limits

- Q is one GPTQ recipe on Qwen3-4B-Instruct-2507 in vLLM; R is three fixed layer-13 structures on Qwen3-0.6B in PyTorch. They are not a common performance scale or a combined compression pipeline.
- Inputs are short English synthetic retrieval/comparison/integer-program tasks, sharing a few templates. Caps of 1,024/512 tokens do not mean those exact lengths were tested. Weak baseline task strata remain in the tables.
- Local teacher reconstruction, baseline KL, gold probability and correctness answer different questions. Native BF16 teacher errors can also be learned by a reconstruction fit. No task-utility non-inferiority tolerance or deployment approval was specified.
- The paired bootstrap describes uncertainty under the small scenario design. It does not certify general task quality, zero regression risk, or equivalence. Q/R datasets, dependent tokens, timing repetitions and reloads are not additional independent test samples.
- R fits weights by ridge regression, without backpropagation or retraining other parameters. It is parameter fitting, not a claim of no learning. No new selector, held-out oracle, GPTQ/SparseGPT algorithm or universal compression rule is proposed.
- Native BF16 arithmetic, shapes and backend matter. The R FP64 fit and BF16 inference are recorded separately. Higher precision analysis does not recover bits lost in native outputs.
- The public scalar checker reconstructs choice scores, norm-based errors, paired outcomes and intervals. Full-vocabulary KL, private hidden/weight arrays, actual kernel execution and complete model reloads require excluded local material. Calculation consistency is not external GPU replication.
- Q one-token request time includes vLLM scheduling, sampling and readback. R prefill plus argmax is a different boundary. No API/server latency or cross-runtime speed ranking is claimed.
- Whole-device peak sampling and CUDA-event timing are not measured. Resident parameter bytes, file bytes and allocator reservations are separate quantities. Short fixed eight-token timing is not a natural long-generation workload.
- Initial Q fixture failures and their compatibility/instrumentation corrections remain in provenance. They are not held-out replicates. A backend failure must not trigger an unreported precision fallback.
- Model weights, full logits/H/Y and build environments stay private. Model redistribution, GitHub publication and deployment are separate decisions.

## Same-record code-task review

The fixed generator links loop count to gold position (`count = 2 + index % 4`, `gold = index % 4`) and makes the correct numeric answer the smallest option. All Q/R code splits have these cues. Q chooses D on all 64 held-out code prompts; every R arm chooses A. Each consequently scores 16/64; Q's full-vocabulary argmax is outside the allowed labels on all 64 prompts in both arms. These observations limit claims about code reasoning and free-generation instruction following. They do not establish that the models used the cues, explain ridge recovery, or demonstrate split leakage. Original inputs/results remain frozen. [Task-wise post-hoc analysis](REPORT.md#analysis) · [한국어](REPORT.ko.md#analysis).
