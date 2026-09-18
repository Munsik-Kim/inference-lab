# Limits

- **Scoped transfer:** the 25% local advantage is observed on this frozen synthetic family. The 50% selectors produce the same module, so the fixed overall verdict is COMPLETED_NO_CLEAR_TRANSFER. This does not negate the measured 25% contrast or establish population equivalence at 50%.
- **One model/layer:** Qwen3-0.6B, zero-based layer 13, English synthetic tasks, 512 tokens, no retraining. The chosen MLP alone loses 25%/50% of its parameters. Other weights remain unchanged.
- **Weak baseline tasks:** BF16 correctness is retrieval 63/64, comparison 15/64, code 16/64. The latter two limit practical task-quality conclusions. Gold-choice NLL and choice accuracy are distinct; neither is free-generation quality. No deployment or non-inferiority assessment exists.
- **Sampled local outputs:** 32 deterministic positions are pooled per prompt, not independent tokens. Sampled standalone and full-prompt GEMM shapes can differ in rounding. Local error is not task-accuracy loss, and lower local error did not uniformly improve model scores.
- **Repeated designs:** twenty fixed random sets per budget reuse each prompt. The 50% selected modules are identical. Neither repetition expands independent sample size.
- **Timing resolution:** all model-prefill speedup intervals cross 1. Only six prompts and three processes on one device were timed. Slow blocks remain. Five-call block p95 is not request/service p95. Whole-device peak and continuous external load were not sampled; allocator peaks include resident comparison modules.
- **Audit boundary:** CPU paths independently recalculate scalar norms, selectors, paired outcomes and intervals. Public records retain prompt-level sufficient statistics and one worst sampled token per method; they cannot independently regenerate hidden tensors or full-vocabulary KL. This is calculation consistency, not third-party GPU replication.
- **History:** the earlier external-GPU resource block and partial package are preserved. Completion resumed only missing frozen stages; historical partial reports remain historical rather than current measurements.
- **Browser scope:** local file:// checks describe their actual OS/browser/viewports. Real iPad remains NOT_TESTED; no service or hosted demo was deployed.

Deployment: **NOT_ASSESSED**. No new model/layer, held-out oracle, recovery training or extra pruning search was run.
