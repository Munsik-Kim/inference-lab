# Short quality lifecycle diagnostic v1

[Case report](../../REPORT.md) · [Protocol](protocol.json) · [Attempts and repeat checks](summary.json) · [Environment](environment.json)

**NOT_REPRODUCED_IN_SMALL_PROBE**. D0 and D1 each ran two fresh-process repetitions for BF16/W4 and generation/continuation/rolling likelihood: **24 attempts, 24 clean, 0 failed**, two short synthetic requests per process. Walltime was 543.3 seconds under the 3,600-second cap. No full-split retries or new environment were used.

D0 retained the original environment and the existing `engine_core.shutdown(timeout=30)` flow. D1 additionally deleted model/request references and collected Python garbage before normal interpreter exit. This is an object-lifetime bundle; adding shutdown was not a fix because the historical code already called it. All core workers reported exit 0 and the external parent independently observed returncode 0 and no remaining recorded workers. Result shape/finiteness and repeated output hashes are retained.

Inputs were frozen synthetic strings under 256 tokens, at most 32 generated tokens, with the real pinned harness APIs. The rolling API ran, but these short strings did not recreate full-context WikiText windows. The original graph/runtime configuration was reused; no latency conclusion is drawn from probe durations. D2 was not selected. Small-probe success does **not** resolve historical full-task failures, which remain four clean/four nonzero exits. Full evaluation would require a separate follow-up authorization.

The public environment audit replaces private roots with semantic placeholders. Private stderr, native library paths and stage logs stay local; public hashes and OS signals are retained. The inherited two-site-packages layout and NumPy 2.5.3 / optional Numba warning are observations, not established causes. No dependency was upgraded.

Run `python -B run.py --private-config /path/to/config.json --output /path/to/new-run` with the verified local checkpoints and frozen environment. The private config supplies `python`, `models` (BF16/W4), `tokenizer` and environment flags; public protocol and runners fix the design. Existing attempt directories are never reused. This is a GPU diagnostic, not a required CPU package installation test.
