# Case015 P2 — Same-observation JSON recovery

A synthetic purchase workflow had already received lookup/multiplication results when D16 malformed the write-call JSON. We implemented checkpoint execution, focal grammar, paired neutral/error-feedback retries and model-free raw replay. C0 is the original response; G constrains its format; N retries without an error explanation; E adds a parse-error explanation.

| Model | C0 | G | N | E |
|---|---|---|---|---|
| BF16 | 7/8 | 7/8 | NOT_APPLICABLE | NOT_APPLICABLE |
| RTN | 8/8 | 8/8 | NOT_APPLICABLE | NOT_APPLICABLE |
| D16 | 0/8 | 8/8 | 0/8 | 0/8 |


All 8 selected D16 failures reproduced exactly. G recovered real write/validation/final completion 8/8; N/E completed 0/8, with one correct stored state under N. The 8 checkpoints come from 4 selected fixtures × two naming versions, not independent held-out tasks. BF16/RTN retries were inapplicable.

[Report](REPORT.md) · [한국어](README.ko.md) · [CPU/GPU reproduction](REPRODUCE.md) · [Runner](execution/run.py) · [Raw](results/gpu_r2/cells) · [Verification](publication/verification/latest.json)

175 new requests = 64 focal + 111 continuation, 6,986 generated tokens, 0 smoke. Saved INT4 is restored to dense BF16. Original A1 failures remain failures; no pooled P1/A1/P2 rate. No main merge or Pages deployment.

```bash
P2=cases/015-tool-interface-sensitivity/followups/json-recovery-p2-v1
OUT=$(mktemp -d /tmp/case015-p2-review-XXXXXX)
python3 -B "$P2/publication/run_checks.py" --output "$OUT/checks" --tests
```

Use `--grammar-python` with existing CPU dependencies for the five grammar tests. GPU reproduction uses pinned source `828ac7358b653626bed7e0a609448e73d120f1bb` and privately held original model/artifacts. The old CPU-only CLI remains closed; execution/run.py is the separate authorized path.
