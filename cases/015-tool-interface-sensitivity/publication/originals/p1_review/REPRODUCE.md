# Reproduce saved evidence or rerun the local model

[한국어](REPRODUCE.ko.md). Run from the extracted Case015 directory. No installation is needed for CPU commands; Python3.10+ standard library is sufficient. Outputs must be new paths.

```bash
export PYTHONPATH="$PWD/src"
python3 -B -m interface_sensitivity.cli --help
python3 -B -m interface_sensitivity.cli validate
python3 -B -m unittest discover -s tests -v
python3 -B -m interface_sensitivity.cli oracle-check --output /tmp/case015-oracle-new.json
python3 -B -m interface_sensitivity.cli analyze --records results/p1_r1/episodes.jsonl --output /tmp/case015-analysis-new.json
python3 -B -m interface_sensitivity.cli audit --records results/p1_r1/episodes.jsonl --output /tmp/case015-audit-new.json
python3 -B -m interface_sensitivity.cli render --records results/p1_r1/episodes.jsonl --output /tmp/case015-explorer-new
```

`validate` checks input/mapping/frozen-source identities. `oracle-check` executes32real CPU oracle fixtures with48state/observation steps. `analyze` computes all96planned cells, missingness, task-paired contrasts, losses/gains and bootstrap. `audit` replays saved raw response text into fresh real fixture roots, recomputes legacy/functional verdicts independently and checks the primary scalar/bootstrap. `render` creates ko.html, en.html and data.json with static scores and escaped episode evidence. Existing output files are never replaced.

For same-interface diova-compare records, run the included export script described in the report; cross-interface records have different prompt hashes and must not be fed to identical-input pairing.

GPU rerun is a separate operation requiring the original read-only4Bcheckpoint, RTN/D16manifest and504rpi4files, and a **fresh** private directory with path resolver `runtime.json` matching the recorded identities. It uses the existing Python3.12.14/Torch2.13.0+cu130/Transformers5.17.0 environment on RTX5080. The bundle contains hashes, not model weights. No new model or quantization is downloaded/generated. The actual entrypoint is:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   /path/to/existing/reference/python -B -m interface_sensitivity.cli run --private /path/to/new-private
```

The original frozen protocol and private runtime resolver hash deliberately prevent silently changing paths or artifacts. A third-party rerun must explicitly register a new path-resolution amendment/source identity and retain the prior freeze; it is not an as-is one-command rerun from a weights-free ZIP. Model requests are greedy192tokens,8calls/6tool attempts,24,576context,900seconds/episode; worker wall7200seconds cumulative includes loading, smoke, decode, failures and I/O. Same request-local BF16 KV/SDPA/TF32/reduction flags as Case013, no across-turn cache or automatic truncation.

The CPU audit checks saved text/action/state/scalars, not excluded logits or the correctness of model inference. P0MonteCarlo, new quantizer, replay matrix, P2/P3, D4, training and remote publication are not run.
