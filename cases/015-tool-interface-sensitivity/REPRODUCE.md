# Recompute the API regression without a model

Uses only the Python3.10+ standard library. Execute saved responses and verify original files, scores and tables without a GPU, model or new installation. Run from the repository root; outputs go to a fresh external directory.

```bash
OUT=$(mktemp -d)
python3 -B cases/015-tool-interface-sensitivity/publication/scripts/run_checks.py --output "$OUT/case015" --tests
```

The wrapper restores the228 original P1 files into a new view and executes the unchanged closed-set verifier. A1 replays 96 recorded episodes into fresh order fixtures, recomputing its separate endpoint and regression table. The 27 P1 regression methods and 18 A1 unique tests remain distinct.

## A1: API definitions → equivalence → raw replay → regression table

```bash
A1=cases/015-tool-interface-sensitivity/followups/order-api-regression-v1
OUT=$(mktemp -d)
python3 -B -m json.tool "$A1/data/schema.json"
PYTHONPATH="$A1/src" python3 -B -m order_regression.cli validate
PYTHONPATH="$A1/src" python3 -B -m order_regression.cli oracle-check --output "$OUT/oracle.json"
PYTHONPATH="$A1/src" python3 -B -m order_regression.cli audit --output "$OUT/audit.json"
PYTHONPATH="$A1/src" python3 -B -m order_regression.cli analyze --output "$OUT/regression.json"
python3 -B "$A1/postprocess/render_review.py" --output "$OUT/explorer"
```

The 34 oracle episodes are 17 fixtures×2 APIs of synthetic contract tests, separate from 96 measured episodes. Open explorer/ko.html or en.html with file://. JSON and model output are displayed as text.

## Separate GPU replay boundary

Publication does not repeat GPU execution. A fresh empty private run root, original snapshot,252 RTN and252 D16 artifacts, original r2 metadata and compatible GPU environment are required. Restore private/runtime.json from the local record containing real paths; public hashes do not reconstruct those files or inputs.

```bash
# GPU_PYTHON and A1_PRIVATE refer to existing validated local inputs.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$A1/src" "$GPU_PYTHON" -B -m order_regression.cli run --private "$A1_PRIVATE"
python3 -B "$A1/postprocess/collect.py" --private "$A1_PRIVATE"
```

Completed run directories are rejected. Check source/spec/input/model/artifact identity first and preserve earlier outputs. The loader never creates missing artifacts. Guards reject the original P1 run_all, solver, encoding, ENUM and enhanced feedback.

[Report](REPORT.md) · [Stage map](STAGE_MAP.md) · [Original P1 commands](publication/originals/p1_review/REPRODUCE.md) · [Model identity](provenance/model_identity.json)

## Display v2 correction

The frozen original A1 renderer is preserved with its JavaScript newline defect. The postprocess wrapper above fixes that display separator and emits external explorer.js without changing raw data. Use this command instead of the frozen render entrypoint. Public v1 can be restored exactly through its separate map; model execution and scores are unchanged.
