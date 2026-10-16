# Compare your own saved answers

[한국어](README.ko.md) · [Package](../../README.md)

This CPU CLI compares already recorded answers on matched prompts and an independent gold rule. Run the models and grade/extract their answers before using it. The two fixture files contain four hand-authored **synthetic_test** records, not GPU measurements.

## Run from the repository root

Python 3.10+; use a new environment and output path for each attempt:

```bash
EXAMPLE_ROOT=$(mktemp -d)
python3 -m venv "$EXAMPLE_ROOT/venv"
"$EXAMPLE_ROOT/venv/bin/python" -m pip install --no-deps ./packages/diova-compare
"$EXAMPLE_ROOT/venv/bin/diova-compare" validate --input packages/diova-compare/examples/compare/baseline.jsonl
"$EXAMPLE_ROOT/venv/bin/diova-compare" validate --input packages/diova-compare/examples/compare/candidate.jsonl
"$EXAMPLE_ROOT/venv/bin/diova-compare" compare --baseline packages/diova-compare/examples/compare/baseline.jsonl --candidate packages/diova-compare/examples/compare/candidate.jsonl --output "$EXAMPLE_ROOT/report"
```

Both validation commands return PASS with 4 records. Comparison returns PASS with 4 pairs, exit 0, and writes `report.md`, `report.json`, `pairs.csv`. Build tools may be fetched during installation; the CLI needs no model, Torch, CUDA or NumPy.

[Baseline](baseline.jsonl) · [Candidate](candidate.jsonl)

## Read the report

Gold is the arithmetic sum, independent of either answer. Hash the exact UTF-8 prompt below without a trailing newline. Extracted answers are whitespace-stripped decimal integer strings.

| ID | Exact prompt | Gold | Baseline → candidate |
|---|---|---|---|
| sum-01 | What is 2 + 2? Reply with an integer. | 4 | 4 → 5 |
| sum-02 | What is 3 + 5? Reply with an integer. | 8 | 7 → 8 |
| sum-03 | What is 6 + 1? Reply with an integer. | 7 | 6 → 9 |
| sum-04 | What is 4 + 5? Reply with an integer. | 9 | 9 → 9 |

Both have 2/4 correct, but `tasks["integer-addition@1"].counts` separates `correct_to_wrong=1`, `wrong_to_correct=1`, `wrong_to_wrong=1`, `all_answer_disagreement=3`. Read per-item rows in `pairs.csv`; JSON retains configuration, summaries and all pairs. Four synthetic pairs do not rank real models.

## Map your records

| Source value | Fields |
|---|---|
| Dataset/version/item key | `task`, `task_version`, `sample_id` |
| Actual input under a shared serialization rule | SHA256 `prompt_hash` |
| Independent accepted answers and extraction/grading rule | `gold`, `gold_definition` |
| Model and checkpoint identifier | `model_id`, `artifact_id`; one combination per file |
| Extracted answer and caller-supplied correctness | `output_type: extracted_answer`, `answer`, boolean `correct` |
| Provenance and schema | `evidence_kind: measurement/historical/synthetic_test`, integer `schema_version: 1` |
| Optional source exit status | integer `process_exit_code` on every row of both inputs, or omitted throughout |

Use `hashlib.sha256(prompt.encode("utf-8")).hexdigest()` and include system/template text in the shared input rule. The CLI checks matching identifiers; it does not authenticate the original prompts or artifact bytes. For extracted answers it consumes correctness supplied by your grading rule, rather than grading arbitrary text itself.

Neither `distribution` nor `scores` is required for answer/correctness comparison. Compatible optional probabilities enable additional diagnostics; full-vocabulary probabilities/logits are not a minimum requirement. [Full schema](../../README.md#record-contract).

## See pairing rejection

After the successful commands, create a mismatched copy:

```bash
"$EXAMPLE_ROOT/venv/bin/python" - "$EXAMPLE_ROOT/mismatch.jsonl" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in Path("packages/diova-compare/examples/compare/candidate.jsonl").read_text().splitlines()]
rows[0]["prompt_hash"] = "0" * 64
Path(sys.argv[1]).write_text("".join(json.dumps(row) + "\n" for row in rows))
PY
"$EXAMPLE_ROOT/venv/bin/diova-compare" compare --baseline packages/diova-compare/examples/compare/baseline.jsonl --candidate "$EXAMPLE_ROOT/mismatch.jsonl" --output "$EXAMPLE_ROOT/rejected-report"
```

Expected: `diova-compare: pair mismatch: prompt_hash`, exit **2**, no report directory. A changed gold definition produces `pair mismatch: gold_definition`. Each input can validate alone while pairing fails; `--intersection` does not repair a changed prompt or grading contract.

## Technical judgment: current design

Chosen: paired losses, gains and disagreement, with source exit status retained. Alternative: mean accuracy alone hides the example’s regression. Evidence: [core](../../src/diova_compare/core.py) and [contract tests](../../tests/test_compare.py). Limit: free-form grading and provenance remain caller responsibilities. This describes the current design, not an additional historical alternatives experiment.

The [historical Case013 exporter](../../README.md#case013-export) is a separate route with a fixed action/extraction contract. Keep private data out of shared examples/reports. No browser upload is involved. Running this fixture is not an external usability test.
