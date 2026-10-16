# diova-compare

[한국어](README.ko.md) · [Project](../../README.md)

I built this **tool for comparing saved AI answers**. Give it two answer files from before and after a model change or compression, and it reports which questions became wrong, became correct, or received different answers. This exposes changes that average accuracy can hide.

Run the models and grade their answers first. This CPU command-line program (CLI) checks that both files use the same questions and grading rules, then summarizes the saved results. It does not run models, grade arbitrary free-form text, or authenticate source data and model files. Python 3.10+; no runtime dependencies, model downloads, Torch or CUDA.

<a id="compare-your-results"></a>
## Start with your own results

[Run the four-pair synthetic example](examples/compare/README.md): prepare saved results → fill fields using the same questions and grading rules → check format (validate) → compare → read report.md / report.json / pairs.csv. The guide covers caller-supplied free-form correctness, a minimal record without distributions, field mapping and an intentional pairing rejection.

Historical experiment reanalysis is a [separate path](#case013-export).

## Install and run

From this directory in a dedicated environment:

```bash
python -m pip install .
diova-compare validate --input /path/to/B.jsonl
diova-compare compare --baseline /path/to/B.jsonl --candidate /path/to/Q.jsonl --output /path/to/new-report
```

The output directory must not exist. The CLI runs from any working directory. Start with the task summary in `report.md`, inspect question-level changes in `pairs.csv`, and use `report.json` for settings, summaries and all comparison rows. A `PASS` result means validation and comparison completed, not that every answer was correct. A local wheel can be built with `python -m pip wheel --no-deps . --wheel-dir /path/to/wheels`; uploading it is a separate action.

## Record contract

JSONL is a text format with one question’s result per line. Each row includes `schema_version: 1`, `sample_id`, `task`, `task_version`, SHA256 `prompt_hash`, `gold` (list of accepted strings), `gold_definition`, `model_id`, `artifact_id`, `output_type` (`choice` or `extracted_answer`), `answer`, boolean `correct`, and `evidence_kind` (`measurement`, `historical`, `synthetic_test`). Here `prompt_hash` is the SHA256 check string for an input under a shared rule; `gold_definition` records how accepted answers were defined and extracted; `artifact_id` identifies the saved model files. `correct` is the caller’s correct/incorrect judgment for extracted free-form answers. `evidence_kind` distinguishes real measurements, historical records and synthetic examples. Missing/duplicate keys, different questions/grading rules or model files mixed within an input are rejected. Intersection pairing is opt-in with `--intersection`; excluded keys remain in the report.

Basic answer comparison needs no probabilities. For optional checks of how probabilities changed, `distribution` contains `kind` (`choice` or `full_vocabulary`), ordered `labels`, `probabilities`, `tokenizer_id`, `prefix_hash`, `dtype` and `normalization: probabilities_sum_to_one`. The validator requires finite nonnegative values summing to one within 1e-8. This is an input representation tolerance, not a tie rule. Labels must have identical order in the paired records. A full distribution also requires `complete: true` and `vocabulary_size` matching every entry. The caller is responsible for providing a complete distribution; the CLI cannot recover omitted vocabulary scores or verify private logits.

NLL measures loss from the probability assigned to the correct answer; Brier measures probability prediction error across choices. Both are lower-is-better and require a single accepted correct choice (gold label). KL measures the difference from original probabilities, in the baseline (B)→candidate direction; a smaller value means closer probabilities, without proving higher accuracy; infinite KL from zero candidate support stays visible as a null numeric field plus an `_infinite` flag. NLL changes are candidate−baseline (negative is better). Free-form extracted answers receive correctness/disagreement metrics, not a forced choice Brier score. Official extraction and normalization own their correctness fields.

`correctness_flip` counts correct→wrong or wrong→correct. `all_answer_disagreement` additionally includes changed wrong answers and, where applicable, different accepted correct answers. `wrong_to_wrong` means changed wrong answers, while `both_wrong` includes unchanged wrong answers. [Definitions and prior work](../../docs/related-work/README.md).

Statistical intervals repeatedly resample matched question pairs within each task (paired item bootstrap); the default is 2,000 replicates with a fixed seed. Separate benchmarks are never averaged. Zero observed events and zero-width descriptive intervals do not establish population equivalence. Use the original evaluation program’s reports (harness) for MMLU subject/group aggregates and WikiText word/byte denominators.

An optional integer `process_exit_code` must be present on all records of both inputs or on neither. It is copied into pair records and task summaries: valid retained numbers do not turn a nonzero source exit into a successful execution.

## Implementation and checks

[Core](src/diova_compare/core.py) · [CLI](src/diova_compare/cli.py) · [Official sample adapter](src/diova_compare/adapters.py) · [Contract tests](tests/test_compare.py).

The adapter converts saved evaluation logs into this input format. It supports logged `acc` (accuracy) and exact-match samples. It does not relabel `acc_norm` (length-adjusted choice scores) or `perplexity` (next-text prediction scores) as ordinary accuracy. Source distributions can be passed through the core contract separately. Historical fixtures remain marked historical. Project license: [Apache-2.0](LICENSE). Codex assisted implementation and tests.

Adaptation provenance: the metric formulas were reimplemented from the study conventions in [`tools/modelpack/numerics.py` at the repository base](https://github.com/Munsik-Kim/inference-lab/blob/9f1e7cbca19334a168561e05be8884bb16815d69/tools/modelpack/numerics.py). The original remains unchanged. This package adds an input format with version numbers, arbitrary choice counts, extracted-answer support, explicit failures when the same questions cannot be matched (pairing) and a standalone CLI; it does not import the frozen GPU tools.

## Version 0.1.1: reviewed bug fixes

[Download the CPU wheel](../../downloads/diova_compare-0.1.1-py3-none-any.whl) and install it in a separate Python environment:

```bash
python -m pip install --no-index --no-deps /path/to/diova_compare-0.1.1-py3-none-any.whl
```

No PyPI upload. Within a task/version, effective NLL/Brier and KL coverage must match across all items. Different tasks may have different metrics. Tiny positive probabilities are retained; exact missing mass uses null plus an infinity flag. [Changelog](CHANGELOG.md).

<a id="case013-export"></a>
## Historical Case013 export

Version 0.1.2 adds a stdlib-only historical Case 013 exporter. From the repository root, after installing this package:

D16 and D4 name two conditions using the exact same INT4 weight files. D16 keeps calculation inputs in BF16; D4 also reduces inputs to INT4 and restores them to BF16. Both restore values for BF16 quality calculations. They are not different model sizes and do not demonstrate faster low-bit execution. This exporter reads the original records, before the later tool-name restriction.

```bash
diova-compare export-case013 --case cases/013-dualquant-reference-quality --output /tmp/case013-export
diova-compare validate --input /tmp/case013-export/agent/base/FREE/RTN.jsonl
diova-compare compare --baseline /tmp/case013-export/agent/base/FREE/RTN.jsonl --candidate /tmp/case013-export/agent/base/FREE/D16.jsonl --output /tmp/case013-agent-report
```

All output directories must be fresh. The exporter verifies the public files it reads, reapplies the original answer extraction and action-checking rules, and retains raw text, tool records and actual test-file state (fixture) in companion files (sidecars). A matching final JSON can still be incorrect when the required edit/validation was not executed. Execution modes, task groups and input lengths stay separate; unexecuted candidate32K conditions are excluded rather than assigned zero scores. Probability loss (NLL), next-text prediction score (PPL) and saved probability difference (KL) aggregates remain in a separate numerical summary. No model, Torch, NumPy or new GPU measurements are needed. See the [Korean workflow](README.ko.md#case013-export).
