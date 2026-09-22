# diova-compare

[한국어](README.ko.md) · [Project](../../README.md)

A small CPU CLI for comparing identified outputs from two model artifacts. It validates paired inputs, reports correctness transitions and answer disagreement, and computes probability diagnostics when a compatible distribution is provided. Python 3.10+; no runtime dependencies, model downloads, Torch or CUDA.

## Install and run

From this directory in a dedicated environment:

```bash
python -m pip install .
diova-compare validate --input /path/to/B.jsonl
diova-compare compare --baseline /path/to/B.jsonl --candidate /path/to/Q.jsonl --output /path/to/new-report
```

The output directory must not exist. The CLI writes `report.json`, `pairs.csv` and `report.md`. It runs from any working directory. A local wheel can be built with `python -m pip wheel --no-deps . --wheel-dir /path/to/wheels`; uploading it is a separate action.

## Record contract

Each JSONL row includes `schema_version: 1`, `sample_id`, `task`, `task_version`, SHA256 `prompt_hash`, `gold` (list of accepted strings), `gold_definition`, `model_id`, `artifact_id`, `output_type` (`choice` or `extracted_answer`), `answer`, boolean `correct`, and `evidence_kind` (`measurement`, `historical`, `synthetic_test`). Missing/duplicate keys, different prompts/gold definitions or artifacts mixed within an input are rejected. Intersection pairing is opt-in with `--intersection`; excluded keys remain in the report.

An optional `distribution` contains `kind` (`choice` or `full_vocabulary`), ordered `labels`, `probabilities`, `tokenizer_id`, `prefix_hash`, `dtype` and `normalization: probabilities_sum_to_one`. The validator requires finite nonnegative values summing to one within 1e-8. This is an input representation tolerance, not a tie rule. Labels must have identical order in the paired records. A full distribution also requires `complete: true` and `vocabulary_size` matching every entry. Its completeness is a caller contract; the CLI cannot recover omitted vocabulary scores or verify private logits.

Choice NLL and Brier require a single accepted gold label. KL is B→candidate; infinite KL from zero candidate support stays visible as a null numeric field plus an `_infinite` flag. NLL changes are candidate−baseline (negative is better). Free-form extracted answers receive correctness/disagreement metrics, not a forced choice Brier score. Official extraction and normalization own their correctness fields.

`correctness_flip` counts correct→wrong or wrong→correct. `all_answer_disagreement` additionally includes changed wrong answers and, where applicable, different accepted correct answers. `wrong_to_wrong` means changed wrong answers, while `both_wrong` includes unchanged wrong answers. [Definitions and prior work](../../docs/related-work/README.md).

Paired item bootstrap stays within each task; the default is 2,000 replicates with a fixed seed. Separate benchmarks are never averaged. Zero observed events and zero-width descriptive intervals do not establish population equivalence. Official MMLU subject/group aggregates and WikiText word/byte denominators belong to the harness reports, not to this generic choice interface.

An optional integer `process_exit_code` must be present on all records of both inputs or on neither. It is copied into pair records and task summaries: valid retained numbers do not turn a nonzero source exit into a successful execution.

## Implementation and checks

[Core](src/diova_compare/core.py) · [CLI](src/diova_compare/cli.py) · [Official sample adapter](src/diova_compare/adapters.py) · [Contract tests](tests/test_compare.py).

The adapter supports logged `acc` and exact-match samples. It does not relabel acc_norm or perplexity as ordinary accuracy. Source distributions can be passed through the core contract separately. Historical fixtures remain marked historical. Project license: [Apache-2.0](LICENSE). Codex assisted implementation and tests.

Adaptation provenance: the scalar definitions were reimplemented from the study conventions in [`tools/modelpack/numerics.py` at the repository base](https://github.com/Munsik-Kim/inference-lab/blob/9f1e7cbca19334a168561e05be8884bb16815d69/tools/modelpack/numerics.py). The original remains unchanged. This package adds a versioned record contract, arbitrary choice counts, extracted-answer support, explicit pairing failures and a standalone CLI; it does not import the frozen GPU tools.
