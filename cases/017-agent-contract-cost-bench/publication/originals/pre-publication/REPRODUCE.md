# Case017 reproduction

CPU verification re-parses saved model responses, reassembles their saved SSE,
executes accepted actions in new fixtures, independently calculates the endpoint,
and recomputes paired tables. It does not run the model, re-tokenize, or prove GPU
output reproducibility. The archive includes the five pinned Case016 source
dependencies needed for source-hash verification. Models and environments are
not included. Python 3.10+ standard library is sufficient for the public CPU path.

From this case directory, use a new external output directory:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
CASE017_CHECK_ROOT=$(mktemp -d)
python3 -B -m agent_contract_cost.verify --output "$CASE017_CHECK_ROOT/replay"
python3 -B -m unittest discover -s tests -v
python3 -B -m unittest discover -s postprocess/tests -v
python3 -B postprocess/verify_recorded_inputs.py --stage P2 --output "$CASE017_CHECK_ROOT/inputs"
python3 -B postprocess/inspect_contract_boundaries.py --stage P2 --output "$CASE017_CHECK_ROOT/annotations"
```

`replay/verification.json` records coverage. `P1-audit`/`P2-audit` contain fresh
endpoint rows; `P1-analysis`/`P2-analysis` are recalculated from those fresh rows.
Saved success flags and submitted summaries are not the endpoint oracle. All
present registered cells are checked. Missing cells remain missing.

Individual entry points also work:

```bash
python3 -B -m agent_contract_cost --help
python3 -B -m agent_contract_cost validate --output "$CASE017_CHECK_ROOT/validate"
python3 -B -m agent_contract_cost oracle-check --output "$CASE017_CHECK_ROOT/oracle"
python3 -B -m agent_contract_cost audit --stage P1 --output "$CASE017_CHECK_ROOT/p1-audit"
python3 -B -m agent_contract_cost.reporting --output "$CASE017_CHECK_ROOT/report"
python3 -B -m agent_contract_cost.review --output "$CASE017_CHECK_ROOT/explorer"
```

`oracle-check` uses scripted synthetic gold actions; it is not measured model
success. `reporting` creates bilingual Markdown and SVG files from audited
measurements. `review` generates `ko.html`, `en.html`, `data.json`, and a render
receipt. The pages work with `file://`; their core tables also work without
JavaScript. Source links are relative to the chosen output directory. The checked
local pages are under [demo](demo/ko.html).

CPU tests: execution contracts, fixture isolation/atomicity, observation ordering,
retry/attempt boundaries, SSE/token accounting, independent endpoint, freeze and
input gates, and display integrity. The optional `tests/tokenizer_probe.py` needs
the pinned local tokenizer and xgrammar; it is separate from standard-library
tests. Its recorded 58 token-mask probes are CPU integration evidence, not 58
additional unit tests or new model responses.

## GPU reproduction boundary

The frozen run uses an existing offline snapshot, vLLM0.32.0 environment, and private
runtime JSON. No model, package installation, new quantization, or GPU rerun is
performed by CPU verification. Read [model identity](provenance/model_identity.json),
[environment identity](provenance/environment_identity.json), [protocol](specs/protocol.json),
and the P0/P1/P2 execution freezes first. `git_commit` is the repository base,
not an execution-source commit; Git commit was outside this task's authorization.

The private runtime keys are `python`, `snapshot`, `case_root`, `original_case`,
`private_root`, `port`, `max_model_len`, `pool_bytes`, `seed`,
`served_model_name`, and `registered_client_context_limit`. Their path values are
not published. `check_runtime` intentionally rejects another configuration hash,
interpreter, source, or dependency identity. Do not edit an old freeze to bypass it.

The actual phase commands are:

```bash
<EXISTING_VLLM_PYTHON> -B -m agent_contract_cost.experiment p0 --runtime <PRIVATE_RUNTIME_JSON> --label execution_freeze_p0_r2
<EXISTING_VLLM_PYTHON> -B -m agent_contract_cost.experiment p1 --runtime <PRIVATE_RUNTIME_JSON>
<EXISTING_VLLM_PYTHON> -B -m agent_contract_cost.experiment p2 --runtime <PRIVATE_RUNTIME_JSON>
```

These completed result roots are immutable and those commands will refuse to
overwrite them. An independent GPU repetition requires a new isolated working
copy/run registration, equivalent pinned dependencies, the full pinned repository
(including Case016's model-identity receipt), a verified local snapshot, an unused
loopback port, and available GPU memory. Register new local path/config identities;
preserve the old source/spec/input hashes and compare tokenization before serving.
`prepare`, `freeze`, and `seal-holdout` are explicit registration commands described
by `experiment --help`. HOLDOUT must be sealed only after P1 completion. This local
review does not claim a second-machine GPU reproduction.

## Local archive

```bash
python3 -B -m agent_contract_cost.package --output "$CASE017_CHECK_ROOT/package"
```

This creates a new ZIP, extracts it elsewhere, checks CRC/path/duplicates/symlinks/
bytes/hash, and actually reruns CPU commands. The final ZIP hash and post-extraction
receipt are adjacent to the archive, avoiding self-hash cycles. No Git or remote
publication is performed.
The ZIP uses lossless LZMA compression to keep all token/SSE records small. Use
Python's `zipfile` or 7-Zip for extraction; built-in ZIP viewers may not support it.

Retained verification: [CPU receipt](results/cpu_verification/receipt.json),
[historical regression](results/cpu_regression/receipt.json),
[P1 input verification](results/p1_input_verification_r1/receipt.json),
[P2 input verification](results/p2_input_verification_r1/receipt.json),
[resource ledger](results/review/resource_ledger.json), and
[decision](results/review/decision.json). These have different evidence scopes;
historical reruns are not additional new Case017 tests.

One secondary-label limitation is preserved: the frozen scorer appends
`terminal_budget_or_infrastructure` to non-success terminal labels including
semantic failures. This extra label is not an exhaustion diagnosis. Read
`terminal_reason` and the actual counters. [The terminal-independent audit](results/terminal_independence_r1/receipt.json)
recalculates all256 endpoints without trusting the saved terminal eligibility;
it changes0 scores. To repeat it:

```bash
python3 -B postprocess/check_terminal_independence.py --output "$CASE017_CHECK_ROOT/terminal"
```
