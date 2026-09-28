# Case011 publication and recorded-data audit

[한국어 소개](../README.ko.md) · [English overview](../README.md) · [Reproduction](../REPRODUCTION.md)

The original study keeps the recurrent execution fixed and compares three final
readouts. This publication adds clearer bilingual navigation and analyses of the
same saved predictions. It performs no new fitting, learned-model forward,
checkpoint selection, codec change or GPU experiment.

## Read the results

- [Original primary results](../results/derived/primary.csv): INT8 MIXED−ORIGINAL
  RMST0 and the original three 98.333…% paired bootstrap intervals.
- [Post-hoc analysis](posthoc/data/summary.json): paired Native/INT8 interaction,
  band-level errors, first-error movement, controls and all 18 solver records.
- [Every input's first-error movement](posthoc/data/items.json): all 1,024 IDs in
  each checkpoint/storage, including longer, shorter and unchanged outcomes.
- [Read-only checkpoint/patch byte audit](checkpoint_head_audit.json): SHORT's
  exported two tensors are identical; MIXED changes 1,158 values in two tensors.

The interaction intervals are descriptive pointwise 95% intervals, separate
from the original primary family. Per-item logits/margins were not retained;
conditional margins on changed-error events are unavailable. Average stored
margin sums cannot supply that missing analysis.

## Two identities, two checks

[original_identity.json](original_identity.json) anchors all 140 original study
files to the 40,777,928-byte review ZIP, SHA256
`dcc8c18abf7333a38024b59e498f20fcace8a95f5a6d810e1e8f5890e1c27d27`.
Edited documents have their original bytes in [original_docs](original_docs/).
Scientific sources, inputs, freezes, measurements, figures and original receipts
remain at their original relative paths. [integrity.py](integrity.py) independently
checks the historical identity even if the publication inventory is regenerated.

[PUBLICATION_SHA256SUMS](PUBLICATION_SHA256SUMS) describes the current Case011 tree,
excluding itself. ZIP bytes and deployment/commit identifiers live in external
metadata and receipts; they are not inserted into a self-hashing archive.
Historical LOCAL_REVIEW and remote-write counts describe the original execution.

## Run without a model

From the repository root, use a Python environment with the pinned
[NumPy dependency](requirements-cpu.txt). Choose a new directory outside the clone:

```bash
python -B cases/011-frozen-state-readout-adaptation/publication/run_checks.py \
  --output /tmp/case011-publication-audit
```

This verifies identities, restores the original paths, recomputes the original
scalar results and the post-hoc analysis, and runs the new publication tests.
`--require-full-repository` additionally requires every protected historical file;
the standalone ZIP contains only the explicitly inventoried Case010 support subset.
Its receipt identifies the full-repository check as not run.

With the separate [CPU Torch requirements](requirements-synthetic.txt), add
`--synthetic` to run the original synthetic suite as well. `--synthetic-only`
runs that suite on the restored snapshot without repeating scalar analysis.
These are synthetic contracts, not new learned-checkpoint inference or fitting.

For direct access to historical files:

```bash
python -B cases/011-frozen-state-readout-adaptation/publication/restore_original.py \
  --output /tmp/case011-original-workspace
```

The new workspace contains the original 140-file case, 74 unchanged Case010
reference files and LICENSE. It is not the entire Case010 distribution.
The original model-free commands work in its Case011 directory. Real model
replay requires the identified private checkpoints, upstream and head patches;
none of those weights is supplied by the public ZIP.

## Verification scope

The original 12 fresh-process reload checks and 30 saved-boundary checks remain
historical receipts. Current publication checks re-read their identities and
recompute saved scalar results. Newly executed test counts and browser checks
are reported separately in CI/local receipts. CI uploads only small JSON/log
receipts, never checkpoints, head tensors, FIT features or runtime state bodies.

한국어: 원형 연구 140개 파일은 문서 사본을 포함해 복원할 수 있습니다. 이번 사후
분석은 같은 기록을 다시 계산한 것이며 새 실험이 아닙니다. 공개 묶음에서는 모델
없는 검산과 합성 CPU 검사를 실행할 수 있고, 원래 모델 및 보정 가중치는 별도로
필요합니다. 실제 게시·배포 상태는 GitHub PR, workflow와 외부 완료 기록에서 확인합니다.
