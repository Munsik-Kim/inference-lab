# Case011 — Recompute records, test contracts, or replay the model

[한국어 보고서](REPORT.ko.md) · [English report](REPORT.md) · [Methods](METHODS.md)

## 1. Model-free publication checks

From the repository or extracted public ZIP root, use Python with NumPy and a **new external output directory**:

```bash
python -B cases/011-frozen-state-readout-adaptation/publication/run_checks.py \
  --output /tmp/diova-case011-publication-check-new
```

This is the current entry point for source preservation, original snapshot restoration, public scalar reanalysis and the separately labelled post-hoc records. It does not fit a head, run a checkpoint, or download a model. Original research receipts and current publication checks remain separate.

To restore the exact original Case011 at its historical relative path, without overwriting an existing directory:

```bash
python -B cases/011-frozen-state-readout-adaptation/publication/restore_original.py \
  --output /tmp/diova-case011-original-new
```

The public package includes a documented unchanged Case010 support subset needed by the CPU contracts. It is not a full copy of the earlier cases. Cross-case narrative links and full-site checks require the repository checkout.

### Direct original scalar auditor

Use Python with NumPy. The recorded run used Python 3.12.14 and NumPy 2.3.5;
plots additionally used Matplotlib 3.11.2. No GPU, Torch, upstream checkout,
checkpoint or head weights are required to recompute the public metrics.
From this Case011 directory, choose a **new** output directory:

```bash
python -B analysis/aggregate.py \
  --input-only-fit results/input_only_fit.json \
  --output /tmp/case011-new-audit --no-figures
```

This independently recalculates first error, RMST0, every-token empirical
T0.05, recovery, accuracy and the paired bootstrap from public predictions.
It verifies all six cells and their ordered sample/checkpoint identities.
CE and margins are recomputed from the retained finite score sums and counts,
not from unreleased full logits. Removing `--no-figures` also builds the plots.
An existing output directory is rejected, so select another directory for a
repeat audit. The output is separate from the retained original results.

For one cell:

```bash
python -B analysis/audit.py \
  --predictions results/fresh/seed0/UNIFORM_8/predictions.npz \
  --summary results/fresh/seed0/UNIFORM_8/summary.json \
  --checkpoint-seed 0 --output /tmp/case011-seed0-audit.json
```

These commands audit the experiment's saved outputs; they do not repeat model
inference. The two HTML tables in `demo/` can be opened directly as local files.

## 2. Synthetic CPU contracts

From the repository or public ZIP root, add `--synthetic` to run the small contract suite with an existing compatible Torch environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -B cases/011-frozen-state-readout-adaptation/publication/run_checks.py \
  --synthetic --output /tmp/diova-case011-synthetic-check-new
```

No original checkpoint, fitted head weights or GPU is used. Synthetic random-model fixtures test the adapter, state/terminal/RNG boundaries, two-tensor patch validation and paired metrics. They are not a reproduction of learned CKDA memory performance.

### Direct original suite

The synthetic suite additionally needs Torch and the exact Case010 reference
subset. The review bundle supplies that subset at its original sibling path,
with an explicit inventory; it is not a complete Case010 distribution. In the
repository the normal full Case010 directory supplies the same files.

```bash
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -v
```

## 3. Actual model replay — separate files and environment required

The original real-checkpoint receipts cover 12 fresh-process patch checks and 30 stored execution boundaries. They describe the completed research run; publication work does not rerun them.

Real-checkpoint replay additionally requires the original upstream commit
`ef9d108d1692387cae37f5b2d539a71826a127c1`, its original compatible CPU environment,
and the three private `training-seed{0,1,2}/final.pt` files. Their required hashes
are in `provenance/adapter_reference_identity.json`. No replacement checkpoint
is trained or downloaded by the tools. Set the following to existing local paths:

```bash
python -B source/preflight.py --upstream "$UPSTREAM" \
  --checkpoint-root "$CHECKPOINT_ROOT" --output /tmp/case011-new-preflight.json

python -B source/reload_check.py --upstream "$UPSTREAM" \
  --checkpoint-root "$CHECKPOINT_ROOT" --patch-root "$HEAD_PATCH_ROOT" \
  --output /tmp/case011-new-reload
```

The head patches require the matching original checkpoint. They are not
standalone models. Each patch stores only `mlp.2.weight` and `mlp.2.bias`
(4,632 value bytes) plus a manifest. Local head tensors remain outside this
review ZIP under the private patch directory; exact manifests and tensor hashes
are retained in `provenance/head_patches/`. The original checkpoints, FIT/DEV
feature matrices, upstream tree and runtime state bodies are also excluded.

### Original bounded runner and freeze sequence

`python -B -m source.run_fit --help` and `python -B -m source.run_eval --help` expose the bounded
research entry points. They operate on an explicitly prepared fresh workspace;
the retained result tree is not a destination for another experiment. The
recorded execution order was input/protocol freeze → SMOKE parity → FIT/DEV
extraction → 18 bounded fits → six selected-head manifests → pre-TEST freeze
→ six shared recurrent rollouts → independent audit. The chronological hashes
are in `configs/prefit_freeze.json`, `selected_heads.json` and
`pretest_freeze.json`.

Fitting is full-batch FP64 CPU L-BFGS in the existing Torch environment, with
FP32 heads used for DEV and inference. No backbone gradients or updated codec
parameters are allowed. Technical resume must validate the exact input,
checkpoint, selected-head and code identities; it must not replace a completed
cell. Private execution checkpoints retain the byte cache and public-output
prefix together. `source/check_execution_checkpoint.py` provides the stricter
offset/cursor/shape audit before reusing any such boundary.

Timing was run after the model/evaluation workers finished:

```bash
python -B source/timing.py --upstream "$UPSTREAM" \
  --checkpoint-root "$CHECKPOINT_ROOT" --patch-root "$HEAD_PATCH_ROOT" \
  --output /tmp/case011-new-timing
```

This is a new measurement, not a scalar audit. It uses two CPU threads, the
retained 16×128 timing cohort, and three repetitions of each individual head.
Run it only in an otherwise idle research environment. The recorded numbers
belong to the Python/NumPy/Torch CPU prototype and are not GPU throughput or
single-request latency.

## 한국어 안내 — 세 가지 실행 목적

1. **기록만 검산:** 저장소 또는 공개 ZIP 루트에서 1번 `publication/run_checks.py` 명령을 실행합니다. Python/NumPy로 원형 보존·예측·스칼라·사후 분석을 확인합니다. `--output`은 아직 없는 외부 디렉터리입니다.
2. **합성 기능 검사:** 2번처럼 `--synthetic`을 붙이면 Torch를 사용하는 작은 계약 검사까지 실행합니다. 원래 학습 모델이나 GPU는 필요하지 않습니다.
3. **실제 모델 재실행:** 3번의 pinned upstream·기존 환경·비공개 원본 checkpoint·head patch가 필요합니다. 이 경로는 새 모델 실행이며 이번 공개 검산의 범위와 다릅니다.

원래 exact-inventory 검사는 외부 scratch에 원형을 복원한 뒤 수행합니다. 공개 문서나 새 분석 파일을 원형 manifest에 추가하지 않습니다. [현재 검사 범위와 원형 identity](publication/README.md).

### 개별 스칼라 명령

개별 `analysis/aggregate.py` 명령은 Case011 디렉터리에서 실행합니다. 공개 prediction/gold에서 최초 실패·평균
연속 정답 길이·경험적 길이·paired 구간을 다시 계산합니다. CE와 margin은 저장한
스칼라 합계와 유효 분모를 사용합니다. 전체 logits를 새로 계산하는 검사는 아닙니다.

합성 검사에는 Torch와 함께 제공한 Case010 참조 파일이 필요합니다. 실제 모델
검사는 원래 비공개 checkpoint와 pinned upstream, 선택된 head patch가 별도로
필요합니다. 기존 결과를 덮어쓰거나 TEST를 다시 골라 보정하는 경로는 제공하지
않습니다. 반복 검산은 새 외부 출력 경로를 사용하고, 실제 재실험은 별도 workspace와
새 실행 기록으로 구분합니다.
