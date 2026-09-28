# Recompute results or rerun the bounded experiment

[한국어 보고서](REPORT.ko.md) · [English report](REPORT.md) · [Methods](METHODS.md)

## 1. Model-free audit — the review ZIP supports this path

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

## 2. CPU contracts and real-checkpoint replay

The synthetic suite additionally needs Torch and the exact Case010 reference
subset. The review bundle supplies that subset at its original sibling path,
with an explicit inventory; it is not a complete Case010 distribution. In the
repository the normal full Case010 directory supplies the same files.

```bash
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -v
```

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

## 3. Experimental runner and the freeze sequence

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

## 한국어 안내

빠른 검산은 1번 명령으로 실행합니다. 공개 prediction/gold에서 최초 실패·평균
연속 정답 길이·경험적 길이·paired 구간을 다시 계산합니다. CE와 margin은 저장한
스칼라 합계와 유효 분모를 사용합니다. 전체 logits를 새로 계산하는 검사는 아닙니다.

합성 검사에는 Torch와 함께 제공한 Case010 참조 파일이 필요합니다. 실제 모델
검사는 원래 비공개 checkpoint와 pinned upstream, 선택된 head patch가 별도로
필요합니다. 기존 결과를 덮어쓰거나 TEST를 다시 골라 보정하는 경로는 제공하지
않습니다. 반복 검산은 새 외부 출력 경로를 사용하고, 실제 재실험은 별도 workspace와
새 실행 기록으로 구분합니다.
