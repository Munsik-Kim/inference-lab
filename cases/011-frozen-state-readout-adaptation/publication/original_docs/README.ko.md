# Case 011 — 상태를 고정한 판독층 보정과 길이 외삽

[English](README.md) · [DIOVA](../../README.ko.md)

같은 recurrent state를 공유하는 세 판독기를 비교하고, 마지막 Linear의 두 tensor만 보정·저장·재로딩하는 도구를 구현했습니다. 기존 CKDA checkpoint 세 개에서 Native와 INT8 저장을 고정한 채 새 1,024개 입력을 길이 2,048까지 읽었습니다.

## 구현한 기능

- [Frozen feature adapter](source/adapter.py): 기존 GELU 출력에서 정확히 분리한 192차원 특징, backbone gradient 0.
- [Head fitting](source/fitting.py) · [patch loader](source/head_patch.py): 최종 1,158개 parameter만 CPU FP64로 적합하고 FP32 두 tensor를 원래 자리에 복사합니다.
- [Paired runner](source/run_eval.py) · [independent audit](analysis/audit.py): 같은 cache에서 판독기 세 개의 최초 오류·정답 점수·control을 비교합니다.
- [Fresh-process reload](source/reload_check.py): 실제 작은 checkpoint에서 logits·features·cache bytes를 대조합니다.

기술: Python · NumPy · PyTorch. 공개 스칼라 검산에는 모델과 GPU가 필요하지 않습니다.

## 확인한 결과

INT8에서 MIXED 보정은 세 checkpoint 모두 평균 연속 정답 길이를 줄였습니다. seed0/1/2의 MIXED−ORIGINAL 변화는 -9.89 / -10.40 / -13.49 token이며, 세 주 비교의 구간은 모두 0 아래입니다. 반면 위치 257–2,048의 평균 정답 CE는 3/3에서 낮아졌고, token 정답률은 2/3에서 높아졌습니다. 정답에 부여한 점수의 개선이 처음부터 정확히 읽는 길이의 증가로 이어지지는 않았습니다.

| Checkpoint | ORIGINAL | SHORT | MIXED | Δ MIXED−ORIGINAL | 98.333% interval |
| --- | --- | --- | --- | --- | --- |
| 0 | 285.84 | 285.84 | 275.96 | -9.89 | [-14.59, -5.41] |
| 1 | 264.59 | 264.59 | 254.19 | -10.40 | [-15.71, -5.16] |
| 2 | 239.80 | 239.80 | 226.31 | -13.49 | [-17.66, -9.38] |

단위는 첫 오류 전 연속 정답 group token의 평균(RMST0)입니다. 각 행은 같은 1,024개 sequence의 paired 비교이며, 5,000회 bootstrap의 98.333% 구간은 세 주 비교에 대한 Bonferroni family 95%를 목표로 한 근사 구간입니다.

SHORT_REFIT의 FP32 tensor는 원형과 동일했습니다. MIXED_REFIT는 두 tensor를 바꿨으며 seed0/1은 사전 200 iteration 상한에 도달한 유한 해입니다. 각 solver 상태와 길이별 점수는 보고서에 함께 남겼습니다.

## CPU로 확인하기

이 Case011 디렉터리에서 Python과 NumPy로 실행합니다. 출력은 아직 없는 새 디렉터리를 지정합니다.

```bash
python -B analysis/aggregate.py --input-only-fit results/input_only_fit.json \
  --output /tmp/case011-new-audit --no-figures
```

이 명령은 공개 prediction·gold·스칼라 합계의 재계산입니다. 실제 모델 fitting/재실행은 별도의 원래 checkpoint·upstream·head patch가 필요합니다. 원래 모델 weights와 fitted head tensor는 검토 ZIP에서 제외하고 정확한 manifest만 제공합니다.

[결과표·분석](REPORT.ko.md) · [Methods](METHODS.md) · [Reproduction](REPRODUCTION.md) · [Protocol](configs/protocol.json) · [로컬 결과 탐색기](demo/ko.html) · [Attribution](NOTICE.md)

현재 상태: 로컬 연구·검토본. 원격 게시·PR·Pages 배포는 실행하지 않았습니다.
