# Case011 — Same State, Different Readout / 같은 상태, 다른 판독기

[한국어 소개](README.ko.md) · [English introduction](README.md)

## 문제 / Problem

같은 모델 상태에서 마지막 판독층만 바꾸면 첫 오류를 늦출 수 있는지 비교합니다. The question is whether a changed final readout delays the first error on the same evolving state.

## 직접 구현 / Implementation

- 같은 저장 방식의 세 판독기에 동일 특징을 전달하는 [adapter](source/adapter.py).
- 최종 weight/bias 두 tensor만 적합하는 [bounded fitter](source/fitting.py), 원본 hash·shape·dtype을 검사하는 [patch loader](source/head_patch.py).
- 공유 state와 독립 S3 정답으로 최초 오류·확률 점수를 비교하는 [runner](source/run_eval.py), 공개 prediction을 다시 읽는 [auditor](analysis/audit.py).

The adapter isolates the existing post-GELU feature. The loader applies two tensors to the original checkpoint; predictions never feed back into recurrence.

## 기술 / Technologies

Python · PyTorch · NumPy. ComplexKDA provides the model/task and Case010 supplies the frozen state codec. No new quantizer or CUDA kernel is introduced.

## 주요 관측 / Findings

마지막 층 1,158개 값의 교체는 추가 recurrent-state 저장 0 B로 실행됩니다. INT8 MIXED의 정답 CE는 보정 범위 밖에서 개선됐지만 평균 연속 정답 길이는 세 checkpoint 모두 감소했습니다. SHORT는 최종 원형과 같은 대조군이며 MIXED 두 선택은 iteration 상한에 도달했습니다.

The two-tensor substitution adds no recurrent-state bytes. Gold CE improved beyond fitting support, while INT8 uninterrupted correct length fell on all three checkpoints. SHORT exported unchanged weights; two selected MIXED fits reached the iteration cap.

## 코드·테스트 / Inspect and run

[Adapter contracts](tests/test_adapter.py) · [Patch and fitting contracts](tests/test_fitting_patch.py) · [Model-free CPU guide](REPRODUCTION.md) · [한국어 결과](REPORT.ko.md#results) · [English results](REPORT.md#results)

## 출처 / Attribution

[NOTICE](NOTICE.md) records ComplexKDA, PyTorch/NumPy, control-task research and Codex assistance. The project contribution is the delimited intervention, fitting/patch integration and paired evaluation, not a new probing principle.
