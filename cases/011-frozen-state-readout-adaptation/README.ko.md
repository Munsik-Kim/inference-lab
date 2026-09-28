# Case 011 — 상태를 고정한 판독층 보정과 길이 외삽

[English](README.md) · [DIOVA 홈](../../README.ko.md)

모델이 저장한 상태를 유지하면서 답을 읽는 마지막 층만 교체하고, 두 tensor를 patch로 저장해 새 프로세스에서 다시 적용하는 도구를 구현했습니다. 같은 상태에서 정답 확률과 첫 오류 시점이 어떻게 달라지는지 비교할 수 있습니다.

[무엇을 만들었나](#implementation) · [핵심 결과](#results) · [직접 확인](#reproduce) · [설계와 실험](#design) · [출처](#sources)

<a id="implementation"></a>
## 무엇을 만들었나

**마지막 층 1,158개 값 · 추가 recurrent-state 저장 0 B · 새 프로세스 patch 재적용**

상태는 입력마다 계속 갱신됩니다. 같은 저장 방식 안에서 ORIGINAL·SHORT·MIXED가 매 시점 같은 상태와 특징을 읽으며, 답이 다음 상태를 바꾸지 않습니다. Native와 INT8끼리 상태가 같다는 뜻은 아닙니다.

- [특징 경계 adapter](source/adapter.py): 원래 GELU 출력과 최종 Linear 사이를 분리합니다. [상태·특징 검사](tests/test_adapter.py).
- [두 tensor patch](source/head_patch.py): 원본 checkpoint hash·shape·dtype을 검사해 weight/bias만 적용합니다. [저장·오류 검사](tests/test_fitting_patch.py).
- [비교 runner](source/run_eval.py)와 [별도 검산기](analysis/audit.py): 같은 입력의 예측·정답 점수·첫 오류를 비교합니다. [집계 검사](tests/test_aggregate.py).

Python · NumPy · PyTorch. 0 B는 **추가 recurrent-state** 비용입니다. patch 파일과 fitting 특징 배열은 [별도 장부](results/byte_ledger.json)에 기록했습니다.

<a id="results"></a>
## 핵심 결과

**정답 확률 점수는 개선됐지만, 처음부터 연속해서 맞히는 평균 길이는 짧아졌습니다.** INT8 MIXED에서 보정 범위 밖인 위치 257–2,048의 정답 교차엔트로피(CE, 낮을수록 좋음)는 세 checkpoint 모두 감소했습니다. 전체 token 정답률은 2/3에서 상승했지만, 평균 연속 정답 길이(RMST0, 길수록 좋음)는 3/3에서 감소했습니다.

| Checkpoint | ORIGINAL | MIXED | 변화 |
| --- | --- | --- | --- |
| 0 | 285.84 | 275.96 | -9.89 |
| 1 | 264.59 | 254.19 | -10.40 |
| 2 | 239.80 | 226.31 | -13.49 |

단위는 첫 오답 직전까지 연속 정답인 **기호 token 수의 평균**입니다. 각 checkpoint는 같은 1,024개 입력을 2,048단계까지 읽었습니다. 자연어 context 길이의 측정이 아닙니다. [주 비교 구간과 Native 결과](REPORT.ko.md#results).

SHORT는 정해진 종료 기준에서 최종 가중치가 원형과 동일한 대조군입니다. 선택된 MIXED의 seed0/1은 200 iteration 상한에 도달했습니다. [선택·solver 상태](REPORT.ko.md#experiments).

<a id="reproduce"></a>
## 직접 확인

저장소 또는 공개 ZIP의 루트에서 실행합니다. Python과 NumPy만으로 공개 예측·스칼라를 검산하며 출력에는 새 외부 디렉터리를 지정합니다.

```bash
python -B cases/011-frozen-state-readout-adaptation/publication/run_checks.py \
  --output /tmp/diova-case011-publication-check-new
```

[모델 없는 검산·합성 검사](REPRODUCTION.md) · [쉬운 결과 화면](https://munsik-kim.github.io/inference-lab/ko/case011.html#overview) · [오프라인 원형 결과표](demo/ko.html)

공개 자료는 코드·측정 기록입니다. 실제 patch 재적용에는 원래 checkpoint와 비공개 head tensor가 필요하며, 이 가중치들은 ZIP에 포함되지 않습니다.

<a id="design"></a>
## 설계와 실험

CKDA 기호 상태추적 과제의 기존 checkpoint 세 개와 두 저장 방식에서 마지막 선형층만 지도 보정했습니다. SHORT/MIXED는 각각 위치 1–32/1–256의 Native 특징 16,384행으로 적합하고 DEV의 정답 CE로 선택했습니다. TEST에서는 여섯 상태 rollout의 특징을 세 판독기가 공유했습니다.

[정식 보고서](REPORT.ko.md) · [Methods](METHODS.md) · [Protocol](configs/protocol.json) · [같은 기록의 사후 분석](REPORT.ko.md#analysis)

<a id="sources"></a>
## 출처

[NOTICE](NOTICE.md)는 ComplexKDA의 모델·과제, Case010의 상태 codec, PyTorch/NumPy와 프로젝트의 개입·저장·검산 구현 및 Codex 지원을 구분합니다. 원래 검토 문서와 검증 기록은 [publication](publication/README.md)에서 추적할 수 있습니다.

[코드·측정 기록 ZIP 다운로드 — 모델 가중치 제외](https://github.com/Munsik-Kim/inference-lab/raw/refs/heads/main/downloads/case011_readout_adaptation_reviewed_publication_v1.zip)
