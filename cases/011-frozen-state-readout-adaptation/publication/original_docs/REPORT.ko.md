# Case 011 — 상태를 고정한 판독층 보정과 길이 외삽

[한국어](REPORT.ko.md) · [English](REPORT.md) · [README](README.ko.md)

같은 recurrent state를 공유하는 세 판독기를 비교하고, 마지막 Linear의 두 tensor만 보정·저장·재로딩하는 도구를 구현했습니다. 기존 CKDA checkpoint 세 개에서 Native와 INT8 저장을 고정한 채 새 1,024개 입력을 길이 2,048까지 읽었습니다.

1. [배경](#s1)
2. [가설](#s2)
3. [이론](#s3)
4. [방법](#s4)
5. [실험](#s5)
6. [결과](#s6)
7. [분석](#s7)
8. [결론](#s8)
9. [레퍼런스](#s9)

<a id="s1"></a>
## 1. 배경
Case010은 저비트 state 저장과 실패를 보존하는 재시작을 구현했고, 고정 FP32 계수의 FP64 산술 승격에서도 별도 진단 입력의 예측·최초 실패가 같았습니다. 그 사실만으로 오류의 원인을 판독기로 확정할 수는 없습니다. Case011은 같은 state에서 최종 판독 가중치를 실제로 바꾸는 별도의 개입입니다.

<a id="s2"></a>
## 2. 가설
사전 주 질문은 INT8에서 MIXED_REFIT가 ORIGINAL보다 RMST0를 늘리는가입니다. SHORT는 위치 1–32의 보정 대조, MIXED는 위치 1–256의 보정이며 둘 다 Native 특징으로만 fitting합니다. 위치 512/1024/2048은 fitting 범위를 넘는 길이 외삽입니다. TEST 후 후보·lambda·seed를 추가하지 않았습니다.

<a id="s3"></a>
## 3. 이론
고정 recurrence와 codec가 만든 state를 원래 normalization·gating·projection·embedding residual·LayerNorm·hidden Linear·GELU로 판독한 결과가 `phi_t`입니다. 후보는 `Wc phi_t + bc`, 원형은 `W0 phi_t + b0`를 계산합니다. 두 계산은 다음 state에 영향을 주지 않습니다.

`tau`는 첫 오답/INVALID의 1-based group-token 위치이고 `RMST0=mean(min(tau−1,2048))`입니다. 미실패는 2048 우측 검열로 기여하며, 이후 정답 복귀는 survival을 되돌리지 않습니다. 경험적 T0.05는 누적 최초 오류 비율이 5% 이하인 마지막 관측 위치이며 이번에는 신뢰지원 horizon 하한을 계산하지 않습니다.

<a id="s4"></a>
## 4. 방법
변경 허용 tensor는 `mlp.2.weight [6,192]`, `mlp.2.bias [6]`뿐입니다. 같은 세 기존 checkpoint를 유지하고 state·계수·hidden feature 생성은 FP32로 고정했습니다. 같은 storage의 세 head는 하나의 feature를 읽습니다. 정답은 독립 S3 정수 left-product 계산기에서 만들며 후보에 전달하지 않습니다. 단순 오답은 계속 실행, 유한 state의 invalid readout은 −1 후 계속 실행, state terminal은 기존 v2 흡수 실패로 보존합니다.

FIT는 각 조건 16,384 supervised 행입니다. SHORT는 첫 32위치, MIXED는 네 band별 8위치를 각 512개 sequence에서 사용합니다. 실제 feature 추출은 둘이 공유한 전체 256길이 Native trajectory이며, 같은 label 행 수를 같은 독립 학습 계산량이라고 주장하지 않습니다. CPU FP64 L-BFGS는 `mean CE + λ(||W−W0||²+||b−b0||²)`를 최소화합니다. λ={1e−4,1e−2,1}, 최대 200 iterations/1000 objective evaluations, FP32 변환 뒤 네 DEV band의 동일 가중 CE, 정확한 동점은 큰 λ 우선입니다. [전체 방법](METHODS.md).

<a id="s5"></a>
## 5. 실험
| Role | Seed | Sequences | Group tokens |
| --- | --- | --- | --- |
| SMOKE | 61601 | 8 | 32 |
| FIT | 61101 | 512 | 256 |
| DEV | 61201 | 128 | 256 |
| FRESH_TEST | 61301 | 1024 | 2048 |

입력은 S3의 여섯 군 원소 모두에서 uniform 생성했습니다. BOS write0은 미채점이며 모든 group prefix를 채점했습니다. 위 각 코호트의 실제 bytes·ID·hash를 먼저 동결했습니다. 새 backbone 학습 0, 새 codec 0, fitting 후보 18개, 선택된 head 6개, primary rollout 6개/논리 arm 18개입니다. 원형 Case010의 512/1024입력을 이번 표본에 합치지 않았습니다.

| Seed | Head | λ | Changed FP32 values | Solver status |
| --- | --- | --- | --- | --- |
| 0 | SHORT_REFIT | 1.0 | 0 | GRADIENT_TOLERANCE_MET |
| 0 | MIXED_REFIT | 0.0001 | 1158 | MAX_ITER_NOT_CONVERGED |
| 1 | SHORT_REFIT | 1.0 | 0 | GRADIENT_TOLERANCE_MET |
| 1 | MIXED_REFIT | 0.0001 | 1158 | MAX_ITER_NOT_CONVERGED |
| 2 | SHORT_REFIT | 1.0 | 0 | GRADIENT_TOLERANCE_MET |
| 2 | MIXED_REFIT | 0.0001 | 1158 | OPTIMIZER_SMALL_CHANGE_OR_DIRECTION_STOP |

SHORT의 초기 gradient가 종료 허용오차 안에 있어 FP32 parameter가 그대로였습니다. MIXED seed0/1은 `MAX_ITER_NOT_CONVERGED`, seed2는 작은 변화/방향에 의한 optimizer 종료입니다. 수렴 미달을 감추거나 TEST를 보고 iteration을 늘리지 않았습니다. 추가 head 진단은 이미 저장한 DEV 특징과 parameter만 읽었으며 선택을 바꾸지 않았습니다.

<a id="s6"></a>
## 6. 결과
### 6.1 주 비교: INT8의 평균 연속 정답 길이

| Checkpoint | ORIGINAL | SHORT | MIXED | Δ MIXED−ORIGINAL | 98.333% interval |
| --- | --- | --- | --- | --- | --- |
| 0 | 285.84 | 285.84 | 275.96 | -9.89 | [-14.59, -5.41] |
| 1 | 264.59 | 264.59 | 254.19 | -10.40 | [-15.71, -5.16] |
| 2 | 239.80 | 239.80 | 226.31 | -13.49 | [-17.66, -9.38] |

INT8에서 MIXED 보정은 세 checkpoint 모두 평균 연속 정답 길이를 줄였습니다. seed0/1/2의 MIXED−ORIGINAL 변화는 -9.89 / -10.40 / -13.49 token이며, 세 주 비교의 구간은 모두 0 아래입니다. 반면 위치 257–2,048의 평균 정답 CE는 3/3에서 낮아졌고, token 정답률은 2/3에서 높아졌습니다. 정답에 부여한 점수의 개선이 처음부터 정확히 읽는 길이의 증가로 이어지지는 않았습니다.

각 checkpoint의 n=1,024 paired sequence, 5,000 bootstrap, 각 98.333% 근사 구간입니다. 세 모델을 합쳐 n=3,072로 재표집하지 않습니다.

| Checkpoint | ORIGINAL T0.05 | SHORT T0.05 | MIXED T0.05 |
| --- | --- | --- | --- |
| 0 | 141 | 141 | 143 |
| 1 | 115 | 115 | 110 |
| 2 | 143 | 143 | 131 |

위 길이는 매-token 경험값이며 CI 하한이 아닙니다. 최초 실패 없이 2,048까지 통과한 입력도 관측 끝에서 검열됩니다.

![INT8 survival](results/derived/figures/int8_survival.png)

![Primary paired RMST0](results/derived/figures/primary_rmst0.png)

### 6.2 Native 보조 비교

| Seed | ORIGINAL RMST0 | MIXED RMST0 | Δ tokens | Pointwise 95% interval |
| --- | --- | --- | --- | --- |
| 0 | 287.43 | 277.23 | -10.20 | [-14.11, -6.50] |
| 1 | 266.61 | 257.08 | -9.54 | [-14.09, -5.03] |
| 2 | 239.70 | 226.49 | -13.21 | [-16.70, -9.83] |

주 비교와 다른 pointwise 95% 구간입니다. INT8 MIXED−SHORT는 SHORT tensor가 원형과 같으므로 점추정이 MIXED−ORIGINAL과 같지만 보조 구간의 confidence/재표집 seed는 별도입니다. 같은 readout의 Native↔INT8 비교 전체는 [secondary.csv](results/derived/secondary.csv)에 있습니다.

### 6.3 INT8의 길이 구간별 정답 점수

| Seed | Positions | ORIGINAL accuracy % | MIXED accuracy % | ORIGINAL CE nats | MIXED CE nats |
| --- | --- | --- | --- | --- | --- |
| 0 | 1–32 | 100.00 | 100.00 | 0.0000 | 0.0002 |
| 0 | 33–128 | 99.92 | 99.95 | 0.0125 | 0.0027 |
| 0 | 129–256 | 95.86 | 96.59 | 0.9255 | 0.1126 |
| 0 | 257–512 | 75.64 | 76.74 | 8.4229 | 1.0975 |
| 0 | 513–1024 | 43.27 | 43.46 | 28.2199 | 5.9880 |
| 0 | 1025–2048 | 21.20 | 21.13 | 48.0973 | 16.1863 |
| 1 | 1–32 | 100.00 | 100.00 | 0.0000 | 0.0006 |
| 1 | 33–128 | 99.55 | 99.61 | 0.1214 | 0.0158 |
| 1 | 129–256 | 92.54 | 92.88 | 2.2983 | 0.2417 |
| 1 | 257–512 | 71.42 | 71.86 | 10.6193 | 1.0782 |
| 1 | 513–1024 | 47.80 | 48.45 | 21.2801 | 2.1044 |
| 1 | 1025–2048 | 36.10 | 36.17 | 26.3626 | 2.5935 |
| 2 | 1–32 | 100.00 | 100.00 | 0.0000 | 0.0000 |
| 2 | 33–128 | 99.96 | 99.94 | 0.0042 | 0.0049 |
| 2 | 129–256 | 96.68 | 96.50 | 0.5672 | 0.3701 |
| 2 | 257–512 | 73.05 | 73.15 | 7.1078 | 4.4274 |
| 2 | 513–1024 | 35.00 | 34.95 | 22.8075 | 14.1163 |
| 2 | 1025–2048 | 19.15 | 19.13 | 31.6552 | 19.5621 |

정답 CE는 낮을수록 좋으며 유효 token 관측당 평균입니다. 정답률은 전체 token 분모를 사용합니다. 257 이상은 fitting 위치 범위 밖이며, 512/1024/2048 한 지점의 값과 구간 평균을 혼동하지 않습니다. 각 분모·invalid 수·margin 및 Native 값은 [scores.csv](results/derived/scores.csv)에 있습니다. 공개 재계산은 저장한 score 합계/분모를 사용하며 비공개 전체 logits를 재구성하지 않습니다.

![Gold CE by position](results/derived/figures/gold_ce_by_position.png)

### 6.4 Control

| Condition | True feature accuracy % | Shuffled accuracy % | Shuffled RMST0 |
| --- | --- | --- | --- |
| INT8 MIXED / seed0 | 43.31 | 16.71 | 1.20 |
| INT8 MIXED / seed1 | 51.22 | 16.68 | 1.20 |
| INT8 MIXED / seed2 | 39.72 | 16.72 | 1.20 |

input-only control은 전체 token 정답률 16.72%, RMST0 1.19 token이었습니다. 첫 group token의 gold는 current token 자체여서 그 위치는 state-dependent 정보가 필요하지 않습니다. 위치 t와 current token 안의 순열은 label을 쓰지 않고 singleton은 그대로 두어 수를 기록했습니다. Head 성과가 control과 다르다는 것은 이 feature 경계에서 state 관련 판독을 지지하지만 정확한 군 표현이나 모든 decoder에서의 복원 가능성을 증명하지 않습니다.

### 6.5 저장량·재로딩·비용

| Object / scope | Bytes |
| --- | --- |
| Native state / stream | 12305 |
| INT8 state / stream | 3137 |
| Model parameter values / before and after | 226120 |
| Final Linear values / before and after | 4632 |
| Original checkpoint file / each retained seed | 233203 |
| Added recurrent-state bytes / refitted head | 0 |
| One FIT feature matrix: 16,384 × 192 × FP32 | 12582912 |

| Seed | Patch | Tensor values B | Manifest B | Total files B |
| --- | --- | --- | --- | --- |
| 0 | SHORT_REFIT | 4632 | 3417 | 8049 |
| 0 | MIXED_REFIT | 4632 | 53024 | 57656 |
| 1 | SHORT_REFIT | 4632 | 3412 | 8044 |
| 1 | MIXED_REFIT | 4632 | 51450 | 56082 |
| 2 | SHORT_REFIT | 4632 | 3412 | 8044 |
| 2 | MIXED_REFIT | 4632 | 27514 | 32146 |

6개 fresh rollout의 terminal stream 수 합계는 0입니다. 이것은 정답 stream 수가 아니라 수치 state 실패 수입니다. Head별 parameter·cache·임시 feature·파일 장부를 분리했습니다. Cell의 shared ledger는 원래 v2 cache/table 계약이며 새 head 선택 manifest는 별도 파일 비용입니다. 전체 artifact의 디스크 크기가 같다는 뜻이 아닙니다. 원래 checkpoint 파일 크기는 [Case010의 파일 identity](../010-ckda-finite-precision-memory-horizon/versions/v2/provenance/v1_checkpoints.json)에서 읽었습니다. 원래 checkpoint와 fitted head tensor는 ZIP에 넣지 않으며 [정확한 patch manifest](provenance/head_patches/)와 loader를 제공합니다.

검사는 실제 실행 기록별로 나눴습니다. 고정 SMOKE의 3 checkpoint×2 storage에서 기존 readout과 feature+head logits가 bitwise 일치했고 최대 차이는 0입니다. [새 프로세스 12건](provenance/head_reload.json)은 logits·labels·features·cache·cursor·terminal/RNG 일치를 통과했습니다. [저장 경계 30건](provenance/execution_checkpoints.json)은 모델 실행 없이 cursor·prefix·inventory·hash를 따로 검증했습니다. 이번 고유 단위 테스트는 79개 PASS, skip 0이며, 여섯 공개 TEST cell과 19개 control 요약의 스칼라를 별도 구현으로 검산했습니다. 검사 횟수는 재실행을 더해 부풀리지 않았습니다. [검사 receipt](provenance/validation_receipt.json), [기존 3,708개 보호 파일의 변경 0 확인](provenance/protection.json).

공유 table·config와 N=1/16/128 저장량, 테스트·적합의 임시 배열은 [전체 byte ledger](results/byte_ledger.json)에 분리했습니다. 프로세스 peak RAM은 미측정입니다.

| Seed | Storage | ORIGINAL ms/token | SHORT ms/token | MIXED ms/token |
| --- | --- | --- | --- | --- |
| 0 | NATIVE_FP32 | 0.02761 | 0.02867 | 0.03521 |
| 0 | UNIFORM_8 | 0.09234 | 0.08859 | 0.08674 |
| 1 | NATIVE_FP32 | 0.02812 | 0.02956 | 0.02920 |
| 1 | UNIFORM_8 | 0.08708 | 0.08766 | 0.09116 |
| 2 | NATIVE_FP32 | 0.02844 | 0.02755 | 0.03158 |
| 2 | UNIFORM_8 | 0.09220 | 0.08766 | 0.08920 |

위 값은 독립 CPU complete-call 3회의 중앙값입니다. 2 threads, 16×128 group tokens, BOS는 실행시간에 포함하고 분모에서는 제외했습니다. 같은 모양의 Linear 하나를 실행하며 head를 동시에 읽는 연구 runner의 시간이 아닙니다. 속도비 유의성·GPU 속도·단일 요청 latency를 주장하지 않습니다. 원 반복값은 [timing.json](results/timing/timing.json)에 있습니다.

<a id="s7"></a>
## 7. 분석
**동일 state에서의 개입을 확인했고, 정답 점수와 첫 오류 수명이 다르게 움직였습니다.** CE는 정답 확률의 음의 로그를 token별로 평균하지만 RMST0는 한 번의 이른 오류에도 줄어듭니다. Fitting의 목적함수는 평균 CE와 가중치 변화 penalty이며 sequence의 최초 오류를 직접 최적화하지 않습니다. 이 실험은 두 목표의 불일치를 보여주지만, 그 차이의 기하학적 원인까지 규명하지는 않았습니다.

INT8의 경험적 T0.05 변화는 seed0: 141→143 / seed1: 115→110 / seed2: 143→131 token입니다. seed0에서는 이 경험적 위치가 늘었지만 RMST0는 줄었습니다. 같은 seed에서도 위험 5%의 위치와 평균 연속 길이는 다른 분포 요약입니다. 이 경험값에 신뢰지원 하한의 의미를 붙이지 않습니다.

SHORT가 바뀌지 않은 결과는 이 fitting 입력·gradient tolerance·FP32 경계에서 추가 보정이 없었다는 뜻입니다. MIXED는 이미 오류가 있는 더 긴 FIT 특징을 포함하고 두 tensor를 바꿨습니다. 두 차이는 fitting 위치 범위의 결과로 해석하되, iteration cap에 도달한 seed0/1에 대해 가능한 모든 선형 판독기의 최적 성능이라고 주장하지 않습니다.

FP64 진단의 재실행, codec 변경, backbone 재학습으로 얻은 결과가 아닙니다. 같은 세 checkpoint에서 하나의 fresh TEST를 사용한 결과이며 다른 학습 seed 모집단이나 LM 문맥 길이 전체로 확대하지 않습니다. 기술적 all-terminal 경계 오류는 합성 검사에서 TEST 전 수정했고, 실제 결과의 부호에 따라 실행 규칙을 바꾸지 않았습니다.

<a id="s8"></a>
## 8. 결론
완성한 결과물은 같은 cache의 세 판독기 비교, 1,158개 parameter의 제한된 지도 보정, 두 tensor patch와 새 프로세스 재실행, 공개 prediction 기반 독립 검산입니다. INT8의 MIXED 보정 후 seed0/1/2의 평균 연속 정답 길이 변화는 각각 -9.89 / -10.40 / -13.49 token이며, 세 주 비교 구간 모두 0 아래입니다. 위치 256 밖의 정답 CE 개선은 관측했지만 처음부터 정확히 읽는 길이는 늘리지 못했습니다. 이 결과는 현재 feature·최종 Linear·fitting 예산의 범위이며, 모든 decoder에서 정보 복원이 불가능하다는 증명이 아닙니다.

현재는 로컬 검토 단계입니다. stage/commit/push/PR/merge/Pages/Drive 업로드는 이번 Case011에서 실행하지 않았습니다.

<a id="s9"></a>
## 9. 레퍼런스
- [Case010 version map](../010-ckda-finite-precision-memory-horizon/VERSION_MAP.md) · [fixed precision/readout source](../010-ckda-finite-precision-memory-horizon/versions/v2/source/precision.py).
- OpenEuroLLM / ComplexKDA, pinned commit `ef9d108d1692387cae37f5b2d539a71826a127c1`; original model/task and coefficients.
- John Hewitt and Percy Liang (2019), [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/). Control motivation; this study does not reproduce that paper’s experimental control task.
- [Protocol and source freeze](configs/protocol.json) · [selected heads](configs/selected_heads.json) · [scalar aggregate](results/derived/aggregate.json) · [head diagnostics](results/head_diagnostics.json) · [NOTICE](NOTICE.md).

Munsik Kim의 연구·구현에 Codex가 코드·검사·분석·문서 작성을 지원했습니다. 기존 모델, recurrence/codec, 수치 라이브러리와 새로운 adapter·fitting·patch·검산 도구의 기여를 [NOTICE](NOTICE.md)에 구분했습니다.
