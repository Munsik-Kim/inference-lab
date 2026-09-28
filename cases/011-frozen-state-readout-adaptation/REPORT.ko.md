# Case 011 — 상태를 고정한 판독층 보정과 길이 외삽

[한국어](REPORT.ko.md) · [English](REPORT.md) · [README](README.ko.md)

같은 recurrent state를 공유하는 세 판독기를 비교하고, 마지막 Linear의 두 tensor만 보정·저장·재로딩하는 도구를 구현했습니다. 기존 CKDA checkpoint 세 개에서 Native와 INT8 저장을 고정한 채 새 1,024개 입력을 길이 2,048까지 읽었습니다.

모델 내부 상태는 이전 입력을 요약하며 매 입력마다 바뀌는 숫자입니다. 여기서 상태를 고정한다는 뜻은 **같은 저장 방식의 세 판독기가 매 시점 동일한 상태·특징을 읽는다**는 것입니다. Native와 INT8의 상태까지 같은 것은 아닙니다. 판독층은 그 특징에서 답을 고르는 마지막 계산층이며, 이번 token은 자연어 단어가 아니라 S3 입력 원소 하나입니다.

**대표 관측:** 위치 256 밖의 정답 확률 점수는 개선됐지만, INT8에서 처음부터 연속으로 맞히는 평균 길이는 세 checkpoint 모두 짧아졌습니다.

1. [배경](#s1)
2. [가설과 검증 질문](#questions)
3. [이론과 지표](#theory)
4. [방법](#s4)
5. [실험](#s5)
6. [실험 결과](#results)
7. [결과 분석](#analysis)
8. [결론](#s8)
9. [레퍼런스 및 기여](#references)

<a id="s1"></a>
<a id="background"></a>
## 1. 배경
Case010은 저비트 state 저장과 실패를 보존하는 재시작을 구현했고, 고정 FP32 계수의 FP64 산술 승격에서도 별도 진단 입력의 예측·최초 실패가 같았습니다. 그 사실만으로 오류의 원인을 판독기로 확정할 수는 없습니다. Case011은 같은 state에서 최종 판독 가중치를 실제로 바꾸는 별도의 개입입니다.

<a id="s2"></a>
<a id="questions"></a>
## 2. 가설과 검증 질문
사전 주 질문은 INT8에서 MIXED_REFIT가 ORIGINAL보다 RMST0를 늘리는가입니다. SHORT는 위치 1–32의 보정 대조, MIXED는 위치 1–256의 보정이며 둘 다 Native 특징으로만 fitting합니다. 위치 512/1024/2048은 fitting 범위를 넘는 길이 외삽입니다. TEST 후 후보·lambda·seed를 추가하지 않았습니다.

<a id="s3"></a>
<a id="theory"></a>
## 3. 이론과 지표
고정 recurrence와 codec가 만든 state를 원래 normalization·gating·projection·embedding residual·LayerNorm·hidden Linear·GELU로 판독한 결과가 `phi_t`입니다. 후보는 `Wc phi_t + bc`, 원형은 `W0 phi_t + b0`를 계산합니다. 두 계산은 다음 state에 영향을 주지 않습니다.

`tau`는 첫 오답/INVALID의 1-based group-token 위치이고 `RMST0=mean(min(tau−1,2048))`입니다. 미실패는 2048 우측 검열로 기여하며, 이후 정답 복귀는 survival을 되돌리지 않습니다. 경험적 T0.05는 누적 최초 오류 비율이 5% 이하인 마지막 관측 위치이며 이번에는 신뢰지원 horizon 하한을 계산하지 않습니다.

정답 확률 점수인 교차엔트로피(CE)는 정답 확률이 높을수록 작아집니다. 전체 token 정답률, 이 CE 평균, 처음부터 첫 오류 직전까지의 길이는 다른 질문에 답합니다. 주 비교는 checkpoint별 INT8 MIXED−ORIGINAL의 RMST0 세 개입니다. 같은 base sequence를 대응시켜 5,000회 재표집하고 seed 63001/63002/63003, NumPy `linear` quantile로 각 98.333…% percentile bootstrap 구간을 계산했습니다. Bonferroni family 95%를 의도한 **근사 구간**이며 exact coverage는 아닙니다. 같은 sequence의 token·head·storage를 독립 표본으로 세지 않습니다.

<a id="s4"></a>
<a id="methods"></a>
## 4. 방법
변경 허용 tensor는 `mlp.2.weight [6,192]`, `mlp.2.bias [6]`뿐입니다. 같은 세 기존 checkpoint를 유지하고 state·계수·hidden feature 생성은 FP32로 고정했습니다. 같은 storage의 세 head는 하나의 feature를 읽습니다. 정답은 독립 S3 정수 left-product 계산기에서 만들며 후보에 전달하지 않습니다. 단순 오답은 계속 실행, 유한 state의 invalid readout은 −1 후 계속 실행, state terminal은 기존 v2 흡수 실패로 보존합니다.

FIT는 각 조건 16,384 supervised 행입니다. SHORT는 첫 32위치, MIXED는 네 band별 8위치를 각 512개 sequence에서 사용합니다. 실제 feature 추출은 둘이 공유한 전체 256길이 Native trajectory이며, 같은 label 행 수를 같은 독립 학습 계산량이라고 주장하지 않습니다. CPU FP64 L-BFGS는 `mean CE + λ(||W−W0||²+||b−b0||²)`를 최소화합니다. λ={1e−4,1e−2,1}, 최대 200 iterations/1000 objective evaluations, FP32 변환 뒤 네 DEV band의 동일 가중 CE, 정확한 동점은 큰 λ 우선입니다. [전체 방법](METHODS.md).

보정은 평균 CE 목적함수를 줄이고 **DEV의 네 위치 band를 같은 비중으로 둔 CE**로 λ를 선택했습니다. TEST의 주지표는 **RMST0**였습니다. 선택 지표와 최종 질문의 차이는 결과를 본 뒤 바꾼 것이 아닙니다. `lambda`는 mean CE에 더한 **합계 제곱 가중치 변화**의 계수이고 feature scaling은 없었습니다. 보정 대상은 최종 선형층뿐인 지도 보정이며 recurrent backbone 학습은 아닙니다.

<a id="s5"></a>
<a id="experiments"></a>
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

<details>
<summary>선택된 solver의 목적함수·gradient·종료 기록</summary>

| Seed | Head | λ | Iteration/evaluation | Objective: initial → final | FIT CE nats: initial → final | Max absolute gradient: initial → final |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | SHORT | 1 | 0/1 | 2.71051e-20 → 2.71051e-20 | 2.71051e-20 → 2.71051e-20 | 6.29e-19 → 6.29e-19 |
| 0 | MIXED | 0.0001 | 200/216 | 0.209249 → 0.0203782 | 0.209249 → 0.0154738 | 0.0262 → 3.63e-06 |
| 1 | SHORT | 1 | 0/1 | 9.18089e-16 → 9.18089e-16 | 9.18089e-16 → 9.18089e-16 | 1.61e-14 → 1.61e-14 |
| 1 | MIXED | 0.0001 | 200/210 | 0.498313 → 0.0418503 | 0.498313 → 0.0368489 | 0.0745 → 0.000249 |
| 2 | SHORT | 1 | 0/1 | 6.60144e-17 → 6.60144e-17 | 6.60144e-17 → 6.60144e-17 | 1.36e-15 → 1.36e-15 |
| 2 | MIXED | 0.0001 | 104/105 | 0.142225 → 0.00501566 | 0.142225 → 0.00078632 | 0.0225 → 6.7e-07 |

계산·적합은 FP64, DEV와 실제 실행은 FP32입니다. SHORT는 첫 gradient가 `1e-7` 이하라 iteration 0에서 멈췄고 FP32 export도 bitwise 원형과 같았습니다. 초기·최종 목적함수가 같은 로그와 [tensor 변위 기록](results/head_diagnostics.json)이 이를 뒷받침합니다. 별도 최종 FP64 tensor 파일은 공개하지 않으므로 이를 새 FP64 weight 감사라고 부르지 않습니다. MIXED seed0/1은 200회 상한에 도달했고 seed2는 104회에 작은 변화/방향 조건으로 종료했습니다. 세 MIXED 모두 gradient 기준의 수렴 인증을 얻지는 않았습니다. 원 solver 상태는 [seed0](results/fit-seed0-MIXED_REFIT.json), [seed1](results/fit-seed1-MIXED_REFIT.json), [seed2](results/fit-seed2-MIXED_REFIT.json)에 있습니다.

</details>

<a id="s6"></a>
<a id="results"></a>
## 6. 실험 결과

아래 6.1–6.5는 원래 고정 TEST의 기록입니다. 이번에 같은 기록으로 추가한 분석은 [7절](#analysis)에 분리했습니다.

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

[INT8 생존곡선 원본](results/derived/figures/int8_survival.png)

![Primary paired RMST0](results/derived/figures/primary_rmst0.png)

### 6.2 Native 보조 비교

| Checkpoint | ORIGINAL | SHORT | MIXED | Δ MIXED−ORIGINAL | Pointwise 95% interval |
| --- | --- | --- | --- | --- | --- |
| 0 | 287.43 | 287.43 | 277.23 | -10.20 | [-14.11, -6.50] |
| 1 | 266.61 | 266.61 | 257.08 | -9.54 | [-14.09, -5.03] |
| 2 | 239.70 | 239.70 | 226.49 | -13.21 | [-16.70, -9.83] |

주 비교와 다른 pointwise 95% 구간입니다. INT8 MIXED−SHORT는 SHORT tensor가 원형과 같으므로 점추정이 MIXED−ORIGINAL과 같지만 보조 구간의 confidence/재표집 seed는 별도입니다. 같은 readout의 Native↔INT8 비교 전체는 [secondary.csv](results/derived/secondary.csv)에 있습니다.

<details>
<summary>두 저장 방식의 경험적 horizon</summary>

| Storage | Seed | ORIGINAL = SHORT T0.05 | MIXED T0.05 |
| --- | --- | --- | --- |
| NATIVE_FP32 | 0 | 141 | 143 |
| NATIVE_FP32 | 1 | 114 | 113 |
| NATIVE_FP32 | 2 | 143 | 131 |
| UNIFORM_8 | 0 | 141 | 143 |
| UNIFORM_8 | 1 | 115 | 110 |
| UNIFORM_8 | 2 | 143 | 131 |

</details>

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

다음은 원래 연구에서 남긴 검사 기록입니다. 고정 SMOKE의 3 checkpoint×2 storage에서 기존 readout과 feature+head logits가 bitwise 일치했고 최대 차이는 0입니다. [새 프로세스 12건](provenance/head_reload.json)은 logits·labels·features·cache·cursor·terminal/RNG 일치를 통과했습니다. [저장 경계 30건](provenance/execution_checkpoints.json)은 모델 실행 없이 cursor·prefix·inventory·hash를 따로 검증했습니다. 원래 연구의 고유 단위 테스트 기록은 79개 PASS, skip 0이며, 여섯 공개 TEST cell과 19개 control 요약의 스칼라를 별도 구현으로 검산했습니다. 검사 횟수는 재실행을 더해 부풀리지 않았습니다. [검사 receipt](provenance/validation_receipt.json), [기존 3,708개 보호 파일의 변경 0 확인](provenance/protection.json).

공유 table·config와 N=1/16/128 저장량, 테스트·적합의 임시 배열은 [전체 byte ledger](results/byte_ledger.json)에 분리했습니다. 프로세스 peak RAM은 미측정입니다.

| Seed | Storage | ORIGINAL ms/token | SHORT ms/token | MIXED ms/token |
| --- | --- | --- | --- | --- |
| 0 | NATIVE_FP32 | 0.02761 | 0.02867 | 0.03521 |
| 0 | UNIFORM_8 | 0.09234 | 0.08859 | 0.08674 |
| 1 | NATIVE_FP32 | 0.02812 | 0.02956 | 0.02920 |
| 1 | UNIFORM_8 | 0.08708 | 0.08766 | 0.09116 |
| 2 | NATIVE_FP32 | 0.02844 | 0.02755 | 0.03158 |
| 2 | UNIFORM_8 | 0.09220 | 0.08766 | 0.08920 |

위 값은 독립 CPU complete-call 3회의 중앙값입니다. 2 threads, 16×128 group tokens, BOS는 실행시간에 포함하고 분모에서는 제외했습니다. 같은 모양의 Linear 하나를 실행하며 head를 동시에 읽는 연구 runner의 시간이 아닙니다. 속도비 유의성·GPU 속도·단일 요청 latency를 주장하지 않습니다. 원래 측정은 3 checkpoint×2 storage×3 head×3반복의 54회이며, 이번 공개 작업에서는 재측정하지 않았습니다. 원 반복값은 [timing.json](results/timing/timing.json)에 있습니다.

<a id="s7"></a>
<a id="analysis"></a>
## 7. 결과 분석
**동일 state에서의 개입을 확인했고, 정답 점수와 첫 오류 수명이 다르게 움직였습니다.** CE는 정답 확률의 음의 로그를 token별로 평균하지만 RMST0는 한 번의 이른 오류에도 줄어듭니다. Fitting의 목적함수는 평균 CE와 가중치 변화 penalty이며 sequence의 최초 오류를 직접 최적화하지 않습니다. 이 실험은 두 목표의 불일치를 보여주지만, 그 차이의 기하학적 원인까지 규명하지는 않았습니다.

INT8의 경험적 T0.05 변화는 seed0: 141→143 / seed1: 115→110 / seed2: 143→131 token입니다. seed0에서는 이 경험적 위치가 늘었지만 RMST0는 줄었습니다. 같은 seed에서도 위험 5%의 위치와 평균 연속 길이는 다른 분포 요약입니다. 이 경험값에 신뢰지원 하한의 의미를 붙이지 않습니다.

SHORT가 바뀌지 않은 결과는 이 fitting 입력·gradient tolerance·FP32 경계에서 추가 보정이 없었다는 뜻입니다. MIXED는 이미 오류가 있는 더 긴 FIT 특징을 포함하고 두 tensor를 바꿨습니다. 두 차이는 fitting 위치 범위의 결과로 해석하되, iteration cap에 도달한 seed0/1에 대해 가능한 모든 선형 판독기의 최적 성능이라고 주장하지 않습니다.

FP64 진단의 재실행, codec 변경, backbone 재학습으로 얻은 결과가 아닙니다. 같은 세 checkpoint에서 하나의 fresh TEST를 사용한 결과이며 다른 학습 seed 모집단이나 LM 문맥 길이 전체로 확대하지 않습니다. 기술적 all-terminal 경계 오류는 합성 검사에서 TEST 전 수정했고, 실제 결과의 부호에 따라 실행 규칙을 바꾸지 않았습니다.

### 7.1 같은 기록의 사후 분석

아래는 공개 준비 중 저장된 prediction·gold·점수 합계에서 추가 계산한 `POST_HOC_SAME_RECORDS` 분석입니다. 새로운 fitting·forward는 0이며 원래 주 비교와 신뢰 family를 바꾸지 않았습니다. [계산 코드](publication/posthoc/analyze.py)와 [원파일 hash가 포함된 결과](publication/posthoc/data/summary.json)가 대응합니다.

**같은 입력에서 저장 방식에 따른 효과 차이.** 각 입력의 `(INT8 MIXED−ORIGINAL) − (Native MIXED−ORIGINAL)`를 먼저 계산한 뒤 평균·구간을 구했습니다. 단위는 token입니다.

| Seed | Native Δ | INT8 Δ | Interaction | Post-hoc 95% interval |
| --- | --- | --- | --- | --- |
| 0 | -10.20 | -9.89 | +0.31 | [-2.47, 3.07] |
| 1 | -9.54 | -10.40 | -0.87 | [-4.36, 2.91] |
| 2 | -13.21 | -13.49 | -0.29 | [-2.10, 1.48] |

이 사후 구간은 5,000회 paired bootstrap(seed 71101/71102/71103, linear quantile)의 pointwise 95%입니다. 세 구간 모두 0을 포함하며 원래 98.333…% 주 구간과 합치지 않습니다.

**첫 오류가 이동한 입력의 분포.** 아래 분모는 각 행 1,024개 base sequence입니다. `d = correct-prefix(MIXED) − correct-prefix(ORIGINAL)`이며 나중에 정답으로 돌아와도 처음 기록한 오류는 바꾸지 않습니다.

| Seed | Storage | Longer | Shorter | Same | Q25 token | Median token | Q75 token |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | NATIVE_FP32 | 314 | 404 | 306 | -15 | 0 | 5.25 |
| 0 | UNIFORM_8 | 313 | 411 | 300 | -15 | 0 | 5 |
| 1 | NATIVE_FP32 | 340 | 427 | 257 | -15 | 0 | 6 |
| 1 | UNIFORM_8 | 324 | 431 | 269 | -16 | 0 | 5 |
| 2 | NATIVE_FP32 | 331 | 494 | 199 | -33 | 0 | 8 |
| 2 | UNIFORM_8 | 320 | 491 | 213 | -32 | 0 | 7 |

INT8에서는 더 짧아진 입력이 각각 411/431/491개로 더 길어진 313/324/320개보다 많았습니다. 중앙값은 모두 0으로, 평균 감소가 모든 입력의 동일한 변화라는 뜻은 아닙니다. [입력별 기록](publication/posthoc/data/items.json)에는 증가·감소·동일 범주를 모두 제공합니다.

**위치 구간별 오류 재배치.** INT8 위치 129–256을 예로 보면:

| Seed | CE nats O→M | Accuracy % O→M | New first errors O→M | Only ORIGINAL correct | Only MIXED correct | Wrong→different wrong |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9255 → 0.1126 | 95.86 → 96.59 | 422 → 457 | 937 | 1896 | 107 |
| 1 | 2.2983 → 0.2417 | 92.54 → 92.88 | 478 → 501 | 1940 | 2388 | 319 |
| 2 | 0.5672 → 0.3701 | 96.68 → 96.50 | 627 → 661 | 1950 | 1713 | 615 |

CE가 세 checkpoint 모두 낮아진 이 구간에서도 최초 오류가 발생한 입력 수는 모두 늘었습니다. 앞 구간의 정답률이 전부 낮아졌다고 단순화할 수는 없습니다. 위치별 token 정답률은 seed0/1에서 높아지고 seed2에서 낮아졌습니다.

표의 정답률·전환 수는 1,024×128=131,072 token 관측, CE는 그중 저장된 유효 점수 관측이 분모입니다. 이번 기록에는 invalid score가 없었습니다. 최초 오류 수는 **전체 1,024입력 중 해당 구간에 처음 틀린 수**이며 직전 생존 입력을 분모로 한 hazard가 아닙니다. 여섯 구간 1–32 / 33–128 / 129–256 / 257–512 / 513–1024 / 1025–2048의 전체 표는 [사후 결과](publication/posthoc/data/summary.json)에 있습니다. 전체 CE·정답률은 구간별 token 수로 가중하며 각 구간에 같은 비중을 두지 않습니다.

**Margin의 확인 범위.** 공개 기록에는 모든 입력의 위치별 margin 합계가 있지만 입력별 logits·margin은 없습니다. 따라서 ORIGINAL 정답→MIXED 오답 사건만 선택한 margin 변화와 full-logit KL은 계산할 수 없습니다. 작은 경계 변화나 특정 클래스 이동이 원인이라는 진단은 남겨 두었습니다.

**대조군의 분모.** 실제 head와 input-only/shuffle은 같은 1,024×2,048=2,097,152개 token 위치를 채점합니다. Shuffle은 같은 시점·current token 안에서 label 없이 순열을 만들고 singleton을 유지합니다. 첫 위치는 gold=current token이어서 대조군도 정답률 100%입니다. 전체 약 1/6이라는 값이 모든 위치의 chance 수준이나 모든 shortcut 부재를 뜻하지 않습니다.

<a id="s8"></a>
<a id="conclusion"></a>
## 8. 결론
완성한 결과물은 같은 cache의 세 판독기 비교, 1,158개 parameter의 제한된 지도 보정, 두 tensor patch와 새 프로세스 재실행, 공개 prediction 기반 독립 검산입니다. INT8의 MIXED 보정 후 seed0/1/2의 평균 연속 정답 길이 변화는 각각 -9.89 / -10.40 / -13.49 token이며, 세 주 비교 구간 모두 0 아래입니다. 위치 256 밖의 정답 CE 개선은 관측했지만 처음부터 정확히 읽는 길이는 늘리지 못했습니다. 이 결과는 현재 feature·최종 Linear·fitting 예산의 범위이며, 모든 decoder에서 정보 복원이 불가능하다는 증명이 아닙니다.

원형 연구 문서와 당시 검증 상태는 [원본 문서 사본](publication/original_docs/)에 보존했습니다. 이번 공개 정리는 같은 prediction·스칼라의 재계산과 사후 분석이며 새 fitting·TEST forward·GPU 실행을 포함하지 않습니다. 현재 공개 검사와 원본 복원은 [publication 안내](publication/README.md)에서 확인합니다.

<a id="s9"></a>
<a id="references"></a>
## 9. 레퍼런스 및 기여
- [Case010 version map](../010-ckda-finite-precision-memory-horizon/VERSION_MAP.md) · [fixed precision/readout source](../010-ckda-finite-precision-memory-horizon/versions/v2/source/precision.py).
- OpenEuroLLM / ComplexKDA, pinned commit `ef9d108d1692387cae37f5b2d539a71826a127c1`; original model/task and coefficients.
- John Hewitt and Percy Liang (2019), [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/). Control motivation; this study does not reproduce that paper’s experimental control task.
- [Protocol and source freeze](configs/protocol.json) · [selected heads](configs/selected_heads.json) · [scalar aggregate](results/derived/aggregate.json) · [head diagnostics](results/head_diagnostics.json) · [NOTICE](NOTICE.md).

기존 모델·과제는 ComplexKDA, 상태 codec은 Case010, 수치 계산과 L-BFGS는 PyTorch/NumPy에 기반합니다. DIOVA는 고정 상태의 판독 개입, 제한된 적합, patch 저장·검사와 paired 결과 분석을 구현했습니다. 원저자·라이선스·Codex 지원은 [NOTICE](NOTICE.md)에 정리했습니다.
