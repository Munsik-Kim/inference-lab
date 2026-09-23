# Case 010 — 저정밀 recurrent state의 저장·재시작과 기억 수명

[English](REPORT.md) · [Case 홈](README.ko.md) · [CPU 재현 안내](REPRODUCTION.ko.md)

하나의 프로젝트에서 실제 저비트 상태 저장, 잔차 전달, 실패를 보존하는 재시작, 판독 수명 측정을 연결했습니다. 단계 A는 저장 방식을 탐색했고, 단계 B는 재시작 계약을 고친 뒤 고정된 소수 설정을 새 입력에서 비교했습니다.

## 목차

[1. 배경](#background) · [2. 가설과 검증 질문](#questions) · [3. 이론](#theory) · [4. 방법](#methods) · [5. 실험](#experiments) · [6. 실험 결과](#results) · [7. 결과 분석](#analysis) · [8. 결론](#conclusion) · [9. 레퍼런스](#references)

<a id="background"></a>
## 1. 배경

Recurrent 모델은 입력을 받을 때마다 상태를 읽고 다시 저장합니다. 고정된 가중치의 양자화와 달리, 매번 저장한 값이 이후 전이의 입력이 됩니다. 상태의 평균제곱오차가 작아도 기호 판독은 틀릴 수 있고, 마지막 token이 맞아도 앞선 오류는 가려질 수 있습니다. 그래서 이 프로젝트는 **최초의 잘못된 기호 판독**, 그전까지 연속 정답인 길이, 함수 호출 사이에 실제로 저장해야 하는 바이트를 함께 측정합니다.

과제는 여섯 원소로 된 순열군 S3의 임의 원소를 입력으로 받습니다. 별도의 정수 곱셈표가 왼쪽 누적곱 `g_t = x_t … x_1`을 계산하고, 원래 모델의 MLP가 매 token 뒤 그 기호를 판독합니다. 공식 평가의 마지막 1/4 구간 보정 정확도와 여기서 추가한 모든 prefix의 생존율은 서로 다른 지표입니다.

<a id="questions"></a>
## 2. 가설과 검증 질문

단계 A의 질문은 자신의 write-back 잔차를 다음 전이로 전달하면 같은 총 바이트 한도에서 더 오래 올바르게 판독할 수 있는가입니다. 전체 차원 잔차는 기전 확인용 기준선이고, 저랭크 잔차는 일부 방향을 버리는 대신 저장량을 줄입니다. 균일·CAL 기반 혼합정밀도는 고정된 저비트 상태 하나보다 강한 비교군입니다.

단계 B는 단계 A의 관측을 본 뒤 설계했습니다. 기능 질문은 유한·terminal·stochastic stream의 실행 의미가 직렬화와 새 프로세스 재시작 뒤에도 같은가입니다. 새 입력의 주 비교는 Rank2의 N=128 저장 한도 안에서 CAL로 정한 Mixed5/6과의 비교입니다. 고정 계수 정밀도 및 후기 상태 추적은 진단입니다. 두 단계 전체를 관측 전에 한 번에 사전등록한 실험으로 취급하지 않으며, 각각의 protocol freeze를 보존합니다.

<a id="theory"></a>
## 3. 이론과 지표 정의

각 head의 물리 좌표 상태는 다음 convention을 따릅니다.

```text
A_t = (I − beta_t k_t k_tᵀ) Diag(alpha_t)
B_t = beta_t k_t v_tᵀ
S_t = A_t S_(t−1) + B_t
attention 판독 = q_tᵀ S_t / sqrt(16)
```

원래 정규화, 부호를 허용하는 gate, 출력 gating·projection·MLP를 유지합니다. 명시적인 물리 좌표 경로를 사용하여 upstream의 누적 부호 gauge와 다른 좌표의 잔차를 섞지 않습니다.

후보는 저장 상태 `Z + Rhat`을 표현합니다.

```text
U_t = A_t (Z_(t−1) + Rhat_(t−1)) + B_t
Z_t = Decode_state(Encode_state(U_t))
E_t = U_t − Z_t
Rhat_t = Decode_residual(Encode_residual(E_t))
```

`E_t`는 후보 자신의 저장에서 생긴 잔차입니다. 후보에 정답·reference state·미래 오차를 주지 않습니다. 저랭크 방식은 CAL에서 고정한 head별 기저의 계수를 저장하며, 복원한 잔차를 전체 전이로 전달한 후 다시 투영합니다. 전이 후에 이전 잔차를 더하는 untransported 설정은 배치 위치를 확인하는 대조군이며, 문헌의 모든 error-feedback 방법을 대표하지 않습니다.

정확한 전체 잔차를 쓰면 같은 affine 갱신을 복원한다는 관계는 계산 항등식입니다. 유한 저장·연산에서는 후보와 reference의 산술 반올림 차이도 포함해야 합니다. 정규화된 key, `|alpha|≤1`, `beta∈[0,2]` 아래의 이상적인 비팽창 전이 norm 상계는 누적 오차를 다루지만 비선형 MLP의 정답을 보증하지 않습니다. 유한군의 ID와 전이표를 저장하는 정확한 과제 전용 대조군도 있으므로, 이 과제에서 길이에 따라 무조건 저장 비트가 증가해야 한다는 하한을 주장하지 않습니다.

| 지표 | 정의 |
| --- | --- |
| 최초 실패 `tau` | 채점하는 group token에서 처음 오답 또는 INVALID가 나온 위치; 나중의 정답 복귀로 지워지지 않음 |
| 생존율 `S(t)` | `tau > t`인 입력의 비율 |
| RMST0 | `mean(min(tau−1, Tmax))`; 미실패는 `Tmax`를 기여; 단위는 처음부터 연속해서 맞힌 group token 수 |
| 경험적 `T0.05` | 관측 최초 실패 비율이 0.05이하인 최대 평가 위치; v1 원래 값은 7개 grid, v2는 모든 token의 경험값도 제시 |
| 신뢰지원 grid 하한 | Bonferroni `alpha/family`로 구한 단측 exact이항 위험 상한이 0.05이하인 최대 사전 grid |
| 저장량 | `B_total(N)=B_shared+N*B_stream`; 공유 기저·codec 설정·실제 token table 포함 |

우측 검열은 최대길이 2048까지 실패를 관측하지 않았다는 뜻입니다. 두 방법의 grid 하한이 다른 것만으로 실제 horizon 차이를 검정한 것은 아닙니다. 입력별 paired 구간은 고정 checkpoint 안에서 따로 계산하며, token·arm·세 checkpoint를 독립 입력으로 합치지 않습니다.

<a id="methods"></a>
## 4. 방법

모델은 고정된 S3/a11_b02 Complex KDA입니다. 한 층, 폭 48, 16×16 상태의 12개 head, short convolution 없음, 48→192→6 MLP 판독기를 사용합니다. Native 계수와 학습 recurrence는 FP32입니다. 한 checkpoint의 모든 방법은 가중치·계수·입력·판독기를 공유합니다. 매 token의 표현 상태를 저장한 뒤 판독하는 CPU 순차 reference 경로입니다.

BOS는 write 0이고 첫 채점 group token은 write 1입니다. BOS의 단순 정오답은 tau에서 제외하지만, BOS에서 terminal이 되면 채점 구간까지 유지됩니다. 동률은 가장 작은 label index로 처리하며 출력 전체의 유한성을 확인합니다.

현재 [실패 보존 adapter](versions/v2/source/online_v2.py)는 세 사건을 다르게 처리합니다.

| 사건 | 상태 처리 | 채점 처리 |
| --- | --- | --- |
| 유한한 오답 | 계속 갱신 | 최초 오류 기록; 이후 token의 정답 복귀 허용 |
| 유한한 state의 비유한 readout | 유한 상태를 commit하고 계속 진행 | 해당 위치만 INVALID `−1` |
| 재구성·전이·저장의 수치 실패 | 해당 행만 terminal; 마지막 commit된 state·잔차·RNG body 유지 | 이후 항상 INVALID; cursor는 계속 증가 |

깨진 checkpoint, 잘못된 shape, programming error, process failure는 실행 오류이며 알고리즘 terminal 발생률로 바꾸지 않습니다. 행별 갱신은 저장 후 유한성까지 확인한 뒤 한 번에 commit합니다. Terminal 행은 추가 RNG를 소비하지 않습니다.

v2 header는 little-endian cursor 8B, terminal code 1B, 최초 terminal write 8B로 총 17 B입니다. v1은 cursor 8B만 저장했습니다. 공유 schema identity도 저장량에 포함합니다. 정답·tau·출력 이력은 [별도 평가 checkpoint](versions/v2/source/checkpoint.py)에 속하며 모델 state가 아닙니다. 임시 디렉터리 검증 후 같은 filesystem에서 rename하는 절차는 원자적 노출을 제공하지만 fsync 기반 crash durability를 입증한 것은 아닙니다.

### 단계 B의 다섯 설정

| 표시 이름 | 원래 식별자 | 저장하는 값 |
| --- | --- | --- |
| Native FP32 | `NATIVE_FP32` | 전체 FP32 상태와 공통 v2 실패 envelope |
| INT8 | `UNIFORM_8` | 모든 상태 채널을 8bit로 저장하고 scale 포함 |
| INT5 | `UNIFORM_5` | 모든 상태 채널을 5bit로 저장하고 scale 포함 |
| Rank2 | `LOWRANK_4_8_R2` | INT4 상태와 고정 rank2 잔차 기저의 INT8 계수 |
| Mixed5/6 | `MIXED_5_6_BUDGET` | 상태를 5bit에서 시작하고 CAL 순위의 29개 head/key 채널을 6bit로 승격; Rank2의 N=128 cap 사용 |

Native와 INT8은 용량·충실도 참고군이며 Rank2 cap 안에 들어가지 않습니다. 다섯 방법은 같은 실패 envelope 계약을 적용하며 모든 recurrent head를 사용합니다.

<a id="experiments"></a>
## 5. 실험과 근거 범위

| 조건 | 단계 A / v1 | 단계 B / v2 |
| --- | --- | --- |
| 모델 | S3/a11_b02 학습 종료 checkpoint 세 개 | 같은 세 checkpoint 재사용; 새 학습 없음 |
| CAL | 64 × 32; seed 1001 | 같은 보정 통계와 기저 재사용 |
| DEV | 128 × 128; seed 2001; 학습 확장 조건 DEV32 | 기존 DEV 및 아래 별도 진단 입력 |
| 주 TEST | 512 × 2048; seed 3001 | fresh 1024 × 2048; seed 50101 |
| 설정 | 원래 26개 + 보충 혼합정밀도 배치 4개 | Native, INT8, INT5, Rank2, Mixed5/6 |
| 동시 신뢰 family | 26×7×3=546; 통합 보충 30×7×3=630 | 5×13×3=195 |
| 신뢰 grid | 32, 64, 128, 256, 512, 1024, 2048 | 32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 512, 1024, 2048 |
| 진단 입력 | 별도 다섯 toy geometry 조건 | 32 × 2048; seed 40101; 정밀도·상태 추적 |
| 시간 측정 | 다른 평가가 끝난 뒤 별도 CPU 측정 | 16 × 128; seed 2002; 3회; 2 threads |
| 연구 단계 | 탐색 및 같은 TEST의 보충 예산 감사 | v1 관측을 참고한 실패 보존 후속 설계 |

세 로컬 checkpoint는 각각 20,000 updates, batch 128, upstream Muon/AdamW 정책과 curriculum 4/6/8/16/32로 학습했습니다. 마지막 예정 update를 선택하고, seed 0의 DEV32 조건과 남은 예산을 만족해 seed 1/2로 확장했습니다. TEST로 checkpoint를 선택하지 않습니다. [v1 protocol](versions/v1/configs/protocol.json), [checkpoint provenance](versions/v1/provenance/checkpoints.json), [v2 protocol](versions/v2/protocol_v2.json)에 정확한 정책과 식별자가 있습니다.

단계 A의 추가 네 배치는 같은 TEST를 재사용한 `POST_HOC_BUDGET_AUDIT_SAME_TEST`입니다. 단계 B의 좁은 메뉴는 이미 본 v1 관측을 참고했고 rank·기저는 다시 적합하지 않았습니다. Mixed5/6은 고정 CAL 중요도와 실제 직렬화 길이로 29개 head/key channel을 5→6bit로 올리고, 다음 승격이 cap을 넘기기 전에 멈춥니다. 공통 1024 입력 평가 batch는 N=128 저장 시나리오와 다른 개념입니다.

원래 실행 receipt에는 이미 수행한 모델 replay·정밀도·시간 측정이 기록되어 있습니다. 이번 통합은 보존 검사, 저장 스칼라 재계산, 합성 CPU 계약 검사만 수행합니다. 현재 통합의 실행 기록과 [v2 당시 validation receipt](versions/v2/provenance/validation_receipt.json)는 별개입니다. 이전 요청에서 지목한 독립 검토 문서·probe는 보존된 근거에서 확보되지 않았고, [가용성 기록](versions/v2/results/historical-reanalysis/review_artifact_availability.json)에 이 제약이 있습니다. 읽지 못한 검토자의 테스트 수를 이번 실행 결과로 쓰지 않습니다.

<a id="results"></a>
## 6. 실험 결과

### A. 단계 A — 저장 방식 탐색과 비교군 보충

원래 26-arm/546-family에서 신뢰지원 하한이 더 긴 잔차 예산 비교는 2/54입니다. 가능한 혼합정밀도 배치 네 개를 더한 30-arm/630-family에서는 0/54입니다. 이는 정해진 후보 cap과 비교 메뉴의 결과이며, 모든 codec의 실패나 개선 불가능성의 증명이 아닙니다. Rank2의 과거 평균 길이 신호와 native의 유한한 수명도 [원래 보고서](versions/v1/REPORT.ko.md)와 [역사적 재계산](versions/v2/results/historical-reanalysis/HISTORICAL_ANALYSIS.ko.md)에 남아 있습니다.

과거 native의 모든 token 기준 경험적 T0.05는 139/108/144이고 원래 7개 grid 값은 128/64/128입니다. 동시 신뢰지원 grid 하한은 세 checkpoint 모두 64입니다. v2에서는 입력·표본수·grid·family가 달라졌으므로 이 하한을 v2 하한에서 빼 기억 개선량으로 쓰지 않습니다.

<a id="restart"></a>
### B. 실패와 재시작

| 근거 | v1 동작 | v2 동작 |
| --- | --- | --- |
| 요청 설명으로 재구성한 합성 fault probe | 함수 안의 실패 표시가 bytes를 다시 읽을 때 사라질 수 있음 | terminal metadata로 부활 방지 |
| 보존된 19개 historical 셀 | uninterrupted 출력이 대조 기준; U2 legacy split 3개는 write 1024 뒤 부활 | 전체 19개 full/split/fresh 예측·최종 hash 대응 |
| 이전에 관측한 수치 실패 4개 | 관측 후 선택한 write 838, 1741, 623, 867 | 같은 위치에서 재현; 실패 직후 별도 프로세스가 body·RNG 고정과 cursor 유지 |
| 유한·stochastic 계약 fixture | 원래 body·RNG가 대조 기준 | 정상 행·깨진 header·chunk 경계·새 프로세스 검사를 별도로 구성 |

[Historical 요약](versions/v2/results/historical-summary/summary.json) · [실제 실패 경계 receipt](versions/v2/results/historical-failure-boundaries/index.json) · [합성 probe](versions/v2/results/v1_failure_probe_from_request.json) · [현재 계약 테스트](versions/v2/tests/test_online_v2.py).

이 실제 모델 대조는 보존된 과거 근거이며 이번 통합에서 추론을 다시 실행한 결과가 아닙니다. 합성 probe는 설명된 결함을 재구성했으며 확보하지 못한 검토자의 원래 script는 아닙니다. Private 최종 상태 body는 포함하지 않으며 공개 감사는 hash 대응과 compact prediction을 확인합니다. Terminal v2 body가 v1 placeholder와 다른 것은 의도한 수정입니다.

<a id="native-int8"></a>
### C. Fresh 입력 — native와 INT8

Native의 stream별 직렬화 크기는 12,305 B, INT8은 3,137 B로 74.5% 감소했습니다. 공유 비용은 아래에서 별도로 계산합니다. RMST0는 첫 오류 직전까지 처음부터 연속해서 맞힌 group token 수의 평균이며 전체 token 정확도가 아닙니다.

| Seed | Native RMST0 | INT8 RMST0 | INT8 − Native (token) |
| --- | --- | --- | --- |
| 0 | 288.57 | 288.12 | -0.45 |
| 1 | 271.15 | 271.57 | +0.42 |
| 2 | 238.28 | 237.49 | -0.79 |

INT8−native의 pointwise 95% paired 구간은 각각 [−2.64, 1.85], [−2.14, 3.01], [−2.00, 0.38] token입니다. 작은 점추정 차이는 이 checkpoint들의 관측이며 사전 정의한 동등성·비열등성 판정이 아닙니다. [원래 paired 표](versions/v2/results/fresh-summary/paired_table.csv).

<a id="fresh"></a>
### D. Fresh 입력 — 다섯 설정과 같은 예산 비교

| Seed | 설정 | RMST0 (token) | 경험적 T0.05 | 신뢰지원 grid T0.05 ≥ | Terminal / 1024 |
| --- | --- | --- | --- | --- | --- |
| 0 | Native FP32 | 288.57 | 135 | 112 | 0 |
| 0 | INT8 | 288.12 | 136 | 112 | 0 |
| 0 | INT5 | 242.85 | 123 | 96 | 0 |
| 0 | Rank2 | 277.47 | 129 | 112 | 0 |
| 0 | Mixed5/6 | 268.20 | 132 | 112 | 0 |
| 1 | Native FP32 | 271.15 | 118 | 96 | 0 |
| 1 | INT8 | 271.57 | 118 | 96 | 0 |
| 1 | INT5 | 220.35 | 96 | 80 | 0 |
| 1 | Rank2 | 219.97 | 112 | 96 | 0 |
| 1 | Mixed5/6 | 240.73 | 110 | 96 | 0 |
| 2 | Native FP32 | 238.28 | 142 | 128 | 0 |
| 2 | INT8 | 237.49 | 141 | 128 | 0 |
| 2 | INT5 | 222.77 | 133 | 112 | 0 |
| 2 | Rank2 | 235.60 | 141 | 128 | 0 |
| 2 | Mixed5/6 | 230.69 | 139 | 112 | 0 |

Terminal 발생 0건은 수치 상태 종료가 없었다는 뜻입니다. 모든 stream은 2048까지 최소 한 번 기호 판독을 틀렸습니다. 각 checkpoint·설정의 분모는 1024를 유지합니다. 같은 입력을 다섯 설정에 적용한 것은 paired 관측이며 5120개 독립 시퀀스가 아닙니다.

| Seed | Rank2 − Mixed RMST0 (token) | Pointwise 95% 구간 |
| --- | --- | --- |
| 0 | +9.27 | [+1.07, +17.35] |
| 1 | -20.76 | [-27.41, -14.20] |
| 2 | +4.91 | [+0.78, +8.94] |

구간은 checkpoint마다 입력을 짝지어 5000회 bootstrap한 pointwise 구간입니다. Rank2의 신뢰지원 하한이 Mixed5/6보다 큰 경우는 3개 checkpoint 중 1개이며 기술통계상의 하한 비교입니다. Uniform5는 보조 기준선으로 남기고 더 강한 예산 대응 Mixed5/6을 대신하지 않습니다. [원래 fresh 표](versions/v2/results/fresh-summary/fresh_table.csv) · [통합 파생 요약](summary/project.json).

![v2 fresh 입력의 checkpoint별 생존곡선; v1/v2를 전후 개선 곡선으로 합치지 않음](versions/v2/figures/fresh_survival.png)

### E. 실제 저장량과 CPU 비용

| 설정 | Stream bytes | Shared bytes | 합계 N=1 | 합계 N=16 | 합계 N=128 |
| --- | --- | --- | --- | --- | --- |
| Native FP32 | 12305 | 30344 | 42649 | 227224 | 1605384 |
| INT8 | 3137 | 30560 | 33697 | 80752 | 432096 |
| INT5 | 1985 | 30560 | 32545 | 62320 | 284640 |
| Rank2 | 2033 | 32409 | 34442 | 64937 | 292633 |
| Mixed5/6 | 2043 | 30965 | 33008 | 63653 | 292469 |

주 예산 비교는 N=128에서 Rank2 292,633 B와 Mixed5/6 292,469 B입니다. 공유·stream 비용이 다르므로 N=1024에서도 같은 cap 관계라고 가정하지 않습니다. 실패 envelope, 실제 코드·scale·RNG, 기저, mixed map, codec 설정, canonical token table을 모두 셉니다. 같은 이전 envelope와 비교해 v2는 stream 9B와 shared 934B를 추가합니다. 공통 모델 parameter 값 226,120 B와 checkpoint 파일 233,203 B는 후보 cache 감소와 별도입니다. 프로세스 peak RAM과 GPU VRAM은 측정하지 않았습니다. [실제 buffer 장부 검사](versions/v2/results/ledger_validation.json).

다음은 CPU 2개 thread, 16개 입력 × 128개 group token, 고정 3회에서 구한 **complete-call 중앙값 ms/group-token**입니다. BOS 작업은 호출 안에 있고 분모는 16×128입니다. 전이·잔차 투영·packing·상태 처리·원래 readout을 포함하며 loading·disk I/O·gold/shadow 진단은 timer 밖입니다.

| Seed | Native FP32 | INT8 | INT5 | Rank2 | Mixed5/6 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0306 | 0.0939 | 0.1209 | 0.1438 | 0.1185 |
| 1 | 0.0288 | 0.0975 | 0.1206 | 0.1491 | 0.1160 |
| 2 | 0.0306 | 0.0946 | 0.1090 | 0.1419 | 0.1119 |

이 prototype codec 호출은 native보다 느렸습니다. GPU 가속이나 단일 요청 latency로 해석하지 않습니다. 연구 worker 종료 후 순차로 측정했으며 운영체제 전체 background 부하는 완전히 통제하지 못했습니다. [원래 시간 기록](versions/v2/results/timing/).

### F. 고정 계수 정밀도와 후기 불안정

| 진단 | 보존된 관측 | 해석 범위 |
| --- | --- | --- |
| D00 FP32 recurrence / FP32 readout | 진단 32개 입력의 RMST0: 290.38 / 279.69 / 236.75 | fresh 1024와 다른 코호트 |
| D10 FP64 recurrence / cast→FP32 readout | seed별 전체 예측·tau가 D00과 같음 | 저장된 FP32 계수를 승격; 다시 생성하지 않음 |
| D01 FP32 recurrence / FP64 readout; D11 FP64 / FP64 | seed별 전체 예측·tau가 D00과 같음 | 고정 표현 가중치; 명시적인 dtype·parity 검사 |
| Rank2 후기 state norm | 최대 ≈ 1.2832e9 / 1.5151e9 / 4.4855e16; 모두 최초 판독 실패 후 | 계속된 rollout과 발생 순서 관측; 원인 규명 아님 |

[정밀도 대응표](versions/v2/results/diagnostic-summary/precision_pairs.csv) · [dtype/readback 검사](versions/v2/provenance/diagnostic_readback.json) · [trace 극값](versions/v2/results/diagnostic-summary/trace_extrema.csv).

같은 32개 진단 입력에서 시간별 norm, self-writeback error, projection leakage, 판독 margin을 기록했습니다. t=512에서는 각 Rank2 checkpoint의 32개 입력이 모두 최초 오류 이후입니다. 전·후 곡선의 조건부 집단은 시간에 따라 변하며 유효 개수를 함께 제시합니다. Terminal scratch 0은 제외하고 gold·native·FP64 shadow는 evaluator만 사용합니다. 산술 승격이 정확한 정답 모델을 만들거나 native 수명 원인을 모두 분리한 것은 아닙니다.

<a id="analysis"></a>
## 7. 결과 분석

Packing과 재시작은 state MSE 그림만으로 드러나지 않는 실행 문제를 다룹니다. 숫자 payload가 넘어가는 경계에서 실패 이력도 함께 넘어가야 합니다. 새 terminal envelope는 그 계약을 수정했고, 기호 기억 길이의 변화는 별도 비교가 판단합니다.

더 강한 N=128 비교군에 대해 Rank2의 평균 차이는 두 checkpoint에서 양수, 한 checkpoint에서 음수입니다. 평균 길이와 5% 위험의 꼬리 지표는 다르게 움직일 수 있습니다. Native 자체도 유한 길이에서 all-prefix 정확도를 잃으므로 후보의 오류를 모두 양자화 탓으로 돌릴 수 없습니다. 고정 FP32 표현값의 산술 승격은 진단 예측을 바꾸지 않았고, 계속된 rollout에서는 큰 후기 norm이 관측됐습니다. 이것만으로 학습 부족·표현 불가능성·보편적인 필요 정밀도를 증명하지 않습니다.

따라서 이 결과는 구체적으로 읽어야 합니다. 구현물은 저장량과 판독 길이의 재계산 가능한 비교를 제공합니다. INT8은 작은 직렬화 상태에서 native에 가까운 평균 점추정을 보였고, 검사한 Rank2 보정은 세 checkpoint에 걸친 일관된 같은 예산 우위를 확보하지 못했습니다. 유리한 평균과 함께 원래 불리한 사례·보충 기준선도 확인할 수 있습니다.

<a id="conclusion"></a>
## 8. 결론

- 실제 저비트 저장, 온라인 잔차 전달, 실패·cursor·RNG를 보존하는 새 프로세스 재시작을 구현했습니다.
- 같은 세 checkpoint의 fresh 입력 비교를 완료한 원기록을 보존했습니다. 이번 통합은 새 추론 없이 그 근거를 재계산합니다.
- Rank2는 Mixed5/6 대비 일관된 같은 예산 기억 수명 우위를 확보하지 못했습니다.
- INT8은 native보다 stream별 직렬화 상태를 74.5% 줄였고, 각 checkpoint의 평균 수명 점추정 차이는 1 token 미만이었습니다.

[CPU 검산 실행](REPRODUCTION.ko.md) · [현재 adapter 읽기](versions/v2/source/online_v2.py) · [원형 버전 추적](VERSION_MAP.md).

<a id="references"></a>
## 9. 레퍼런스와 기여

문헌 목록은 보존한 [v1 NOTICE](versions/v1/NOTICE.md)와 [v2 NOTICE](versions/v2/NOTICE.md)를 따릅니다. 이번 통합에서는 새 문헌 결과나 재현 주장을 추가하지 않았습니다. Complex KDA는 모델·과제·native recurrence를 제공하고, recurrent-state 양자화·corrected-state feedback·상태 오류 분석에는 선행연구가 있습니다. DIOVA는 adapter·바이트 계약·비교 protocol·감사와 표시 도구를 구현했으며 Codex가 구현·문서화를 지원했습니다. [통합 출처와 라이선스](NOTICE.md).
