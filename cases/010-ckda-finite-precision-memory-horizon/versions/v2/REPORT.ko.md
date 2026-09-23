# Case010 v2 — 실패 상태 보존과 동일 저장 예산의 판독 길이

[English](REPORT.md) · [처음으로](README.ko.md)

수치적으로 실패한 recurrent stream의 상태·커서·난수를 저장하고, 새 프로세스에서도 종료 상태를 유지하는 codec를 구현했습니다. 같은 세 학습 체크포인트에서 저비트 저장 방식과 최초 판독 실패까지의 길이를 새 입력으로 비교했습니다.

## 목차

[배경](#배경) · [검증 질문과 가설](#검증-질문과-가설) · [이론](#이론) · [방법](#방법) · [실험](#실험) · [실험 결과](#실험-결과) · [결과 분석](#결과-분석) · [결론](#결론) · [레퍼런스](#레퍼런스)

## 배경

v1의 numerical failure mask는 함수 호출 안에만 있었습니다. 상태 bytes에 그 이력이 없어서, 새 호출은 실패 후 저장된 0 placeholder를 정상 상태로 읽을 수 있었습니다. v2는 마지막 정상 body를 유지하고 실패 원인·최초 write 위치를 직렬화합니다. 기존 실험의 원자료와 판정은 수정하지 않았습니다.

## 검증 질문과 가설

1. 정상·실패·확률적 반올림 상태를 새 프로세스로 옮겨도 예측과 최종 bytes가 같은가?
2. 같은 FP32 계수 표현값에서 recurrence/readout의 산술 정밀도만 바꾸면 native 판독 수명이 달라지는가?
3. 새 입력에서 rank2 residual이 같은 N128 저장 상한을 쓰는 혼합5/6비트보다 5% 위험 판독 길이를 늘리는가?

후보 목록은 이미 본 v1 관측을 참고한 후속 설계입니다. 새 TEST로 rank·기저·코덱·체크포인트를 고르지 않았습니다. [protocol_v2](protocol_v2.json)에 입력bytes와 비교·구간 계산을 고정했습니다.

## 이론

물리 좌표 recurrence는 $S_t=A_tS_{t-1}+B_t$입니다. 보정 후보는 자신의 저장 표현 $Z+\hat R$를 전체 전이에 전달하고, 갱신값에서 생긴 residual을 다시 저장합니다. 정확한 full residual의 transport 항등식은 일반 corrected-state feedback의 성질이며 새 정리가 아닙니다. 정확한 affine 갱신에서 residual 표현 오차를 eta_t로 쓰면 e_t=A_t e_(t−1)+eta_t입니다. 유한 산술에서는 같은 dtype이어도 계산 반올림의 차이를 eta_t에 포함해야 하며, 계수를 바꾸면 별도 오차항이 필요합니다. Rank 제한·유한 산술·MLP 판독을 함께 쓰는 이번 결과에 무한 기억이나 보편적 비트 하한을 부여하지 않습니다.

$\tau$는 첫 채점 group token의 오답 또는 INVALID 시점입니다. 한번 틀린 뒤 정답으로 돌아와도 survival에는 복귀하지 않습니다. $S(t)=P(\tau>t)$이고 RMST0는 첫 실패 전 연속 정답 token 수의 평균입니다. 실패 token은 제외하며 관측 끝까지 맞으면 2048을 기여합니다. Token accuracy는 회복을 포함하는 별도값입니다.

## 방법

S3의 6개 원소를 균일하게 뽑고, 별도 정수 곱셈표로 왼쪽 누적곱 정답을 계산합니다. 기존 single-layer CKDA (d_model 48, 12 heads, head dimension 16), 세 최종 체크포인트, upstream `ef9d108…`를 그대로 사용했습니다. 새 학습 0회입니다. Native도 동일한 FP32 token table을 사용하는 순차 물리 좌표 경로이며 fused CUDA 실행이 아닙니다. CAL 64 × 32 (seed1001)의 기저·중요도를 재사용했습니다.

상태 header는 little-endian cursor 8 B + terminal code 1 B + 최초 terminal write 8 B입니다. ACTIVE는 최초 위치 UINT64_MAX를 씁니다. BOS는 write 0, 첫 채점 token은 write 1입니다. 성공·종료 no-op 모두 cursor를 증가시킵니다. 종료 이후 state/residual/RNG body는 바뀌지 않습니다.

유한 logits의 오답은 계산을 계속합니다. 상태가 유한한데 readout만 비유한하면 그 출력만 −1로 남기고 계속합니다. 재구성·transition·저장 범위·새 상태 판독의 수치 실패는 해당 행만 종료하며 state/residual/RNG 갱신을 모두 되돌립니다. 잘못된 checkpoint·shape·프로그래밍 예외·프로세스 장애는 실험 실행 오류로 따로 다룹니다. Gold와 tau는 evaluator 기록에만 있고 codec 입력에 들어가지 않습니다.

공유 config·basis·token table과 각 stream의 codes·scales·RNG·header를 실제 serialize해 계산했습니다. 혼합 기준은 모두 5 bit로 시작해 기존 CAL 점수 순서대로 29개 head/key channel을 6 bit로 올렸습니다. 다음 승격이 rank2의 N128 상한을 넘기므로 그 전에 멈췄습니다. Uniform8/native는 고정밀 참고군입니다.

| Arm | Stream B | Shared B | Total N1 / N16 / N128 |
| --- | --- | --- | --- |
| NATIVE_FP32 | 12305 | 30344 | 42649 / 227224 / 1605384 |
| UNIFORM_8 | 3137 | 30560 | 33697 / 80752 / 432096 |
| UNIFORM_5 | 1985 | 30560 | 32545 / 62320 / 284640 |
| LOWRANK_4_8_R2 | 2033 | 32409 | 34442 / 64937 / 292633 |
| MIXED_5_6_BUDGET | 2043 | 30965 | 33008 / 63653 / 292469 |

모든 seed의 크기는 같습니다. N=128은 저장 예산의 동시 stream 가정이며 표본수 1024나 실제 평가 batch 1024와 다릅니다. N1/16/128의 실제 buffer 길이도 [별도로 대조](results/ledger_validation.json)했습니다. Mixed는 더 작은 shared 비용과 조금 큰 stream payload를 쓰므로 N1024에서도 같은 cap에 들어간다는 주장은 하지 않습니다. 원래 공통 가중치 값 226,120 B와 checkpoint 파일 233,203 B는 별도입니다. v2는 stream당 9 B, 공유 metadata 934 B를 더 씁니다. 혼합 기준의 v1-envelope 값은 동일한 새 map에 옛header를 적용한 산술 대조이며 과거 측정 arm이 아닙니다. 한 FP32 임시 state 배열은 N1024에서 12,582,912 B이며 여러 임시 배열이 공존합니다. 실제 process RAM peak/GPU VRAM은 측정하지 않았습니다.

## 실험

- **[A] v1 재계산:** toy 125, learned 78, supplement 12. 원래 26-arm/family546와30-arm/joint630을 분리했습니다.
- **[B] 구현 대조:** 합성 fault injection, fresh-process bytes/RNG 검사, 과거 16개 TEST × 2048의 4 arm × 3 checkpoint, DEV 추가 3 arm, 실제 실패 4개 입력. 합성 실패는 실제 실패 빈도에 넣지 않았습니다.
- **[C] 새 평가:** fresh seed 50101의1024 × 2048 입력을 5 arm × 3 checkpoint가 공유합니다. 별도 seed 40101의32 × 2048 입력은 정밀도·trace 진단용입니다.

Fresh 신뢰계산은 13개 고정 grid × 5 arm × 3 checkpoint = 195 family, alpha 0.05의 exact one-sided binomial 상한입니다. Grid 밖 경험적 곡선은 같은 동시 신뢰하한이 아닙니다. 실패 0/1024의 상한은0.0080424여서 보조 1% 위험 하한도 표본수상 가능하지만, 실제 실패수로 다시 판단합니다. Paired RMST 구간은 각 checkpoint 안에서 5000회 입력 단위 bootstrap한 pointwise 구간입니다. 세 checkpoint의 표본을 합치지 않았습니다.

평가 job은 runtime과 누적 tau·INVALID·출력 구간을 별도 checkpoint로 저장합니다. 임시 디렉터리 inventory를 검증한 뒤 같은 filesystem에서 rename합니다. 이는 atomic visibility이며 fsync 기반 crash durability 보장은 아닙니다. 실제 runtime payload는 private에 두고 공개본에는 초기 0 payload, 최종hash와 compact예측을 남겼습니다.

## 실험 결과

### [A] 역사적 v1

원래 26-arm family에서는54개 budget비교 중 2개에서 supported horizon이 컸고, 더 강한 혼합 비교군을 넣은 30-arm/joint630에서는0/54였습니다. 두 기록을 섞지 않았습니다. Native의 모든token 경험적T0.05는139/108/144였으며, 원래 7개 grid 표의128/64/128과 다른 계산입니다. [전체 재계산](results/historical-reanalysis/HISTORICAL_ANALYSIS.ko.md)과 [역사적 그림](figures/historical_v1_30arm_budget.png)을 별도로 제공합니다.

### [B] 실패와 재시작

프롬프트의 계약을 바탕으로 만든 v1 합성 probe에서 실패 stream이 재시작 뒤 정상 출력으로 돌아오는 불일치를 재현했습니다. 요청된 독립 검토ZIP/원래probe 파일은 찾지 못했으므로 그 스크립트를 실행했다고 기록하지 않았습니다. 실제 체크포인트 과거 입력의 19개 대조가 통과했습니다. 정상경로는 v1/v2 예측·body가 같고, v2는 split/fresh-process 예측·최종 bytes가 같습니다. 원래 cut 1024 뒤에 발생한 Uniform3의 실패도 빠뜨리지 않도록, 별도 고정 부록에서 네 실제 실패 직전·직후를 다시 직렬화했습니다. 실패 직후 새 프로세스로 넘긴 4개 검사도 통과했습니다. [실패 경계 부록](results/historical-failure-boundaries/index.json). 실제 실패 후 v1과 다른 body는 이전 정상 body+terminal flag를 유지하는 의도한 변경입니다. [실패 probe](results/v1_failure_probe_from_request.json), [역사적 대조](results/historical/seed0/index.json), [회귀 테스트](tests/test_online_v2.py).

### [C] 고정 계수 정밀도 진단

| Seed | Mode | RMST0 | Empirical T0.05 | S128 / S512 | Token accuracy |
| --- | --- | --- | --- | --- | --- |
| 0 | D00 | 290.38 | 129 | 100.00% / 6.25% | 45.43% |
| 0 | D10 | 290.38 | 129 | 100.00% / 6.25% | 45.43% |
| 0 | D01 | 290.38 | 129 | 100.00% / 6.25% | 45.43% |
| 0 | D11 | 290.38 | 129 | 100.00% / 6.25% | 45.43% |
| 1 | D00 | 279.69 | 141 | 96.88% / 6.25% | 52.53% |
| 1 | D10 | 279.69 | 141 | 96.88% / 6.25% | 52.53% |
| 1 | D01 | 279.69 | 141 | 96.88% / 6.25% | 52.53% |
| 1 | D11 | 279.69 | 141 | 96.88% / 6.25% | 52.53% |
| 2 | D00 | 236.75 | 157 | 96.88% / 0.00% | 39.14% |
| 2 | D10 | 236.75 | 157 | 96.88% / 0.00% | 39.14% |
| 2 | D01 | 236.75 | 157 | 96.88% / 0.00% | 39.14% |
| 2 | D11 | 236.75 | 157 | 96.88% / 0.00% | 39.14% |

D00은 FP32/FP32, D10은 FP64 recurrence 뒤 상태를 FP32로 내려 원래 판독, D01은 FP32 state를 올린 명시적 FP64 판독, D11은 FP64/FP64입니다. q/k/v/alpha/beta/g/e를 만드는 계수 경로는 FP32 표현값으로 고정했습니다. D01/D11은 그 값과 고정 weight를 승격한 뒤 출력 sigmoid(g), 출력 projection, 정규화와 MLP 판독을 실제 FP64로 계산합니다. 계수 생성의 재계산과 판독 산술의 승격을 구분하며, dtype trace와 FP32 reference parity를 검사했습니다. 네 모드는 각 seed에서 전체 예측·tau가 같았습니다. [진단 검산](provenance/diagnostic_readback.json).

![정밀도별 생존과 현재 token 정확도](figures/precision_survival_accuracy.png)

### [C] Fresh 저장 방식 비교

| Seed | Arm | Empirical T0.05 | Supported T0.05 ≥ | RMST0 | Terminal /1024 |
| --- | --- | --- | --- | --- | --- |
| 0 | NATIVE_FP32 | 135 | 112 | 288.57 | 0 |
| 0 | UNIFORM_8 | 136 | 112 | 288.12 | 0 |
| 0 | UNIFORM_5 | 123 | 96 | 242.85 | 0 |
| 0 | LOWRANK_4_8_R2 | 129 | 112 | 277.47 | 0 |
| 0 | MIXED_5_6_BUDGET | 132 | 112 | 268.20 | 0 |
| 1 | NATIVE_FP32 | 118 | 96 | 271.15 | 0 |
| 1 | UNIFORM_8 | 118 | 96 | 271.57 | 0 |
| 1 | UNIFORM_5 | 96 | 80 | 220.35 | 0 |
| 1 | LOWRANK_4_8_R2 | 112 | 96 | 219.97 | 0 |
| 1 | MIXED_5_6_BUDGET | 110 | 96 | 240.73 | 0 |
| 2 | NATIVE_FP32 | 142 | 128 | 238.28 | 0 |
| 2 | UNIFORM_8 | 141 | 128 | 237.49 | 0 |
| 2 | UNIFORM_5 | 133 | 112 | 222.77 | 0 |
| 2 | LOWRANK_4_8_R2 | 141 | 128 | 235.60 | 0 |
| 2 | MIXED_5_6_BUDGET | 139 | 112 | 230.69 | 0 |

| Seed | Candidate − baseline | Δ RMST0 | Pointwise 95% interval |
| --- | --- | --- | --- |
| 0 | LOWRANK_4_8_R2 − MIXED_5_6_BUDGET | 9.27 | [1.07, 17.35] |
| 0 | LOWRANK_4_8_R2 − UNIFORM_5 | 34.62 | [26.26, 43.03] |
| 0 | UNIFORM_8 − NATIVE_FP32 | -0.45 | [-2.64, 1.85] |
| 1 | LOWRANK_4_8_R2 − MIXED_5_6_BUDGET | -20.76 | [-27.41, -14.20] |
| 1 | LOWRANK_4_8_R2 − UNIFORM_5 | -0.38 | [-7.13, 6.24] |
| 1 | UNIFORM_8 − NATIVE_FP32 | 0.42 | [-2.14, 3.01] |
| 2 | LOWRANK_4_8_R2 − MIXED_5_6_BUDGET | 4.91 | [0.78, 8.94] |
| 2 | LOWRANK_4_8_R2 − UNIFORM_5 | 12.83 | [8.81, 17.00] |
| 2 | UNIFORM_8 − NATIVE_FP32 | -0.79 | [-2.00, 0.38] |

Rank2−mixed 평균 차이는 seed0/1/2에서 9.27 / -20.76 / 4.91 tokens입니다. 더 높은 supported 5% 길이는 1/3 checkpoint에서 관측됐습니다. 이는 서로 다른 지원하한의 기술통계 비교이며 방법 간 위험 차이의 다중비교 확증검정이 아닙니다. Fresh 15개 cell 전체에서 수치 terminal은 0/15,360 stream-arm 기록입니다. 이 15,360개를 독립 입력이나 독립 학습 모델로 세지 않습니다.

![Fresh 생존](figures/fresh_survival.png)
![동일 예산과 판독 길이](figures/fresh_budget_horizon.png)
![Paired RMST](figures/fresh_paired_rmst.png)

### 후기 상태와 비용

![Rank2 시간별 진단](figures/rank2_time_diagnostics.png)

그림의 실패 전·후 곡선은 시점마다 구성원이 달라지는 조건부 평균입니다. t128에서 seed0/1/2의 전·후 표본 수는 31/1, 28/4, 31/1이고 t512에는 모두 0/32입니다. 각 지표의 유효 표본 수는 [시간별 표](results/diagnostic-summary/trace_fixed_times.csv)에 있습니다. Rank2의 큰 후기 norm·오차·projection 손실은 첫 판독 오류 뒤에도 계산을 계속한 값입니다. Scratch 0과 terminal 이후 값은 유효 norm에 포함하지 않았습니다. [진단 원자료와 정의](results/diagnostic-summary/summary.json)는 native/shadow 오차와 자기 write-back 오차를 구분합니다. Gold margin은 evaluator에서만 계산했습니다. 첫 오답·수치 실패·큰 norm의 선후 관계는 관측이며 단독 인과 증명이 아닙니다.

다른 본 작업의 평가가 끝난 뒤 CPU 2 threads로 16 × 128, 세 반복을 순차 측정했습니다. 아래는 **complete-call ms/group-token의 중앙값**입니다. BOS 비용은 call에 포함되고 분모는 16 × 128입니다. Transition·projection·packing·상태 검사·원래 readout을 포함하며 loading·I/O·gold/shadow는 제외합니다.

| Seed | NATIVE_FP32 | UNIFORM_8 | UNIFORM_5 | LOWRANK_4_8_R2 | MIXED_5_6_BUDGET |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0306 | 0.0939 | 0.1209 | 0.1438 | 0.1185 |
| 1 | 0.0288 | 0.0975 | 0.1206 | 0.1491 | 0.1160 |
| 2 | 0.0306 | 0.0946 | 0.1090 | 0.1419 | 0.1119 |

[시간 원기록](results/timing/seed0/timing.json)은 active update와 terminal no-op 횟수를 나눕니다. 이는 Python/NumPy/PyTorch CPU 경로의 비용이며 단일 요청 latency나 GPU 가속 수치가 아닙니다. OS 배경 부하를 완전히 통제한 측정도 아닙니다.

## 결과 분석

v2의 주된 소프트웨어 성과는 종료 의미가 호출 경계를 넘어 유지되는 것입니다. 이것을 모델 기억 수명의 개선으로 세지 않습니다. 같은 값의 FP32/FP64 경로가 이번 32개 입력에서 같은 label을 냈다는 관측은 native의 이른 첫 실패를 산술 precision 하나로 설명하기 어렵게 하지만, 학습 부족·표현력 한계·필수 비트 하한을 단독으로 입증하지 않습니다.

평균 연속 정답 길이, 5% 위험 길이, token accuracy, 후기 상태 norm, CPU 비용은 서로 다른 목표입니다. Native도 장기 외삽에서 실패하므로 양자화 후보의 모든 오답을 저장 오류 탓으로 돌리지 않습니다. 기존 v1의 joint 0/54는 그대로이며, 새 family와 표본수로 바뀐 신뢰하한 자체를 방법 개선으로 해석하지 않습니다.

## 결론

실패·난수·커서를 보존하는 packed state와 evaluator 재시작 경로를 구현하고, 실제 과거 실패를 재현했습니다. 새 입력에서 같은 저장 상한의 비교와 전체 곡선을 남겼습니다. Rank2의 평균 차이와 지원하한 결과를 함께 제시했으며, 유리한 한 seed나 늦은 token 정확도를 최초 실패 목표 대신 쓰지 않았습니다. 새 학습 0, GPU 실행 0, 원격 게시 0입니다.

## 레퍼런스

[NOTICE](NOTICE.md) · [unchanged v1 sources and hashes](provenance/v1_source_identity.json) · [upstream/checkpoint provenance](provenance/v1_checkpoints.json) · [protocol freeze](provenance/protocol_freeze.json)

ComplexKDA와 기존 state-quantization/error-feedback 문헌의 대응은 v1 고지를 유지합니다. 모델·학습·기존 코덱은 upstream/v1 구현이고, 이번 기여는 failure-aware adapter, restart 계약과 그 후속 검증입니다. Codex가 구현·검사·보고서 작성을 지원했습니다.
