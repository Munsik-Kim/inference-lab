# Case019 — 손실은 그대로인데 학습은 진행될까?

두 층 선형망의 방향·크기 시간 분리 재현과 방향 개입 CPU 실험

손실이 거의 변하지 않는 동안에도 모형 내부의 방향은 목표에 가까워질 수 있습니다. 두 층 선형망을 CPU로 독립 구현해 이 현상을 재현하고, 학습 중 예측 크기를 보존한 채 방향을 바꾸어 이후 손실 경로가 어떻게 달라지는지 확인했습니다.

실험은 whitened 두 층 선형망의 FP64 gradient flow를 유한 차원에서 푼 것입니다. 방향은 목표 matrix의 singular subspace와의 관계, 크기는 주로 예측 Z=UVᵀ의 singular value, 손실은 목표와의 제곱 오차입니다.

1. 작은 초기화에서는 평균 정렬과 손실 변화 시점이 갈라졌습니다. 큰 초기화에서는 손실 변화가 먼저였습니다.
2. 좁은 경계에서는 평균에 작게 드러나는 느린 방향이 오래 남았습니다. R2의 긴 지연과 재하강을 별도 계산으로 확인했습니다.
3. 같은 예측 크기에서도 방향 개입 뒤 경로가 달라졌습니다. 초기 이득이 나중에 줄거나, 교란 뒤 손실 순서가 뒤집히는 조건도 있었습니다.

처음부터 목표 방향에 놓은 aligned 대조군에서도 작은 크기가 자라는 지연이 남았습니다. 방향 정렬과 크기 성장은 따로 관측했습니다.


## 시간 분리는 어느 조건에서 나타났나?

| g | rho | T_align_half median | T_visible10 median | S median | R |
| --- | --- | --- | --- | --- | --- |
| 0.4 | 0.01 | 2.260022 | 1.642175 | -0.622727 | 8/8 |
| 0.4 | 0.0001 | 2.247329 | 4.158588 | 1.908582 | 8/8 |
| 0.4 | 1e-06 | 2.247202 | 6.622546 | 4.373821 | 8/8 |
| 0.1 | 0.01 | 2.297640 | 1.641471 | -0.661481 | 8/8 |
| 0.1 | 0.0001 | 2.284612 | 4.158919 | 1.873717 | 8/8 |
| 0.1 | 1e-06 | 2.284482 | 6.621101 | 4.338883 | 8/8 |
| 0.025 | 0.01 | 2.307623 | 1.640637 | -0.672651 | 8/8 |
| 0.025 | 0.0001 | 2.294509 | 4.158098 | 1.862787 | 8/8 |
| 0.025 | 1e-06 | 2.294378 | 6.620116 | 4.327983 | 8/8 |

시간은 gradient-flow 단위입니다. T_align_half는 오른쪽 평균 방향 오차가 처음 절반으로 줄어든 시점이고, T_visible10은 rank 제한 바닥을 뺀 학습 가능 손실이 10% 줄어든 시점입니다. half 도달은 완전 정렬이 아닙니다. rho는 U₀=0,V₀=√(2rho)R의 매개변수이며 진폭 자체가 아닙니다. 같은 R8 개를 rho/g 전체에 공유했습니다.

## 평균이 숨긴 느린 방향

g=.025,rho=1e-6,R2의 worst-half는 83.061701(저장 bracket83.0–83.2)이었습니다. 원 81 개 137,862시점의 유한-rho graph 계산과 추가 원 U/V RK4 네 번이 긴 tail과 비단조 경로를 확인했습니다. 큰 rho=.01의 mode32는 2.815288에 .1을 넘고 7.859546에 다시 내려간 뒤 39.969691부터 재진입했습니다. gradient-flow 시간 10 단위 동안 grid상 유지된 첫 구간은 39.969691–49.969691입니다. P1 최초 crossing과 P2 사후 유지 진단은 별도 열입니다.

g는 32·33번째 target singular value를 .6±g/2로 바꾸는 boundary-pair 설계입니다. 다른 62 개 값과 basis를 유지하지만 경계 amplitude·target norm·최적 손실 L_min도 변합니다. 각 target의 실제 바닥을 사용했습니다.

![R2 tail and persistence](publication/figures/p2-01_R2_tail_and_persistence.png)

## 같은 크기에서 방향을 바꾸기

기존 16 개 P1 경로에서 t*=1/5를 나눠 32 parent, 네 arm의 128 개 continuation을 만들었습니다. CONTINUE는 그대로 새 solver로 재시작합니다. GAUGE는 숨은 좌표를 함께 회전해 Z와 물리 부분공간을 그대로 두며 Gram은 직교 conjugacy를 따릅니다. ALIGN은 현재 singular value를 target 상위 pair에 배치하는 oracle입니다. ROTATE는 target은 두고 ambient 방향을 고정 회전합니다. ALIGN/ROTATE는 실제 두 Gram과 예측 singular value를 보존했습니다.

ALIGN의 즉시 손실 하한은 target SVD를 쓰는 구성 성질입니다. 이후 경로는 실제로 다시 적분했습니다. 네 arm의 시작 손실과 gradient는 같게 맞추지 않았습니다. 하나의 회전 triple과 R8·target·parent를 공유합니다.

J는 변환 직후 차이, G는 그 뒤 CONTINUE 대비 누적 손실 감소량의 차이, A=J+G는 해당 시점의 총 차이입니다. 모두 parent의 원래 D=L_orig0−L_min을 공통 분모로 사용합니다. 아래는 tau=5에서 R별 짝 차이를 먼저 구한 평균입니다.

| g | t* | arm | mean J/D | mean G/D | mean A/D |
| --- | --- | --- | --- | --- | --- |
| 0.4 | 1 | ALIGN | 7.0715178e-07 | 0.017319575 | 0.017320282 |
| 0.4 | 1 | ROTATE | -3.4774016e-06 | -0.021678511 | -0.021681988 |
| 0.4 | 5 | ALIGN | 0.00011944011 | 0.0030392404 | 0.0031586805 |
| 0.4 | 5 | ROTATE | -0.0053405449 | -0.19634004 | -0.20168059 |
| 0.025 | 1 | ALIGN | 7.1230319e-07 | 0.017424919 | 0.017425631 |
| 0.025 | 1 | ROTATE | -3.4937498e-06 | -0.021810957 | -0.02181445 |
| 0.025 | 5 | ALIGN | 0.00011943048 | 0.0064885595 | 0.00660799 |
| 0.025 | 5 | ROTATE | -0.0053611267 | -0.19549883 | -0.20085995 |

g=.025,t*=5의 ALIGN 평균 A/D는 약 0.6608%의 원래 학습 가능 손실 차이입니다. 학습속도 개선율이 아닙니다. 초기 이득에는 이후 감소 G도 포함됐으나 g=.4에서는 장기에 차이가 roundoff 수준으로 좁혀졌습니다. g=.025의 유한 horizon에는 잔여 차이가 남았습니다. 더 좋은 시작에서 줄일 손실이 적어 G가 음수가 될 수 있습니다.

ROTATE는 tau=5의 32 parent 모두에서 손실이 더 높았습니다. 반면 g=.025,t*=1,tau=20은 5 개가 더 낮고 3 개가 더 높았습니다. t*=5에서는 2 개가 더 낮고 6 개가 더 높았습니다. 경로 차이를 최종 optimum 개선이나 모든 방향 교란의 평균 효과로 바꾸어 말하지 않습니다.

![Paired immediate and subsequent effects](publication/figures/p2-04_J_G_A_paired_points.png)

## 방법·구현·해석 단위

P0는 scalar6+작은 matrix9+64 차원 2의 17 개 pilot입니다. P1은 random72+고유 aligned9의 81 개이며 aligned를 seed8 개에 복제하지 않았습니다. P2의 128 개 branch는 같은 8 개 R에서 나온 16 개 원 경로·32 parent를 공유합니다. 단계별 수치 적분 수를 독립 dataset 수로 합치지 않습니다. 모든 crossing은 개별 궤적에서 먼저 구한 저장 bracket의 선형 보간입니다. 미도달은 null/NOT_REACHED로 남깁니다. sorted prediction singular value와 signed mode coefficient lᵢᵀZrᵢ도 구분합니다.

원문 부분 재현은 gradient flow·구조화 초기화·유한 차원 방향/크기 시간 분리의 수치 대조입니다. boundary-pair, worst-angle, checkpoint 방향 개입은 자체 통제 실험입니다. 원문의 작은-rho·장시간·고차원 극한과 유한 64 차원 실험을 구분합니다. 평가한 값은 target matrix에 대한 학습 손실이며 업무 정확도나 일반화 risk가 아닙니다.

전체 원 자료와 frozen source는 exact bytes로 별도 evidence에 보존했습니다. 공개 quick 검사는 작은 slice, full recorded 검사는 소스+명시적 evidence를 사용합니다. 저장하지 않은 전체 factor 경로를 복원하지 않고, GAUGE/CONTINUE는 저장 오차 series와 남은 checkpoint 범위를 검산합니다. 원P0의 최초 실패, P1 launcher 오류, P2 사후 선택·helper 작성 시점은 이력으로 보존합니다.

[Reproduction](REPRODUCE.ko.md) · [Source map](SOURCE_MAP.md) · [Reuse](REUSE_MAP.md) · [Distribution](publication/distribution.json) · [Facts](publication/facts.json)
