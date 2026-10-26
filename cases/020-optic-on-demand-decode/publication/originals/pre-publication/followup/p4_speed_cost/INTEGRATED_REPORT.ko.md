# Case020 — 요청 복원은 메모리를 줄였고, 중복 제거의 속도 이득은 재현되지 않았다

**필요한 임베딩만 복원하면 메모리는 얼마나 줄고, 언제 느려지는가?**

LARGE/Z32에서 전체 재구성 table A는 256MiB, 요청 복원 B/C의 상주 tensor는 약0.524MiB였다. 그러나 작은 상주량이 낮은 호출 비용을 보장하지는 않았다. P4에서 같은 process·context에 A/B/C를 함께 두고 순서를 교차한 주 측정 H에서는 C가 B보다 빠른 block이 0/30이었다. 준비와 실제 N회 호출까지 합치면 짧은 N128에서는 A/B의 우열이 섞였고, N1024와4096에서는 A가 모든 block에서 빨랐다.

PLE는 각 token이 각 layer에서 사용하는 추가 임베딩이다. OPTIC는 여러 layer가 공유하는 token code와 작은 벡터 사전인 codebook을 저장하고, layer별 affine decoder로 임베딩을 복원한다. resident는 계속 남아 있는 tensor, workspace는 연산 중 라이브러리가 사용하는 작업 공간이다. 여기서 A는 원래 미압축 PLE가 아니라 **같은 압축 표현을 미리 복원한 table**이다.

## 메모리가 부족할 때의 선택

다음은 원 P2 최종 run `20261026-1235`의 LARGE/M2048/U512, 세 block에서 같은 값이 나온 method-isolated 결과다. P4의 공존 process 메모리는 넣지 않았다.

| method | same NPZ bytes | resident tensor bytes | warm allocated MiB | timed allocated peak MiB | timed reserved peak MiB |
|---|---|---|---|---|---|
| A | 550998 | 268435456 | 288.016 | 304.016 | 338.000 |
| B | 550998 | 548928 | 32.540 | 80.790 | 82.000 |
| C | 550998 | 548928 | 32.540 | 52.571 | 102.000 |


B/C의 상주 tensor는 A보다 약489배 작지만, 전체 VRAM 절감률은 아니다. ID16KiB, 반환 출력16MiB, FP32 중간 복원, allocator와 라이브러리 비용을 따로 고려해야 한다. 이 조건에서 C는 B보다 timed allocated peak를 약34.9% 줄였지만 reserved peak는 더 컸다. 따라서 B는 전체 table을 유지하기 어려울 때 단순한 공식 요청 복원 경로로 의미가 있고, C는 더 낮은 allocated peak가 필요한지와 호출 비용을 함께 판단해야 한다. 원 측정의 이 메모리 장점은 보존되지만 P4에서 속도 이득이 함께 성립한 조건은 없었다.

## 같은 process에서 반복된 호출 비용

H는 호출 전 synchronize를 마친 뒤 fn→완료 synchronize까지 잰 host wall이다. 아래 값은 block별 `C median / B median`의 중앙값이며, 내부120회를 pooled 표본으로 합친 값이 아니다. 양의 C−B는 C가 더 느리다는 뜻이다.

| Z | M / U | blocks | C/B H median ratio* | block ratio range | C−B median difference* (us) | C faster |
|---|---|---|---|---|---|---|
| 32 | 1 / 1 | 6 | 2.333 | 1.742–2.595 | +138.17 | 0/6 |
| 32 | 2048 / 2048 | 6 | 1.936 | 1.864–2.155 | +140.92 | 0/6 |
| 32 | 2048 / 512 | 6 | 1.777 | 1.511–1.920 | +114.27 | 0/6 |
| 128 | 2048 / 2048 | 3 | 1.942 | 1.799–1.943 | +265.92 | 0/3 |
| 128 | 2048 / 512 | 3 | 1.800 | 1.745–1.817 | +119.73 | 0/3 |
| 644 | 2048 / 2048 | 3 | 1.409 | 1.350–1.470 | +130.53 | 0/3 |
| 644 | 2048 / 512 | 3 | 1.142 | 1.115–1.306 | +56.13 | 0/3 |


*별표 열은 block별 비율 또는 차이의 중앙값이다. 모든 원점은 [paired_block_comparison.csv](summary/paired_block_comparison.csv), 절대시간의 mean/median/p95/min/IQR은 [controlled_latency.csv](summary/controlled_latency.csv)에 있다.*

core의 Z32 세 입력은 각각6개 fresh process에서 비교했다. C가 B보다 빠른 block은 세 조건 모두0/6이었다. M2048/U2048의 S/B 중앙 비율도1.763으로, 정렬 후 M행을 그대로 복원한 대조가 B보다 느렸다. C/S는1.133이었다. S와 C는 정렬·할당·operator 구성이 달라 이 차이를 unique의 순수 비용으로 분리할 수 없다. U=M에서는 복원 행 수 감소 자체가 없다는 점만 분명하다.

폭 확장에서는 Z128/G32와 Z644/G161을 추가했다. Z644/U512는 Z32/U512보다 C/B 차이가 작았지만 여전히1.142배, 세 block 모두 C가 느렸다. Z32→128→644의 변화는 단조롭지 않았고, 폭과 함께 groups·codebook·상주량·연산량도 바뀌었다. 폭 하나의 순수 인과 효과나 학습 artifact의 특성으로 해석하지 않는다. 고정한 10%·방향 일치 문턱을 충족한 조건은 없으며, 이것은 구현이나 실험 완료 여부와 별개다.

![같은 context 호출 비용](figures/01_shared_context_latency.png)

같은 context를 쓰더라도 절대시간의 흔들림은 남았다. 예를 들어 core M1에서 B의 block median은82.27–480.79us, C는187.00–837.36us였다. 느린 원점도 제외하지 않았다. 그림의 각 marker는 한 process block이며 짧은 막대는 block 중앙값이다. 공유 상태·교차 순서가 과거 P2와 다르므로 두 결과를 한 표본으로 합치거나 변화 원인을 process 격리 하나로 단정하지 않는다.

## 두 타이머와 과거의 혼재된 승리

원 최종90 raw를 다시 읽으면 C/B의 host median 승리5/30, event mean 승리4/30, 공통0/30, 방향 불일치9/30으로 전달값과 일치했다. host 승리 중3개는 U=M이고 세 block 모두 host 승리한 조건은0/10이었다. median끼리 맞추면 host5/event4/공통1/불일치7, mean끼리 맞추면 host8/event4/공통1/불일치10이다. 통계량 혼합만으로 설명되는 문제는 아니며, 원 event는 K32/8/1의 별도 pass다.

P4 D는 **같은 호출**의 host/event를 저장하지만 host에 event 기록2회와 wait가 들어간다. E는 공통K16의 stream elapsed/16이며 launch gap이나 C 내부 wait의 영향을 포함할 수 있다. H의 지표를 D로 대체하거나 event를 순수 kernel 합으로 읽지 않는다. 평균·중앙값·p95는 각 pass와 clock을 구분한 CSV에 모두 남겼다. event bracket 평균의 p95는 단일 호출 tail p95가 아니다.

![폭 변화](figures/02_width_comparison.png)

## table을 얼마나 오래 재사용할 것인가

동일 LARGE/Z32/M2048/U512를 반복하는 새 수명주기다. 각 N·method·block마다 별도 process를 실행했고 warm-up 없이 첫 호출도 N에 포함했다. session은 artifact 읽기 전부터 ID H2D, A의 전체 복원·compact 정리, 실제 N회 동기 호출, 최종 소유 tensor 정리까지다. parent worker는 import/context/종료도 포함한다. 아래27개 session 원점과 parent wall을 모두 표시한다.

| N | method | session s: b0 / b1 / b2 | median session s | parent worker s: b0 / b1 / b2 |
|---|---|---|---|---|
| 128 | A | 0.196851 / 0.169624 / 0.196438 | 0.196438 | 2.072030 / 1.898246 / 1.904284 |
| 128 | B | 0.168025 / 0.190640 / 0.185433 | 0.185433 | 2.009308 / 1.854379 / 1.887632 |
| 128 | C | 0.224380 / 0.229143 / 0.245484 | 0.229143 | 2.086732 / 2.058524 / 2.039747 |
| 1024 | A | 0.252050 / 0.223652 / 0.195350 | 0.223652 | 2.060568 / 2.041846 / 1.878002 |
| 1024 | B | 0.410242 / 0.410329 / 0.318430 | 0.410242 | 2.180281 / 2.144023 / 2.216770 |
| 1024 | C | 0.507840 / 0.655918 / 0.632942 | 0.632942 | 2.318672 / 2.321995 / 2.431254 |
| 4096 | A | 0.442087 / 0.530962 / 0.342683 | 0.442087 | 2.117734 / 2.402362 / 2.118920 |
| 4096 | B | 0.941224 / 1.514844 / 0.832234 | 0.941224 | 2.781958 / 3.278795 / 2.605198 |
| 4096 | C | 1.234773 / 1.125446 / 1.362115 | 1.234773 | 3.017688 / 2.867949 / 3.179739 |


N128에서 B가 A보다 빠른 block은2/3이고 중앙값도 B가 낮았다. N1024·4096에서는 A가3/3 빠르다. 관측한 중앙값 순서는128과1024 사이에서 바뀌지만 정확한 운영 교차 호출 수는 만들지 않는다. 과거 setup+N×warm host-median 시나리오의 약920–2319회는 당시 계산으로 유지한다. 새 세션에서는 cold B 첫 호출 약92–119ms, C 약122–164ms가 관측됐고 A 첫 gather는 약1.2–2.1ms였다. A는 setup에서 이미 decoder 전체 복원을 수행한다. warm 선형 시나리오와 실제 cold 세션의 차이를 보고하되, 첫 비용의 세부 원인을 profiling 없이 특정하지 않는다.

C의 session 중앙값은 세 N 모두 B보다 높았다. 개별 N4096 block 하나에서는 C가 B보다 낮았으므로 모든 실행에서 C가 불리했다고 쓰지 않는다. 각 method/N의3점을 유지해 변동과 선택 조건을 함께 보여준다. session/N은 setup을 나눈 평균 상각 비용이며 per-call p95가 아니다. parent wall의 대부분을 차지하는 공통 초기화도 service의 GPU busy time으로 해석하지 않는다.

![실측 호출 누적 비용](figures/03_actual_lifecycle.png)

## 32MiB의 발생 단계

원90 raw 모두에서 library 초기화 후 baseline allocated가 이미33,554,432 bytes였다. 원 warm endpoint에서 baseline·module·ID를 빼면 잔여는0–952 bytes다. 원 raw에는 별도 after-load와 after-output-release endpoint가 없어 과거 단계 전체를 복원할 수는 없다.

새 method별 진단3개에서는 context 직후0→작은 FP16 GEMM 후32MiB→module/ID 제거 후32MiB가 관측됐다. 제거 뒤 남은 active block의 C++ stack에 `at::cuda::setWorkspaceForHandle`, `getCurrentCUDABlasHandle`, FP16 cuBLAS GEMM helper가 함께 있었다. 따라서 새 replay의32MiB는 **Torch cuBLAS handle workspace**로 소유자를 확인했다. 과거 raw의 stack은 없으므로 새 stack을 과거 allocation의 직접 증거로 소급하지 않는다. 원 REPORT의 기록을 덮어쓰지 않고 이번 통합 설명에서 단계·근거를 보완한다.

![메모리 단계](figures/04_memory_attribution.png)

## 완료 범위와 재현

P4B30/30 timing worker·96/96 method-block, P4C27/27 lifecycle worker, preflight3·smoke1·memory3을 완료했다. 실패·재시도·누락은0이며 선택 profile은0이다. 모든 layer의 원 허용 오차를 통과했고 추가 allocated peak는2GiB gate 아래였다. 전용 preflight와 source/config/input freeze를 통과한 뒤 주 측정을 실행했다. 후속 설계임을15:03:17 KST freeze에 기록했고, preflight/smoke에서 이미 본 범위를15:04:16 KST main freeze에 남겼다.

GPU 관련 parent wall은 원196 worker 457.424303870s + 신규64 worker 265.775915436s = 723.200219306s이며, 전체1800초 중 1076.799780694s가 남았다. wall만 합산했고 원 diagnostic/final/P4 성능 표본은 합치지 않았다. 공통 작은 GEMM probe64회는 이64 worker 안에 포함되므로 별도 worker로 중복 차감하지 않았다.

원 Case747파일과 외부18파일의 bytes/hash를 보존했다. 원 manifest의 coverage 검사는 원 파일만 복사한 별도 임시 범위에서 수정 없는 verifier로 통과했다. CPU 독립 감사, 외부 출력 재생성, 단위·manifest 변조 거절과 그림4개 육안 점검 결과는 [재현 안내](REPRODUCE.ko.md)와 provenance에 있다. 패키지 설치·환경 업그레이드·원격 CI·게시를 실행하지 않았다.

## 해석 범위

측정 범위는 RTX 5080 한 대의 eager operator다. 같은 synthetic artifact와 같은 GPU ID를 반복하는 warm-cache 호출이며 raw unscaled BF16의 모든 16 layer를 반환한다. 실제 LLM, 학습된 OPTIC artifact, 언어 품질, TTFT/TPOT, KV 재배분, 서버 동시성은 측정하지 않았다. Z644는 E4B의 latent 폭에 대응할 뿐 V/L/H와 품질을 재현하지 않는다. 파일 cache는 warm일 수 있고 GPU/CPU clock을 고정하지 않았다. WSL의 device 전체 표본은 연속 peak 또는 process별 완전한 attribution이 아니다. P2 isolated 결과와 P4 shared-context 결과는 서로 다른 계약으로 유지한다. block은 process 반복 단위이며 내부 호출을 독립 모델·사용자 과제로 세지 않는다. 전력 표본으로 에너지나 비용을 추정하지 않는다.

공식 codec와 Torch GEMM/allocator는 재사용했다. 직접 만든 부분은 fixture·A/C/S wrapper·교차 순서·시간/수명주기 계측·독립 FP64 참조와 CPU 감사·그림·문서다. [SOURCE_MAP](SOURCE_MAP.md), [REUSE_MAP](REUSE_MAP.md), [상세 보고서](REPORT.ko.md)에 구현과 근거가 연결돼 있다.
