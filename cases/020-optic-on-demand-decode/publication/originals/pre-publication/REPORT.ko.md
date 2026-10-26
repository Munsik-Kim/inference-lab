# Case020 — OPTIC 압축 PLE 실행 비용

상주 module payload는 줄었고 동기화를 포함한 operator latency는 process block 사이에서 크게 변했다.90개 본 cell-block 모두 수치·자원 gate를 통과했다.30개 짝 중5개에서 C가 B보다 빨랐고 세 block 모두 C가 빨랐던 조건은0개다. 판정: `OPERATOR_VALIDATED, MEMORY_TRADEOFF_OBSERVED, DEDUP_UNRESOLVED_BLOCK_VARIATION`.

| Shape | A resident bytes | B/C resident bytes | A / B payload ratio | NPZ bytes / arrays bytes |
|---|---:|---:|---:|---:|
| SMALL | 16,777,216 | 149,568 | 112.17× | 151634 / 149504 |
| LARGE | 268,435,456 | 548,928 | 489.02× | 550998 / 548864 |

| LARGE M2048/U512 | A | B | C |
|---|---:|---:|---:|
| Timed allocated peak MiB | 304.02–304.02 | 80.79–80.79 | 52.57–52.57 |
| Timed reserved peak MiB | 338.00–338.00 | 82.00–82.00 | 102.00–102.00 |

Payload ratio and allocated/reserved/device memory are separate measurements.


| Block | A host median µs | B host median µs | C host median µs | C/B |
|---:|---:|---:|---:|---:|
| 0 | 29.55 | 123.14 | 869.78 | 7.064 |
| 1 | 60.16 | 126.41 | 247.07 | 1.955 |
| 2 | 25.35 | 178.65 | 860.85 | 4.819 |


## 메모리 회계와 환경

parameter/buffer 전부를 순회하고 non-persistent group_offsets도 합산했다. alias/view는 underlying storage 기준으로 중복을 제거했다. upstream compact_resident_bytes는 두 shape에서 실제 B/C module storage보다64bytes 작다. NPZ bytes·배열 payload·CPU metadata·공통 int64 ID·BF16 출력·C unique/inverse/중간/최종 tensor를 구분한다. 출력만 M*L*H*2bytes다. A table은 같은 압축 artifact의 재구성이며 원래 미압축 PLE가 아니다.

warm-up 출력 참조를 없애고 synchronize 후 peak를 reset했다. timed loop에는 gc/empty_cache가 없다. reserved-start에는 allocator의 warm-up 이력이 남는다. setup/cold/timed allocated peak를 raw와 memory.csv에 별도 저장한다. module storage, allocated/reserved, 전체 VRAM은 같은 수치가 아니다. warm-up 뒤 module+ID inventory 외에 약32MiB의 allocated 잔류량이 있었지만 정확한 소유자를 추적하지 않았으므로 특정 kernel 비용으로 단정하지 않는다. context/library와 장치 전체 표본도 별도다. NVML 표본은 연속 최고치가 아니며 WSL의 process attribution에는 제한이 있다.

관측 장치: RTX5080(sm_120),16303MiB, driver621.01; Ryzen7 9800X3D/16 logical CPUs; Ubuntu24.04/WSL2. operator는 기존 Python 3.12.14·NumPy 2.3.5·Torch 2.13.0+cu130 환경을 읽기 전용으로 재사용했다. 독립 clean environment 검증은 아니다. 그림은 기존 Python 3.10.19·NumPy 2.2.6·Matplotlib 3.10.8에서 만들었다. import 경로·CPU/RAM/disk·BLAS/thread 설정은 provenance에 있다. OMP/MKL/OpenBLAS thread1과 cache/bytecode 설정은 자식 process에만 적용했다. 설치·업그레이드는 없었다.

GPU FP16 GEMM의 실제 FP32 out_dtype·finite·전체 layer/shape/dtype를 확인했고 별도 trace에서 lookup/GEMM/cast를 확인했다. TF32=false, FP16 accumulation=false다. fp16/bf16 reduced-precision reduction은 runtime 기본 true에서 false로 바꿔 FP32 누적 정책을 유지했다. 이 기본값과의 차이를 공개한다. 다른 dtype/CPU fallback·새 kernel·compile·CUDA graph는 없다.2GiB gate는 초기 context baseline 대비 allocated peak이며 setup/cold/main을 포함한다. reserved/context/장치 전체 사용량과 섞지 않는다.

GPU 전 compute-app 목록은 비었고 utilization0%, 장치 메모리1844MiB였다. 첫 진단 run 종료 후에도 목록은 비었고 utilization0%,1854MiB였다. 최종 본 run 종료 상태는 provenance/gpu_state_after.json에 별도로 있다. 다른 process 종료·reset·설정 변경은 하지 않았다. 이는 해당 시점의 장치 관측이며 모든 서비스의 무변경 감사는 아니다.


## 방법과 계측 계약

A는 공식 decoder로 전체 token을256개 고정 chunk로 복원해 GPU BF16[V,L*H] table을 만든다. setup에서 decoder와 table이 공존한 peak를 저장하고, compact 객체·임시 tensor를 제거한 뒤 측정한다. gc/empty_cache는 setup에만 있다. B는 고정 upstream forward를 그대로 호출한다. C는 매번 unique(sorted=True,return_inverse=True)→공식 decoder(U개)→inverse gather(M개)를 수행한다. install_compact_ple의 sqrt(H) 배율은 사용하지 않는다.

D는 저장된 FP16 배열을 FP64로 올려 코드 lookup과 affine 식을 계산하는 NumPy 참조다. P1은 작은 scalar loop와 einsum을 비교하고, 독립 감사는 layer별 행렬곱으로 다시 계산한다. 원래 미압축 모델의 정답이나 latency 경쟁 대상은 아니다.

P1 fixture는 V257/L3/H16/Z8/G2/D4/K256/8bit, seed0/1/2이다. 모든257개 token·전체3 layer, token0/256, 뒤섞임·반복·단일·다차원·빈 ID, weight0+bias 대조, 서로 다른 layer bias와 정확히0 출력을 검사했다. NPZ는 allow_pickle=False로 읽고 정수 코드·배열 bytes/hash를 확인했다.387개 산술/대응 기록은 독립 실험387개를 뜻하지 않는다. 먼저 공식 V32/L3/H16 예제를 실행했고 stdout/stderr/종료코드/wall을 남겼다.

공통 입력 경계는 CPU에서 만든 int64 IDs의[0,V) 범위를 확인한 뒤 측정 밖에서 GPU로 옮긴다. 음수/범위 밖은 CPU에서 공식 경로와 공통 경계가 모두 거절했다. 공식 API는 int32/float를 long으로 묵시 변환하지만 공통 경계는 거절한다. 빈 출력은 공식·adapter 모두 BF16[0,L*H]이다. meta device 불일치 probe의 관측 행동은 P1 contracts에 별도 남겼으며 지원하는 공통 입력 계약에 포함하지 않는다. GPU illegal-index assert는 실행하지 않았다.

동결 gate: FP64 loop/vector rtol1e-10·atol1e-12, BF16 overall/layer별 relativeL2≤1e-2, finite 필수,0-reference maxabs≤1e-6. BF16 반올림 값에 FP64 gate를 적용하지 않는다. 수정 본 P2는 모든 입력의 직접 A/B/C 대조를 별도 process에서 먼저 수행하고 각 cell도 같은 gate를 통과했다. A/B/C는 실제90개 method-pair 기록에서 bit-identical했으나 이를 필수 조건으로 정하지 않았다.

SMALL은 V8192/L8/H128/Z32/G8/D4, LARGE는 V32768/L16/H256/Z32/G8/D4이다. dense table은16/256MiB다. 실모델 shape가 아니며 Z32는 공식 E4B의 Z644와 다르다. payload 배율은 학습된 압축 품질을 뜻하지 않는다.

artifact/input/order seed61026020/61026021/61026022를 분리하고 protocol의 SeedSequence mapping으로 입력을 생성했다. M1은 U1 한 조건, M128/2048은 U=M 또는 U=M/4이다. repeat75는 unique ID를 정확히4회씩 넣어 섞었고 실제 U와 bytes/hash를 저장했다. shape마다 artifact는 하나다.3 block은 같은 artifact의 process 반복이며 실모델3개나 사용자 과제3개가 아니다.

block마다30 cell 순서를 고정 shuffle하고, cell마다 새 process를 순차 실행했다. autograd OFF, eager, GPU ID/출력 계약이다. 검산·I/O·H2D/D2H·profiler·로그·gc/empty_cache는 timed function 밖이다. 출력 할당은 포함한다. warm-up20회, host100회, event100 bracket이며 M1/128/2048의 K는 모든 방법에 공통32/8/1이다. p95는 NumPy linear다. host는 완료 synchronize 비용을 포함하고 event/K는 stream 경과시간이다. event에는 launch gap/unique host wait로 인한 idle이 들어갈 수 있다. bracket p95/K는 단일 요청 tail이 아니다. 계측 floor를 저장하고 빼지 않았다.

C 내부 비용은 본 계측 뒤10회 동기화 진단 pass로 측정했다. 추가 동기화가 있어 성분 합계가 원래 C 시간과 같지 않아도 된다. B의 lookup/GEMM/bias/cast trace는 별도 profiler 실행이다.첫 run의6 smoke에서도 full timing 진단 루프를 보았음을 수정 freeze에 공개했다. 수정 run도 본 측정 전6 smoke를 실행했고 이후 protocol을 바꾸지 않았다. 모든 smoke는 본 표에서 제외했다. 본 측정에 맞춰 shape·seed·허용 오차·loop를 바꾸지 않았다.


## 전체 핵심 표

중앙 latency는3개 block 통계의 중앙값, 대괄호는 block 최솟값–최댓값이다. memory peak는 block 범위다. host p95는 각 block의100개 호출에서 계산했다. event 열은 block 안100개 bracket/K의 평균을 구한 뒤 block 중앙값으로 표시한다. host tail이나 순수 kernel 합계가 아니다. shape별 NPZ/배열 bytes는 상주량 표에 있고 모든 block 값은 CSV/raw에 남아 있다.

| Shape | M/U | Method | GPU module resident MiB | timed allocated peak MiB | timed reserved peak MiB | host median µs [block range] | host p95 µs [range] | event bracket mean/K µs [range] | setup ms [range] | gate |
|---|---:|---|---:|---:|---:|---|---|---|---|---|
| SMALL | 1/1 | A | 16.0000 | 48.00–48.00 | 50.00–50.00 | 33.93 [29.10–53.60] | 46.14 [37.48–68.32] | 31.98 [13.22–40.25] | 144.65 [127.95–153.44] | PASS |
| SMALL | 1/1 | B | 0.1426 | 32.15–32.15 | 34.00–34.00 | 98.58 [97.50–99.45] | 183.90 [155.33–591.41] | 161.63 [144.88–188.63] | 5.87 [5.34–5.91] | PASS |
| SMALL | 1/1 | C | 0.1426 | 32.15–32.15 | 34.00–34.00 | 174.59 [161.65–279.13] | 369.61 [289.94–927.43] | 402.46 [266.30–493.31] | 5.99 [5.91–6.04] | PASS |
| SMALL | 128/128 | A | 16.0000 | 48.25–48.25 | 50.00–50.00 | 101.01 [39.92–105.17] | 123.55 [48.27–131.53] | 9.93 [9.79–50.68] | 135.07 [129.29–163.54] | PASS |
| SMALL | 128/128 | B | 0.1426 | 32.91–32.91 | 34.00–34.00 | 85.52 [75.97–92.91] | 171.38 [161.53–588.55] | 189.95 [93.84–206.42] | 5.95 [5.85–5.98] | PASS |
| SMALL | 128/128 | C | 0.1426 | 32.91–32.91 | 34.00–34.00 | 714.93 [160.49–756.62] | 846.32 [295.11–1028.36] | 372.82 [314.67–508.17] | 5.70 [5.38–6.26] | PASS |
| SMALL | 128/32 | A | 16.0000 | 48.25–48.25 | 50.00–50.00 | 31.23 [24.40–102.62] | 46.00 [45.29–127.29] | 10.46 [9.03–13.66] | 128.27 [126.66–159.95] | PASS |
| SMALL | 128/32 | B | 0.1426 | 32.91–32.91 | 34.00–34.00 | 94.71 [85.48–100.17] | 159.71 [144.34–377.01] | 156.89 [101.05–248.16] | 5.82 [5.24–6.15] | PASS |
| SMALL | 128/32 | C | 0.1426 | 32.46–32.46 | 34.00–34.00 | 318.42 [168.05–926.39] | 900.84 [298.74–1033.89] | 219.79 [203.67–491.71] | 5.52 [5.16–5.83] | PASS |
| SMALL | 2048/2048 | A | 16.0000 | 52.02–52.02 | 70.00–70.00 | 19.44 [19.24–32.75] | 45.39 [34.70–45.51] | 122.51 [15.73–179.52] | 150.47 [132.84–161.94] | PASS |
| SMALL | 2048/2048 | B | 0.1426 | 44.41–44.41 | 54.00–54.00 | 90.94 [85.96–367.97] | 157.95 [150.12–416.94] | 96.40 [88.61–528.30] | 5.80 [5.77–5.81] | PASS |
| SMALL | 2048/2048 | C | 0.1426 | 44.44–44.44 | 54.00–54.00 | 176.95 [168.66–859.39] | 320.25 [297.53–1107.72] | 460.73 [203.69–805.47] | 5.36 [5.36–5.58] | PASS |
| SMALL | 2048/512 | A | 16.0000 | 52.02–52.02 | 70.00–70.00 | 29.45 [28.73–30.26] | 57.79 [52.86–58.75] | 117.86 [13.18–170.29] | 149.14 [145.70–161.82] | PASS |
| SMALL | 2048/512 | B | 0.1426 | 44.41–44.41 | 54.00–54.00 | 89.12 [85.24–530.82] | 474.41 [154.69–635.57] | 88.15 [80.88–110.47] | 5.94 [5.17–6.03] | PASS |
| SMALL | 2048/512 | C | 0.1426 | 37.19–37.19 | 54.00–54.00 | 221.57 [166.47–236.37] | 967.19 [267.19–1023.57] | 803.29 [608.06–1019.15] | 5.67 [5.36–5.81] | PASS |
| LARGE | 1/1 | A | 256.0000 | 288.01–288.01 | 290.00–290.00 | 32.18 [31.64–99.49] | 42.03 [39.25–124.10] | 37.65 [26.92–38.64] | 159.55 [151.47–232.09] | PASS |
| LARGE | 1/1 | B | 0.5235 | 32.56–32.56 | 34.00–34.00 | 90.05 [74.74–486.26] | 153.04 [151.32–635.16] | 140.80 [140.22–174.60] | 6.35 [6.09–6.43] | PASS |
| LARGE | 1/1 | C | 0.5235 | 32.56–32.56 | 34.00–34.00 | 156.50 [155.42–869.30] | 286.09 [259.93–975.83] | 397.51 [298.15–488.06] | 6.69 [6.08–7.29] | PASS |
| LARGE | 128/128 | A | 256.0000 | 289.00–289.00 | 310.00–310.00 | 103.99 [33.91–104.70] | 125.65 [51.26–131.00] | 12.92 [9.77–27.66] | 145.29 [134.53–156.06] | PASS |
| LARGE | 128/128 | B | 0.5235 | 35.54–35.54 | 54.00–54.00 | 94.02 [80.54–362.85] | 185.65 [160.83–494.17] | 114.29 [84.97–161.88] | 5.83 [5.53–6.17] | PASS |
| LARGE | 128/128 | C | 0.5235 | 35.54–35.54 | 56.00–56.00 | 172.06 [166.99–922.78] | 1012.69 [304.75–1080.81] | 339.44 [322.65–435.78] | 6.44 [5.78–6.63] | PASS |
| LARGE | 128/32 | A | 256.0000 | 289.00–289.00 | 310.00–310.00 | 21.27 [18.34–33.29] | 42.55 [34.39–62.11] | 9.89 [8.75–30.94] | 154.92 [152.32–160.69] | PASS |
| LARGE | 128/32 | B | 0.5235 | 35.54–35.54 | 54.00–54.00 | 82.84 [77.30–592.29] | 177.04 [166.85–658.62] | 162.67 [114.34–235.35] | 6.14 [5.98–6.26] | PASS |
| LARGE | 128/32 | C | 0.5235 | 33.78–33.78 | 56.00–56.00 | 168.66 [163.91–945.74] | 294.45 [286.61–1062.38] | 407.17 [384.72–492.28] | 6.28 [5.66–6.51] | PASS |
| LARGE | 2048/2048 | A | 256.0000 | 304.02–304.02 | 338.00–338.00 | 25.69 [25.58–26.51] | 52.14 [42.36–119.33] | 18.21 [16.50–21.02] | 146.28 [138.72–158.13] | PASS |
| LARGE | 2048/2048 | B | 0.5235 | 80.79–80.79 | 82.00–82.00 | 487.59 [125.62–538.09] | 580.99 [423.40–641.36] | 403.84 [139.31–413.80] | 6.25 [5.70–6.79] | PASS |
| LARGE | 2048/2048 | C | 0.5235 | 80.82–80.82 | 114.00–114.00 | 267.75 [266.74–383.98] | 889.25 [393.55–910.40] | 249.58 [222.77–428.29] | 6.75 [5.77–6.95] | PASS |
| LARGE | 2048/512 | A | 256.0000 | 304.02–304.02 | 338.00–338.00 | 29.55 [25.35–60.16] | 58.45 [39.85–84.50] | 25.35 [19.54–94.81] | 147.49 [137.28–159.88] | PASS |
| LARGE | 2048/512 | B | 0.5235 | 80.79–80.79 | 82.00–82.00 | 126.41 [123.14–178.65] | 394.14 [176.42–558.16] | 274.93 [135.26–355.89] | 6.48 [6.30–6.53] | PASS |
| LARGE | 2048/512 | C | 0.5235 | 52.57–52.57 | 102.00–102.00 | 860.85 [247.07–869.78] | 1008.38 [346.12–1054.56] | 229.83 [186.01–986.72] | 5.85 [5.45–6.02] | PASS |


## 짝 비율과 절대차

같은 shape/ID/block끼리 짝지었다. b0,b1,b2 순서다. 다른 block을 합쳐 하나의 표본으로 만들거나 setup을 요청 latency에 섞지 않았다.

| Shape | M/U | B/A host b0,b1,b2 | C/B host b0,b1,b2 | C−B µs b0,b1,b2 |
|---|---:|---|---|---|
| SMALL | 1/1 | 3.351, 1.839, 2.931 | 1.791, 2.832, 1.625 | 77.090, 180.553, 62.194 |
| SMALL | 128/128 | 0.722, 2.327, 0.847 | 9.411, 1.727, 8.847 | 638.968, 67.580, 671.091 |
| SMALL | 128/32 | 4.104, 2.737, 0.923 | 3.179, 1.966, 9.781 | 218.256, 82.567, 831.683 |
| SMALL | 2048/2048 | 11.234, 4.468, 4.677 | 2.335, 1.962, 1.946 | 491.419, 82.701, 86.012 |
| SMALL | 2048/512 | 2.894, 3.102, 17.544 | 2.599, 1.868, 0.445 | 136.330, 77.353, -294.457 |
| LARGE | 1/1 | 2.846, 4.887, 2.322 | 9.654, 0.320, 2.094 | 779.248, -330.837, 81.755 |
| LARGE | 128/128 | 2.773, 0.774, 3.466 | 1.830, 2.073, 2.543 | 78.037, 86.454, 559.933 |
| LARGE | 128/32 | 32.290, 2.322, 3.896 | 0.277, 2.182, 11.416 | -428.374, 91.356, 862.898 |
| LARGE | 2048/2048 | 20.298, 19.063, 4.891 | 0.498, 0.547, 3.057 | -270.339, -220.851, 258.352 |
| LARGE | 2048/512 | 4.166, 2.101, 7.048 | 7.064, 1.955, 4.819 | 746.644, 120.660, 682.203 |


![C−B 총차이와 별도 성분 진단](figures/03_unique_cost.png)


## 이력·보존·실행 범위

**Protocol 정정:** 첫 run20261026-1215는 worker별 FP64 대조는 timing 전에 했으나 직접 P2 A/B/C 대조는 뒤에 했다.90 cell-block을 DIAGNOSTIC_ONLY_PROTOCOL_ORDER로 보존하고, 본 run20261026-1235에서 shape별 정확성 전용 process2개로10조건의 A/B/C/FP64를 먼저 확인했다. 각 timed worker는 입력/source가 맞는 preflight hash를 확인한다. shape·seed·ID·산술·허용 오차·timing loop·실행 순서는 그대로다. 이미 본 첫 run 값과 변경 이유를 새 freeze에 공개했고 두 run을 같은1800초 장부에 더했다.

P0 PASS, P1 PASS, P2 COMPLETE, P3 로컬 공개 후보 완료. 계획30 cell/90 cell-block 대비 실제30/90, 실패0·누락0이다. 두 run의 GPU worker는 총 **196개**(수정 본 run 99개, 이전 진단97개), 누적 **457.424초**,1800초 중 **1342.576초**가 남았다. import/context/setup/종료가 포함되며 검산 순서 오류 정정으로 전체 grid를 한 번 재실행했으며 장부를 초기화하지 않았다. source는 고정된 실행/doc/license7파일과 최종 고지1파일만 받았고 모델·dataset·wheel·의존성 다운로드는 없었다. 사용자 Windows→WSL 설정 변경으로 중단된 구간을 따로 기록하며 calendar wall을 GPU 시간으로 바꾸지 않는다.

처음 Windows native의 C:\home\mskim\Codex_main은 없었다. WSL에서 실제 부모와 /home/mskim/inference-lab-public 공통 Git 저장소를 찾았다. 원 checkout의 보관 정리로 생긴 삭제 상태를 복구/reset하지 않았다. main3e6f2c6…을 fetch 확인했고 공개001–019, 원격40 branch·열린 PR0·로컬 미공개 번호를 확인한 뒤 Case020 worktree/branch를 새로 만들었다. stage/commit/push는 없다. 보존 snapshot은 기존 tracked Case tree/Git 상태와 실제 참고한4파일 bytes/hash 범위이며 전 디스크 감사가 아니다.

P1 관측 뒤 source/config/input을 freeze했다. probe/smoke 원 source와 freeze는 별도로 보존했다. 본 실행 전 요청 BF16 raw 보존과 source identity receipt만 추가했고 산술·loop·허용 오차·비교 경로는 바꾸지 않았다. 이미 본6 smoke의 전체 timing 진단값은 freeze에 밝히고 본 표에서는 제외했다. 후처리·보고서 code는 측정 후 작성했으며 최종 PUBLIC_MANIFEST로 식별한다.

초기 환경 조회 실패는 Windows cwd 부재, sandbox WSL/NVML 거절, operator runtime의 Matplotlib 부재였다. 기존 환경에 설치하지 않고 해결했으며 GPU worker/수치 실패와 구분한다. 공식 CPU 예제의 원 stdout/receipt와 첫 P1은 보존했고 두 번째 P1에서 BF16 raw를 추가했다. 그림 QA에서 처음 C−B 음수 점이 잘리는 것을 발견해 축 범위를 고쳤고 초기 그림은 소유 private 경로에 보존했다. raw 표본은 바꾸지 않았다. 초기 감사 코드가 pair 수를45로 잘못 예상했으나 설계상3 block×10입력×3비교=90임을 바로잡았다.10개 짝에서 C가 빨랐으므로 초기 no-benefit 표지도 block 변동 미확정으로 고쳤다.

독립 감사는 frozen source/input·계획/실제 cell·raw hash·M/U·timing quantile/짝 비율·dtype별 payload/고유 storage·수치 gate를 재검산하고 보존 BF16에서 별도 FP64 식을 계산한다. 변조/정상 fixture4개는 timing bytes 변경·U 변경·payload 단위 변경의 거절과 Torch 없는 정상 경로를 확인한다. 새 외부 출력에서 summary와 그림6파일을 재생성해 같은 환경에서 byte 동등성을 확인했다. 이는 과학 gate나 원격 CI PASS를 대신하지 않는다. 다른 머신/원격 CI는 실행하지 않았다.

같은 ID를 반복하는 warm-cache operator 측정이다. Zipf 요청·장문 prefill·실모델 decode·언어 능력·학습된 OPTIC 품질·KV 재배분·보편적 속도/VRAM 이득은 뜻하지 않는다. host/event 변동의 원인을 clock/scheduler 하나로 단정하지 않았다.3 process block과100회 내부 반복으로 작은 신뢰구간을 만들지 않고 원점·범위를 제공한다. 모델/학습 데이터 다운로드, generation, 학습, serving overlay/vLLM 변경, 유료 API, Drive/Pages, 게시를 실행하지 않았다.


[Protocol](configs/protocol.json) • [Raw](results/20261026-1235) • [판정](results/summary/decision.json) • [Source map](SOURCE_MAP.md) • [재현](REPRODUCE.ko.md)

Case020 • run `20261026-1235` • local candidate, uncommitted.

Base: `3e6f2c6b36fb2a3853cc3677bee84904d28567cf`  
Branch: `research/case020-optic-on-demand-decode-20261026-1215`  
Upstream: `344c712aeb6620e39eb78930fb922447f0e123f3`  
Main source/input/config bundle: `c43ea3a0a27cd28be7b9e8e26ce5cd65a42af28c070a49cd1f1b06cd77bc489b`  
Protocol SHA256: `c930534682204b99990bc58540c170925bd32ada3045a0027367e50130f34987`  
GPU worker SHA256: `fd0716a262f3444b4bbec49131ed06ecd04532337f5a962119b3367a5fdf5b77`
