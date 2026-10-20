# Case016 — RTX 5080에서 NVFP4 KV 캐시 검증: 같은 메모리로 더 긴 문맥을 수용할 수 있을까?

vLLM0.32.0과 FlashInfer의 실제 압축 KV 경로를 연결하고, 저장·읽기·고정 메모리 용량·동시 요청 비용을 검증하는 도구를 구현했습니다. Qwen3-4B의 가중치는 BF16으로 유지했습니다.

같은 약4GiB pool에서 B16의29,120슬롯을 NVFP4의103,552슬롯으로 늘려 token 수용량 3.5560×. 등록한16K/C2·C4 workload의 세 라운드 모두에서 serving gain을 관측했습니다.

1. [배경](#background)
2. [질문](#questions)
3. [기술 원리](#principles)
4. [방법](#method)
5. [실험](#experiment)
6. [결과](#results)
7. [분석](#analysis)
8. [결론](#conclusion)
9. [재현·출처](#sources)

<a id="background"></a>

## 1. 배경

원래 장문 agent 탐색에서는 기준선 완주율이 낮았습니다. 원 실패를 보존하고, 이번 후속은 실제 압축 KV의 저장·읽기와 수용량·서빙 비용이라는 기술 질문으로 한정했습니다. 새 agent 과제·학습·가중치 양자화를 추가하지 않았습니다.

<a id="questions"></a>

## 2. 질문

TECH_PATH_VALID: 실제 SM120/36층에서 packed KV를 저장·읽는가? TECH_NUMERIC_VALID: 같은 압축 캐시의 두 dequant 해석과 실제 attention 유한성이 확인되는가? CAPACITY_VALID: 같은 pool의 실제 bytes와 슬롯이 일치하는가? SERVING_MEASURED: 고정 요청 집합을 전 조건에서 관측했는가? 속도 향상을 완료 조건으로 삼지 않았습니다.

<a id="principles"></a>

## 3. 기술 원리

B16: 2×36×8×128×2=147456B/token. F4: packed data36864 + FP8 block scales4608 =41472B/token. 비율147456/41472=3.5556이며 실제 page 정렬 뒤 슬롯 비율과 조금 다릅니다. 공유 storage의 view는 중복 계수하지 않았습니다. F4는 uint8 packed values와 FP8 E4M3 block scale, 고정 global scale1을 사용했습니다. 짧은 profile은 NVFP4 writer와 압축 데이터를 읽는 FlashInfer FA2 경로를 보여줍니다. BF16 fallback과 FA2 read를 구분합니다.

<a id="method"></a>

## 4. 방법

모델 revision은 `cdbee75f17c01a7cc42f958dc650907174af0554`,36층·Q32/KV8·head_dim128입니다. 두 조건에4GiB 목표 pool·context28672·max_num_seqs4·chunked prefill2048·prefix caching OFF·async scheduling OFF·BF16 weights를 공통 적용했습니다. 두 CUDA graph gate가 통과해 실제 compilation mode FULL_AND_PIECEWISE를 공통 사용했습니다. B16/F4 모두 decode는 XQA를 사용했고 F4 prefill은 NVFP4 FA2였습니다. XQA 호출은 packed NVFP4 data와 block scale을 받습니다. `nvfp4_fa2=True`와 `impl.use_xqa=None`은 decode 미사용의 근거가 아닙니다.

<a id="experiment"></a>

## 5. 실험

정확히8192/16384개의 입력 token IDs를 Qwen template 전체(roles·assistant prefix 포함)로 동결해 completions endpoint에 전달했습니다. 합성 filler는 사전 생성에서만 조정했고 질문·사실은 자르지 않았습니다. 각 길이의 같은8 prompt를 C1/2/4에서 세 번 사용합니다. 순서는 B16→F4 / F4→B16 / B16→F4입니다. 출력은 ignore_eos=True,128토큰이며 자연 종료 sanity와 구분합니다. 각 요청 timeout180초에는 client slot 대기도 포함됩니다.

Makespan은 닫힌8요청 집합의 launch→마지막 client 응답 종료입니다. 실제 timestamp는 HTTP 수신 후 raw SSE 파싱까지 포함하며 HTTPclose만의 별도 시각은 기록하지 않았습니다. req/s=성공 요청/makespan, output token/s=관측 output tokens/makespan. Client TTFT는 전송→첫 nonempty token-ID event, decode 평균은 마지막−첫 token-bearing event를 n−1로 나눈 값입니다. SSE chunk 간격을 token ITL로 쓰지 않았습니다. 엔진 queue/decode는 전용 서버 histogram sum/count의 block 구간 차이입니다. 메모리는 배경 점유를 포함하는 device 표본과 Torch allocated/reserved를 분리했습니다.

<a id="results"></a>

## 6. 결과

| KV | bytes/token | actual pool bytes | blocks ×16 | token slots |
|---|---|---|---|---|
| B16 | 147456 | 4293918720 | 1820 | 29120 |
| F4 | 41472 | 4294508544 | 6472 | 103552 |

| Input | C | B16 req/s | F4 req/s | F4/B16 median [3-round range] | success B16 / F4 |
|---|---|---|---|---|---|
| 8K | 1 | 0.396 | 0.397 | 1.001 [1.000, 1.003] | 24/24 / 24/24 |
| 8K | 2 | 0.563 | 0.561 | 0.996 [0.992, 1.000] | 24/24 / 24/24 |
| 8K | 4 | 0.630 | 0.712 | 1.125 [1.125, 1.131] | 24/24 / 24/24 |
| 16K | 1 | 0.251 | 0.247 | 0.986 [0.985, 0.991] | 24/24 / 24/24 |
| 16K | 2 | 0.251 | 0.303 | 1.209 [1.205, 1.211] | 24/24 / 24/24 |
| 16K | 4 | 0.251 | 0.340 | 1.351 [1.350, 1.357] | 24/24 / 24/24 |

| Input | C | B16/F4 median TTFT s | B16/F4 decode ms/token | B16/F4 queue s | B16/F4 preemptions | B16/F4 device peak MiB |
|---|---|---|---|---|---|---|
| 8K | 1 | 0.844/0.923 | 13.182/12.534 | 0.000/0.000 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15102/15103 |
| 8K | 2 | 0.873/0.973 | 21.015/20.326 | 0.083/0.092 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15096/15100 |
| 8K | 4 | 1.895/1.353 | 28.072/33.124 | 1.402/0.629 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15101/15102 |
| 16K | 1 | 2.121/2.378 | 14.646/13.093 | 0.000/0.000 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15102/15101 |
| 16K | 2 | 6.100/2.467 | 14.735/32.372 | 3.337/0.273 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15097/15104 |
| 16K | 4 | 13.987/3.020 | 14.662/67.868 | 8.804/1.718 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15102/15103 |

| Input | B16 copy exact /4 | F4 copy exact /4 | equal token sequences /8 |
|---|---|---|---|
| 8K | 4 | 4 | 4 |
| 16K | 4 | 4 | 4 |

| Phase | model request attempts | responses received | observed output tokens |
|---|---|---|---|
| p0_numeric | 2 | 2 | 64 |
| p0_graph | 8 | 8 | 128 |
| warmup | 12 | 12 | 1536 |
| sanity | 32 | 32 | 921 |
| timed | 288 | 288 | 36864 |

| Owned server | Mode | Load/compile/warmup ready seconds | Total attempt wall seconds |
|---|---|---|---|
| p0_B16 | EAGER | 23.310 | 29.545 |
| p0_F4 | EAGER | 41.275 | 48.242 |
| graph_B16 | GRAPH | 36.330 | 37.096 |
| graph_F4 | GRAPH | 35.291 | 36.040 |
| R1_B16 | GRAPH | 18.284 | 198.163 |
| R1_F4 | GRAPH | 17.284 | 184.655 |
| R2_F4 | GRAPH | 17.284 | 152.574 |
| R2_B16 | GRAPH | 16.290 | 166.072 |
| R3_B16 | GRAPH | 17.285 | 167.496 |
| R3_F4 | GRAPH | 17.292 | 152.470 |


작은 NHD/HND 검사에서 같은 압축 cache의 두 dequant 방식 최대차이는0이었습니다. BF16 원본 대비 상대 norm 오차는 각각 0.095461, 0.095341. 이 값은 양자화 오차이며 decoder 간 불일치가 아닙니다. 짧은512-token P0 eager 입력과32-token 생성에서 실제 attention36층은 두 조건 모두 유한했습니다. 전체 모델 logits의 독립 유한성·전체 KL/PPL은 미측정입니다.

<a id="analysis"></a>

## 7. 분석

같은 pool에서 F4의 여유 슬롯과 정상 attention kernel 선택이 queue·선점·시간과 함께 달라집니다. 전체 구성의 관측 차이이며 양자화 산술만의 원인으로 분리하지 않습니다. 세 round 비율과 범위를 모두 보여주고 요청 latency 재표집으로 makespan의 좁은 CI를 만들지 않았습니다. 동일8 prompt의 반복을24개 독립 과제로 세지 않습니다.

계측 주석: 재사용 CacheInspector의 `cudagraph_policy`는 과거 eager 설명 상수였습니다. raw 필드는 보존하고 실제 `compilation_mode`, startup capture 로그와 동결 launch 설정으로 graph를 판정했습니다. 모델 파라미터 비교는 모든 named parameter의 앞뒤8개 값에 대한 signature이며 전체 GPU weight hash가 아닙니다. 파일 단위 모델 shard hash는 원 snapshot과 다시 일치했습니다.

<a id="conclusion"></a>

## 8. 결론

등록한16K/C2·C4 workload의 세 라운드 모두에서 serving gain을 관측했습니다.

이전 agent 연구는 B16 0/48·F4 6/48, 같은12개 DEV 과제의 메뉴/cap 반복입니다. 기준선 gate NOT_MET·원 실패·amendment01~08·소스 hash와 CPU 감사 원점수 변경0을 보존했습니다. F4의 세 task 성공을 일반 업무 우위로 확대하지 않습니다.

| Scope | Status |
|---|---|
| Native NVFP4 KV storage/read ·36 layers | VERIFIED |
| Fixed4GiB capacity ·36 timed blocks/288requests | MEASURED |
| Copy canary/open-text sanity | LIMITED ·16 paired prompts |
| Full KL/PPL · agent-quality improvement ·100K context | NOT_EVALUATED |
| Agent P2/P3 · weight W4 · training · prefix reuse | NOT_RUN |
| GitHub Pages | BUILDER_READY · MANUAL_DEPLOY_NOT_RUN |


<a id="sources"></a>

## 9. 재현·출처

[CPU / GPU commands](REPRODUCE.ko.md) · [Stage map](STAGE_MAP.md) · [Raw rows](followups/nvfp4-technical-r1/results/) · [Protocol](followups/nvfp4-technical-r1/specs/protocol.json) · [Execution source freeze](followups/nvfp4-technical-r1/provenance/execution_freeze.json)

실행 source SHA256: `2c36f2cb0f0394f4f9661733344226d05c099f262e248dd4f94ec135e4fc4aec`

Git base·실행 소스 hash·모델 revision·page build revision은 다른 identity입니다. 원 연구449개 파일 중447개를 손실 없는 gzip 또는 동일 바이트로 공개 보존했습니다. 개인 scratch 경로를 담은 패키징 helper2개는 원본 hash만 기록하고 공개에서 제외했습니다. 원 사용자 worktree의 기존 파일은 변경하지 않았습니다.

[vLLM v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) · [Pinned FlashInfer backend](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flashinfer.py) · [Completion protocol](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/openai/completion/protocol.py) · [NOTICE](NOTICE.md)

NVFP4 커널은 upstream vLLM/FlashInfer 제공입니다. 직접 기여는 통합·실제 경로/용량 검사·고정 workload 계측·CPU 감사·문서입니다. Codex가 구현과 별도 계산 검토를 보조했으며 사람의 독립 심사나 외부 GPU 재현으로 표시하지 않습니다.

이번 누적 GPU worker wall: 1183.314s /7200s. 모델 load·작은 검사·graph gate·warmup·sanity·metric·I/O를 포함하며 GPU busy나 serving 가속률이 아닙니다. 설치·모델 다운로드·새 weight artifact 생성은0회입니다.
