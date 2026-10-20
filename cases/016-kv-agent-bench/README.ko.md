# Case016 — RTX 5080에서 NVFP4 KV 캐시 검증: 같은 메모리로 더 긴 문맥을 수용할 수 있을까?

vLLM0.32.0과 FlashInfer의 실제 압축 KV 경로를 연결하고, 저장·읽기·고정 메모리 용량·동시 요청 비용을 검증하는 도구를 구현했습니다. Qwen3-4B의 가중치는 BF16으로 유지했습니다.

같은 약4GiB pool에서 B16의29,120슬롯을 NVFP4의103,552슬롯으로 늘려 token 수용량 3.5560×. 등록한16K/C2·C4 workload의 세 라운드 모두에서 serving gain을 관측했습니다.

**16K/C2: 1.21× [1.21, 1.21] · 16K/C4: 1.35× [1.35, 1.36]**

위 비율은 성공 HTTP 요청/s의 F4/B16 비율이며3라운드 중앙값과 범위입니다. 전체 VRAM 감소율·agent 성공률·100K 단일 문맥 지원을 뜻하지 않습니다.

[Report](REPORT.ko.md) · [CPU / GPU reproduction](REPRODUCE.ko.md) · [Runner](followups/nvfp4-technical-r1/src/nvfp4_technical/runner.py) · [Raw / CSV](followups/nvfp4-technical-r1/results/) · [Stage map](STAGE_MAP.md)

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

처음 보는 용어: KV 캐시는 모델이 읽은 내용의 중간 계산값을 저장하는 공간입니다. BF16은16비트 기준선, NVFP4는4비트 값과 block scale을 함께 저장하는 형식입니다. C는 동시에 보내는 client 요청 수, token은 모델이 읽는 글 조각입니다. req/s는 초당 성공 요청 수로 높을수록 좋습니다.

이전 agent DEV는 B16 0/48·F4 6/48, 같은12개 과제의 반복입니다. 기준선 gate NOT_MET과96개 원 판정·CPU 감사는 [원형 안내](STAGE_MAP.md)에 보존했습니다. Agent P2/P3는 실행하지 않았습니다. 한영 Pages builder에는 연결했으며 수동 배포는 이번 범위에서 실행하지 않습니다.
