# Case017 — 도구 호출 규칙을 지키면 AI가 일을 더 정확히 끝낼까?

가상 기록을 읽고 계산하거나 수정한 뒤, 실제 저장된 상태와 마지막 답이 모두 맞는지 시험했습니다. 상품별 개수와 한 개 가격으로 합계를 구하는 작업, 창고 사이에 물건을 옮기는 작업, 문서의 상태·태그·담당자를 바꾸는 작업입니다. 기본 모델 가중치는 그대로 두고, 출력 규칙 네 가지와 대화 기억(KV cache)의 저장 정밀도 두 가지를 비교했습니다.

| KV | C0 FREE | G GRAMMAR | N NEUTRAL | E ERROR |
| --- | --- | --- | --- | --- |
| K0 | 16/24 | 16/24 | 16/24 | 16/24 |
| K1 | 7/24 | 15/24 | 9/24 | 13/24 |

분모는 새 합성 HOLDOUT24입니다. NVFP4에서 문법 제약은 자유 생성보다 8개를 더 완주했고 손실은 0개였습니다. BF16의 완주수는 네 정책 모두 16/24로 같았습니다. 문법 제약은 잘못된 상태값이나 오래된 검증으로 완료하는 행동까지 고치지는 못했습니다.

A: 여러 상품의 개수와 한 개 가격으로 할인 적용 합계를 계산·저장합니다. B: 두 창고의 재고를 읽고 물건을 원자적으로 옮깁니다. C: 문서의 현재 수정 번호를 읽고 상태·태그·담당자를 바꿉니다. 전부 소유한 가상 JSON 기록입니다.

직접 구현: 엄격 JSON 계약·4정책 실행기, 안전 fixture·관측 추적, 독립 endpoint 감사·비교·한영 탐색기. 엔진과 NVFP4 커널은 vLLM/FlashInfer upstream입니다.

[Report](REPORT.ko.md) · [English report](REPORT.md) · [한국어 탐색기](demo/ko.html) · [English explorer](demo/en.html) · [CPU/GPU reproduction](REPRODUCE.md)

```bash
PYTHONPATH=src python3 -B -m agent_contract_cost.verify --output /tmp/case017-review-new-output
```

출력은 아직 없는 새 경로를 사용합니다. 이 명령은 저장 응답을 재생하며 모델을 호출하지 않습니다.

### DEV / P1

| KV | C0 FREE | G GRAMMAR | N NEUTRAL | E ERROR |
| --- | --- | --- | --- | --- |
| K0 | 5/8 | 5/8 | 5/8 | 5/8 |
| K1 | 2/8 | 5/8 | 2/8 | 4/8 |

P0 22 requests; P1 233; P2 775. Total 1030 API generation attempts, 50585 output tokens; worker wall 1035.148s.

K0는 BF16 KV, K1은 native NVFP4 KV입니다. **가중치는 두 조건 모두 BF16**입니다. G는 공개 JSON schema로 생성 token을 제한하고, N/E는 최초 계약 실패에만 한 번 재시도합니다. 같은 약4GiB pool의 저장 슬롯 비율3.556배는 전체 VRAM 절감이나 이번 업무의 속도 향상이 아닙니다. P3·가중치 INT4·학습·동시 serving·원격 게시·Pages는 미실행입니다.

[Protocol](PROTOCOL.md) · [Reuse](REUSE_MAP.md) · [Notice](NOTICE.md) · [Roadmap](ROADMAP.md) · [Source freeze](provenance/execution_freeze_p2.json)
