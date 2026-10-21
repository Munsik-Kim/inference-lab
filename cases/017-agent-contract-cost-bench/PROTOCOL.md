# Case017 — Agent Contract–Cost Bench: 실행 계약

## 질문과 등록 범위

같은 모델이 같은 업무를 수행할 때, 출력 형식을 제한하거나 잘못된 호출을 한 번 다시 작성하게 하면 실제 기록과 최종 답까지 더 잘 완료하는지 비교합니다. 이어서 가중치는 BF16으로 유지하고 KV 캐시만 BF16과 native NVFP4로 바꾸어 성공·실패·호출·토큰·시간을 함께 측정합니다.

정본은 [specs/protocol.json](specs/protocol.json)과 단계별 execution freeze입니다. 제공된 사용자 지시는 0–5절까지입니다. 이후 단계의 구체적인 실행 순서, 분석 seed와 전체 worker 7,200초 ceiling 등은 모델 관측 전에 추가한 **로컬 사전 등록**이며, 사용자가 이미 정했던 항목으로 소급 서술하지 않습니다. 이 문서는 계약 설명이고, 실제 완료·미완료 상태는 결과 receipt에서 확인합니다.

## 고정 모델과 실행 경계

| 항목 | 등록 값 |
|---|---|
| 모델 | Qwen/Qwen3-4B-Instruct-2507 |
| revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| weights / TP | 모든 조건 BF16 / 1 |
| 엔진 | 기존 vLLM 0.32.0 + FlashInfer, xgrammar 0.2.8 |
| K0 / K1 | 실제 BF16 KV / packed NVFP4 KV + FP8 block scales |
| 요청 경로 | 모든 정책 `/v1/chat/completions`, 동일 template와 단일 content JSON envelope |
| sampling | temperature 0, top_p 1, n 1, thinking OFF, EOS 준수 |
| 메모리·스케줄러 | 약 4GiB 공통 KV pool 목표, server context 28,672, max sequences 4, prefill 2,048, prefix cache OFF, async scheduling OFF |
| 품질 실행 | episode concurrency 1, 매 턴 전체 history 재전송, 요청 내부 KV 사용 |

실제 설정은 tensor/config receipt로 확인합니다. 같은 pool 목표라도 정렬 후 실제 bytes가 조금 다를 수 있어 각 조건의 실제 저장량·block 수를 기록합니다. NVFP4는 저장 형식뿐 아니라 attention kernel 경로도 달라질 수 있습니다. 비교는 우선 그 전체 serving 구성의 관측 차이입니다.

## 업무를 이해하기 위한 용어

- **Fixture:** 실제 서비스와 연결되지 않은 작은 합성 JSON 기록입니다. 각 episode는 새 임시 root의 원상태에서 시작합니다.
- **주문 정산(A):** 구매 내역의 물건 수·개당 가격·할인율을 읽고, 계산 도구가 반환한 최종 금액을 대상 기록에 저장합니다. 금액은 소수점이 없는 최소 회계 단위입니다.
- **창고 이동(B):** 출발·도착 기록의 재고와 revision을 읽고 요청 수량을 한 번에 이동합니다. 양쪽 합계와 다른 기록을 보존합니다.
- **문서 metadata(C):** 현재 revision을 읽고 요청한 상태·태그·담당자 ID를 갱신합니다. 제목과 다른 문서는 보존합니다.
- **Validation:** 공개된 구조·보호 필드·업무 불변식을 확인합니다. 숨긴 정답을 대신 계산해 주는 채점 도구가 아닙니다.
- **Final:** 실제로 받은 최신 validation을 근거로 최종 값을 반환하는 등록 행동입니다. 단순한 완료 문장과 다릅니다.

## 공통 출력과 엄격 parser

매 응답은 다음과 같은 **한 개의 완결 JSON object**입니다. 아래 ID는 형식 설명용이며 fixture의 정답이 아닙니다.

```json
{"tool":"read_order","arguments":{"order_id":"example-record"}}
```

최상위 key는 `tool`과 `arguments`만 허용합니다. 등록된 정확한 이름, 도구별 필수 key·자료형·공개 범위를 검사합니다. JSON 앞뒤의 허용 공백 외에 prose, Markdown fence, 여러 객체, 잘린 JSON, 중복 key를 허용하지 않습니다. 숫자 문자열·소수 표현·bool을 정수로 고치지 않으며 `true`와 `1`은 다릅니다. 첫 번째 유효 substring만 골라 실행하지 않습니다.

유효한 정수나 ID라고 정답인 것은 아닙니다. 도구는 실제 받은 인자를 적용하고 schema-valid 잘못된 값도 숨긴 oracle로 교정하지 않습니다. 실행 오류 결과는 모든 정책에서 같은 방식으로 모델에게 돌려줍니다.

## 네 정책

| 정책 | 응답 생성 | 첫 계약 실패 처리 |
|---|---|---|
| C0 FREE | 공통 prompt, 문법 제약 OFF | 실행 없이 종료 |
| G GRAMMAR | 처음부터 final까지 매 요청에 같은 공개 도구별 schema 적용 | host 계약이 거절하면 종료, 별도 retry 없음 |
| N NEUTRAL RETRY | C0와 같은 자유 생성 | 실패 원문을 assistant 이력에 보존하고 중립 user 문장을 넣어 한 번 다시 생성 |
| E ERROR RETRY | C0와 같은 자유 생성 | 같은 발동·횟수·예산에서 실제 공개 error code/path/expected만 user 메시지로 제공 |

N의 문장은 다음과 같이 고정합니다.

> 직전 호출은 실행되지 않았습니다. 동일 작업의 호출을 다시 작성하세요.

E는 [contracts.py](src/agent_contract_cost/contracts.py)의 공개 오류 객체 `{"code":...,"path":...,"expected":...}`를 직렬화합니다. 틀린 실제 값, 정답 금액·ID·도구 순서나 hidden oracle을 안내에 붙이지 않습니다. 이 host 피드백은 도구 결과가 아니므로 새 tool event/관측 ID를 만들지 않습니다.

Retry는 episode 전체에서 첫 **계약 실패** 한 번에만 발동합니다. 정상 도구 실행에서 반환한 업무 오류는 전용 retry 조건이 아닙니다. N과 E는 각각 독립 자유 실행하므로 최초 실패나 후속 경로가 같다고 가정하지 않습니다. E−N은 이 정책 비교이며 오류 설명 정보만의 고립된 인과효과는 아닙니다.

### G의 실제 보장 범위

공식 요청 field `structured_outputs.json`을 사용합니다. `oneOf` 분기로 등록 도구명과 해당 인자 구조·필수 key·자료형·공개 정적 범위를 표현합니다. ID는 자유 문자열이고 금액은 공개 범위의 자유 정수입니다. Gold ID·금액 const/enum, 정답 다음 행동, target whitelist는 없습니다.

[CPU tokenizer/matcher 검사](provenance/grammar_cpu_validation_r2.json)에서 설치된 backend가 배열 `uniqueItems`를 강제하지 않는 것을 확인했습니다. 따라서 서버 grammar에서 이 keyword를 제거하고 `TOOL_BRANCH_SCHEMA_HOST_ONLY_UNIQUE_ITEMS`로 등록했습니다. 태그 중복은 모든 정책의 같은 host validator가 검사합니다. CPU compile/mask 통과와 서버가 실제 제약을 적용했다는 P0 증거는 구분합니다.

## 예산과 비용 회계

| 한도 | 공통 값 |
|---|---:|
| 모델 요청당 출력 cap | 512 tokens |
| episode 모델 요청 시도 | 8 |
| parsed 도구·final 시도 합계 | 6 |
| episode 출력 총량 | 4,096 tokens |
| retry | 최초 계약 실패에 최대 1회 |
| 요청 / episode deadline | 180초 / 900초 |
| client history/context | 24,576 tokens, 자동 truncate 없음 |

실제 요청 cap은 `min(512, 남은 episode 토큰, 24576−실제 prompt tokens)`입니다. 요청 시도는 HTTP 응답을 기다리기 전에 차감합니다. Timeout·부분 stream도 요청 시도와 관측된 출력 토큰을 소비하고, 완결 응답 수는 별도입니다.

파싱 불가·배열·scalar는 모델 요청과 출력 비용을 소비하지만 parsed 행동 시도는 아닙니다. 파싱한 JSON object의 envelope·이름·schema 오류는 parsed 시도 1회를 소비하며 callable 실행은 없습니다. `final`도 parsed 시도 1회입니다. 여섯 번째 final은 허용하고, 여섯 번째가 nonfinal이면 그 행동을 실행한 뒤 `TOOL_CALL_LIMIT`으로 종료합니다. 일곱 번째 요청에는 실행 가능한 등록 행동 예산이 없어 생성하지 않습니다. 이 명시된 정책은 과거 Case016의 final 회계와 다릅니다.

`model_request_attempts`, `model_responses_received`, `server_abort_requests`, `parsed_tool_attempts`, `contract_rejections`, `callable_executions`, `side_effect_writes`, `final_attempts`, `retry_triggered`, `retry_consumed`, 출력 토큰을 분리합니다. 발동한 retry가 context/deadline 사전 검사에서 막히면 실제 retry 소비는 0입니다.

전체 P0/P1/P2 worker ceiling 7,200초는 load·compile·실패·I/O를 포함하는 추가 로컬 운영 한도입니다. CPU 구현·자료 준비 시간과 GPU busy 시간은 별도입니다. 미종결 worker attempt가 있으면 wall 장부를 복구하기 전 새 GPU 작업을 막습니다. 다음 episode를 완전히 실행할 잔여시간이 없으면 새 셀을 열지 않고 PARTIAL과 누락을 기록합니다.

## 데이터·단계·동결

| 묶음 | 업무 수 | 비교 | 본 분모 |
|---|---:|---|---:|
| R | 과거 Case015 기록 | 회귀 참고·원형 보존 | 새 성공률에 미포함 |
| DEV / P1 | A3+B3+C2 = 8 | 2 KV × 4 정책 | 64 episode |
| HOLDOUT / P2 | A8+B8+C8 = 24 | 2 KV × 4 정책 | 192 episode |
| COST / P3 | A2+B2+C2 = 6 | 설계용, GPU 미실행 | 0 |

P0는 캐시·실제 attention·raw token·문법 적용·별도 smoke를 확인하는 기술 단계이고, 주 성공률과 합치지 않습니다. P1은 K0→K1, P2는 K1→K0 순서이며 각 KV 안의 task/policy 순서는 고정 seed로 shuffle합니다.

Seed는 DEV `6102001`, HOLDOUT `6102002`, COST `6102003`, P1 순서 `6102011`, P2 순서 `6102012`, bootstrap `6102013`, 엔진 `6102007`입니다. DEV 실제 최초 입력은 1,726–1,784 tokens이고, 약 2K라는 목표에 맞추려고 내용이나 token을 잘라내지 않았습니다.

HOLDOUT generator와 seed는 P1 이전에 동결합니다. 실제 HOLDOUT 자료는 P1 64셀 완료와 기존 freeze 불변을 확인한 뒤 생성·봉인합니다. DEV/COST와 task·초기 상태·record ID 중복을 검사하고 tasks/model-input hash, P1 완료 receipt hash를 seal에 넣습니다. P2는 이 seal과 P1 freeze를 다시 확인합니다. 한 generator의 세 업무 유형으로 만든 자료이며 독립 회사나 외부 업무 모집단으로 표현하지 않습니다.

## 관측과 주 endpoint

매 generation 시작 시 도구 결과 event ID 목록을 고정합니다. 읽기 결과를 보지 않은 계산·이동·갱신은 값이 우연히 맞더라도 관측 의존성을 만족하지 않습니다. 모델에게 전달하는 도구 관측은 `event_id`, `tool`, `status`, `result`뿐입니다. 원 초기 상태·hidden oracle·감사용 상태 snapshot은 모델 입력에 넣지 않습니다.

`VERIFIED_WORKFLOW_SUCCESS`는 다음을 함께 확인합니다.

1. 업무별 요구 행동이 실제 실행되고 필요한 read/calculation 결과를 받은 뒤 의존 행동이 생성됨.
2. 실제 최종 기록·revision이 독립 기대 상태와 일치함.
3. 모든 중간 상태에서 보호 필드와 다른 기록을 보존함.
4. 현재 상태·version에 대응하는 성공 validation을 final 생성 전에 실제로 관측함.
5. 정확한 자료형·key·값의 final을 예산 안에 반환함.

파일만 맞음, 답만 맞음, 근거 없는 완료 주장, 최초 계약 오류, 최종 종료 이유는 각각 기록합니다. 이전 오류 뒤 유효한 사슬로 회복하면 성공할 수 있지만, 무단 변경을 되돌려 원상태로 만든 경우는 전 과정 보존 위반으로 남습니다.

## 감사와 분석

[audit.py](src/agent_contract_cost/audit.py)는 raw를 다시 파싱해 새 fixture에서 실행하고 event·관측·상태·요청·예산을 대조합니다. 별도 endpoint는 Decimal half-up 계산, 재고 이동·revision, 보호 필드, 의존 사슬과 최종 답을 다시 계산합니다. 저장 success bool이나 보고서 숫자를 정답으로 읽지 않습니다. 저장 input IDs/hash 대조는 실제 tokenizer로 다시 토큰화한 검증과 구분합니다.

[analysis.py](src/agent_contract_cost/analysis.py)는 감사한 행에서 KV별 G−C0/N−C0/E−C0/E−N, 정책별 K1−K0를 계산합니다. 유지 성공·손실·회복·양쪽 실패와 family 표를 먼저 제시합니다. Bootstrap 10,000회는 family 안에서 task를 재표집하고 모든 조건에 같은 task index를 적용하는 탐색적 pointwise 95% 구간입니다. 결측은 0으로 채우지 않고 관측 분모·사유·binary 최악/최선 범위를 표시합니다. 폭 0의 구간도 동등성·비열등성 보장이 아닙니다.

P1과 P2의 분모를 합치지 않습니다. 독립 자유 rollout이 갈라진 뒤의 턴을 같은 관측 비교로 강제하지 않습니다. Episode wall 합계를 동시성 throughput이나 KV 산술만의 인과효과로 해석하지 않습니다.

## 보존과 종료

새 모델 다운로드·학습·재양자화·D16/RTN 로드는 없습니다. 이전 Case015/016의 소스·원응답·점수·실패·freeze는 보존합니다. 실행 소스 hash, 입력 bundle hash, model revision, Git base, artifact/환경 hash는 서로 다른 식별자입니다. 새 commit을 만들지 않으므로 Git base가 실행 소스 전체를 이미 포함한다고 표시하지 않습니다.

기술 gate가 막히거나 예산이 소진되면 실제 완료 셀·attempt·누락·원인을 기록하고 로컬 검토본으로 끝냅니다. 어떤 성공률이 나와도 이번 단계에서 P3, 새 sweep, GitHub push/PR/merge, Pages나 Drive 게시로 자동 진행하지 않습니다. 재사용과 권리 범위는 [REUSE_MAP](REUSE_MAP.md), [NOTICE](NOTICE.md), 후속 설계는 [ROADMAP](ROADMAP.md)에 있습니다.
