# Case015 P2 — 동일 관측 JSON 복구 진단: CPU 준비 보고서

기존 실패의 직접적인 실행 중단 지점은 **저장 도구 호출의 JSON 파싱**이었습니다. 모델이 합계 숫자를 텍스트에 썼더라도 유효한 인자로 해석되지 않아 파일은 바뀌지 않았습니다. 동일한 입력에서 형식 제약 또는 한 번의 재시도가 실제 업무 완료까지 회복시키는지는 아직 측정하지 않았습니다.

[배경](#배경과용어) · [확인 결과](#확인된기존기록) · [체크포인트](#공통체크포인트) · [개입](#사전등록한개입) · [예산](#예산) · [검사](#cpu검사) · [남은 조건](#새추론전gate) · [재현](#출처와재현)

<a id="배경과용어"></a>
## 1. 배경과 용어

A1은 소유한 합성 JSON 파일을 사용하는 시험입니다. 각 과제에는 대상 구매 기록과 변경하면 안 되는 대조 기록이 있습니다. 모델은 조회 도구에서 **수량**과 **개당 가격**을 받고, 곱셈 도구로 **합계**를 계산한 뒤 파일에 저장하고 검증해야 합니다. 데이터의 `unit_price_minor`는 개당 가격, `total_minor`는 합계를 나타내는 정수 필드입니다. 여기서 ‘주문’은 이 합성 기록의 이름이며 실제 거래를 뜻하지 않습니다.

API_V1과 API_V2는 같은 네 도구를 제공하며 이름의 snake_case/lowerCamelCase만 다릅니다. BF16은 원래 모델, RTN과 D16은 기존 INT4 저장본을 BF16으로 복원해 실행했던 품질 조건입니다. 이번 작업은 그 저장 기록을 CPU에서 읽었습니다.

**체크포인트**는 첫 조회와 곱셈이 끝나고 세 번째 생성(0-based 2)을 시작하기 직전의 대화, 받은 도구 결과, 파일 상태, 사용한 예산입니다. 같은 파일 상태만으로 같은 입력이라고 간주하지 않고 결과 사건 ID와 대화까지 대조했습니다.

<a id="확인된기존기록"></a>
## 2. 확인된 기존 기록

분석 기준은 `51385286063eb55e791f13a0b97f1e341ce6a879`입니다. 작업 시 원격 main도 같았으며 선행 P2 구현은 발견하지 못했습니다. 동봉 ZIP은 조사한 위치에서 발견되지 않아 이 커밋의 공개 원자료로 재구성했습니다.

원 A1의 **96 episode / 463 생성 요청 / 378 도구 시도 / 368 실제 사건**을 원 fixture에서 다시 실행했습니다. 개별 worker 파일, 합쳐진 episode, 별도 tool-event 파일을 대조했고, 원 scorer와 별도 `independent_endpoint`의 전체 업무 완료 판정이 모두 일치했습니다. 원 raw UTF-8 hash와 history/tools, 생성 전 관측 사건 ID, 상태 digest/version도 검사했습니다.

전체 malformed marker slot은 10개입니다. 그중 실패 8건은 아래와 같습니다. 길이는 저장된 output ID 수입니다. 모두 마지막 ID가 **151645/EOS**, 모델 요청 **3/8**, 도구 시도 **3/6**, 실제 조회·곱셈 사건 **2개**, 파일 version **0**입니다.

| 시나리오 | API | 토큰 | JSON 결함 | 닫는 marker 존재 |
|---|---|---:|---|---|
| A1-00 | V1 | 43 | 금액 키 소실 + quoted digits | 없음 |
| A1-00 | V2 | 43 | 금액 키 소실 + quoted digits | 없음 |
| A1-06 | V1 | 43 | 금액 키 소실 + quoted digits | 있음 |
| A1-06 | V2 | 43 | 금액 키 소실 + quoted digits | 있음 |
| A1-08 | V1 | 48 | total_minor 앞 extra quote | 있음 |
| A1-08 | V2 | 48 | total_minor 앞 extra quote | 있음 |
| A1-14 | V1 | 47 | total_minor 앞 extra quote | 있음 |
| A1-14 | V2 | 43 | 금액 키 소실 + quoted digits | 있음 |

구문 JSON, wrapper, 인자 키·타입·값, EOS/cap, 예산, 실제 실행은 [분류 원행](results/cpu_preparation/malformed_slots.json)과 [실패 CSV](results/cpu_preparation/failure_table.csv)에 분리했습니다. 파싱되지 않은 인자의 정확성은 `NOT_EVALUABLE_UNPARSED`입니다. 텍스트에 정답 숫자가 보이는 것을 인자 성공으로 세지 않았습니다.

원 raw의 `BUDGET_EXHAUSTED`는 초기 terminal 값이 파싱 실패 분기에 남은 runner 결함과 대응합니다. 이번 해석은 `PARSE_ERROR`이며 원 raw와 실패 판정은 그대로 보존했습니다. 192토큰 cap이나 실제 호출 예산 고갈로 해석하지 않습니다.

나머지 2개는 **D16 API_V2 A1-01/10의 mixed response**입니다. 같은 응답의 정상 write가 실행되어 최종 성공했습니다. 오류 피드백을 받은 뒤 회복한 사례가 아닙니다.

원 parser는 slot 안에서 **첫 유효한 dict**를 찾습니다. 전체 문자열이 JSON이 아니거나 뒷부분이 지저분해도 채택할 수 있습니다. 닫는 marker가 없다는 사실만으로 실패를 판정하지 않았습니다.

<a id="공통체크포인트"></a>
## 3. 공통 체크포인트

선택된 네 fixture A1-00/06/08/14 × API 2종에서 prefix의 실제 조회·곱셈을 새 fixture root에 재실행했습니다. 새 prefix를 모델로 생성하지 않았습니다. [체크포인트 8개](data/checkpoints.json)와 [허용된 입력 스냅샷 8개](data/model_request_snapshots.json)는 서로 hash로 묶었습니다.

각 지점에서 BF16/RTN/D16의 다음 항목이 같지 않으면 준비가 실패하도록 했습니다.

- messages/tools의 값, key 직렬화 순서와 hash, 전체 history hash.
- 모델이 이미 받은 결과와 `available_result_event_ids`, 공통 관측 hash.
- 파일 상태 digest/version, 모델 2회·도구 2회 사용량.
- 저장된 입력 token hash/길이.

추가로 로컬 Qwen tokenizer JSON과 저장 chat template를 CPU에서 사용해 **8개 고유 입력을 재토큰화했고 24개 방법별 저장 해시와 모두 일치**했습니다. 저장된 출력 토큰도 **463개 raw/text로 정확히 복호화**됐습니다. 이는 새 C0 응답을 재생성했다는 뜻이 아닙니다.

실제 모델 입력은 `messages`와 `tools`만 복사합니다. 원 정답, 대조 기록 정보, 내부 semantic ID, 다른 방법의 후속 write와 oracle은 감사용 checkpoint/별도 historical 파일에 남고 프롬프트에는 들어가지 않습니다. 조회·곱셈 도구로 이미 받은 값은 원래 관측으로 유지합니다.

이 8지점은 **4개의 이미 선택된 DEV fixture**에서 왔습니다. 새로운 8개 독립 과제나 held-out end-to-end 평가가 아닙니다.

<a id="사전등록한개입"></a>
## 4. 사전 등록한 개입

| 조건 | 최초 checkpoint 응답 / 재시도 | 모델에 주는 도움 |
|---|---|---|
| C0 ORIGINAL | 원 messages/tools, 원 malformed-only 종료 정책 | 추가 도움 없음 |
| G SCHEMA | C0와 같은 messages/tools, 첫 응답에만 실제 token mask | 선언된 네 도구의 이름별 JSON key/type 계약 |
| N NEUTRAL_RETRY | 같은 **새 C0** malformed-only 뒤 1회 | 중립 retry 문장 |
| E ERROR_FEEDBACK | N과 같은 새 실패·관측·상태·예산에서 1회 | N에 파싱 오류 설명만 추가 |

N user 문장:

```text
No tool was executed. Retry the previous response.
```

E user 문장:

```text
PARSE_ERROR: Tool-call JSON could not be parsed. No tool was executed. Retry the previous response.
```

N/E는 새 C0 raw를 그대로 assistant content로 남깁니다. host feedback은 user 메시지이며 가짜 tool event가 아닙니다. 관측 결과 ID와 파일 상태는 늘지 않습니다. 정상 호출도 있는 mixed response에는 retry하지 않고 원 실행 순서를 유지합니다. 과거 D16 raw를 BF16/RTN이 새로 생성한 응답으로 넣는 경로는 거절합니다.

G는 프롬프트 힌트가 아니라 설치되어 있던 `lm-format-enforcer 0.11.3`를 이용한 실제 CPU allowed-token mask입니다. 네 도구를 모두 선택할 수 있고, ID는 자유 문자열, 금액은 자유 정수, final의 validated는 true/false 모두 가능합니다. gold 금액 const/enum, 대상 ID whitelist, write 강제, 사후 JSON 수선은 없습니다. 복수 호출·같은 응답의 write→validate를 지원합니다. 원 parser에 맞춰 **완성된 호출 JSON은 닫는 marker 없이도 EOS 가능**, 미완성 JSON은 EOS 차단입니다.

실제 로컬 토크나이저에서 두 API의 합성 경로 10개와 미완성 prefix 2개를 검사했습니다. 이는 지원 경로에 대한 CPU 검사이며 모든 토큰 조합이나 실제 GPU decoding을 입증한 것은 아닙니다. 후속 generation에는 문법 제약을 확대하지 않습니다. unmasked logits의 유한성 확인 후 grammar가 의도한 `-inf`를 적용하는 순서를 별도로 검사했습니다.

<a id="예산"></a>
## 5. 예산과 주 평가 계약

모든 분기는 prefix 사용량 **모델 2 / 도구 2**에서 시작합니다. 최대 모델 8회, 도구 시도 6회, 요청당 새 토큰 192, context 24,576, deadline 900초입니다. malformed marker도 도구 시도에 포함합니다. N/E는 공유한 C0의 모델·도구·시간 비용을 다시 초기화하지 않습니다. final 또는 validation을 추가 요청해도 모델 예산에 포함됩니다.

정확한 과거 prefix wall은 저장되지 않았습니다. 따라서 같은 지점 세 방법의 **원 전체 episode wall 중 최댓값(6.427–7.229초)**을 공통 prefix 차감으로 등록했습니다. 뒤쪽 작업까지 포함한 보수적인 상계이며 정확한 prefix 실측이 아닙니다. 새 C0, host 처리, 분기 continuation/idle 시간도 추가 차감해야 합니다. 새로 900초를 지급하지 않습니다.

F는 **새 C0에서 실제로 재현되고 남은 예산상 재시도가 가능한 malformed-only 수**입니다. 전체 malformed와 budget-ineligible도 따로 기록합니다.

| 새 요청 장부 | 식 | F=8이라는 자원 가정 | 절대 상한 F=24 |
|---|---|---:|---:|
| 최초 진단 응답 | 24 C0 + 24 G + 2F | 64 | 96 |
| 이후 continuation 요청 상한 | 5(24−F)+5×24+8F | 264 | 312 |
| 모든 새 모델 요청 상한 | 288+5F | 328 | 408 |
| 새 토큰 상한 | 모든 요청 상한×192 | 62,976 | 78,336 |

64는 예측 결과도, 전체 모델 호출 수도, 독립 episode 수 64개도 아닙니다. 이번 CPU 등록은 미래 GPU worker wall 총예산을 승인하거나 설정하지 않았습니다.

주 endpoint는 원 `assess`와 `independent_endpoint`의 **VERIFIED_WORKFLOW_SUCCESS**입니다. 올바른 실제 write, 원 관측 순서, 다른 기록·불변 필드 보존, 최종 상태에 대응하는 validation을 받은 뒤의 strict final이 모두 필요합니다. 파싱 성공이나 파일 상태만 맞는 결과는 주 성공으로 세지 않습니다. 잘못된 ID, 정답±1 정수, 숫자 문자열, bool을 int로 사용한 경우를 원 fixture/parser/scorer 기준대로 검사했습니다.

## 6. 새 복구 결과

아래는 측정 결과가 아니라 아직 실행하지 않은 대조 목록입니다. [96개 조건부 계획행](results/cpu_preparation/recovery_results.json)의 success는 전부 null입니다. N/E는 C0 trigger가 없으면 향후에도 적용 대상이 아닙니다.

| 방법 / API | C0 | G | N | E |
|---|---|---|---|---|
| BF16 / V1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| BF16 / V2 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| RTN / V1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| RTN / V2 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| D16 / V1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| D16 / V2 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

G가 올바른 인자와 전체 업무 완료까지 회복하면 이 선택된 사례의 직렬화 병목을 지지합니다. 형식만 회복하면 인자/후속 업무 문제를 따로 남깁니다. 오류 설명의 고유 효과는 E와 N을 비교해야 합니다. 어떤 결과도 원 D16 자유 생성 실패를 없던 것으로 만들지 않습니다.

<a id="cpu검사"></a>
## 7. CPU 검사와 실제 준비 중 오류

| 검사 | 이번 실제 실행 | 구분 |
|---|---:|---|
| 기존 P1 | 27 PASS | 과거 고유 검사의 회귀 재실행 |
| 기존 A1 | 18 PASS | 회귀 재실행 |
| 기존 publication | 6 PASS | 회귀 재실행 |
| 새 checkpoint / 요청 / retry / 예산 / endpoint 계약 | 20 PASS | 신규 고유 검사 |
| 새 문법 / key·type / 복수 호출 / EOS·finite 순서 | 5 PASS | 신규 고유 검사 |
| 실제 tokenizer 통합 | 입력 8/24, 출력 복호화 463, mask 합성 경로 10 | 모델 성능 표본 아님 |

검사 이름과 수는 [실제 로그에서 집계](results/cpu_preparation/test_inventory.json)했습니다. 기존 manifest는 457개 원 파일만 복원한 독립 view에서 원 명령으로 검증했습니다. 새 후속 파일 때문에 생기는 closed-set 차이를 허용하도록 원 검사를 수정하지 않았습니다.

최종 CPU 명령 12개의 subprocess wall 합은 **19.252초**이며 여기에는 마지막 파일 hash 재검사 **12.185초**가 포함됩니다. 앞선 별도 로컬 파일 hash 검사 **12.109초**도 보존했습니다. 이 수치를 전체 준비 시간 또는 GPU busy time으로 쓰지 않습니다. 모델 snapshot 12개 파일(가중치 shard 3개 포함), RTN/D16 각 252개 파일의 bytes/hash가 두 검사에서 같고 기존 로컬 manifest와도 일치했습니다. RTN payload 합 2,043,746,208B, D16 합 2,274,662,304B입니다. 실제 모델 로딩·payload deserialize는 하지 않았습니다.

준비 중 두 이력을 보존했습니다.

- 최초 의존성 탐색에서 lm-format-enforcer의 optional import가 Torch를 불러왔습니다. 모델 객체/forward 및 명시적 GPU API 호출은 없었고, framework 내부 초기화는 계측하지 않았습니다. 이후 optional library를 불러오기 전 import guard를 설치했으며 정식 CLI·CPU 검사에는 Torch/Transformers가 로드되지 않았습니다. [기록](results/cpu_preparation/attempts/dependency_probe.json)
- 첫 CPU 문법 probe는 wrapper의 free-string shortcut 전달 누락으로 느려져 **이 작업이 만든 해당 프로세스만** 247.570초에 중단했습니다(exit 143). shortcut/cache 전달을 수정한 소스를 동결하고 검사를 통과했습니다. 다른 프로세스는 종료하지 않았습니다. [실패 이력](results/cpu_preparation/attempts/grammar_attempt01_interrupted.json)

<a id="새추론전gate"></a>
## 8. 새 추론 전 gate와 현재 판정

| 영역 | 상태 |
|---|---|
| 기존 실패 재생·공통 checkpoint·CPU 요청 계약 | 확인됨 |
| 로컬 모델/RTN/D16 파일 전체 identity | LOCAL_IDENTITY 확인; 공식 외부 digest 비교 아님 |
| C0 24개 입력 hash | CPU 재토큰화 확인 |
| 문법 지원 | 실제 tokenizer CPU 경로 확인; GPU processor 연결 미실행 |
| 실제 loaded module coverage·small weights·beta_inverse/W_pre·runtime flags | 새 모델을 불러오지 않았으므로 NOT_RUN |
| 새 C0 output_token_ids 24개 재현 | 새 추론 필요, NOT_RUN |
| 실제 복구·추가 비용·원인 효과 | NOT_RUN / NOT_ESTABLISHED |
| GPU 실행 | 이번 범위에서 CLOSED_SCOPE |

향후에는 revision `cdbee75f17c01a7cc42f958dc650907174af0554`, tokenizer/template/generation config, RTN/D16 각 252개 hash·bytes·실제 module coverage, 원 beta_inverse/W_pre 경로와 small weights, BF16/SDPA/KV/TF32 설정을 확인해야 합니다. 새 C0 출력 ID가 원 기록과 다르면 먼저 환경·실행 차이를 분리하고 개입 효과의 원인 해석을 보류합니다. GPU 연결에서 FiniteBudget → grammar mask의 순서, 첫 응답에만 개입, 원 stop token과 전체 endpoint도 검증해야 합니다.

`run`은 이 CPU 단계에서 exit 3으로 거절됩니다. 승인 우회 옵션이 없습니다. 현재 제공하는 것은 동작하는 CPU 상태 기계·request builder·문법 마스크와 dry-run이며, 실제 모델 adapter를 연결한 GPU 실행기는 다음 별도 승인 범위입니다.

<a id="출처와재현"></a>
## 9. 출처·원형 보존·재현

분석 Git base와 새 소스 SHA는 다른 식별자입니다.

- Git base: `51385286063eb55e791f13a0b97f1e341ce6a879`
- CPU source/spec/data freeze: `03829dc8549ad45a0949c7f372568ff2fef76e02f4bbb1d5e22131f61e4f4c80`
- [파일별 source freeze](provenance/code_freeze.json), [사전 등록 시각·입력 hash](provenance/preregistration.json)
- [검사 명령·exit·wall](results/cpu_preparation/command_receipts.json), [원자료 재계산](results/cpu_preparation/independent_record_audit.json)
- [보존 확인](results/cpu_preparation/preservation.json), [검증 판정](results/cpu_preparation/verification.json), [남은 gate](results/cpu_preparation/gates.json)
- [실제로 실행한 CPU 재현 명령](REPRODUCE.ko.md), [재사용·라이선스](NOTICE.md)

원 P1/A1의 코드·입력·응답·사건·scorer·parser·protocol·manifest, Case001–014, 공통 package와 사이트 파일을 변경하지 않았습니다. 모델/환경은 읽기만 했으며 설치·업그레이드·fitting·재양자화·학습·모델 forward·게시를 수행하지 않았습니다. 전체 환경 파일의 byte hash 비교를 했다는 뜻은 아닙니다. 원 모델·INT4 payload·대형 logits를 새 후속 폴더에 복사하지 않았습니다.

Codex가 구현과 별도 CPU 계산을 보조했습니다. 이번 감사는 저장된 응답과 도구 실행의 재현이며 독립 GPU 추론이나 사람의 외부 심사가 아닙니다.
