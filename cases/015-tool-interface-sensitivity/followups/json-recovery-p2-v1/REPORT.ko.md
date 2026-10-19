# Case015 P2 — 같은 정보를 받은 상태에서 JSON 호출 실패 복구하기

**D16의 원래 JSON 실패 8/8이 새 추론에서도 그대로 재현됐고, 첫 응답에 도구 계약 문법을 적용한 G는 8/8에서 실제 작업 완료를 회복했습니다.** 중립 재시도 N과 오류 설명 E는 각각 0/8로 완료하지 못했습니다. N의 1건은 올바른 금액 저장까지 진행했으나 검증 예산이 남지 않았습니다.

[한눈에 보기](#overview) · [문제](#question) · [고정 조건](#contract) · [실험](#experiment) · [결과](#results) · [실패와 복구](#errors) · [비용](#costs) · [결론](#conclusion) · [재현·출처](#reproduce)

<a id="overview"></a>
## 1. 한눈에 보기

| 모델 / Model | C0 | G | N | E |
|---|---|---|---|---|
| BF16 | 7/8 | 7/8 | NOT_APPLICABLE | NOT_APPLICABLE |
| RTN | 8/8 | 8/8 | NOT_APPLICABLE | NOT_APPLICABLE |
| D16 | 0/8 | 8/8 | 0/8 | 0/8 |


분모 8은 **이미 실패 분석에 사용한 합성 구매 기록 4개 × API 이름 표기 2개**의 checkpoint입니다. 서로 독립인 8개 업무나 새 end-to-end 성공률이 아닙니다. BF16/RTN에는 새 C0의 malformed-only 오류가 없어 N/E가 적용되지 않았습니다.

주 판정은 원 `VERIFIED_WORKFLOW_SUCCESS`입니다. JSON을 읽을 수 있거나 파일 금액만 맞는 것으로 성공을 주지 않습니다. 실제 저장, 올바른 최신 validation 결과 수신, 엄격한 최종 답변과 보호 기록 보존이 모두 필요합니다.

<a id="question"></a>
## 2. 문제와 용어

A1은 **구매 기록 읽기 → 수량과 개당 가격 곱하기 → 합계를 파일에 저장하기 → 저장 결과 확인 → 최종 답변**의 합성 작업입니다. `order_id`는 구매 기록의 ID, `unit_price_minor`는 개당 가격, `total_minor`는 합계입니다. 금액은 정수로 취급합니다. JSON은 도구 이름과 입력값을 전달하는 텍스트 형식입니다.

여기서는 조회·곱셈 결과를 이미 받은 다음, 합계를 저장할 도구 호출을 작성하기 직전으로 돌아갑니다. 그 대화·이미 받은 결과·파일 상태·남은 예산을 함께 고정한 지점이 checkpoint입니다. prefix는 새 모델로 생성하지 않고 원 호출을 실제 fixture에 재생했습니다.

원 A1 전체 96 episode의 463 생성 요청·378 도구 시도·368 실제 사건은 CPU에서 재생했습니다. 선택한 D16 실패 8건은 43–48토큰, EOS151645 종료이고, 원 파일 version0·모델3/8·도구3/6이었습니다. 원 `BUDGET_EXHAUSTED` 표기는 초기 terminal 값 잔존이며 해석은 PARSE_ERROR입니다. 원 raw와 실패 판정은 바꾸지 않았습니다. 나머지 malformed slot2개는 정상 호출과 섞여 성공한 기존 사례이며 피드백 회복 사례가 아닙니다.

<a id="contract"></a>
## 3. 고정 조건과 개입

Qwen/Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`, 기존 BF16/RTN/D16을 사용했습니다. RTN/D16 각252개 저장 projection을 읽고 D16의 **FP32 beta_inverse 입력 변환 → BF16 W_pre** 경로를 유지했습니다. 모델 하나씩 상주시켰고, greedy·thinking=False·192토큰·batch1·SDPA·use_cache=True·턴 사이 cache 없음·logits_to_keep=1·TF32/reduced reduction=False입니다.

| 조건 | 바뀐 부분 | 바뀌지 않은 부분 |
|---|---|---|
| C0 | 없음 | 원 messages/tools·자유 생성·malformed-only 종료 |
| G | 최초 checkpoint 응답에만 실제 decoding grammar mask | 같은 messages/tools, 후속 생성은 원 정책 |
| N | 같은 새 C0 실패 뒤 user 중립 문장으로 1회 retry | 같은 실패 raw·도구 관측·fixture·시도 예산 |
| E | N에 PARSE_ERROR 설명만 추가 | N과 동일한 새 C0 실패와 관측 |

N: `No tool was executed. Retry the previous response.`

E: `PARSE_ERROR: Tool-call JSON could not be parsed. No tool was executed. Retry the previous response.`

G는 선언된 네 도구를 모두 허용하고 이름별 key/type만 제한합니다. ID는 자유 문자열, 금액·곱셈은 자유 정수, final의 validated는 true/false 모두 가능합니다. 정답 ID/금액 whitelist·const/enum, write 강제, 사후 JSON 수선은 없습니다. 여러 호출과 같은 응답의 write→validate도 허용합니다. 원 parser에 맞춰 완성 dict는 closing marker 없이 EOS 가능하고 미완성 JSON의 EOS는 차단합니다. 원 FiniteBudget이 **unmasked logits의 유한성·deadline을 먼저 검사**하고 그 뒤 mask의 의도된 -inf를 적용합니다.

N/E의 실패 raw는 해당 방법의 **새 C0**에서 왔습니다. feedback은 user 메시지이며 tool result가 아닙니다. 모델에 보내는 것은 messages/tools뿐이고 oracle·다른 방법의 후속 답·숨겨진 fixture·semantic ID는 넣지 않았습니다. C0/G의 전체 입력은 같고 N/E의 전체 입력은 feedback 때문에 다르므로 공통 관측·상태 hash와 intervention/request hash를 분리했습니다.

prefix 모델2/도구2를 차감한 상태에서 합계 모델8·도구6·요청192토큰·context24576·deadline900초를 지켰습니다. malformed marker도 도구 시도를 소비합니다. 정확한 과거 prefix 시간 대신 등록한 공통 보수적 차감(원 전체 episode wall의 방법별 최댓값)을 유지했습니다. 새 C0와 branch의 실제 active wall도 차감했습니다.

<a id="experiment"></a>
## 4. 실행과 재현 gate

첫 응답 전 실행 소스를 commit했습니다. 첫 attempt는 sandbox의 `/dev/dxg` 미노출로 CUDA 초기화에서 실패했고 **모델 요청0회**였습니다. 같은 Python의 장치 접근 실행에서 cuInit0을 확인한 뒤, 이전 비용을 차감하는 새 소스 commit으로 재개했습니다. 패키지·드라이버·모델은 바꾸지 않았습니다.

- 최초 소스/실패: `447bda9c6311678ea17780c100e7d9a03b12b3b6` / [보존 receipt](results/gpu_r1_infra/receipt.json).
- 실제 생성 소스: `828ac7358b653626bed7e0a609448e73d120f1bb` / [실행 코드](https://github.com/Munsik-Kim/inference-lab/blob/828ac7358b653626bed7e0a609448e73d120f1bb/cases/015-tool-interface-sensitivity/followups/json-recovery-p2-v1/execution/run.py).
- 실행 source hash: `0a9cd6e3c3d97c6da698ab9bc945bad533f7e826b1b816f2593a7d17bf9c8770` / [동결 파일](execution/source_freeze.json).
- [변경 사유](execution/amendment_01.json), [원 CPU 등록](specs/protocol.json), [추가 실행 등록](execution/spec.json).

**C0 24/24의 입력 해시와 최초 output token ID가 원 기록과 일치한 뒤에만 G/N/E를 실행했습니다.** 아래 표는 raw에서 재계산한 것입니다.

<details><summary>24개 입력·출력 재현 전체 표</summary>

| Model | API | Checkpoint | Input hash | Focal output IDs |
|---|---|---|---|---|
| BF16 | API_V1 | A1-00 | True | True |
| BF16 | API_V1 | A1-06 | True | True |
| BF16 | API_V1 | A1-08 | True | True |
| BF16 | API_V1 | A1-14 | True | True |
| BF16 | API_V2 | A1-00 | True | True |
| BF16 | API_V2 | A1-06 | True | True |
| BF16 | API_V2 | A1-08 | True | True |
| BF16 | API_V2 | A1-14 | True | True |
| D16 | API_V1 | A1-00 | True | True |
| D16 | API_V1 | A1-06 | True | True |
| D16 | API_V1 | A1-08 | True | True |
| D16 | API_V1 | A1-14 | True | True |
| D16 | API_V2 | A1-00 | True | True |
| D16 | API_V2 | A1-06 | True | True |
| D16 | API_V2 | A1-08 | True | True |
| D16 | API_V2 | A1-14 | True | True |
| RTN | API_V1 | A1-00 | True | True |
| RTN | API_V1 | A1-06 | True | True |
| RTN | API_V1 | A1-08 | True | True |
| RTN | API_V1 | A1-14 | True | True |
| RTN | API_V2 | A1-00 | True | True |
| RTN | API_V2 | A1-06 | True | True |
| RTN | API_V2 | A1-08 | True | True |
| RTN | API_V2 | A1-14 | True | True |


</details>

새 결과는 64개 관측 분기와 32개 NOT_APPLICABLE 조건으로 구성됩니다. F는 새 C0의 eligible malformed-only 수인 **8**로 실제 관측됐습니다. 결과가 나쁘다고 재생성하거나 과거 응답을 새 baseline으로 섞지 않았습니다.

<a id="results"></a>
## 5. 결과

다음 parse는 원 parser의 첫 유효 dict 채택입니다. 인자 열은 최초 진단 응답에서 **실제 사건으로 평가 가능한 호출**을 분모로 합니다. 파싱 불가 인자는 NOT_EVALUABLE이며, 호출이 없는 조건을 정답률100%로 채우지 않았습니다. N/E의 첫 retry가 read를 다시 요청한 경우도 parse 성공입니다. 이를 write 수선 성공으로 읽으면 안 됩니다.

| Model | API | Arm | Focal parse | Name / calls | Keys+types / calls | Values / evaluable calls | Workflow |
|---|---|---|---|---|---|---|---|
| BF16 | API_V1 | C0 | 4/4 | 4/4 | 4/4 | 4/4 | 3/4 |
| BF16 | API_V1 | G | 4/4 | 4/4 | 4/4 | 4/4 | 3/4 |
| BF16 | API_V1 | N | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE |
| BF16 | API_V1 | E | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE |
| BF16 | API_V2 | C0 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| BF16 | API_V2 | G | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| BF16 | API_V2 | N | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE |
| BF16 | API_V2 | E | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE |
| RTN | API_V1 | C0 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| RTN | API_V1 | G | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| RTN | API_V1 | N | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE |
| RTN | API_V1 | E | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE |
| RTN | API_V2 | C0 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| RTN | API_V2 | G | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| RTN | API_V2 | N | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE |
| RTN | API_V2 | E | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE |
| D16 | API_V1 | C0 | 0/4 | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE | 0/4 |
| D16 | API_V1 | G | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| D16 | API_V1 | N | 4/4 | 4/4 | 4/4 | 4/4 | 0/4 |
| D16 | API_V1 | E | 4/4 | 4/4 | 4/4 | 4/4 | 0/4 |
| D16 | API_V2 | C0 | 0/4 | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE | 0/4 |
| D16 | API_V2 | G | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| D16 | API_V2 | N | 4/4 | 6/6 | 6/6 | 6/6 | 0/4 |
| D16 | API_V2 | E | 4/4 | 4/4 | 4/4 | 4/4 | 0/4 |


### 같은 checkpoint의 paired 결과

| Model | Contrast | Both success | Lost | Recovered | Both fail | Unobserved |
|---|---|---|---|---|---|---|
| BF16 | C0 -> G | 7 | 0 | 0 | 1 | 0 |
| BF16 | N -> E | 0 | 0 | 0 | 0 | 0 |
| RTN | C0 -> G | 8 | 0 | 0 | 0 | 0 |
| RTN | N -> E | 0 | 0 | 0 | 0 | 0 |
| D16 | C0 -> G | 0 | 0 | 8 | 0 | 0 |
| D16 | N -> E | 0 | 0 | 0 | 8 | 0 |


D16의 C0→G는 8회 회복, 손실0입니다. N/E는 같은8개 기회에서 모두 실패했습니다. N/E가 같다는 이 관측을 동등성 검정으로 바꾸지 않았습니다. BF16/RTN에는 retry 기회가 없었습니다.

<a id="errors"></a>
## 6. 어디에서 회복하고 어디에서 다시 실패했나

- **G / D16:** 여덟 지점 모두 올바른 ID·정수 금액으로 실제 write를 실행하고 최신 validation을 받은 뒤 final까지 완료했습니다. 형식만 개선된 결과가 아닙니다.
- **N / D16:** 처음에는 모두 read를 다시 요청했습니다. 7건은 후속 write JSON에서 다시 실패했습니다. API_V2 A1-06은 read/multiply/write/validate를 한 응답에 요청했지만 남은 도구 예산3회로 앞 세 호출만 실행됐습니다. 올바른 금액은 저장됐으나 validation이 실행되지 않았고, 다음 validation 요청도 예산에 막혔습니다. 추가 read 자체는 허용된 대체 경로이며, 예산과 완료 계약 때문에 실패했습니다.
- **E / D16:** 여덟 지점 모두 다시 read를 요청했고 후속 write JSON에서 실패했습니다. 추가 오류 설명의 고유한 workflow 회복은 N 대비 관측되지 않았습니다.
- **BF16:** API_V1 A1-06에서 validation에 틀린 구매 ID를 사용한 실패가 C0/G에 남았습니다. 올바른 파일 상태만으로 성공을 주지 않았습니다. 최초 write 문법 제약이 후속 ID 오류까지 해결하지는 않았습니다.

원 parser는 slot의 첫 유효 dict를 찾습니다. G의 **후속 무제약 응답**에서도 추가 중괄호 등 꼬리가 있는3개 slot은 원 parser가 채택했습니다. N/E에도 그런 채택이 있습니다. parser를 완화하거나 모든 응답이 strict JSON이라고 주장하지 않았습니다. [원문과 사건 64개](results/gpu_r2/cells), [전이 JSON/CSV](results/gpu_r2/diagnostics/error_transitions.json).

<a id="costs"></a>
## 7. 비용과 검산

| Model | Arm | New requests | Focal | Continuation | New tokens |
|---|---|---|---|---|---|
| BF16 | C0 | 24 | 8 | 16 | 965 |
| BF16 | G | 24 | 8 | 16 | 965 |
| RTN | C0 | 24 | 8 | 16 | 960 |
| RTN | G | 24 | 8 | 16 | 960 |
| D16 | C0 | 8 | 8 | 0 | 358 |
| D16 | G | 24 | 8 | 16 | 948 |
| D16 | N | 23 | 8 | 15 | 945 |
| D16 | E | 24 | 8 | 16 | 885 |


새 생성 요청은 **175회 = 최초 진단64 + continuation111**, 출력은 **6,986토큰**입니다. smoke0, 생성 요청 실패0이며, 별도로 pre-generation CUDA load 실패1회를 보존했습니다. N/E가 공유한 C0는 고유 request ID로 한 번만 셌습니다. [장부](results/gpu_r2/costs_by_arm.json).

성공 attempt worker wall은 **519.837초**입니다. 이전 실패·장치 진단을 합한 측정 합은 **542.134초**, 과거 source-check의 미계측 오버헤드1초까지 보수적으로 차감한 예산 사용량은 **543.134/7,200초**입니다. load/hash/decode/readback·CPU 처리·I/O를 포함하며 GPU busy나 serving 속도가 아닙니다.

Torch 요청별 peak allocated **8,418,611,200B**, reserved **8,671,723,520B**입니다. 요청 끝 CUDA device used 최대 표본은 **10,158,145,536B**로 다른 GPU/Windows 사용도 포함합니다. 연속 NVML 계측과는 다릅니다.

- raw 응답 재파싱·새 fixture 실행·원 assess와 independent_endpoint 재계산: **64개 분기 /175개 고유 요청 PASS**.
- 독립 CPU tokenizer/template: **175개 input ID/hash와 output ID→raw/text 일치**.
- G 실제 tokenizer mask 재계산: **24개 focal /1,118개 생성 토큰 일치**, 후속 mask 없음.
- 기존 P1 27/A1 18/publication6 및 CPU준비20/문법5는 회귀로 구분합니다. 새 실행 연결 검사9개와 공개 보존 검사4개는 별도입니다. 최종 실행 결과는 [공개 검증 receipt](publication/verification/latest.json)에 있습니다.
- snapshot12개 파일과 RTN/D16 각252개 파일의 전체 bytes/hash가 실행 전후 일치했습니다. 실제 GPU module·W_pre·beta readback, unchanged parameter, 매 요청 parameter identity 검사도 통과했습니다. [모델 보존](results/gpu_r2/model_preservation.json).

공개 CPU 감사는 모델 logits·GPU 추론 자체를 독립 재현하지 않습니다. 직접 의존성은 실행 전에 동결했으며 전체 venv의 byte hash를 검사한 것은 아닙니다. 게시용 CPU dependency lock은 기존 설치본의 버전을 별도로 기록했습니다.

<a id="conclusion"></a>
## 8. 결론과 적용 범위

**복구 관측:** 이 D16 저장본과 네 선택된 fixture에서, 동일한 관측 뒤 최초 tool-call의 계약 기반 직렬화 제약은 정확한 인자와 workflow 완료까지 회복했습니다. 이는 출력 직렬화/계약 경계가 이 실패를 막는 데 유효한 개입 지점이라는 근거입니다. 내부 표현의 유일한 원인이나 모든 INT4의 문제를 입증한 결과는 아닙니다. RTN은 C0부터8/8이었습니다.

**남은 문제:** N은 일부 실행만 회복했고 E의 추가 설명은 N보다 완료를 늘리지 못했습니다. 후속 무제약 write의 표현 오류, 중복 행동의 예산 비용, BF16의 validation ID 오류가 남았습니다. 기존 자유 생성 실패는 그대로 유지합니다.

| 범위 | 판정 |
|---|---|
| 같은 관측의 최초 출력 재현 | 24/24 확인 |
| 선택된 D16 checkpoint의 G workflow 회복 | 8/8 관측 |
| N/E의 추가 workflow 회복 | 각각0/8; N은 파일 상태1/8 부분 회복 |
| 새로운 업무·held-out·모델/quantizer 일반화 | 미확립 |
| 실제 INT4 kernel·serving·속도·VRAM 이득 | 미평가; dense BF16 품질 경로 |
| 기존 P1/A1 연구 | 원 판정·raw·source/manifest 보존; P2와 분모를 합치지 않음 |
| 원격 완료 범위 | branch push + PR; main 병합·Pages 배포 제외 |

<a id="reproduce"></a>
## 9. 재현·원형·출처

[CPU 재현](REPRODUCE.ko.md) · [English report](REPORT.md) · [기계 판독 decision](results/gpu_r2/decision.json) · [원자료 표 생성 코드](publication/derive.py) · [공개 manifest](publication/PUBLIC_MANIFEST.json).

기존 Case015 안내6개와 P2 준비 안내4개만 교체/연결했고 원 바이트는 [대응표](publication/preservation_map.json)에 보존했습니다. 원형 검사는 manifest의 옛 hash를 바꾸지 않고 정확한 경로로 복원한 closed view에서 실행합니다. 따라서 과거 CPU 준비 시점의 ‘7,499개 변경0’을 현재 안내 수정까지 포함한 주장으로 재사용하지 않습니다. [CPU 당시 원 보고서](publication/originals/cpu_preparation/REPORT.ko.md)는 byte 보존본이며 상대 링크는 복원 view 기준입니다.

개인 경로가 있는 metadata/log7개는 의미 label로 정리했고 raw 응답은 그대로입니다. [원본↔공개 hash 대응](publication/sanitization.json). 모델·.rpi4·snapshot·private runtime·환경·full logits는 포함하지 않았습니다. 출처와 라이선스는 [기존 NOTICE](../../NOTICE.md)와 [P2 NOTICE](NOTICE.md)를 유지합니다. Codex가 구현·실행·별도 CPU 재계산을 보조했으며 사람의 외부 심사나 저자 승인이 아닙니다.
