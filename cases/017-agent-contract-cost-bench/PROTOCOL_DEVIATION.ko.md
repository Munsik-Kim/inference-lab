# 확대 기준 누락과 P2의 해석

확대 기준 이탈: 설정 점검용 DEV에서 BF16 KV+C0의 완주는 5/8로 원안의 6/8 기준에 미달했습니다. 당시 로컬 계획과 실행 코드가 이 성공 조건을 누락하고 완료·동결·봉인만 검사해 P2를 진행했습니다. 별도 면제 승인 기록은 확인하지 못했습니다. 이 경위를 공개하며, P2는 해당 기준을 통과한 확인시험이 아닌 신규 합성 과제의 탐색적 측정으로 제시합니다.

이 설명은 원 점수를 수정하거나 폐기하지 않습니다. `COMPLETE`는 등록된 실행 행렬의 완료를 뜻하며 ‘준비 기준 통과’나 ‘gate를 충족한 확인시험’이라는 뜻이 아닙니다. 측정 재계산, 출력 전 입력 봉인 근거, 확대 기준 준수는 별개로 판단합니다.

## 원안과 실제 local 실행

| 항목 | 근거와 판단 |
| --- | --- |
| 원안 | 이번에 전달받은 ChatGPT 측 보존 발췌는 BF16 KV+C0 ≥6/8, 실제 모델 실패로 미달하면 HOLDOUT 자동 실행 중단을 명시합니다. |
| local 계획 | 원 `PLAN.ko.txt` 6–8행은 “사용자 지시는 0~5절까지 전달되었다”고 기록하며, 13행은 P1 완료·코드 동결 뒤 P2를 진행한다고 적었습니다. |
| local 등록 | `specs/protocol.json`, version `v1.1-local-registration-r1`에는 성공수 6/8 조건이 없습니다. |
| 실제 코드 | `src/agent_contract_cost/experiment.py` 134–149행 `check_p2_gate`와 417–439행 `seal-holdout`은 P1 완료64개·freeze·seal을 확인하며 성공수를 읽지 않습니다. |
| DEV 실패3개 | `DEV-B00/B01/B02_K0_C0`: 첫 응답에 조회 JSON 두 개를 생성해 엄격한 단일 object 계약에서 PARSE_ERROR. 실제 사건0개, 상태 version0, 예산 소진 아님. |
| 면제 근거 | 별도 사용자 면제 승인·결정문은 조사한 local 계획·provenance·attempt 기록에서 찾지 못했습니다. Codex의 local 진행을 사용자 승인으로 바꾸지 않습니다. |
| 미확인 | local Codex가 실험 전에 원안의 어디까지 수신했는지, 전달/등록 중 어느 단계에서 조건이 빠졌는지는 미확인입니다. |

[원 local PLAN](PLAN.ko.txt) · [동결 protocol](specs/protocol.json) · [동결 실행 코드](src/agent_contract_cost/experiment.py) · [보완 조사 사본](publication/reviews/CASE017_REVIEW_ADDENDUM_KO.sanitized.txt)

## 시점과 원형 보존

아래는 모두 2026-10-20 UTC의 local 자기기록입니다. 외부 독립 시각 인증으로 표현하지 않습니다.

| 순서 | 시각 | 원 기록 |
| --- | --- | --- |
| protocol 등록 | 07:03:28.201933 | `specs/protocol.json` |
| P1 freeze | 07:21:27.507465 | `provenance/execution_freeze_p1.json` |
| P1 완료 | 07:25:17.672620 | `results/p1_r1/completion.json` |
| HOLDOUT input manifest | 07:26:44.012959 | `provenance/holdout_input_manifest.json` |
| HOLDOUT seal, outputs_observed=0 | 07:26:44.040365 | `provenance/holdout_seal.json` |
| P2 freeze | 07:26:59.253593 | `provenance/execution_freeze_p2.json` |
| P2 worker 시작 | 07:27:24.042065 | `provenance/worker_journal.jsonl` |
| 첫 P2 cell 시작 | 07:27:42.423255 | `results/p2_r1/attempts.jsonl` |
| 첫 SSE created | 07:27:42Z, 초 단위 | `HOLDOUT-B05_K1_N`의 첫 응답 |

manifest 시간은 생성 시작 시각이 아닌 생성 후 기록 시각이며, SSE created는 정확한 client 수신 시각이 아닙니다. 기록상 생성·봉인·P2 freeze는 첫 P2 모델 응답보다 앞섭니다. P1/P2 freeze 공통 **35개 파일은 byte/hash 동일**하고, P2는 HOLDOUT 입력2개와 seal만 추가했습니다. 원 source·protocol에 6/8을 사후 삽입하지 않았습니다.

## 이번에 추가로 전달된 원안 근거

출처는 `Case017_Fresh_20261020.txt`, version1의 제7절 발췌입니다. 사용자가 제시한 ChatGPT 측 보존 파일 생성 기록은 2026-10-20 15:42:13 KST, 재확인·이번 전달은 2026-10-21입니다. [전달 발췌](publication/reviews/gate_excerpt_received_20261021.ko.txt)를 별도 보존했습니다.

이는 원안에 6/8 기준이 있었다는 근거이며, local Codex가 실험 전에 전문을 수신했다는 증거는 아닙니다. 보완본의 “원문 전문 미확보”는 당시 local 조사 범위를 설명하므로 원 보완본은 고치지 않았습니다. 공개 보완 사본은 개인 경로만 표시용 placeholder로 바꿨으며 [변환 receipt](publication/reviews/addendum_copy_receipt.json)에 원본·사본 hash를 분리했습니다.

## 공개 판정

- 유지: 원 P1/P2 점수, 256개 전체 원응답 재생·별도 endpoint·paired 산술, 기록상 출력 전 HOLDOUT 봉인과 공통 source 일치.
- 공개할 이탈: DEV 5/8, 확대 기준 6/8 미달, local 성공 gate 누락, 별도 사용자 면제 승인 기록 미발견.
- 해석 범위: 신규 합성 과제의 탐색적 정책·KV 구성 비교. 일반 업무 확인시험·비열등성·동등성 인증으로 쓰지 않습니다.

이 문서는 2026-10-21의 설명 보완입니다. 과거 시점의 새 승인서·수정 사전 등록이 아니며 새 실험을 지시하지 않습니다.
