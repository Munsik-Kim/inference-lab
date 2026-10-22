# Case018 P2 — 신규 HOLDOUT24의 동일 설정 비교

이 명세는 HOLDOUT 생성과 모델 출력 관측 전에 작성합니다. [P0 과학 protocol](specs/protocol.json)과 P1의 공개 system/tool/schema·strict parser·가드·상태 전이·채점·후속 절차·episode 예산을 그대로 사용합니다. [P2 실행 설정](specs/p2_execution.json)이 단계 연결과 회계를 추가하며 과거 등록 문서를 수정하지 않습니다. 원 P1 결과는 DEV6에서 BF16 Q6/W2, NVFP4 Q5/W0으로 R0/R1이 같았습니다. 이를 개선하도록 과제나 feedback을 고치지 않습니다.

## 생성·봉인

P0에 동결된 `generator.generate("HOLDOUT", gate)`를 seed61022002로 정확히 한 번 호출합니다. A/B/C 각8,24개이며 네 조건96개 key는 생성 전부터 정할 수 있습니다. 성공수 gate는 null입니다. P1 matrix24 완료·실제 기술 gate·source/원자료 보존·독립 audit·등록 generator를 확인해야 생성합니다. 등록 generator가 중복/부적합 task를 만들면 원 산출물을 남기고 중단하며 seed를 다시 고르지 않습니다.

DEV와 같은 세 family template를 공유합니다. ID를 제외한 family·원 수량 vector·사건 수량/phase의 scenario group이 DEV와 겹치지 않고 HOLDOUT 안에서도 유일한지 검사합니다. 서로 다른 회사/독립 template를 의미하지 않습니다. 과제마다 clean/no-recovery/scripted-correct 세 CPU control(총72)을 실행하고 독립 상태 계산으로 확인합니다. terminal/reference 후속 branch는 각각144개입니다. 별도24개 guard-twice script는 예산 여유용 합성 검사이며 모델 episode와 control 분모에 합치지 않습니다.

실제 pinned tokenizer로 최초24개 입력과 control/script history를 검사하고 원 순서 schema를 컴파일합니다. P1의 실행 전 compiled grammar와 일치해야 합니다. 원문을 자르거나 정답 script를 모델 입력에 넣지 않습니다. `pre_generation_freeze.json`, input manifest/seal에 source·plan·task/상태/오류/목표·공개 입력·실제 IDs·분석 계약을 고정합니다. seal의 `holdout_outputs_observed=0`과 이미 관측한 DEV24/응답180/TECH6을 함께 기록합니다. 로컬 시각은 외부 시각 인증이 아닙니다.

## 같은 모델·서버·정책

Qwen3-4B-Instruct-2507 revision cdbee75f17c01a7cc42f958dc650907174af0554, 양쪽 BF16 weights와 P1의 기존 vLLM0.32.0 환경만 사용합니다. TP1/GRAPH/FlashInfer, max-model-len28672, sequences4/batched2048, prefix/async OFF, episode C1, TF32 OFF를 유지합니다. 약4GiB pool도 동일합니다. runtime JSON에서 바뀌는 것은 소유 private output 경로와 비어 있는 port8197뿐입니다. 새 weight SHA 전체를 계산했다는 주장을 하지 않습니다.

기술 서버 K1→K0, HOLDOUT K0→K1 순서입니다. 마지막 K0를 재사용하고 K1 reload에도 실제 cache/36층·finite attention·원 grammar mask·raw IDs/usage/kernel을 fresh 확인합니다. 정상적으로는 TECH 생성3회와 direct forward6회, 상한 TECH8입니다. P1과 같은 timeout/drain 코드·runtime이면 취소 probe를 다시 만들지 않습니다. 새 worker의 확인이 실패하면 fallback 없이 중단합니다.

R0는 schema-valid final을 받고 사후 판정합니다. R1은 공개 scope/digest/version과 생성 전 관측에서 현재 성공 validation만 확인합니다. 첫 두 부적격 final에 고정 feedback, 세 번째 GUARD_LIMIT입니다. final 수량 자동 대조·추가 read 안내·oracle 보정은 넣지 않습니다. 요청/parsed action16, 응답512/episode8192 tokens, context24576, 요청180초/episode900초는 유지합니다. 문법은 모든 응답에 같고 정답 ID/수량/다음 행동을 강제하지 않습니다.

## 비용과 중단

HOLDOUT 생성 요청 상한1536, TECH8 포함1544입니다. 이전 worker439.692244164초를 차감한 공통 잔여3160.307755836초에서 load/compile·실패·I/O·후속·소유 종료를 셉니다. 과거 취소2건의 실제 생성 token 비용은 미관측으로 유지합니다. P2의 관측치를 더해도 누적 출력은 과거 관측 하한10744+새 관측 하한이며 전체 정확한 token 합계라고 쓰지 않습니다.

정상 업무 실패는 원 raw와 종료 상태를 보존해 다음 예정 key로 진행합니다. 모델/인프라 실패 자동 retry는 없습니다. 불완전 응답은 도구를 실행하지 않고 token 비용을 null로 남깁니다. timeout 뒤 drain이 불확실하면 중단합니다. 일부만 완료하면 예정/시작/종료/누락을 표시합니다. 이번 CLI는 기존 attempt root를 자동 재시작하지 않으며 완료 cell 재생성을 허용하지 않습니다.

## 채점·검산·분석

D/B/P/Fresh/C1/C2/Q/W는 P1과 같습니다. Q가 주 지표, W는 정확한 final과 관측까지 포함한 workflow입니다. rejected recovered:true 선언도 D이며 accepted final과 다릅니다. final이 없어도 종료 상태의 Q를 평가하고 Fresh는 마지막 생성 전 진단입니다. 두 후속 프로그램은 독립 복제에서 현재 revision을 공개 read로 얻고 고정 인자로 실행합니다. 모델의 자율 후속 행동이나 모든 미래 동등성 증명이 아닙니다.

원 ReferenceState와 별도 SSE assembler를 재사용해96개 endpoint·원문·예산·관측·이력을 다시 계산합니다. parser/serialization/공개 builder/wire rendering 공유 범위는 [P2 재사용 대응표](P2_REUSE_MAP.md)에 적습니다. 실제 tokenizer 재인코딩·원 순서 compiled grammar의 prefix/EOS 검사는 별도 CPU 경로입니다. sorted freeze JSON으로 문법을 재컴파일하지 않습니다.

셀 분모24/family8, DEV와 HOLDOUT을 별도로 표시합니다. 주 비교 KV별 Q(R1−R0), 보조 W/구성요소/KV 차이입니다. paired task family-stratified bootstrap10000/seed61022018에서 각 task의 네 조건을 함께 재표집합니다. percentile은 rank(n−1)p의 선형 보간입니다. pointwise 탐색95% 구간이며 폭0/차이0은 일반 동등성 증명이 아닙니다. 불완전 matrix에서는 CI를 계산하지 않습니다. 비용/Q·비용/W는 실패 포함 전체 비용을 나누며 성공0은 undefined입니다.

이번 단계는 P2에서 종료합니다. P3/UI·사이트·Git stage/commit/원격 게시를 실행하지 않습니다. 첫 HOLDOUT 뒤 작성하는 결과 보고서는 사전등록으로 부르지 않습니다.
