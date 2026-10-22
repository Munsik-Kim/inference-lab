# Case018 P1 — 실제 서버 연결과 DEV24 실행 계약

과학 계약은 P0의 [protocol.json](specs/protocol.json) 그대로입니다. 수량·checkpoint·오류·보호·continuation·Q/W·R1 조건을 바꾸지 않았습니다. P0 원형은 외부 `case018-p1-evidence-20261022-101706/p0-original/`에 exact bytes로 보존했습니다. 이 문서는 [P1 실행 명세](specs/p1_execution.json)의 설명이며 모델 결과 보고서는 사전등록이 아닙니다.

## 고정 비교

Qwen/Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, BF16 weights. 기존 vLLM0.32.0/Torch2.13.0/FlashInfer0.7.0.post1/xgrammar0.2.8/Transformers5.19.0 환경을 그대로 사용합니다. 실제 설치 METADATA와 작은 config/tokenizer/index 파일 hash, weight 파일 목록·크기 및 역사적 hash를 대조했습니다. weight 전체를 새로 hash했다는 뜻은 아닙니다. 로딩 경로·실제 파라미터 dtype/샘플·cache는 worker에서 확인합니다.

K0=BF16 KV, K1=native NVFP4 KV. R0=형식에 맞는 final을 받고 사후 채점, R1=현재 상태의 성공 검증을 관측하기 전 final을 최대2 feedback 후 거절합니다. 6개 DEV task×4조건=24 episode입니다. 자연 모델 오류가 아니라 동일한 CPU 주입 오류에서 복구합니다. 성공률 확대 기준은 `null`이며 결과가 나빠도 유효하게 완료된 실행은 보존합니다.

공통 TP1, GRAPH, FlashInfer, max_model_len28672, 고정 목표4GiB KV pool, max_num_seqs4, batched tokens2048, chunked prefill ON, async scheduling/prefix caching OFF, episode C1입니다. TF32 OFF/BF16 reduction ON을 관측합니다. 커널 경로 차이도 포함한 serving 구성 비교이며 반올림만의 인과 효과가 아닙니다.

## 모델 입력과 실제 토큰

P0의 공개 payload에서 messages와 공개 도구 설명만 사용합니다. native tools/Hermes/reasoning parser 없이 tool 선언 JSON을 기존 system 내용 뒤에 같은 방식으로 붙입니다. 네 조건 모두 같은 wire messages, `structured_outputs.json`의 원 P0 공개 schema, host strict parser를 사용합니다. ID·수량 정답·다음 행동·policy/KV 이름은 grammar/prompt에 추가하지 않습니다.

실제 pinned tokenizer/chat template의 rendered prompt와 IDs/hash를 매번 저장하고 서버 prompt text/IDs/usage와 일치시킵니다. application content가 parser 입력이며, special token을 유지한 decode와 원 token IDs는 별도입니다. EOS를 비용에서 빼지 않습니다. 실제 토큰은 P0의 UTF8/4 simulated units와 다른 필드에 저장합니다.

Greedy temperature0/top_p1/n1/seed61022007, enable_thinking=False, ignore_eos=False입니다. 요청 cap은 min(512, 잔여8192 output, context24576−실제 prompt tokens)입니다. 잘라내기·padding·repair·retry/fallback은 없습니다. 부분 stream·DONE/IDs/usage 누락은 불완전 응답이며 도구를 실행하지 않습니다.

## 기술 gate와 순서

TECH는 별도 식별자의 공개 schema/transport/cache 입력입니다. DEV 과제의 성공률을 미리 보는 입력이 아닙니다.

- G1: 실제 로컬 모델/tokenizer/template·BF16 weights, 동일 task의 최초 payload/상태/오류/실제 IDs 대응.
- G2: 실제 tokenizer+xgrammar의 API8/type/key/범위/EOS/여분 JSON 경계 검사. 실제 worker의 원 `apply_grammar_bitmask`를 호출하는 읽기 관측 wrapper로 호출·차단·남은 유한 logit을 기록합니다. 올바른 JSON 하나를 생성했다는 사실만으로 mask를 인정하지 않습니다.
- G3: all36 실제 cache storage와 NVFP4 data/FP8 scale view·reader 경로·pool/slot 산술. 기술 전용 direct dummy forward에서 prefill8/decode1, graph NONE으로 관측 가능한 attention 출력 유한성을 확인합니다. 이는 생성 API와 별도이고 main serving GRAPH를 바꾸지 않습니다. 짧은 실제 TECH 요청의 profiler kernel 이름도 별도 보존합니다.
- G4: raw SSE/IDs/usage, 실제 token cap와 deadline, stream close 뒤 소유 engine running/waiting gauge가 연속2회0인 drain. 존재하지 않는 abort API는 사용하지 않습니다. P0 정답/가드 script를 CPU에서 실제 tokenizer로 길이 검산하며 모델 성공으로 세지 않습니다.

기술 서버 순서는 K1→K0입니다. 양쪽 기술 gate 완료 뒤 DEV는 P0에 등록한 K0→K1을 유지하며 K0 서버를 바로 재사용합니다. block 안 task/R0/R1은 seed61022011의 단일 RNG로 block별 shuffle을 사전에 고정했습니다. K1은 DEV에서 다시 로드할 때 실제 cache/config/파라미터 샘플을 대조합니다. 조건별 다른 pool·eager 전환·다른 서버 이용은 없습니다.

기술 생성은 최대32, DEV는 최대384, 합계416 requests입니다. 등록된 기술 계획은 KV별 정상 짧은 요청1회와 timeout/drain 요청1회입니다. 직접 forward2회/KV와 엔진 내부 load/compile/warmup은 API 생성 요청과 구분합니다. 실패·취소도 journal과 공통 worker wall에 포함합니다.

첫 DEV 출력 전 `execution_freeze.json`에 source/spec/input/dependency/runtime identity, 24개 예정 key와 실제 TECH 관측 수를 저장합니다. `dev_outputs_observed=0`은 TECH 미관측이라는 뜻이 아닙니다. 이후 source가 바뀌면 같은 run으로 이어붙이지 않습니다.

## 실행·종료·비용

각 episode는 새 error fixture에서 시작합니다. 생성 전 available_result_event_ids는 고정하며 결과를 소급 관측으로 넣지 않습니다. 도구는 원 인자를 적용합니다. terminal snapshot을 고정한 뒤 F1/F2를 독립 복제에서 실행합니다. 후속 결과를 모델에 다시 주지 않습니다.

상한 requests16/parsed attempts16, response512/episode8192 tokens, request180초/episode900초는 유지합니다. 마지막 final·거절·오류도 공통 예산을 소모합니다. 요청 timeout은 연결/대기/전체 stream과 episode·worker 잔여를 포함합니다. timeout 후 drain이 불명확하면 다음 cell을 시작하지 않습니다. 모델 오답·계약/보호·예산 실패는 정상 관측이며 자동 재생성하지 않습니다.

공통 GPU worker3600초에서 기술·load/compile·실패·I/O·소유 서버 종료를 차감합니다. episode wall은 초기 fixture 준비부터 terminal 고정까지이며 CPU continuation/audit와 worker wall은 별도입니다. 겹치는 시간을 합쳐 총비용으로 부르지 않습니다. 입력 tokens는 매 턴 history 재전송 합계, output은 반환 IDs 합계입니다. 미관측은 null/누락 사유이며0으로 채우지 않습니다. 비용/Q 또는 비용/W는 실패 포함 전체 비용을 해당 성공수로 나눕니다. 성공0은 undefined입니다.

전체 raw를 CPU에서 별도 SSE assembler와 독립 ReferenceState/endpoint로 재구성합니다. 실제 tokenizer 재인코딩·grammar token-path 검사는 기존 GPU환경의 별도 CPU 경로이며 새 GPU 재현이 아닙니다. DEV6의 paired Q/W 개수와 family별 결과는 기술 통계입니다. 두 continuation을 독립 모델 표본·모든 미래 동등성·자율 후속 능력으로 세지 않습니다.

P2/HOLDOUT 생성·봉인·관측·P3/UI·원격 게시를 실행하는 단계가 아닙니다. 남은 worker 예산과 기술 준비 상태만 보고하며 P2 조건이 갖춰져도 자동 진입하지 않습니다.

## 실행 후 확인한 관측 구현과 기록 읽기

이 절은 실행 뒤의 설명이며 새 사전등록이나 과거 receipt의 수정이 아닙니다. 실제 기본 runner는 `vllm.v1.worker.gpu.model_runner.GPUModelRunner`입니다. legacy runner로 전환하지 않았습니다. G2는 이 V2 runner의 `structured_outputs_worker.apply_grammar_bitmask`를 관측했습니다. G3는 실제 `FlashInferImpl.forward`의 반환/출력 버퍼를 읽었고, 기술 전용 forward에서만 공식 `skip_compiled=True`, `CUDAGraphMode.NONE`, `is_profile=True`를 사용했습니다. 바인딩은 finally에서 복원하며 DEV의 GRAPH 설정은 유지했습니다. 관측 경로를 연결하는 과정의 실패와 source별 기록은 `results/p1_r1`부터 `p1_r4`, 별도 amendment01–04에 보존했습니다. 이 네 시도에는 DEV 출력이 없습니다.

최종 실행 소스·명세는 `results/p1_r5/execution_freeze.json`입니다. 문법은 원 `specs/grammar.json`의 **속성 순서를 포함한 bytes**에서 컴파일했습니다. freeze JSON은 표시 직렬화에서 key를 정렬하므로, 그 안의 schema object를 다시 컴파일하면 xgrammar가 허용하는 속성 출력 순서가 달라질 수 있습니다. 원 schema 파일 hash와 실행 전 compiled grammar를 함께 대조해야 합니다. 실제 후처리 검사의 이 오류와 수정은 새 CPU 검사 receipt로 남겼고 실험 소스·응답·점수는 바꾸지 않았습니다.

TECH 취소 두 건은 HTTP 200 뒤 SSE/usage를 받기 전에 timeout됐습니다. 원 client의 `output_tokens=0`은 수신 ID가 없다는 기록입니다. 엔진의 실제 생성 비용이 0이라는 뜻이 아니며, 파생 비용 표에서 해당 token 비용은 null/미관측으로 둡니다. 연결을 닫은 뒤 실제 running/waiting gauge의 연속 두 번 0으로 drain을 확인했습니다.
