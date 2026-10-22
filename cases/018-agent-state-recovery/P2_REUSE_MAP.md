# P2 재사용·변경 경계

P1 기존 파일은 수정하지 않습니다. `p2_contract`, `p2_prepare`, `p2`, `p2_audit`, `p2_analysis`와 P2 전용 test/spec/doc를 추가합니다. 원 P1 source39파일·원자료343파일 및 기존606파일의 대응은 외부 start inventory와 최종 보존 receipt로 확인합니다. 기존 P1 identity와 새 P2 working source identity는 별개입니다.

| 그대로 호출 | P2 역할 |
|---|---|
| generator.generate/check_p2_gate | 등록된 seed·규칙으로 한 번 생성. P2 wrapper가 P1 audit/보존 확인을 추가 |
| public.initial_payload/grammar/definitions, contracts.parse | 원 prompt·도구·schema·strict parser. 변경/repair/정답 누출 없음 |
| runner.run_episode, world.World, scoring.assess/continuations/fresh_validation | 같은 API·가드·예산·state·최종/후속 endpoint |
| live_client.LiveClient.prepare, request_body, Case017 chat/tokenize | 같은 실제 tokenizer/template·content JSON·raw SSE/IDs/usage |
| server.create_server/rpc/drain, inspector.RecoveryInspector | 같은 소유권·GRAPH/actual KV/finite/mask 관측 |
| audit.audit_episode/ReferenceState, transport_audit.verify | 같은 독립 상태/관측/예산/SSE 재생 |

P2LiveClient의 respond는 P1의 작은 바깥 회계 부분을 명시적으로 이식했습니다. 원 DEV384/TECH32 상수를 HOLDOUT1536/TECH8로 바꾸고 P2 journal에 분류합니다. 원 request body·prepare·chat·raw 저장/입력 ID 검사는 그대로입니다. per-episode 예산이나 정책 변경이 아니며, 이 부분을 설정만 추가했다고 표현하지 않습니다.

P2 auditor는 원 audit_episode를 호출하되 DEV24 전용 source/coverage wrapper 대신 새96key·input seal·누적 비용/미관측 검사를 추가합니다. 별도 모델 없는 tokencheck는 보존된 P1 retokenize_saved_r2의 원 schema bytes/속성 순서·compiled grammar 대조를 재사용합니다. 라이브 호출은 하지 않습니다. source/science와 실행 뒤 문서/주석 자료를 분리합니다.

공유하는 strict parser·serialization·공개 입력 builder/wire rendering까지 독립적으로 검증했다는 뜻은 아닙니다. 상태 전이·보호·freshness·후속 절차·raw SSE 조립은 원 별도 경로입니다. 새 분석기는 family-stratified paired task bootstrap과 P2 CSV만 추가합니다. CPU 가짜 응답/정답 script는 SYNTHETIC_CONTRACT_TEST이며 모델 결과에 합치지 않습니다.

P2 audit wrapper는 split 분리 확인을 위해 읽기 전용 DEV loader와 scenario fingerprint/gate metadata도 재사용합니다. 이 import 경로에 generator/world 모듈이 연결되지만, endpoint 재계산은 `ReferenceState` 경로이며 World/scoring/generator의 계산 결과나 저장 bool을 답으로 사용하지 않습니다. CPU auditor 전체 프로세스가 그 모듈들을 전혀 import하지 않는다고 표현하지 않습니다.

Qwen/vLLM/FlashInfer/xgrammar와 실제 NVFP4 커널은 upstream입니다. 직접 작성한 부분은 Case018의 평가 환경·가드·관측/회계·단계 연결·독립 검산/분석입니다. Codex가 구현과 문서 작업을 지원했습니다. 기존 저장소 라이선스와 모델/외부 소프트웨어의 별도 라이선스를 구분합니다.
