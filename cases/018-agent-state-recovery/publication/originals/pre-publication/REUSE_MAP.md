# 재사용과 독립성

P0에서 기존 Case017은 읽기 전용 개념 참조이며 다른 case를 import하거나 `sys.path`에 추가하지 않았습니다. P1의 명시적 live 경로만 아래의 읽기 전용 helper를 import합니다. 원 코드·parser·scorer·실측 자료는 수정하지 않습니다. 새 Case018 코드는 저장소의 [Apache-2.0 라이선스](../../LICENSE)를 따르며 외부 논문/상표/모델 라이선스는 별개입니다. Codex가 작성·검사 작업을 지원했습니다.

| 참조 | 실제 참고한 부분 | Case018 |
|---|---|---|
| Case017 `contracts.py`, `parse_action`, strict type/schema 검사 | 단일 object, 중복 key 거절, bool/int 분리 | 새 `contracts.py`의 작은 schema subset/strict parser. 함수 복사나 import 없음 |
| Case017 `runner.py`, `run_episode` | immutable generation 관측 목록, 요청/응답과 callable 회계 분리 | 새 CPU `runner.run_episode`; content-only envelope. 서버·tokenizer·vLLM client 재사용 없음 |
| Case017 `scoring.py`, `assess` 및 REPORT의 Fresh 설명 | 실행 직전 상태와 생성 전 관측 구분, 최신 digest/version | `scoring.fresh_validation`, R1 가드. Case017 점수·등록 endpoint를 가져오지 않음 |
| Case017 `audit.py`의 별도 상태 계산 원칙 | raw 재파싱 + 독립 endpoint | `audit.ReferenceState`, `audit_episode`: 새 재고/보상/원장 수식·continuation·지표 구현 |
| Case017 model/environment receipt·protocol·실행 freeze·CPU receipt | 기록된 Qwen revision/환경과 실행 관측 경계 | `provenance/recorded_model_identity.json`은 역사 기록의 사본 요약. 현재 artifact/모델 환경 검증·import·load가 아님 |

## P0의 Case018 내부 독립 audit 경계

runner와 auditor가 **공유**하는 것은 표준 JSON/canonical hash, 공개 tool/schema, strict parser 및 allowlist 입력 builder입니다. parser 결함까지 완전히 독립적으로 검증했다는 뜻은 아닙니다. 별도 strict 반례 테스트로 경계를 검사합니다.

auditor는 `world`, `runner`, `scoring`, `FakeClient`, generator를 import하지 않습니다. 원응답에서 도구를 다시 파싱하고, 별도 `ReferenceState` 전이/공개 validation/보호 규칙/복구·후속 상태/관측·예산/종료를 재계산합니다. checkpoint 생성과 주입 산술도 origin 및 사건에서 다시 확인합니다. 저장된 성공 bool/terminal label은 대조 대상으로만 씁니다. 새 root에 독립 재구성한 terminal 상태를 씁니다.

로컬 CPU 교차검산이며 외부 심사자·다른 머신 재현이 아닙니다. 손 계산20/5 예제와 DEV별 역산 검사가 generator/auditor가 같은 오식을 그대로 공유하는 위험을 줄입니다. 실제 tokenizer grammar backend·native KV 및 모델 inference는 공유/재사용 검증 범위에 포함되지 않습니다.

원형 보존 근거는 `provenance/reference_manifest.json` 및 `provenance/preservation_after.json`입니다. 선택한 원 연구 source/spec/핵심 receipt·보고서와 공개 참조의 before/after만 확인합니다. 모델 weights 전체 재해시·기존 전체 검사는 수행하지 않습니다.

## P1에서 실제 재사용한 읽기 전용 코드

| helper | P1 사용 | 변경 경계 |
|---|---|---|
| Case017 `client.chat`, `tokenize_messages` | lazy live adapter의 content-only raw SSE/반환 IDs 및 tokenizer/template | 원 파일 변경 없음. Case018은 공개 tool 설명을 system text로 직렬화하고 원 schema를 structured_outputs.json으로 보냄. 모델/seed/request cap은 Case018 명세 |
| Case017 `server_manager.OwnedServer` | 새 loopback port·소유 process group의 start/stop | subclass가 model name·Case018 worker extension·child-only source/cache/env를 지정. 다른 서비스에 attach/종료 안 함 |
| Case016 `CacheInspector`, `ProfileInspector`, `TechnicalInspector`와 package 초기화 5개 | all36 cache storage/scale/config·파라미터 샘플·짧은 kernel 이름 관측 | 원 코드 변경 없음. Case018 worker extension이 실제 V2 grammar mask와 `FlashInferImpl.forward` 출력 관측, 기술 direct forward를 추가. upstream 커널/모델 수정 없음 |

P1의 runtime helper 의존성과 원 파일 hash는 새 execution freeze에 별도 포함합니다. 서버는 이 source bytes를 사용합니다. P0의 23개 source bundle과 P0 결과는 외부 exact-byte snapshot으로 재생하며, 현재 확장된 source를 과거 원 실행으로 위장하지 않습니다.

P1 CPU auditor는 Case017 transport를 호출하지 않습니다. 별도 stdlib SSE assembler로 raw content/IDs/usage/DONE을 다시 조립하고, P0의 독립 `ReferenceState`와 예산·freshness·continuation을 재계산합니다. 공유 범위에 공개 정의의 wire rendering을 추가했으며, 모델/tokenizer/서버를 import하거나 요청하지 않습니다. 기존 runtime을 사용하는 실제 재토큰화 검사는 별도 CPU receipt입니다.

P1에서 실제 참조한 dependency는 execution freeze의 8개 파일입니다. Case017 세 파일은 현재 worktree의 공개 source, Case016 다섯 파일은 Case017 원 연구 worktree에 보존된 구현에서 읽었습니다. 기존 환경의 vLLM/FlashInfer/xgrammar/Transformers는 upstream 구현입니다. 새로 작성한 부분은 Case018 adapter·strict 단일 content JSON 연결·token/예산 회계·공개 freshness 가드·독립 재생과 DEV 분석입니다. NVFP4 알고리즘·커널·모델을 직접 개발했다는 뜻이 아닙니다.

P1 검산은 원 parser/공개 schema·serialization·입력 builder/wire rendering을 공유합니다. 상태 전이·보호·freshness·continuation·endpoint와 SSE assembler는 별도 경로입니다. 실제 재토큰화와 grammar prefix 검사는 기존 runtime을 사용한 로컬 CPU 검사이며, 외부 GPU 재현이나 제3자 검증이 아닙니다. 추가 검사기는 실행 뒤 작성한 별도 evidence source로 식별합니다.
