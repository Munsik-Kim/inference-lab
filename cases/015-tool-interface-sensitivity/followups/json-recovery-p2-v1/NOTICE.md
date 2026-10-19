# 재사용과 출처

분석 대상은 inference-lab `51385286063eb55e791f13a0b97f1e341ce6a879`의 Case015 A1입니다. 새 코드가 직접 재사용하는 원 Python 파일, fixture/schema, raw records와 identity 문서는 [source_manifest.json](provenance/source_manifest.json)에 byte/hash로 고정했습니다. 원 Case015 457개 파일 전체는 [original_case_files.json](provenance/original_case_files.json)에 따로 고정했습니다.

원 `order_regression.common/registry/fixture/scoring/audit/runner`의 파서·history·실제 fixture·원 scorer·독립 endpoint를 재사용합니다. 기존 파일은 수정하지 않습니다. 새로운 부분은 공통 checkpoint 재구성, allowlist request builder, 한 번의 paired retry, 예산을 이어받는 CPU session, schema/envelope/token mask, tokenization 검사와 후속 전용 CLI입니다. GPU model adapter는 연결하지 않았습니다.

원 [Case015 NOTICE](../../NOTICE.md)와 저장소 [Apache-2.0 LICENSE](../../../../LICENSE)를 유지합니다. 모델 및 RTN/D16 payload는 복사·재배포하지 않습니다. 논문·CIPR 학습 또는 새 양자화법을 구현한 작업이 아닙니다.

선택적인 CPU 문법 검사는 설치되어 있던 lm-format-enforcer 0.11.3의 공개 CharacterLevelParser/JsonSchemaParser/TokenEnforcer API를 사용합니다. 해당 배포의 MIT LICENSE를 확인했습니다. tokenizers 0.23.2의 설치 metadata는 Apache 분류, Jinja2 3.1.6은 BSD LICENSE를 표시합니다. 라이브러리 소스·binary는 이 후속 폴더에 묶지 않습니다. 로컬 Transformers의 chat-template `tojson` filter 의미를 읽어 Jinja 렌더를 맞췄으며 실제 24개 입력 hash 일치로 해당 snapshot에 대한 대응을 확인했습니다. Transformers 모델 실행이나 일반 template 호환성을 재현한 것은 아닙니다.

Codex가 구현·검사·문서화를 보조했습니다. 별도 CPU 계산은 원 사건과 scalar/문자열 재검증이며 사람의 외부 심사나 독립 GPU 재현으로 표시하지 않습니다.
