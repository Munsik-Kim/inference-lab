# Case014 인계

- 현재 P0 결과·코드·한영 보고서는 로컬 검토본이다. 원격 게시는 승인되지 않았다.
- P0 고유15개 테스트 PASS. 동일 seed MC의 반복을 별도 독립 표본으로 세지 않는다.
- P1은 `BLOCKED_STORAGE_IDENTITY`. D 원본 경로를 확인하지 못했고 기존 WSL 캐시의 읽기 전용 사용 가능 여부를 사용자에게 질문했다. 경과 시간을 동의로 처리하지 않았다.
- 모델 revision과 config/tokenizer hash, safetensors header는 확인했다. 모델 가중치 전체를 재hash하지 않았다. 모델 forward는0회이다.
- 입력은 private run root의 `inputs/`에 있다. 공개본에는 hash/offset/내용 grouping 근거만 포함했다. 경로 mapping은 ZIP 밖 private identity에 있다.
- P1을 실행할 때는 기존 source/tests를 다시 만들지 말고 `REPRODUCE.md`의 실제 CLI를 사용한다. 미완료 결과 출력 경로는 재사용하지 않는다.
- runner는 6표적×5강도를 모두 계획하며 CAL 선택 후 비CAL forward를 시작한다. 기존 RTN prefix codec과 모든 alpha의 artifact 저장·reload를 거친다. 실제 GPU 동작 검증은 아직 되지 않았다.
- 성공 결과가 생기면 `scripts/analyze.py`와 독립 `scripts/audit.py`로 재계산한다. `build_docs.py`는 P1 결과가 생긴 상태에서 오래된 미실행 문서를 생성하지 않도록 중단한다. 실제 관측으로 보고서·화면을 갱신해야 한다.
- P2/P3/신규 모델/설치/기존 코드 수정/stage/commit/push/PR/merge/Pages/Drive는 수행하지 않는다.
