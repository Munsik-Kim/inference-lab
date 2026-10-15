# 실행 설계 확인사항

- Case014 번호는 비어 있었다. 기존 작업을 유지한 별도 worktree를 main `09194c3f2e2c991feff5b6abab0cf8ab81a07052`에서 만들었다.
- 논문 본문 HTML/PDF를 확보했다. 이전의 “본문 미확보”를 이번 상태로 계승하지 않는다.
- 데이터는 기존 PG19의 책 URL 단위 32/16/32문서와 기존 Pile ArXiv의 16개 content group이다. 과거 WikiText 진단을 재활용하지 않는다.
- Pile에는 원 논문 ID가 없어 article-disjoint라고 표시하지 않는다. 선택 window의 5gram 근접 중복은 검사했으나, 전체 문서의 근접 중복과 사전학습 비노출은 보장하지 않는다.
- 실제 입력 폭은 hidden_size1024와 다르다. o_proj2048, down_proj3072를 헤더에서 확인했다.
- D 원본을 기록에서 특정하지 못해 기존 WSL 캐시 사용 여부를 확인 중이다. SM120 실행 실패나 OOM이 아니다.
- 최초14개 테스트 뒤 요구된 독립 Torch/NumPy 분해 검산1개를 추가하고15개를 실행했다. MC 재실행에서 동일 seed별 경량 SSE를 저장했다. 반복 실행을 새 고유 테스트나 새 독립 seed로 세지 않는다.
- 기존 데이터·case·환경을 수정하지 않았다. 새 자료는 이 case와 외부 private scratch에만 저장한다.

## P1 재개와 실제 실행 — 2026-10-15

사용자가 기존 WSL snapshot의 읽기 전용 사용을 명시 승인하여 D 원본 확인 조건을 이번 P1에서 대체했다. 이전 중단·P0 freeze를 수정하지 않고 provenance/p1_r2_authorization.json에 승인 시각과 정책 예외를 기록했다.

30/30 핵심 조건과 후순위 연속 대조를 실제로 완료했다. quantizer·fit·scorer는 동일하며 wrapper/run identity, 누적 예산, 실행 수 기록과 smoke 경계 확인만 보강했다. 원래 준비본과 보고서·화면·closed manifest 전체69파일은 publication/original_pre_p1에 bytes 그대로 보존했다. 새 입력 추출·P0 MC 재실행은 없다.

원래 rough forward estimate3865와 실제 loop cardinality3845의 차이는 독립 auditor에 정정 기록했다. 핵심30조건과 명시된 controls는 모두 존재하며 누락 cell은 없다. 이것은 평가값을 보고 표본이나 조건을 줄인 변경이 아니다.

작은 Δ의 산술 해석은 보수적 norm bound를 사용하는 별도 사후 진단으로 추가했다. 원래 FP64 A/B/C/D 분류나 CAL 선택을 바꾸지 않았다.
