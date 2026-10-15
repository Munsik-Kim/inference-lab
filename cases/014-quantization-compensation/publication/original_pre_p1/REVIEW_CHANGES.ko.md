# 실행 설계 확인사항

- Case014 번호는 비어 있었다. 기존 작업을 유지한 별도 worktree를 main `09194c3f2e2c991feff5b6abab0cf8ab81a07052`에서 만들었다.
- 논문 본문 HTML/PDF를 확보했다. 이전의 “본문 미확보”를 이번 상태로 계승하지 않는다.
- 데이터는 기존 PG19의 책 URL 단위 32/16/32문서와 기존 Pile ArXiv의 16개 content group이다. 과거 WikiText 진단을 재활용하지 않는다.
- Pile에는 원 논문 ID가 없어 article-disjoint라고 표시하지 않는다. 선택 window의 5gram 근접 중복은 검사했으나, 전체 문서의 근접 중복과 사전학습 비노출은 보장하지 않는다.
- 실제 입력 폭은 hidden_size1024와 다르다. o_proj2048, down_proj3072를 헤더에서 확인했다.
- D 원본을 기록에서 특정하지 못해 기존 WSL 캐시 사용 여부를 확인 중이다. SM120 실행 실패나 OOM이 아니다.
- 최초14개 테스트 뒤 요구된 독립 Torch/NumPy 분해 검산1개를 추가하고15개를 실행했다. MC 재실행에서 동일 seed별 경량 SSE를 저장했다. 반복 실행을 새 고유 테스트나 새 독립 seed로 세지 않는다.
- 기존 데이터·case·환경을 수정하지 않았다. 새 자료는 이 case와 외부 private scratch에만 저장한다.
