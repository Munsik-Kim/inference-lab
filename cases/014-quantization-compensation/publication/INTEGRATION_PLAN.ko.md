# Case014 등록 준비 — 원격 게시 미승인

카드: **보정한 오차, 새 문서에도 도움이 될까?**

보정 fit부터 INT4 저장·재로딩, 오차 분해와 전체 모델 비교까지 실행했다. Qwen3-0.6B의6개 표적에서 CAL로 고른 강도는 새 EVAL의 합산 국소 오차를 모두 늘렸고, 다음 토큰 NLL은 표적에 따라 개선과 악화가 나뉘었다. 같은 입력에서 국소 오차와 모델 점수를 구분하는 도구와 실제 기록을 제공한다.

English: **Does compensation transfer to new documents?** Executed fitting, INT4 save/reload, error decomposition and full-model evaluation. On six Qwen3-0.6B targets, CAL-selected strengths increased aggregate EVAL local error while next-token NLL moved in different directions. The tool and records connect local error with downstream scores on matched inputs.

읽기 경로: README → demo/p1_r2/ko 또는 en → 보고서9절 → 전체120행 요약·2,880행 원자료·CPU 검산.

현재 결과는 packed INT4 저장과 dense BF16 품질 reference다. 공개 승인 후에만 root repository README/Casebook/Pages에 연결한다. 이번에는 다른 case·공통 package·site·원격을 수정하지 않았다.
