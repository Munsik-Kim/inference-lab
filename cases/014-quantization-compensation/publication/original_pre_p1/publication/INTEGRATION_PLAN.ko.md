# Case014 등록 준비 — 게시 미승인

예정 카드: **양자화 오차 보정, 다시 저장하면 무엇이 남을까?**

보정 fit, INT4 저장·재로딩과 오차 분해를 구현했다. 정확한 작은 예에서는 보정이 기존 상쇄를 깨뜨릴 수 있었고, 독립 noise를 갖는 선형 대조에서는 평균 오차가 줄었다. LLM 관측은 P1 실행 뒤 추가한다.

English: **What survives when compensated weights are quantized again?** Built fitting, INT4 save/reload and error decomposition. Exact small examples expose lost cancellation; independent-noise linear controls reduce average error. LLM findings remain pending P1.

읽기 경로: README → demo/ko 또는 en → 보고서9절 → source/CPU검산.
원격 게시 승인과 P1 검토 후에만 root README/Casebook/site에 연결한다. 이번 작업은 이 case 밖의 파일을 수정하지 않는다.
