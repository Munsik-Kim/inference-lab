# Case015 — 같은 도구, 다른 이름

같은 작업에 제공하는 도구의 이름과 순서를 바꾸고, 모델이 실제로 작업을 끝내는지 비교하는 도구를 구현했습니다. 정확한 이름 변환, 단계별 동등성 검사, 실제 fixture 실행과 저장 응답 재감사를 연결했습니다.

96개 FREE episode를 완료했습니다. D16은 불투명 이름에서 두 patch 실패를 회복했지만, lowerCamelCase에서 새 손실이 생겼습니다. 주 signed interaction의 구간은 0을 포함합니다.

[English](README.md) · [ 결과·사건 탐색 ](demo/ko.html) · [ 정식 보고서 ](REPORT.ko.md) · [ CPU 검산 ](REPRODUCE.ko.md) · [Code](src/interface_sensitivity/cli.py)

| Method | I0 original | I1 order | I2 lowerCamelCase | I3 opaque |
|---|---:|---:|---:|---:|
| BF16 | 8/8 | 8/8 | 8/8 | 7/8 |
| RTN | 8/8 | 8/8 | 8/8 | 8/8 |
| D16 | 6/8 | 6/8 | 5/8 | 8/8 |


각 칸은 같은 8개 기존 DEV 과제의 실제 기능 성공 수입니다. 96개는 독립 과제 수가 아닙니다. ENUM·개선된 오류 안내·alias는 모두 껐습니다.

- 이름·순서 변환과 의미 ID의 정확한 대응: [registry](src/interface_sensitivity/registry.py), [transform](src/interface_sensitivity/transform.py).
- 도구·인자·실제 상태·검증의 단계별 확인: [runner](src/interface_sensitivity/runner.py), [scoring](src/interface_sensitivity/scoring.py).
- 모델 없이 원시 응답·도구 사건·paired 효과 재계산: [audit](src/interface_sensitivity/audit.py), [analysis](src/interface_sensitivity/analysis.py).

P0 CPU 고유 검사 27개와 32개 oracle episode가 통과했습니다. GPU 요청은 본실험 228회와 별도 smoke 4회이며, worker wall은 모델 로딩·decode를 포함해 452.013 s. 실행은 기존 INT4 저장본을 BF16으로 복원한 품질 reference이며 serving 속도 비교가 아닙니다.

[계약·계획](specs/protocol.json) · [원자료](results/p1_r1/episodes.jsonl) · [독립 CPU 재계산](results/p1_r1/independent_audit.json) · [판정·범위](results/p1_r1/decision.json) · [후속 제안](ROADMAP.md)
