# Case015 P2 — 같은 관측에서 JSON 호출 실패 복구하기

구매 기록을 읽고 합계를 계산한 AI가, 저장 도구에 보낼 JSON의 따옴표·항목 이름을 잘못 써 멈춘 경우를 시험했습니다. **같은 조회·계산 결과를 받은 지점**에서 원래 생성(C0), 첫 응답 문법 제약(G), 중립 재시도(N), 오류 설명(E)을 비교하는 실행기와 저장 기록 감사 도구를 구현했습니다.

| 모델 / Model | C0 | G | N | E |
|---|---|---|---|---|
| BF16 | 7/8 | 7/8 | NOT_APPLICABLE | NOT_APPLICABLE |
| RTN | 8/8 | 8/8 | NOT_APPLICABLE | NOT_APPLICABLE |
| D16 | 0/8 | 8/8 | 0/8 | 0/8 |


D16의 기존 실패 8/8을 새 추론에서 그대로 재현했고, G는 올바른 실제 저장·검증·최종 답변까지 8/8 회복했습니다. N/E는 각각 0/8 완료했으며 N은 금액 저장만 1건 회복했습니다. 분모는 **선택된 합성 기록 4개 × API 2종**입니다. BF16/RTN의 N/E는 적용할 오류가 없어 NOT_APPLICABLE입니다.

[결과와 한계](REPORT.ko.md) · [English](README.md) · [CPU 재현](REPRODUCE.ko.md) · [실행 코드](execution/run.py) · [raw](results/gpu_r2/cells) · [검증](publication/verification/latest.json)

새 생성 요청 175회(최초 진단 64 + 후속 요청 111), 출력 토큰 6,986개, smoke 0회입니다. 저장본을 dense BF16으로 복원한 품질 실험이며 기존 A1의 자유 생성 실패 판정은 그대로입니다. 원 A1/P1과 P2의 분모를 합치지 않습니다.

```bash
P2=cases/015-tool-interface-sensitivity/followups/json-recovery-p2-v1
OUT=$(mktemp -d /tmp/case015-p2-review-XXXXXX)
python3 -B "$P2/publication/run_checks.py" --output "$OUT/checks" --tests
```

모델 없이 raw 재생·원 판정·표를 확인합니다. 문법5개 검사는 기존 의존성 Python을 `--grammar-python`으로 지정합니다. GPU 재현은 별도의 pinned 소스 `828ac7358b653626bed7e0a609448e73d120f1bb`와 기존 로컬 모델/artifact가 필요합니다. [CPU 당시 등록](specs/protocol.json)·[원형 대응표](publication/preservation_map.json)·[원형 복원 방식](publication/verify.py)을 보존했습니다. 현재 승인된 GPU CLI는 execution/run.py이며 옛 CPU CLI의 run 거절은 유지합니다. main 병합·Pages 배포는 이 작업 범위가 아닙니다.
