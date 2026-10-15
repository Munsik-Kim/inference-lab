# Case 014 — 같은 INT4 저장량으로 모델 품질 개선하기

[English](README.md) · [Pages](https://munsik-kim.github.io/inference-lab/ko/case014.html#overview)

양자화된 모델의 실제 입력 경로를 따라 가중치를 보정하고, 같은 INT4 형식으로 저장하는 도구를 구현했습니다. Qwen3-0.6B의196개 Transformer projection에 고정된 약한 보정을 적용해, 새 PG19 64책의256-token 평가 구간에서 RTN 대비 PPL을12.92% 낮췄습니다. Projection 저장량은 그대로 유지했습니다.

**ArXiv에서는 같은 이득이 확인되지 않았습니다. 이 결과는 저장본을 BF16으로 복원한 품질 평가이며, 실행 속도·VRAM 절감은 별도로 측정하지 않았습니다.**

| 실제 비교 | RTN PPL | QUARTER PPL | 변화 |
| --- | ---: | ---: | --- |
| PG19 ·64 책의 구간 | 57.600 | 50.156 | -12.92%; NLL64/64 개선 |
| ArXiv ·16 content groups | 12.452 | 12.687 | +1.89% 점추정, NLL 구간0포함 |

**동일 INT4 projection 저장량:247,733,136 bytes/method.**
원 BF16의 PG19 PPL은 45.405.

[결과 비교하기](REPORT.ko.md#results) ·
[구현과 재현](REPRODUCE.md) ·
[P0→P2 연구 과정](STAGE_MAP.md)

## 구현한 기능

실제 입력 확인 → 약한 보정 적합 → INT4 저장 → 새 프로세스 재로딩 → 새 문서의 다음 토큰 점수 비교.

- [Native capture / own-trajectory runner](followups/p2_core/source/run_p2.py)
- [CPU FP64 full-input fitting](followups/p2_core/source/p2_fit.py)
- [INT4 codec](scripts/codec.py) · [Independent decoder](followups/p2_core/source/independent_codec.py)
- [Three-term decomposition](scripts/decompose.py) · [Raw-to-display exporter](publication/scripts/derive.py)
- [CPU tests](followups/p2_core/tests/test_p2.py) · [Public checks](publication/scripts/run_checks.py)

## 범위와 출처

P0의 대수·codec, P1의 독립 표적 개입, P1b의 독립 강도 선택, P2의 자체 경로 전층 보정을 한 사례에 묶었습니다. 각 단계의 부정 결과와 원형·선택·판정은 보존했습니다.

구현은 Guo의 선행 연구를 참고한 ridge/RTN reference입니다. 공식 QEP 전체 재현이나 새 quantizer가 아닙니다. 자체 구현과 upstream 역할은 [PORTFOLIO](PORTFOLIO.md), [NOTICE](NOTICE.md), [source audit](references/source_audit.json).
