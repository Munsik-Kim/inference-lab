# P1b 검토 반영과 P2의 변경 범위

P1b의 원 PLAN·REPORT·12개 선택·462개 구간·decision은 보존했습니다. 별도의
`CASE014_P1B_INDEPENDENT_REVIEW_KO.md`와 `independent_analysis.json`은 현재 Case014
경로에서 찾지 못했습니다. 사용자 지시의 검토 항목을 원 machine-readable 기록과
대조했으며, 이 문서를 독립 검토자가 작성한 문서로 표시하지 않습니다.

| 검토 항목 | 반영 |
| --- | --- |
| D_macro −0.360780 | 여섯 고정 개입의 (SELECT_SSE−OLD_FIT)/alpha0 SSE 평균. 전체 모델 오차36% 감소 또는 OLD_FIT 대비 상대36% 감소가 아님 |
| alpha0 대비 SSE−10.172%, NLL−0.001055 | 서로 다른 prefix의 여섯 개입 평균이며 하나의 전층 모델 결과가 아님 |
| 고정0.25 대비 SSE 추가1.122 baseline-SSE pp | 네 표적은0.25와 동일 artifact; 추가 감소는 block4/20 o_proj의0.5 선택에서 발생 |
| 고정0.25 대비 NLL | 보조 pointwise CI가0을 포함; 추가 품질 이득 불확정 |
| ArXiv | 두 selector 모두 alpha0보다 평균 SSE/NLL 악화; 기존 실패를 유지 |
| 선택 비용 | argmin 약0.005초와 SELECT full-forward/metrics 약152.325초를 구분 |
| 사후 선택 빈도 | sensitivity diagnostic일 뿐 새 selector나 학습 결과로 채택하지 않음 |
| 드문 BF16 부호 반전 | 기존 EVAL1536비교 중1개 방향 불일치 및 bound 비인증 기록을 보존; 전체 결론 오류로 확대하지 않음 |

P2는 이 관측 뒤의 새 설계입니다. 이전 FULL 제안 대신 고정 QUARTER를 primary로,
고정 HALF를 보조로 사용합니다. K와 calibration input은 각 방법의 실제 현재
경로에서 새로 만들고 기존 여섯 K/선택 alpha를 전층에 복사하지 않습니다.

구축은 원 native CausalLM 전체 forward를 그룹마다 다시 실행합니다. 별도 blockwise
shortcut이 없어 그 동작을 원 forward와 같다고 추정할 필요가 없습니다. 실제 hook으로
qkv·gate/up 공통 입력, 같은 block 내 변화, 원형 복원과 미래 block의 비영향을 검사합니다.

기존 자료의 불변과 새 자료의 spec/source/input/artifact 동결을 따로 기록합니다.
새 EVAL을 열기 전에 BF16/RTN/QUARTER/HALF 전체 모델의 구축을 마칩니다.
