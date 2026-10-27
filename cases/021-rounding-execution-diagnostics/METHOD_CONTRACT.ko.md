# 방법·용어 계약

[English equations and exact dtype/layout](METHOD_CONTRACT.md)

artifact는 실행을 다시 구성할 code·scale·metadata 묶음입니다. INT4는 weight code의4비트 저장 형식이며 D16은 기존 W4A16 설정명입니다. dense는 가중치를 미리 복원한 행렬곱, packed/Marlin은 압축 code·scale을 GPU kernel에 넘기는 경로입니다. BF16/FP32는 숫자 정밀도이며 중간값과 최종 출력의 dtype는 별개입니다.

D0는 `(codes×gamma)×alpha_inv` 순서, D1은 FP32 combined scale→BF16 scale→BF16 weight, P는 같은 code/scale의 Marlin 실행입니다. bias는 없으며 beta는 입력에 정확히 한 번 적용합니다. P5는 beta 이후 Z를 core에 직접 전달합니다. signed nibble과 Marlin의 c+8 encoding, group64 scale permutation을 구분했습니다.

activation은 다음 계산에 전달되는 값, norm은 크기 정규화, RoPE는 위치를 반영하는 Q/K 회전입니다. R1은 이 묶음의 중간값만 FP32로 유지하고 최종 Q/K는 BF16입니다. 같은 native cos/sin을 승격하며 주파수·GEMM·모델 전체 정밀도를 바꾸지 않습니다.

KL/JSD는 분포 차이이며 정답률이 아닙니다. 원 full-vocabulary 계산은16위치→문서 평균 순서입니다. 공개 검산은 저장된 scalar를 재집계합니다. relative_RMSE=√(SSE/reference_energy), energy0은 UNDEFINED입니다. ULP는 표현 가능한 인접값 간 단계이며 kernel 반올림 횟수가 아닙니다. INCONCLUSIVE는 평균 KL 감소 방향을 확정하지 못한 판정입니다.

P5의 A=R(a×R(c×s)), B=R(a×c×s)는 단일항 후보식입니다. ΔG=ΔN+ΔC의 부호 분류 tolerance는 max(1e-12,1e-10×해당 셀 G/N/C 최대 절대값)입니다. tolerance 부근은 별도이고 원 모델0.02 gate와 다른 검사입니다. K는 mask 예산이며 실제 비영 수·입력 norm과 함께 보고합니다. [상세 식·축·분모](METHOD_CONTRACT.md), [모든 표](REPORT.ko.md)를 참조하십시오.
