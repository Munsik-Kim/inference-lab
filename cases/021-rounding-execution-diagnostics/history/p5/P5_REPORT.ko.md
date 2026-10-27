# Case021 P5 — 실제 값의 단일 곱과 여러 항 결합

두 실제 projection의 고정 probe에서 단일항 D1/P 차이는 없었습니다. 반면 같은 실제 값을 여러 항으로 결합하면 차이가 나타났습니다. 이것은 실행 차이가 나타나는 입력 범위를 좁힌 결과이며, kernel 내부 산술의 단일 원인을 확정한 결과는 아닙니다. 기존 전체 모델의 **INCONCLUSIVE**는 유지합니다.

**P5A: 기존 국소 gap의 변화.** P4의 같은 block input·같은 좌표 자료128조합(DEV32/EVAL96)을 읽었습니다. Q/K, 층25/27, 문서와 head/position 분모를 분리했습니다. ΔG=ΔN+ΔC를 검산했습니다. EVAL의 gap 증가34개는 ΔN<0·ΔC>0이13개, 둘 다 양수가17개, ΔN>0·ΔC<0이4개였습니다. tolerance 근처 증가는0개입니다. P4에서 음의 cross가 약해진24개는 이 분류와 다른 질문이며 12/12/0으로 포함됩니다. 새 부호 tolerance는 max(1e-12,1e-10×해당 셀의 G/N/C 최대 절대값)로 파생 표를 보기 전에 동결했습니다. 원 P4의 exact-sign 증가 여부도 열로 보존했습니다. 이 에너지 항등식은 전체 모델 KL의 인과 기여율이 아닙니다.

![국소 에너지 변화](figures/01_gap_components.png)

기존 DEV8×층0/25/27 첫 block의 25,165,824원소 중 7,823개가 수치상 달랐습니다(0.031086%). BF16 간격1/2/그 이상은7,496/133/194개, 최대는27,892였습니다. 음수 정렬과 ±0의 수치 동일성을 처리했습니다. 0 부근/부호를 가로지르는 큰 간격을 내부 반올림 횟수로 해석하지 않습니다. P5A에서 midpoint 원인을 새로 검산하지는 않았습니다.

**P5B: 산술 후보를 구분하는 단일항.** A=R(a×R(c×s)), B=R(a×c×s)이며 R은 BF16 nearest-even입니다. c=3,s=1.0078125에서 a=1이면 둘 다3.03125, a=0.125이면 둘 다0.37890625, a=1.5이면 A4.5625/B4.53125입니다. 정확한 이진 Fraction 식과 직접 BF16 반올림을 대조했습니다. 이 숫자는 설명용 합성이며 실제 artifact에 있다고 가정하지 않았습니다.

CPU 후보 pool은 projection별 최대4096, 실제 DEV 두 문서의 고정 순서·seed와 고정 계수로 제한했습니다. GPU 결과 전 첫 구분 후보를 선택했습니다. pool에는 dev_code_00/01이 모두 들어갔지만 실제 GPU 단일항의 첫512 실제 계수와 challenge는 dev_code_00에서 선택됐습니다. 이를 두 문서의 GPU 표본으로 세지 않습니다. 층0·25의 q_proj, I1024/O2048/group64/M512, bias 없음, 이미 beta가 한 번 적용된 Z를 두 core에 그대로 전달했습니다.

각 projection의 ±1,±1/8, 고정 비거듭제곱, 실제 계수, 구분 challenge 다섯 fixture 모두512 유효 행입니다. 10조건의 첫 block 10,485,760원소에서 D1/P와 D1/A 불일치는0개였습니다. P의 A_only=406,626, B_only=0, both=10,079,134, neither=0입니다. 선정 challenge좌표1,024개도 모두 A_only였습니다. 공개 candidate_fit 표의 ALL_VALID/실제계수/선정좌표 범위는 겹치므로 합산하지 않습니다. 3 process 반복의40 hash 대조가 일치했습니다. B와 구분되는 좌표에서 A와 양립한다는 결론이며 모든 내부 반올림 순서를 증명하지 않습니다.

![단일항 후보 대응](figures/02_single_term_candidates.png)

**P5C: 사전 조건 D에 따른 희소 replay.** 단일항의 실제 계수 결과가 같고 반복이 안정적이어서, GPU 결과 전에 생성한 nested mask를 실행했습니다. 남긴 BF16 값은 원 bit 그대로이고 나머지만0이며 재정규화하지 않았습니다. 두 층×두 DEV문서×K={1,2,8,64,256,1024}입니다. 표의 각 원소 수 분모는512×2048입니다.

| 층 | 문서 | 처음 차이를 관측한 K | K=1/2/8/64/256/1024의 불일치 원소 수 |
|---|---|---:|---|
| 0 | dev_code_00 | 64 | 0 / 0 / 0 / 16 / 97 / 353 |
| 0 | dev_code_01 | 8 | 0 / 0 / 1 / 11 / 90 / 332 |
| 25 | dev_code_00 | 64 | 0 / 0 / 0 / 17 / 97 / 365 |
| 25 | dev_code_01 | 64 | 0 / 0 / 0 / 20 / 88 / 338 |

K와 입력 norm, 실제 비영 개수가 함께 변합니다. 첫 차이의 K는 보편적인 임계값이 아닙니다. 최대 ULP는 K와 단조 증가하지 않았습니다. 96개 반복 hash가 일치했고, K1024의 입력과 두 방법 출력8개는 P4 저장값과 bit가 같았습니다. P4 대조는 다른 시점·process의 대응입니다.

두 FP64 참조는 실제 D1 BF16 weight를 승격한 dense64와 code×실제 BF16 scale의 exact64입니다. 어느 것도 원 미압축 모델의 품질 정답이 아닙니다. chunked matmul 결과를 두 방법 모두와 비교하고 전체 SSE/energy를 math.fsum으로 재집계했습니다. 사전 좌표와 사후 mismatch/불확실 좌표를 합한768 oracle audit 행은 정확한 dyadic 합과 대조했습니다. exact midpoint tie264행, 감사 subset의 FP64 최종 반올림 불일치0행입니다. error bound로 판정이 확정되지 않고 exact subset에도 없는231,110 oracle-조건 좌표는 UNRESOLVED로 남겼습니다. 이는 같은 좌표의 두 oracle을 독립 표본으로 센 값이 아닙니다.

**계약·비용·이력.** eager/inference_mode, TF32와 FP16/BF16 reduced reduction OFF, Marlin FP32 reduce=true/atomic=false를 유지했습니다. 설치된 torch.ops._C.marlin_gemm과 CUDA Marlin symbol을 별도4회 profiler로 확인했습니다. native INT4 MMA 또는 새 kernel 구현이라고 부르지 않습니다. 원 runtime은 읽기 전용으로 재사용했고 clean environment 검증은 아닙니다. 최고 PyTorch allocated peak는48.321MiB로2GiB gate 안이며 장치 전체 VRAM과 다릅니다.

본 core 호출은 B60+C144=204회, beta/module-core 기술 대조16회, profiler4회, 합224회입니다. 새 전체 모델 forward·입력 추출 forward는0회입니다. GPU compute worker12개 외에 gate 거절1건, 읽기 전용 대기1건, 종료 확인1건이 장부에 있습니다. 외부 compute PID가 있을 때 C worker는 생성되지 않았고, 약131초 대기 후 gate를 다시 통과했습니다. 전후 표본은 연속 독점 증명이 아닙니다.

새 GPU parent wall은229.252607s, 기존 campaign과 합2282.963944s/7200s입니다. 새 CPU worker wall은61.077458s이며 문서 편집 전체 경과는 별도 NOT_MEASURED입니다. wall은 GPU busy time이나 서비스 지연시간이 아닙니다. 이 문서의 비용은 [장부 요약](summary/budget.json)의 시점입니다.

첫 NumPy scalar shift 검사 실패, Fraction의 -0/+0 비교 실패, C 외부 실행문 dict 해석 실패를 보존했습니다. 각각 dtype 명시, 수치 zero 동일성 적용, key 명시로 고쳤고 source/입력/허용오차의 과학적 의미를 바꾸지 않았습니다. 첫 두 실패는 새 GPU 출력 전에 해결했습니다. C 첫 gate 거절은 실행문 실패와 별개입니다. [실패 및 identity 기록](records/identity.json), [구현 재사용](REUSE_MAP.md), [보존](summary/preservation.json), [재현](REPRODUCE.ko.md)을 연결합니다.

**이번 범위의 끝.** 새 모델·설치·학습·자유 생성·norm/RoPE 개별 분리·새 HOLDOUT·속도 sweep·Git/원격 게시·Pages는 실행하지 않았습니다. 실제 activation/code/scale/단일항 출력은 로컬에만 있습니다. 원 P0–P4 source/raw/실패/보고서와 두 artifact·선택 runtime 및 Git 상태는 명시한 범위에서 그대로입니다. 다른 프로젝트 전체나 모델 weight 전체를 새 감사한 것은 아닙니다.

두 oracle의 반대 관측도 보존했습니다. 24개 고정 조건에서 SSE의 수치상 순서는 dense64 기준 P가 더 작은11개/D1이 더 작은2개/동일11개, exact64 기준7/6/11개였습니다. 이는 크기 문턱을 적용하지 않은 작은 차이의 기술적 집계이며 우수성 판정이 아닙니다. K1024의 exact64 참조에서는 층0 두 문서의 D1 SSE가 더 작고, 층25 두 문서에서는 P SSE가 더 작았습니다. 한 oracle만 선택해 유리한 결론을 만들지 않았습니다.
