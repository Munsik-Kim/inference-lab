# Case014 — 양자화 오차 보정의 상쇄·재양자화·일반화 분석

[English](REPORT.md)

1. [배경](#s1)
2. [질문·가설](#s2)
3. [이론과 구현 계약](#s3)
4. [방법](#s4)
5. [실험](#s5)
6. [결과](#s6)
7. [분석](#s7)
8. [결론](#s8)
9. [재현과 참고자료](#s9)

<a id="s1"></a>
## 배경

양자화 오차를 보정한 뒤 다시 저장하면 무엇이 남는지 확인하는 도구입니다. 앞에서 전달된 오차, 기존 오차의 상쇄, 재양자화의 변화를 같은 입력에서 분리하고, 보정에 쓰지 않은 문서의 실제 모델 출력과 연결하도록 설계했습니다.

<a id="s2"></a>
## 질문·가설

Q1: 유리한 상쇄를 깨뜨릴 수 있는가? Q2: 실제 재양자화가 방향을 바꾸는가? Q3: CAL 선택이 새 문서로 전이하는가? P0는 Q1의 가능성과 분해 도구를 확인합니다. Q2/Q3의 모델 관측은 P1 실행이 필요합니다.

<a id="s3"></a>
## 이론과 구현 계약

W[out,in], X/Z[in,tokens]. CAL only: G=ZZᵀ, λ=10⁻³trace(G)/d, (G+λI)Kᵀ=[W(X−Z)Zᵀ]ᵀ. δW=αK, W*=W+δW.

CPU FP64 Cholesky + triangular solves, full-input Gram, row chunks128. Quantizer input is FP32(W*). Reloaded FP32 What supplies algebra; BF16(What) supplies dense quality execution. Bias remains unchanged.

A=W(X−Z), C=δWZ, B=(What−W*)Z, B₀=(What₀−W)Z, R=A−C−B, R₀=A−B₀.

Δ=||R||²−||R₀||²=T_inherited+T_cancellation+T_requant.

T_inherited=||C||²−2⟨A,C⟩; T_cancellation=2⟨B₀,C⟩; T_cont=||C||²−2⟨R₀,C⟩; T_requant=||ΔB||²−2⟨R₀−C,ΔB⟩.

T_cont holds B₀ fixed: a counterfactual, not a measured continuous model. Actual BF16 arithmetic residual and weight-cast residual are separate. Ridge H is not generally a projector. Evaluation never refits K, scale or codes.

<a id="s4"></a>
## 방법

encoder는 signed INT4[-8,7], input group64, output-row scale FP32 absmax/7, ties-to-even입니다. zero group만 scale1을 쓰며 비영 underflow는 거절합니다. header36bytes, scale4bytes/row/group, low-nibble-first code0.5byte/value입니다. padding을 포함합니다. 실제 모델 artifact는 아직 생성하지 않았습니다.

계획된 표적별 core payload는 o_proj1,179,684bytes, down_proj1,769,508bytes입니다. **shape로 계산한 값이며 모델 artifact 실측이 아닙니다.** 동일 shape의 alpha들은 같은 core bytes를 사용합니다.

<a id="s5"></a>
## 실험

P0: 유리수 두 부호,4-code 전수 loss, NumPy/Torch FP64 항등식, codec 경계와 독립 reload, ridge/non-projector, CAL/EVAL 누출 경계를 실행했습니다. Dither는 depth2/8/16×step0.01/0.1×alpha0/0.5/1, 조건별2,000 seed입니다.

P1: 기존 snapshot revision, 6개 실제 shape, PG19 CAL32/DEV16/EVAL32, Pile ArXiv SHIFT16, SMOKE1을 확인했습니다. Pile는 원 논문 ID 없이 content hash로 묶었으므로 article-disjoint를 주장하지 않습니다. 과거 사용 기록의 조사 범위 밖과 사전학습 노출은 미확인입니다. 모델 forward0회,30개 필수 cell 모두 미실행입니다.

<a id="s6"></a>
## 결과

**현재 완료:** 정확한 유리수 예, INT4 저장·새 프로세스 재로딩, 독립 NumPy/Torch FP64 검산, 조건별2,000 seed dither 대조. 고유 CPU 테스트15개 PASS.

**P1 미실행:** 6표적×5강도의 코드와 문서 분할은 준비했지만 D 원본 위치가 확인되지 않아 모델 실행은 보류 중입니다. LLM의 개선·악화·일반화 결과는 아직 없습니다.

고정 오차 예에서 ε=0.01이면 BASE 오차는0.0001, 보정 뒤는0.0101입니다. 앞·뒤 오차가 서로 줄여주던 조합을 보정이 깨뜨릴 수 있습니다. 반대 부호 조합에서는0.0201→0.0101로 좋아집니다.

독립4-code 예에서는 teacher=3.016, BASE=3, COMP=4여서 오차가0.016→0.984입니다. 두 경로의 마지막 입력은3과2로 다릅니다. 2bit 스칼라 도식이며 INT4 LLM 빈도를 뜻하지 않습니다.

![Exact independent examples](figures/fig1_exact.svg)

|Depth|Step|alpha|Mean SSE|Paired Δ vs0|SE|95% MC interval|
|---:|---:|---:|---:|---:|---:|---|
|2|0.01|0|0.0041733059|0|0|[0, 0]|
|2|0.01|0.5|0.0026607099|-0.0015125961|2.9735e-05|[-0.00157088, -0.00145432]|
|2|0.01|1|0.0020843085|-0.0020889974|3.2047e-05|[-0.00215181, -0.00202619]|
|8|0.01|0|0.017054555|0|0|[0, 0]|
|8|0.01|0.5|0.0028151943|-0.014239361|0.00014652|[-0.0145265, -0.0139522]|
|8|0.01|1|0.0021202522|-0.014934303|0.00014596|[-0.0152204, -0.0146482]|
|16|0.01|0|0.033926054|0|0|[0, 0]|
|16|0.01|0.5|0.0028265192|-0.031099535|0.00029548|[-0.0316787, -0.0305204]|
|16|0.01|1|0.0021048641|-0.03182119|0.00029528|[-0.0323999, -0.0312424]|
|2|0.1|0|0.41820752|0|0|[0, 0]|
|2|0.1|0.5|0.26557389|-0.15263364|0.0030126|[-0.158538, -0.146729]|
|2|0.1|1|0.21140704|-0.20680048|0.0032177|[-0.213107, -0.200494]|
|8|0.1|0|1.7273954|0|0|[0, 0]|
|8|0.1|0.5|0.28456538|-1.44283|0.015438|[-1.47309, -1.41257]|
|8|0.1|1|0.21303847|-1.5143569|0.015393|[-1.54453, -1.48419]|
|16|0.1|0|3.4784585|0|0|[0, 0]|
|16|0.1|0.5|0.28332275|-3.1951357|0.032556|[-3.25895, -3.13133]|
|16|0.1|1|0.21112725|-3.2673312|0.032548|[-3.33113, -3.20354]|

<a id="s7"></a>
## 분석

dither 대조에서는 모든 비영 강도의 paired 평균 SSE가 alpha0보다 낮았습니다. 조건부 평균0과 고정 step을 갖춘 작은 선형 네트워크의 관측입니다. 실제 finite INT4 양자화나 비선형 LLM의 품질 보장은 아닙니다. MC 구간은 동일 입력에서 독립 noise seed를 반복한 표준오차 구간이며 문서 일반화 구간이 아닙니다.

세 항 분해의 역할은 단일 원인을 확정하는 것이 아니라, 보정 자체·기존 상쇄·재양자화의 기여를 같은 입력에서 계량하는 것입니다. CAL의 연속 잔차가 줄더라도 새 입력과 suffix NLL은 달라질 수 있습니다. 현재 LLM 자료가 없어 A/B/C/D 발생 건수나 CAL→EVAL 순위는 미측정입니다.

<a id="s8"></a>
## 결론

P0 구현은 PASS입니다. 보정이 항상 좋은 것은 아니라는 정확한 예와, 특정 noise 조건에서 좋아지는 양성 대조를 함께 남겼습니다. P1 현상, P2 품질 신호, 실용 효용과 packed 속도는 별도 판정이며 이번에는 미확립입니다.

P1의 정확한 장애는 모델 저장 경로 정책입니다. 기존 WSL snapshot과 작은 파일 hash는 확인했으나 D 원본은 특정하지 못했습니다. 사용 확인 또는 D 경로가 해결되면 같은 동결 입력과 코드로 실행할 수 있습니다.

<a id="s9"></a>
## 재현과 참고자료

실제 명령은 [REPRODUCE](REPRODUCE.md), 입력은 [manifest](provenance/input_manifest.json), source 상태는 [audit](references/source_audit.json), 실행 범위는 [decision](results/local_r1/decision.json)에 있습니다. CPU scalar 검산은 비공개 W/X/Z/full logits의 GPU 계산을 재검증하지 않습니다.

이론 출처: [Guo2026 v1](https://arxiv.org/html/2610.16020v1). 사용자 제공 독립 finite-codebook 수치와 이번 ridge/RTN 도구는 저자 공식 QEP 전체 재현과 구분합니다. 저자가 이미 실험한 HALF·calibration 전이를 새 발견으로 제시하지 않습니다. source/code/license 식별은 source audit에 있습니다.

기존 case·공통 코드·환경은 보존합니다. 이번 구현·문서·자동 검사는 Codex의 도움으로 작성했으며 독립 인간 검토로 표시하지 않습니다.

