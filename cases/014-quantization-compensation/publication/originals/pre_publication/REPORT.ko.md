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

양자화는 모델의 숫자를 적은 비트로 저장하는 방법입니다. 보정은 앞부분에서 전달된 오차를 줄이도록 뒤쪽 가중치를 조정합니다. 이 도구는 그 조정값도 다시 INT4로 저장했을 때 무엇이 남고, 새로운 문서에서도 도움이 되는지 확인합니다.

보정값 계산부터 INT4 저장·재로딩, 오차 분해, 동일 입력의 전체 모델 비교까지 실행했습니다. CAL에서 고른 강도는 국소 오차를 줄였지만, 새 EVAL 문서에서는 여섯 표적 모두의 합산 국소 오차가 증가했습니다. 모델의 다음 토큰 점수는 표적에 따라 다르게 움직였습니다.

<a id="s2"></a>
## 질문·가설

Q1 상쇄 / Q2 실제 재양자화 / Q3 CAL→새 문서 전이를 따로 판정했습니다. 작은 스칼라 반례와 dither는 기존 P0 결과이며 이번 LLM 결과와 합치지 않습니다.

<a id="s3"></a>
## 이론과 구현 계약

W[out,in], X/Z[in,tokens]. CAL only: G=ZZᵀ, λ=0.001 trace(G)/d; (G+λI)Kᵀ=[W(X−Z)Zᵀ]ᵀ; δW=αK; W*=W+δW.

A=W(X−Z), C=δWZ, B=(What−W*)Z, B₀=(What₀−W)Z, R=A−C−B, R₀=A−B₀.

Δ=||R||²−||R₀||²=T_inherited+T_cancellation+T_requant.

T_inherited=||C||²−2⟨A,C⟩; T_cancellation=2⟨B₀,C⟩; T_cont=||C||²−2⟨R₀,C⟩; T_requant=||B−B₀||²−2⟨R₀−C,B−B₀⟩.

W*를 FP32로 바꿔 signed INT4[-8,7], input group64, output-row별 FP32 absmax/7, nearest-even으로 재양자화했습니다. low-nibble-first, header36B, zero-group scale1 규칙입니다. 실제 파일을 저장·읽은 What(FP32)를 대수에, BF16(What)를 실제 모델에 사용했습니다. 양자화 코드·scale 계산과 fit/scorer는 P0 준비본과 동일합니다.

이것은 packed INT4 **저장 + dense BF16 품질 reference**입니다. K를 런타임 추가 연산으로 붙이지 않았습니다. FP32 연속 대조는 FP32 가중치·입력과 BF16 출력 cast를 사용한 별도 진단이며 상한이 아닙니다.

<a id="s4"></a>
## 방법

원형 BF16 projection 입력 X와 선행 블록만 RTN으로 바꾼 Z를 실제 full forward에서 캡처했습니다. 표적은 block4/12/20의 o_proj/down_proj, 강도는0/0.25/0.5/0.75/1입니다. 표적 하나씩 독립 개입하고 나머지 같은 블록·suffix는 원형으로 유지했습니다.

CAL32책에서 CPU FP64 full Gram/Cholesky/삼각 solve와 row chunk128을 사용했습니다. 여섯 K·lambda와 30개 artifact·선택 강도를 동결한 뒤 비CAL 문서를 열었습니다. DEV에서 재선택하지 않았고, EVAL 최적 강도는 사후 oracle로만 남겼습니다.

각 문서는256토큰, 다음255토큰 NLL/argmax flip, 고정16위치의 full-vocabulary KL입니다. chat/BOS/EOS 추가 없이 batch1, use_cache=False, SDPA, TF32 및 BF16 reduced reduction=False, 전체256위치 LM head를 공통 사용했습니다.

<a id="s5"></a>
## 실험

PG19의 서로 다른 book URL로 CAL32/DEV16/EVAL32를 나누고, Pile ArXiv16개 content group을 SHIFT로 사용했습니다. 별도 SMOKE1은 평가 분모에서 제외합니다. 동일 문서·token hash·offset·mask·position을 유지했습니다. Pile 원 논문 ID가 없어 article-disjoint를 보장하지 않으며, 조사 범위 밖의 과거 사용과 사전학습 노출은 미확인입니다.

핵심30조건×96문서=2,880행입니다. 이는 독립 문서2,880개가 아닙니다. EVAL의 독립 단위는32책, SHIFT는16 content group입니다. 전체 forward3,845회에는 teacher·prefix·smoke·연속 대조가 포함됩니다.

측정은 RTX5080 SM120, Torch2.13.0+cu130, Transformers5.17.0에서 수행했습니다. 보수적 worker wall은 **358.877초**, runner 내부 구간은344.235초입니다. GPU busy 시간·TTFT·TPOT는 측정하지 않았습니다.

실측 파일 크기: o_proj **1,179,684B**, down_proj **1,769,508B**, 각 표적의5개 alpha에서 동일합니다. Torch peak allocated는1,646,827,520B, peak reserved는1,845,493,760B입니다. CUDA free/total의 동기화된 사후 표본 중 최대 사용량은3,331,915,776B이며, 연속 감시한 Windows 전체 peak가 아닙니다.

<a id="s6"></a>
## 결과

표의 Δ는 같은 표적의 alpha0 대비입니다. 각 baseline은 선행 블록과 해당 표적만 RTN으로 바꾼 hybrid 모델이며, 전체 모델 RTN 또는 과거 Case013 baseline과 다릅니다. SSE는 합산 국소 오차, NLL은 nats/token이며 둘 다 음수 변화가 개선입니다.

CAL 선택값의 EVAL 합산 SSE는6/6에서 증가했습니다. EVAL NLL은 o_proj3개에서 감소, down_proj3개에서 증가했습니다. SHIFT NLL은6/6에서 증가한 점추정치입니다. 일부 NLL 구간은0을 포함합니다. 아래 구간은 문서 pair10,000회 bootstrap, seed6101516의 탐색적 pointwise95% 구간입니다.

|Block / projection|CAL α|CAL ΔSSE %|EVAL ΔSSE %|EVAL ΔNLL [95% CI]|SHIFT ΔNLL [95% CI]|
|---|---:|---:|---:|---|---|
|4 / down_proj|1|-53.34|+45.40|+0.017570 [+0.009267, +0.025416]|+0.047380 [+0.028125, +0.070095]|
|4 / o_proj|1|-52.06|+10.98|-0.006263 [-0.012331, -0.000537]|+0.010130 [+0.002822, +0.018001]|
|12 / down_proj|1|-55.72|+36.68|+0.011532 [+0.006298, +0.017117]|+0.016129 [+0.005145, +0.027515]|
|12 / o_proj|1|-55.25|+43.26|-0.004538 [-0.009533, +0.000430]|+0.005138 [-0.001055, +0.011251]|
|20 / down_proj|0.75|-45.33|+14.29|+0.003528 [-0.000984, +0.008394]|+0.056429 [+0.043199, +0.071002]|
|20 / o_proj|1|-66.15|+5.11|-0.007653 [-0.011391, -0.003748]|+0.023418 [+0.014054, +0.033364]|

![All six target curves](figures/p1_r2/fig2_components.svg)

[All120 target/strength/split summaries](results/p1_r2/summary.csv) · [Raw2880 rows](results/p1_r2/cells)

![CAL/EVAL ranks](figures/p1_r2/fig3_transfer.svg)

<a id="s7"></a>
## 분석

A는 상속오차 개선이 fixed-B 상쇄 손실에 뒤집힌 계산 패턴, B는 재양자화가 fixed-B 개선을 뒤집은 패턴, C는 frozen 연속 보정의 새 입력 전이 손실, D는 국소 SSE 개선과 NLL 악화가 함께 나온 패턴입니다. 분류는 겹치며 분모는 문서×표적×비영 강도 비교입니다.

EVAL의 A1건은 block4/down_proj/alpha0.5, 문서hash `4f835d09…be32`입니다. T_inherited=−3.673586, T_cancellation=+4.965093, T_cont=+1.291507, T_requant=+4.018703, Δ=+5.310211입니다. 실제 BF16 module SSE 차이는+5.491078이었습니다. 다만 보수적 산술 오차 bound67.454096보다 Δ가 작아 **실행 경로에 대한 부호 보장은 불확정**입니다.

추가 사후 진단은 |SSE_run−SSE|≤2||R||||e||+||e||²를 양 arm에 적용했습니다. 비영 강도2,304행 중2,195행은 Δ 부호가 이 bound를 넘었고109행은 불확정입니다. 원래 FP64 분류를 바꾸거나 원인을 단정하지 않습니다. B12건 중 해당 Δ 부호 bound를 통과한 것은1건입니다.

|Split|Comparisons (nonzero α)|A|B|C|D|
|---|---:|---:|---:|---:|---:|
|CAL-FIT|768|0|0|0|205|
|DEV-SELECT|384|0|8|169|81|
|EVAL-ID|768|1|4|339|172|
|SHIFT|384|0|0|367|14|

![All target local error versus downstream NLL](figures/p1_r2/fig4_local_nll.svg)

<a id="s8"></a>
## 결론

이 고정 설정에서 CAL의 국소 오차만 보고 고른 강도는 EVAL의 국소 오차 순위를 유지하지 못했습니다. 보정·재양자화·새 입력·downstream 점수가 서로 다른 현상임을 실제 저장 파일과 전체 forward로 연결한 것이 이번 결과입니다. 모든 보정이 나쁘거나 다른 모델에도 같은 빈도로 일어난다는 결론은 아닙니다. P1 완료는 좋은 점수와 별개입니다.

<a id="s9"></a>
## 재현과 참고자료

기존 P0 고유15개 테스트를 이번에 다시 통과했습니다. 새 고유15개를 추가한 것이 아닙니다. P0 MC는 재실행하지 않았으며 당시 lambda=0입니다. 독립 stdlib scalar auditor는2,880행과 선택·분모·집계를 검산했고, 새 CPU 프로세스는30개 artifact를 독립 디코딩했습니다. 비공개 작은 표본8건은 Torch CPU FP64로 전체 행렬과 cast를 재검산했습니다. 원 weight/X/Z/full logits가 제외된 검토본은 GPU NLL/KL 연산을 새로 재현하지 않습니다.

사용자 승인은 기존 WSL snapshot의 읽기 전용 재사용에 한정합니다. 실제 전체 로컬 weight SHA256은 기존 cache blob 식별자와 일치했습니다. 외부 공식 digest를 새로 확보한 인증과 구분합니다. 원본 snapshot·기존 P0/실패·환경을 보존하며 P2/P3와 원격 게시는 미실행입니다.

Execution source SHA256: `da2a0368bb4a284a0af1a5896121ca0e828a59d9392f49b954092664c1cab960`. Base Git SHA: `09194c3f2e2c991feff5b6abab0cf8ab81a07052`. These identify different objects.

[Reproduce](REPRODUCE.md) · [Independent audit](results/p1_r2/independent_audit.json) · [Source audit](references/source_audit.json) · [Original P0 report](publication/original_pre_p1/REPORT.ko.md).

[Guo2026 v1](https://arxiv.org/html/2610.16020v1) motivates this independent ridge/RTN diagnostic. This is not official full QEP reproduction or a new quantizer. Codex assisted implementation and reporting; no independent human review is claimed.

