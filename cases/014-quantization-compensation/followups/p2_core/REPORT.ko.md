# 약한 양자화 보정은 전체 모델에서도 도움이 될까?

Case014 P2 · 2026-10-15 · `COMPLETED_P2_CORE` · [English](REPORT.md)

## 한눈에 보기

양자화된 모델이 실제로 전달하는 입력을 다시 측정하고, 그 입력에 맞춘 약한 보정을
동일한 INT4 형식에 저장하는 도구를 구현했습니다. 각 블록의 q/k/v → attention → o →
residual/norm → gate/up → down 순서를 원래 모델 전체 forward로 따라가며 구축합니다.

Qwen3-0.6B의 전체196 projection에 적용한 QUARTER는 새 PG19 64책에서 RTN보다
평균 NLL이 **-0.138374 nats/token**, 95% CI **[-0.149414, -0.127515]**로 낮았습니다.
PPL은 **57.600 → 50.156 (-12.923%)**였고, 64책 모두 NLL이 낮았습니다.
반면 ArXiv16개에서는 QUARTER의 점추정이 악화했고 구간에0이 포함됐습니다. HALF의
ArXiv NLL은 RTN보다 높은 구간으로 확인됐습니다. **PG19 이득과 도메인 전이 한계를 함께 관측했습니다.**

RTN·QUARTER·HALF의 실제 projection 저장량은 각각 **247,733,136 bytes**로 같습니다.
평가는 저장본을 BF16으로 복원한 dense 품질 경로입니다.

[원행·집계](../../results/p2_r1/summary.json) · [독립 CPU 감사](../../results/p2_r1/independent_audit.json) ·
[재현 명령](REPRODUCE.md) · [코드](source/run_p2.py) · [간단 결과표](demo/index.html)

목차: [질문](#질문) / [구현](#구현) / [데이터·실험](#데이터실험) / [결과](#결과) /
[분석](#분석) / [결론](#결론) / [재현·출처](#재현출처)

## 질문

P1b에서 여섯 개 독립 개입의 국소 SSE가 개선됐지만, 이들을 합친 전체 모델의 성적은
측정하지 않았습니다. P2는 그 결과를 본 뒤 설계한 후속입니다. 기존 FULL 제안 대신
고정0.25를 주 후보로, 고정0.5를 보조 후보로 등록했습니다. 새 selector를 적합하지 않았습니다.

| 확인할 질문 | 이번 관측 |
| --- | --- |
| 자체 경로의 전층 약한 보정이 같은 형식 RTN보다 새 PG19 NLL을 낮추는가? | QUARTER의 등록된 주 비교 `LOWER_NLL` |
| 국소 P1b 이득이 전층에서도 유효한가? | 고정 CAL32·모델·이번 PG19 표본에서 전층 품질 이득을 직접 관측. 여섯 국소 효과를 합산한 예측값은 아님 |
| 다른 출처에도 유지되는가? | ArXiv QUARTER는 점추정 악화·구간 불확정, HALF는 NLL 악화 |

## 구현

BF16은 원형 모델입니다. RTN은 원래196개 projection을 그대로 INT4로 저장합니다.
QUARTER/HALF는 각자 원형에서 시작하고, 현재까지 바뀐 자신 경로의 CAL 입력 Z와
완전 BF16 teacher의 X를 사용합니다. q/k/v와 gate/up은 같은 입력의 Gram factor만
그룹 안에서 공유합니다. 다른 강도의 Z·K를 공유하지 않습니다.

**기준 구현은 native CausalLM 전체 forward입니다.** 그룹마다 모델 전체를 실행하므로
attention·QK norm·RoPE·residual·MLP를 생략한 블록 단축 경로가 없습니다. 실제 hook으로
qkv/gate-up 공통 입력, 같은 블록 q 변화의 o 입력 전파, 미래 블록의 비영향을 확인했습니다.

<details><summary>적합·저장·오차 분해의 정확한 계약</summary>

W[out,in], X/Z[in,tokens]. 원 BF16 W의 표현값을 FP64로 올립니다.
G=ZZᵀ, λ=0.001 tr(G)/d_in, (G+λI)Kᵀ=[W(X−Z)Zᵀ]ᵀ를 CPU FP64
Cholesky와 triangular solve로 풉니다. full-input Gram, output-row chunk128입니다.
Wstar=W+αK이며 원래 W에 한 번만 더합니다. QUARTER/HALF는 각각196개의 새 K와
112번의 factorization을 갖습니다. 첫 블록 qkv는 X=Z, K=0, 저장 code/scale도 RTN과 같습니다.

FP32(Wstar)를 input-axis group64·output-row/group FP32 absmax/7로 양자화합니다.
signed[−8,7], nearest-even, 실제 zero group만 scale1, low-nibble-first, header36B입니다.
모든 파일은 실제 pack→새 파일→두 decoder로 reload한 후 BF16(What)를 설치했습니다.
runtime에 K/residual을 별도 실행하지 않습니다.

CAL에서 A=W(X−Z), C=αKZ, B=(What−Wstar)Z, B0=(Q(W)−W)Z입니다.
Δ=||A−C−B||²−||A−B0||²=T_inherited+T_cancellation+T_requant.
B0는 **각 방법 자신의 현재 Z에서의 국소 alpha0 대조**입니다. 별도 전체 RTN 모델의
오차가 아닙니다. 392개 CAL 분해의 최대 상대 gap은 2.294e-15, 최대 solve residual은
1.970e-14로 사전1e−10 기준을 통과했습니다. 고정 first-o SMOKE의 실제 BF16
module output/cast/arithmetic residual은 [별도 기록](../../results/p2_r1/smoke_matrix_verification.json)에 있습니다.

</details>

embedding/lm_head의 tie, norm·RoPE·attention·bias 정책은 원형을 유지했습니다.
모든 조건은 BF16/batch1/seq256/SDPA/eval/no_grad/use_cache=False, TF32 및
BF16/FP16 reduced reduction OFF, native 전체256위치 head를 사용했습니다.

## 데이터·실험

| 용도 | 원본 단위·수 | 이번 사용 |
| --- | --- | --- |
| CAL-FIT | 원 P1 PG19 32책 | 기존 token·offset 그대로, P2 자체 경로 K만 새 적합 |
| EVAL-ID | 새 PG19 64책 | 16,320 valid next tokens/method |
| SHIFT | 새 ArXiv 16 content group | 4,080 valid next tokens/method, ID와 별도 |
| SMOKE | 기존 별도1개 | 명시적 재사용, 평가 분모 제외 |

기존209개 고유 입력(PG19 177책·ArXiv32 group)을 확인 가능한 manifest 합집합으로
제외했습니다. 새 source 전송은0 bytes입니다. PG19의 canonical book ID·content/token
hash 및 정규화 창, 5-gram 중복을 검사했습니다. 창 비교19,880회의 최대 Jaccard는
0.014799였습니다. 전체 원문 근접 중복과 사전학습 미노출은 확인하지 않았습니다.
Pile는 article ID가 없어 content-group 기준입니다.

spec → 입력 동결 → CPU/SMOKE → RTN → HALF → QUARTER → 전체 artifact 동결
순서였습니다. **모든 저장본을 동결할 때 fresh forward는0회**였습니다. 그 뒤 BF16 →
QUARTER → HALF → RTN 순으로 새 ID·SHIFT를 평가했습니다. 데이터 seed6101521,
구축/평가 순서6101522, bootstrap6101523입니다. CAL 밖 입력은 fit 함수가 거절합니다.

add_special_tokens=False, chat/BOS/EOS 추가 없이 문서당256 input tokens입니다.
위치0..254의 원문 다음255 tokens를 NLL/argmax로 평가하고, 기존16개 고정 위치에서
full-vocabulary KL(BF16∥method)을 측정했습니다. 자유 생성이나 agent 성공률은 아닙니다.

## 결과

### 새 PG19 64책

| Method | NLL (nats/token) ↓ | PPL ↓ | KL(BF16∥method) ↓ | Argmax flip | ΔNLL vs RTN |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 3.815629 | 45.405313 | 0.000000 | 0.000% | -0.237892 |
| RTN | 4.053521 | 57.599922 | 0.438809 | 31.740% | +0.000000 |
| QUARTER (0.25) | 3.915147 | 50.156465 | 0.290688 | 25.447% | -0.138374 |
| HALF (0.5) | 3.911792 | 49.988432 | 0.236665 | 24.179% | -0.141730 |

![PG19 NLL/PPL](../../figures/p2_r1/01_pg19.png)

QUARTER는 BF16 대비 RTN의 추가 NLL 손실 중 58.17%를 회복했습니다.
HALF는 59.58%였습니다. 두 후보 모두 BF16보다 NLL이 높습니다.
이 회복률의 분모는 RTN−BF16 NLL gap이며, 모델 정확도 회복률과 다릅니다.

### 새 ArXiv 16개

| Method | NLL (nats/token) ↓ | PPL ↓ | KL(BF16∥method) ↓ | Argmax flip | ΔNLL vs RTN |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2.348638 | 10.471299 | 0.000000 | 0.000% | -0.173230 |
| RTN | 2.521869 | 12.451842 | 0.320093 | 22.549% | +0.000000 |
| QUARTER (0.25) | 2.540615 | 12.687474 | 0.310653 | 23.162% | +0.018747 |
| HALF (0.5) | 2.633587 | 13.923621 | 0.318018 | 24.975% | +0.111718 |

![ArXiv NLL/PPL](../../figures/p2_r1/02_arxiv.png)

### 문서 paired 비교

| Corpus | Candidate − RTN | Mean ΔNLL | Paired 95% CI | PPL change | Lower / higher / tie docs | Verdict |
| --- | --- | ---: | --- | ---: | --- | --- |
| PG19 | QUARTER (0.25) | -0.138374 | [-0.149414, -0.127515] | -12.923% | 64 / 0 / 0 | LOWER_NLL |
| PG19 | HALF (0.5) | -0.141730 | [-0.156537, -0.126159] | -13.214% | 63 / 1 / 0 | LOWER_NLL |
| ArXiv | QUARTER (0.25) | +0.018747 | [-0.005723, +0.044885] | +1.892% | 5 / 11 / 0 | INCONCLUSIVE |
| ArXiv | HALF (0.5) | +0.111718 | [+0.072058, +0.153853] | +11.820% | 2 / 14 / 0 | HIGHER_NLL |

주 비교는 PG19 QUARTER−RTN 한 개입니다. 나머지는 보조 pointwise 구간입니다.
문서 쌍을 함께10,000회 재표집했고, 원 CAL32·모델·고정 강도에 조건화한 문서
불확실성입니다. 새로운 CAL·모델 모집단까지의 구간은 아닙니다. PPL은 exp(totalNLL/count)이며
문서 PPL 평균을 사용하지 않습니다. PPL1%는 프로젝트 참고값이고, 이번 주 비교의 크기는
-12.923%입니다. HALF의 PG19 점추정이 조금 낮아도 주 후보를 교체하지 않았습니다.

![전체 문서 차이](../../figures/p2_r1/03_paired_document_nll.png)

### 저장량·offline 비용

| Item | Actual measurement |
| --- | ---: |
| INT4 core files per method | 196 |
| Core bytes: RTN = QUARTER = HALF | 247,733,136 B each |
| Header total / scales total / packed codes total | 7,056 / 27,525,120 / 220,200,960 B |
| BF16 projection tensors, calculated from actual tensors | 880,803,840 B |
| Common unchanged parameters, counted once with tied weights | 311,296,000 B |
| RTN offline construction | 9.898 s |
| QUARTER offline construction | 546.999 s |
| HALF offline construction | 515.160 s |
| Cumulative worker wall, all 3 processes | 1141.530 s / 10,800 s |
| Torch peak allocated / reserved | 1,634,240,512 / 1,845,493,760 B |
| Maximum observed post-forward whole-device used sample | 3,331,915,776 B |
| Native whole-model forwards / fresh metric rows | 7,535 / 320 |


원 모델의 공통 부분은 별도 장부이며 core 파일만으로 standalone 모델이 되지 않습니다.
구축 시간에는 native CAL capture·FP64 적합·양자화·분해 진단·I/O가 포함됩니다.
전체 worker는 main 1130.317초(새 프로세스 reload 포함)와 등록된
보조 SMOKE 11.214초의 합입니다. 데이터 준비 15.606초와 CPU 감사/그림/문서화는 별도입니다.
내부 단계 시간은 포개지는 값이므로 다시 합산하지 않습니다. Torch peak와 장치 사용
표본은 다른 지표이며, 장치 표본을 연속 Windows 전체 peak로 부르지 않습니다.

![저장량과 offline 구축비용](../../figures/p2_r1/04_storage_and_offline_cost.png)

## 분석

PG19에서는 약한 전층 보정이 같은 INT4 bytes에서 RTN 손실을 상당 부분 줄였습니다.
ArXiv에서는 QUARTER의 KL이 RTN보다 조금 작아도 gold NLL은 점추정상 높았습니다.
HALF는 PG19에서 더 낮은 점추정과 함께 ArXiv에서 더 큰 손실을 보였습니다.
teacher 분포 근접도·정답 NLL·도메인 전이는 구분해서 볼 필요가 있습니다.

이 결과는 PG19 CAL에 맞춘 보정의 도메인 의존과 일관됩니다. 개별 층의 유일한
원인이나 보편 법칙을 입증한 것은 아닙니다. CAL의 세 항 분해는 국소 계산을
검산하며 전층 NLL 기여율로 더하지 않습니다. P1b의 여섯 효과와 이번 전층 효과도
측정 단위가 다릅니다. [P1b 검토 반영](REVIEW_RESPONSE.ko.md)에 기존 분모·한계를 정리했습니다.

## 결론

등록된 PG19 주 질문은 `LOWER_NLL`로 답할 수 있습니다. 동일 저장 형식에서 고정0.25의
전층 자체 경로 보정이 실제로 이득을 냈습니다. SHIFT의 결과를 고려하면 즉시 다른
도메인·모델·agent로 확대할 근거는 제한적입니다. P2 핵심 비교는 결과와 관계없이
4조건·320문서 측정 모두 완료했고, 누락/실패 cell은0개입니다.

| 판정 축 | 완료 범위 |
| --- | --- |
| 구현·저장·전층 품질 | 196 projections × 3 quantized models, 588 files, 392 K fits, PASS |
| 주 통계 비교 | PG19 QUARTER−RTN: LOWER_NLL |
| SHIFT | QUARTER INCONCLUSIVE with higher point estimate; HALF HIGHER_NLL |
| 검산 | 새 고유 P2 tests9 PASS; 기존15+P1b10은 회귀 재실행. 독립 scalar 감사320행·10비교·20구간 변환 PASS |
| 별도 실제 검증 | 새 GPU 프로세스588 files 독립 decode·4모델 SMOKE full-logit SHA 일치; CPU full SMOKE 행렬3건·고정 CAL normal equation8건 PASS |
| 보존 | 기존 Case014423파일·private4482파일·repo tracked6351파일·model6파일(전체 weight 포함)·P1b ZIP hash 일치, index/HEAD/remote 변경 없음 |
| 실행 경계 | packed storage → dense BF16 품질 reference. TTFT/TPOT/GPU busy/native packed speed 미측정 |
| 제외 | P3·4B·agent·새 selector·패키지 변경·Git stage/commit/push/PR/merge·Pages/Drive 미실행 |

## 재현·출처

[실제 CLI와 CPU/GPU 범위](REPRODUCE.md), [등록 spec](specs/p2.json),
[원자료](../../results/p2_r1/metrics/), [판정](../../results/p2_r1/decision.json),
[보존 검사](../../results/p2_r1/preservation.json), [자원 기록](../../results/p2_r1/resource_summary.json).

작업 branch는 research/case014-quantization-compensation, Git base/HEAD는
09194c3f2e2c991feff5b6abab0cf8ab81a07052입니다. 실행 source SHA256은
`0bb627be395672209a1fc7aa5d6c9801a8486b0511bc21545ad252ddedb1eca4`입니다. source tree hash·Git commit·모델 revision·개별
artifact hash를 구분합니다. 모델 revision은 c1899de289a04d12100db370d81485cdf75e47ca,
전체 local weight file SHA256은 f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b입니다.
이는 실제 로컬 identity 검증이며 새 공식 digest 또는 D 원본 대조는 아닙니다.

Guo의 [논문v1](https://arxiv.org/html/2610.16020v1)과 과거
[source audit](../../references/source_audit.json)은 이론·선행 실험의 출처입니다.
이번 P2는 프로젝트의 ridge/RTN 자체 구현이며 저자 공식 QEP 전체 재현이 아닙니다.
문헌 HTML/PDF 확보·절 대조는 기존 P0 기록을 인용하며 P2에서 새 문헌 실험으로 세지 않습니다.
Codex가 구현·분석·문서화를 보조했습니다. 사람의 독립 심사를 거쳤다고 표시하지 않습니다.

검토 ZIP은 코드·경량 원행·spec·manifest·보고서·그림을 포함합니다. model/K/INT4
weights/X/Z/logits/원문은 제외합니다. 공개 묶음의 scalar 감사는 외부 GPU 계산 자체의
재현과 다릅니다. 로컬 private 경로와 전체 재현재료 inventory는 ZIP 밖에 보존합니다.
