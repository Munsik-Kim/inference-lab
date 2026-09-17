# Project card

## English

**Decision Stability Audit for Low-Precision Attention**

**Problem.** Similar aggregate accuracy can hide changes in confidence, individual answers and wrong-to-wrong choices. Local attention error alone does not determine whether a task answer improves or regresses.

**Implementation.** A scoped Qwen3-0.6B layer-13 prefill intervention compares native BF16 with two pinned SageAttention bundles. Fact-first synthetic retrieval, comparison and restricted-code tasks supply independent gold. Exact token manifests, full-output finite checks, kernel-route probes and own-cache generation keep the comparison controlled. Python analysis and an offline HTML explorer connect retained option logits to individual outcomes.

**Technical decisions.** Score one observed-prompt next token, never a forward containing future answers. Preserve baseline-conditioned stress separately from the unfiltered standard set. Keep all other layers and decode BF16. Distinguish gold regression/gain from baseline disagreement. Require complete validity evidence rather than a successful sample or hardcoded boolean. Measure the actual model-prefill boundary instead of reusing an operator speedup.

**Measured evidence.** Among 192 standard scenarios, A_PUBLIC/V4 changed 5/8 choices, including 2/4 gains and 3/4 wrong-to-wrong flips, with no observed standard regressions. Mean paired gold-NLL intervals spanned zero. The 46 selected stress scenarios included 2/1 regressions. The separate 24-scenario structured-generation diagnostic had no strict schema-valid outputs in any arm. [Analysis](ANALYSIS.md) reports model timing and uncertainty without a deployment recommendation.

**Limits.** One small model, one prefill layer and a narrow synthetic family. BF16 code accuracy was 15/64. Four-option scores do not measure general instruction following. No non-inferiority or broad quality-preservation claim follows from a small difference. The generation floor constrains that secondary diagnostic.

**Contribution and reuse.** The case contributes intervention integration, paired gold-based metrics, explicit validity gates, CPU audit paths and an offline explorer. Qwen, Transformers and SageAttention supply the model and kernels. Cases004/005 informed small interface/reference components. Codex assisted implementation, execution, analysis and writing; no independent human or third-party replication is claimed.

**Reproduce:** [commands](REPRODUCTION.md), [raw evidence](results/README.md), [offline explorer](demo/index.html).

## 한국어

**문제:** 전체 정답률이 비슷해도 개별 답, 정답 확률, 틀린 답의 종류는 달라질 수 있다. attention 출력 오차와 실제 답변 변화 사이를 같은 입력으로 확인했다.

**구현:** Qwen3-0.6B의 13번 층에서 프롬프트 prefill attention만 바꾸었다. 독립적으로 계산한 정답, 동일 토큰 입력, 전체 출력 유효성 검사, 실제 커널 호출 기록을 묶었다. 이후 decode와 다른 층은 BF16을 유지했다. 저장한 logits와 스칼라 기록을 CPU에서 다시 계산하고 HTML로 항목별 근거를 확인할 수 있게 했다.

**측정:** 표준 192개에서 A_PUBLIC/V4의 선택은 5개/8개 바뀌었다. 정답으로 바뀐 사례는 2개/4개였고, 정답에서 오답으로 바뀐 사례는 관측되지 않았다. 그러나 별도 경계 스트레스 46개에서는 2개/1개가 정답에서 오답으로 바뀌었다. 평균 정답 NLL 변화 구간은 두 설정 모두 0을 포함했다. 별도의 짧은 JSON 생성은 모든 설정에서 형식 유효율 0/24였다.

**한계:** 자체 합성 입력, 작은 모델 하나와 층 하나의 결과다. BF16 코드 문제 정답은 15/64로 낮았다. 일반적인 품질 보존, 실서비스 가속, 안전성 또는 배포 적합성을 입증하지 않는다.

**기여:** 새 양자화 방법이나 커널을 개발한 것이 아니라, 기존 커널의 모델 내 개입을 통제하고 정답·기준선·유효성·비용을 구분하는 평가와 근거 탐색 도구를 구현했다. Codex가 코드·실행·분석·문서 작성을 지원했다.

## 30-second walkthrough

1. Open the offline explorer and read the standard-set NLL intervals and four correctness cells.
2. Select a gain or a wrong-to-wrong flip; inspect the exact prompt, gold and B/A_PUBLIC/V4 probabilities. An absent regression category in standard data stays empty.
3. Switch to the BF16-conditioned stress set, then inspect the separate generation and model-cost records. Selection-conditioned rates and variable-length generation timing answer different questions.

## Evidence-backed resume bullets

- Implemented a scoped, one-layer Qwen prefill intervention with actual kernel-route checks, full-output validity gates and cache-separated generation diagnostics.
- Measured paired gold scores and decisions on 192 standard and 46 baseline-conditioned stress scenarios; retained gains, regressions, wrong-to-wrong flips and the failed structured-output diagnostic.
- Built CPU scalar/bootstrap verification and a network-free item-level HTML explorer from immutable measurement records and exact input manifests.

## Post-hoc readout follow-up

The [separate supplement](supplemental/readout-ties-v1/PORTFOLIO.md) adds exact-tie partitions and a same-hidden-state FP32 output-head diagnostic. It reuses 238 existing scenarios, preserves native operational outcomes, and distinguishes output-rounding evidence from other readout arithmetic differences. This is controlled evaluation and reproducible numerical diagnosis, not a new quantizer or an improved deployment configuration.

한국어: [보충 포트폴리오 카드](supplemental/readout-ties-v1/PORTFOLIO.md)는 기존 238개 시나리오의 동률과 동일 hidden state 출력 계산을 분리해 설명한다. 새로운 확인 표본이나 품질 개선 결과로 소개하지 않는다.
