# Portfolio card

## English

**Interaction-Aware Structured MLP Pruning**

**Problem.** Individually small channel contributions can reinforce or cancel when removed together. Does interaction-aware selection improve which groups to remove at an identical budget?

**Implementation.** Decomposed one Qwen3-0.6B SwiGLU MLP into 16 groups, estimated signed interactions, exhaustively selected deletion sets, and physically sliced gate/up/down projections. Fixed calibration/development/held-out inputs, mapping tests, scoped restoration, native model scores, paired timing and independent CPU checks connect the selection to an actual smaller module.

**Measured evidence.** Both importance-based selections have lower mean local error than each of 20 frozen random removal sets per budget; this is a descriptive local reference, not random full-model evaluation or global optimality. On 192 held-out synthetic prompts, 25% pairwise selection lowered mean local error by 0.2003 percentage points [0.1145, 0.2848] versus independent selection. At 50% both chose identical groups. The frozen overall result is COMPLETED_NO_CLEAR_TRANSFER. Local MLPs were faster, while whole-model prefill intervals crossed 1. At 25%, PAIRWISE reduces baseline flips from 48/192 to 12/192 and conditional regressions from 15/94 to 2/94, while gold NLL is worse than INDEPENDENT by +0.069860 nats in the existing post-hoc comparison. Preserving the baseline and improving gold scores are different objectives. [Random and fidelity–gold tables](README.md#preserving-b-is-different-from-scoring-the-correct-answer).

**Contribution and limits.** The contribution is the controlled harness, structural surgery and auditable comparison, assisted by OpenAI Codex. Qwen/Transformers supply the model; HOPE is separate MoE motivation. This is one model/layer, 512-token synthetic prompts and no retraining. Weak comparison/code baseline performance limits utility claims; deployment is NOT_ASSESSED.

**Walkthrough.** Open the [offline explorer](demo/index.html), filter a task and prompt ID, then compare INDEPENDENT_4 and PAIRWISE_4 groups, local error, gold transitions and retained scores. Inspect the same 50% selection and the timing intervals; follow [CPU reproduction](REPRODUCTION.md) to the distinct scalar checker. Configuration timing is from a six-prompt subset, not each displayed request.

## 한국어

**상호작용을 고려한 구조화 MLP 압축 실험**

**문제.** 각각 작은 채널 기여도 함께 삭제하면 상쇄되거나 커질 수 있습니다. 같은 예산에서 상호작용을 고려한 선택이 더 나은 삭제 집합을 만드는지 조사했습니다.

**구현.** Qwen3-0.6B의 SwiGLU 한 층을 16그룹으로 분해하고 부호가 있는 상호작용을 추정했습니다. 삭제 집합을 전수 탐색하고 gate/up/down 행렬을 실제로 축소했습니다. 입력 분할 고정, 인덱스 검사, 안전한 모듈 교체·복원, 모델 점수, 대응 시간 측정과 독립 CPU 검산을 연결했습니다.

**측정 근거.** 두 중요도 선택은 예산별 고정 random 집합20개 각각보다 평균 국소 오차가 낮았습니다. 이는 유한한 국소 reference의 기술통계이며 random 전체 모델 평가나 전역 최적성이 아닙니다. 미관측 합성 입력 192개에서 25% 상호작용 선택의 평균 국소 오차는 독립 선택보다 0.2003퍼센트포인트 낮았습니다(감소량 95% 구간 [0.1145, 0.2848]). 50%는 같은 그룹을 선택해 고정 종합 판정은 COMPLETED_NO_CLEAR_TRANSFER입니다. MLP 자체는 빨라졌지만 전체 모델 prefill 구간은 1배를 포함했습니다. 25%에서 PAIRWISE는 기준선 대비 선택 변경을48/192에서12/192, 조건부 회귀를15/94에서2/94로 줄였지만, 기존 사후 직접 비교의 정답 NLL은 INDEPENDENT보다 +0.069860 nats 나빴습니다. 기준선 보존과 정답 점수 개선은 다른 목표입니다. [무작위 기준과 충실도·정답 표](README.ko.md).

**기여와 한계.** 프로젝트 기여는 비교 도구, 실제 구조 축소와 추적 가능한 검증이며 Codex가 구현·실행·분석·문서를 지원했습니다. Qwen/Transformers 모델과 HOPE의 별도 MoE 발상을 구분합니다. 한 모델·한 층·길이 512 합성 입력의 재학습 없는 연구입니다. 비교·코드 과제의 낮은 기준선 정답률이 실용적 해석을 제한하며 배포는 NOT_ASSESSED입니다.

**시연.** [영어 UI 탐색기](demo/index.html)에서 같은 과제·프롬프트 ID의 INDEPENDENT_4와 PAIRWISE_4를 고르고 그룹, 국소 오차, 정답 전환과 점수를 비교합니다. 50%의 동일 선택과 속도 구간을 확인하고 [CPU 재검산 안내 — 영어](REPRODUCTION.md)로 이동합니다. 표시되는 설정별 시간은 6개 입력 측정값이며 개별 화면 요청의 시간은 아닙니다.
