# Portfolio card

## English

**Problem:** Small changes in answer-choice counts can conceal ties and score sensitivity at the output head.

**Implementation:** Added an exact-tie audit and same-hidden-state full-vocabulary FP32 shadow readout to a controlled one-layer Qwen study. Reused verified B hidden vectors, bounded candidate replay, retained native results and checked a separate FP64 four-label path.

**Evidence:** All 5/8 native standard-set flips involved a top tie. Shadow readout produced 3/3 flips, including new changes. All 476 replays matched original native full vectors. This distinguishes native decisions from a post-hoc arithmetic diagnostic without claiming better quality.

**Limits:** One small model, one layer, one GPU and synthetic templates; 238 existing scenarios, not new confirmation. No new generation or timing. Full tensors are private; public scalar reanalysis checks calculation consistency.

**Contribution and reuse:** Diagnostic harness, exact-key joins, independent scalar audit and offline explorer with Codex assistance; Qwen, SageAttention and the original Case006 adapters are reused and attributed. [Reproduction](README.md#inspect-and-reproduce).

30 seconds: read the native/shadow table, select a native flip, switch to the labelled FP32 view, then inspect all four scores and the original tie set. Empty categories remain empty.

- Audited native tie involvement and separate output-head arithmetic on 238 fixed Qwen scenarios without replacing the original evidence.
- Verified 476 replayed full native vectors and 714 CPU FP64 option checks under a frozen bounded supplement.
- Built CPU-recomputable paired tables and a measured-data offline explorer tested through file:// in Edge.

## 한국어

문제: 답변 선택 변화가 동률 처리와 출력 head의 계산 정밀도에 얼마나 연결되는지 구분했다.

구현: 기존 Qwen 단일 층 실험에 정확한 동률 분류, 동일 hidden state의 FP32 전체 head 계산, 별도 FP64 선택지 계산과 오프라인 근거 탐색기를 추가했다. 기준 hidden state는 재사용하고 누락된 후보만 제한적으로 replay했다.

결과: 표준 집합의 원래 선택 변화 5/8건은 모두 최고값 동률을 포함했다. FP32 그림자 readout에서는 각각 3건이며 새로운 변화도 있었다. 성능 향상이나 품질 보존을 주장하지 않는다.

기여: Codex 지원으로 분석·유효성 확인·재계산·표현을 구현했다. Qwen/SageAttention의 모델과 커널은 원저자 작업이다. 238개는 기존 합성 입력이며 새로운 확인 자료가 아니다.
