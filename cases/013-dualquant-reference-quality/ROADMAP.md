# Follow-up priorities / 후속 연구

모든 항목은 **PROPOSED / NOT_RUN**입니다. 이번 공개 작업은 저장된 기록의 정리와 CPU 검산만 수행했습니다. 새 GPU 실험·formal TEST·upstream issue는 실행하거나 발행하지 않았습니다.

1. **P1 — Identical-history first agent divergence.** Compare correct and actually emitted tool-identifier token log-probabilities under BF16/RTN/D16/D4 at the first shared-history branch; inspect reselection after identical error feedback. This is originalDEV cause diagnosis, not generalization validation. Logits are currently absent.
2. **P2 — Aligned probability diagnostics.** First use saved per-token losses for window-paired NLL and gold NLL at the existing KL positions. Such readback is possible; full new candidate logits are not assumed available. Any additionalGPU calculation needs a separate registered experiment.
3. **P3 — Packed artifact A/B/C.** Original dense versus actual-format dense versus Marlin, with blockwise first differences, matched dtype/reduction/rescale. Validate full-model correspondence before testing packed4B quality.
4. **P4 — Minimal controlled variants.** Compare inner mode and BF16 scale projection separately; add RTN W4A4 to identify activation-side effects. Register each method change and record any DEV-based selection history.
5. **P5 — Common serving comparison.** StockGPTQ with matched engine/cache/parser and requests. Only after validated deployment should formal120task data/scorers/criteria be frozen and executed. Current records establish none of that stage's speed/capacity benefits.

Do not select settings from already observedTEST, add candidate-specific tool aliases to old scores, or overwrite old failures. These priorities propose tests of hypotheses in [failure analysis](FAILURE_ANALYSIS.md); they do not establish causes.
