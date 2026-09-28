# Additional analysis of the same Case011 records

This layer reads the saved predictions, per-position score sums and solver logs. It does not load a model, fit a head or run new inputs. Its evidence label is `POST_HOC_SAME_RECORDS`; the original primary comparison is recomputed separately with its unchanged interval definition.

Run from the Case011 directory with Python and NumPy. The output directory must not already exist:

```bash
python -B publication/posthoc/analyze.py --output /tmp/case011-posthoc-check
python -B -m unittest discover -s publication/tests -p 'test_posthoc*.py' -v
```

- [Protocol](protocol.json): fixed before this additional calculation. The interaction uses 5,000 jointly paired base-sequence resamples, seed `71101 + checkpoint_seed`, linear percentiles and descriptive pointwise 95% intervals. These intervals are separate from the original primary 98.333…% intervals.
- [Summary](data/summary.json): all three primary recomputations, 18 arm summaries, three storage-interaction rows, 36 position-band comparisons, six first-error shift distributions, control checks and all 18 solver traces summarized.
- [Per-input first-error movement](data/items.json): all 6,144 checkpoint/storage/input rows, including longer, shorter and unchanged cases. These rows reuse 1,024 paired base sequences and are not 6,144 independent inputs.
- [Position-band table](data/bands.csv), [interaction table](data/interaction.csv), [first-error shifts](data/error_shifts.csv), [solver table](data/solver.csv).
- [Primary change figure](data/primary_rmst0.svg) and [CE change figure](data/ce_by_band.svg). The corresponding numeric values are in the summary and tables.

CE uses the stored sum divided by its valid-score observation count. Accuracy and answer transitions use all recorded token positions, including invalid predictions as incorrect. A band's first-error count is divided by all 1,024 sequences; it is not a hazard conditioned on prior survival. The error-shift distribution uses the number of consecutive correct tokens before the first error, with the original censoring rule.

Per-item margins and full logits were not retained. Per-position margin sums therefore cannot answer how margins changed specifically at newly wrong items. This is reported as unavailable; no logits, conditional margin or KL values are inferred.

The current read-only tensor comparison is linked through [the head audit](../checkpoint_head_audit.json). The original solver logs show that SHORT performed no optimizer update. For MIXED, seeds 0 and 1 reached the 200-iteration cap; seed 2 used the optimizer's small-change/direction stop. None of these selected MIXED heads met the gradient tolerance. The final FP64 parameter arrays were not retained in the public records, so an FP64 displacement value is not invented.

Source paths, byte counts and hashes are recorded in `summary.json`; each generated file is identified in [the output manifest](data/output_manifest.json). Re-running into a new directory produces deterministic JSON, CSV and SVG files.

## 한국어 안내

이 사후 분석은 저장된 예측·위치별 점수 합·solver 로그만 읽습니다. 모델 실행과 재적합은 하지 않습니다. 위 명령으로 기존 주 비교를 재계산하고, Native와 INT8의 효과 차이·위치별 오류 변화·입력별 최초 오류 이동을 확인할 수 있습니다.

같은 입력을 함께 재표집한 저장 방식 상호작용은 기술적 95% 구간으로 표시합니다. 원래 주 비교의 98.333…% 구간을 바꾸지 않습니다. 위치 구간의 최초 오류 수는 전체 1,024개 입력이 분모이며, 그 직전까지 살아남은 입력만을 분모로 하는 hazard가 아닙니다.

입력별 margin과 전체 logits는 저장되지 않아, 새로 틀린 입력만 골라 margin을 분석하는 항목은 미확인으로 남깁니다. SHORT의 최종 FP32 값은 원형과 같고, MIXED의 고정 iteration 종료 상태도 그대로 공개합니다. 점수 개선을 최초 오류 지연이나 수렴한 최적 판독기의 증명으로 바꾸지 않습니다.
