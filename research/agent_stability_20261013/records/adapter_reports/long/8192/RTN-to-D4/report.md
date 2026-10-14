# Paired comparison

Paired items within each task. Intervals are pointwise descriptive; separate benchmarks are not averaged.

## case013.long.8192@case013-r2-first-json-exact-and-required-fixture-actions-v1

Pairs: 12; evidence kinds: historical.

```json
{
  "n_pairs": 12,
  "counts": {
    "baseline_correct": 11,
    "candidate_correct": 8,
    "both_correct": 7,
    "both_wrong": 0,
    "all_answer_disagreement": 5,
    "correct_to_wrong": 4,
    "wrong_to_correct": 1,
    "wrong_to_wrong": 0,
    "correctness_flip": 5
  },
  "metrics": {
    "delta_accuracy": {
      "mean": -0.25,
      "pointwise_95_interval": [
        -0.5833333333333334,
        0.08333333333333333
      ],
      "unit": "fraction"
    }
  },
  "regression_denominator": 11,
  "gain_denominator": 1,
  "source_process_exits": {
    "baseline": [
      0
    ],
    "candidate": [
      0
    ],
    "scope": "Retained scalar calculations; nonzero source exits remain execution failures."
  }
}
```
