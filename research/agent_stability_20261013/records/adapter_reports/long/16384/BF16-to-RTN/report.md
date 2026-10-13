# Paired comparison

Paired items within each task. Intervals are pointwise descriptive; separate benchmarks are not averaged.

## case013.long.16384@case013-r2-first-json-exact-and-required-fixture-actions-v1

Pairs: 12; evidence kinds: historical.

```json
{
  "n_pairs": 12,
  "counts": {
    "baseline_correct": 12,
    "candidate_correct": 10,
    "both_correct": 10,
    "both_wrong": 0,
    "all_answer_disagreement": 2,
    "correct_to_wrong": 2,
    "wrong_to_correct": 0,
    "wrong_to_wrong": 0,
    "correctness_flip": 2
  },
  "metrics": {
    "delta_accuracy": {
      "mean": -0.16666666666666666,
      "pointwise_95_interval": [
        -0.4166666666666667,
        0.0
      ],
      "unit": "fraction"
    }
  },
  "regression_denominator": 12,
  "gain_denominator": 0,
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
