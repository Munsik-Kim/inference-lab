# Paired comparison

Paired items within each task. Intervals are pointwise descriptive; separate benchmarks are not averaged.

## case013.long.16384@case013-r2-first-json-exact-and-required-fixture-actions-v1

Pairs: 12; evidence kinds: historical.

```json
{
  "n_pairs": 12,
  "counts": {
    "baseline_correct": 10,
    "candidate_correct": 7,
    "both_correct": 7,
    "both_wrong": 2,
    "all_answer_disagreement": 4,
    "correct_to_wrong": 3,
    "wrong_to_correct": 0,
    "wrong_to_wrong": 1,
    "correctness_flip": 3
  },
  "metrics": {
    "delta_accuracy": {
      "mean": -0.25,
      "pointwise_95_interval": [
        -0.5,
        0.0
      ],
      "unit": "fraction"
    }
  },
  "regression_denominator": 10,
  "gain_denominator": 2,
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
