# Paired comparison

Paired items within each task. Intervals are pointwise descriptive; separate benchmarks are not averaged.

## case013.agent.base.FREE@case013-r2-first-json-exact-and-required-fixture-actions-v1

Pairs: 8; evidence kinds: historical.

```json
{
  "n_pairs": 8,
  "counts": {
    "baseline_correct": 8,
    "candidate_correct": 0,
    "both_correct": 0,
    "both_wrong": 0,
    "all_answer_disagreement": 7,
    "correct_to_wrong": 8,
    "wrong_to_correct": 0,
    "wrong_to_wrong": 0,
    "correctness_flip": 8
  },
  "metrics": {
    "delta_accuracy": {
      "mean": -1.0,
      "pointwise_95_interval": [
        -1.0,
        -1.0
      ],
      "unit": "fraction"
    }
  },
  "regression_denominator": 8,
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
