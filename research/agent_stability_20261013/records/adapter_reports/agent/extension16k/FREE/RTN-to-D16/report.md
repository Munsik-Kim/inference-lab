# Paired comparison

Paired items within each task. Intervals are pointwise descriptive; separate benchmarks are not averaged.

## case013.agent.extension16k.FREE@case013-r2-first-json-exact-and-required-fixture-actions-v1

Pairs: 4; evidence kinds: historical.

```json
{
  "n_pairs": 4,
  "counts": {
    "baseline_correct": 4,
    "candidate_correct": 3,
    "both_correct": 3,
    "both_wrong": 0,
    "all_answer_disagreement": 0,
    "correct_to_wrong": 1,
    "wrong_to_correct": 0,
    "wrong_to_wrong": 0,
    "correctness_flip": 1
  },
  "metrics": {
    "delta_accuracy": {
      "mean": -0.25,
      "pointwise_95_interval": [
        -0.75,
        0.0
      ],
      "unit": "fraction"
    }
  },
  "regression_denominator": 4,
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
