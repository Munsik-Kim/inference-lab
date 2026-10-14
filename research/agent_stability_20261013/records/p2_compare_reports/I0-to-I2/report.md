# Paired comparison

Paired items within each task. Intervals are pointwise descriptive; separate benchmarks are not averaged.

## case013.followup.P2.original8.FREE@diagnostic-dev-v1:a717480d30c679013b8b4772e0391daac3fbe095d4f4cde63ce7cb3c8fb3bb08

Pairs: 8; evidence kinds: measurement.

```json
{
  "n_pairs": 8,
  "counts": {
    "baseline_correct": 6,
    "candidate_correct": 4,
    "both_correct": 4,
    "both_wrong": 2,
    "all_answer_disagreement": 2,
    "correct_to_wrong": 2,
    "wrong_to_correct": 0,
    "wrong_to_wrong": 0,
    "correctness_flip": 2
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
  "regression_denominator": 6,
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
