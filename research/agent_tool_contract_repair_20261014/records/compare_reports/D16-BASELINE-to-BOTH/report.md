# Paired comparison

Paired items within each task. Intervals are pointwise descriptive; separate benchmarks are not averaged.

## case013.tool-contract.original8.FREE@original8-mitigation-dev-v1:e648790f18cea8086f3678b0ee81815ef2bea6403d1d4b6016d6065b9b9346fa

Pairs: 8; evidence kinds: measurement.

```json
{
  "n_pairs": 8,
  "counts": {
    "baseline_correct": 6,
    "candidate_correct": 8,
    "both_correct": 6,
    "both_wrong": 0,
    "all_answer_disagreement": 0,
    "correct_to_wrong": 0,
    "wrong_to_correct": 2,
    "wrong_to_wrong": 0,
    "correctness_flip": 2
  },
  "metrics": {
    "delta_accuracy": {
      "mean": 0.25,
      "pointwise_95_interval": [
        0.0,
        0.625
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
