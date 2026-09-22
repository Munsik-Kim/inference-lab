# Changelog

## 0.1.1

- Reject inconsistent effective metric coverage within a task/version before writing output. Distribution-derived NLL/Brier and the kind of KL are part of the contract. Correctness-only tasks and different contracts across tasks remain supported. Item order and dictionary key order do not change acceptance.
- Compute positive-probability KL using differences of logarithms, retaining subnormal probabilities. Exact missing candidate mass still produces null plus an explicit infinity flag. JSON never contains NaN/Infinity literals.
- Homogeneous historical counts, sample identities and source process exits are unchanged. Tiny FP64 rounding differences from the stable KL formula are software changes, not new model measurements.

## 0.1.0

Historical reviewed implementation. Two edge defects above were reproduced; do not use the historical wheel for a new installation.
