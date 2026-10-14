# Changelog

## 0.1.2

- Add the stdlib-only `export-case013` command. Export frozen long-context and agent records with verified public-source hashes, method/artifact identities, original input pairing, canonical extraction and retained process exits.
- Preserve required file-edit/validation actions even when a final JSON matches gold. Keep cohorts, modes and length variants separate; record candidate32K exclusions without inventing zero scores.
- Keep PPL/NLL and previously measured KL in a separate continuous summary. This adapter adds no inference, logits or formal quality eligibility.

## 0.1.1

- Reject inconsistent effective metric coverage within a task/version before writing output. Distribution-derived NLL/Brier and the kind of KL are part of the contract. Correctness-only tasks and different contracts across tasks remain supported. Item order and dictionary key order do not change acceptance.
- Compute positive-probability KL using differences of logarithms, retaining subnormal probabilities. Exact missing candidate mass still produces null plus an explicit infinity flag. JSON never contains NaN/Infinity literals.
- Homogeneous historical counts, sample identities and source process exits are unchanged. Tiny FP64 rounding differences from the stable KL formula are software changes, not new model measurements.

## 0.1.0

Historical reviewed implementation. Two edge defects above were reproduced; do not use the historical wheel for a new installation.
