# Attribution and scope

This is a separate local follow-up to Case010 v1. It implements a persistent numerical-failure state in the serialized cache, resumable evaluation history, model-free audits, fixed-checkpoint inference and bounded precision/state diagnostics. Codex assisted source inspection, implementation, testing and writing. An independent calculation path within this project is not an external independent reproduction.

Project-authored code follows the repository's [Apache-2.0 LICENSE](../../LICENSE). Upstream model software and the referenced research retain their own authorship and terms. ComplexKDA is consumed from a separately supplied checkout pinned at `ef9d108d1692387cae37f5b2d539a71826a127c1`; its task, model, recurrence and original readout are upstream work. The upstream checkout and the three locally trained checkpoint files are not redistributed in this v2 package. Any separate distribution of upstream software must preserve its own license and notices.

The unchanged local v1 files under [source/v1_reference](source/v1_reference/) are project reference copies, not a new implementation of the upstream model. Their original paths and byte identities are recorded in [v1_source_identity.json](provenance/v1_source_identity.json). The original review ZIP and full v1 result tree remain separate; the v2 package contains only the new follow-up artifacts and the stated reference copies.

## References inherited from v1

The following bibliography and its role are carried from the retained v1 notice. This v2 work did not perform a fresh paper search, newly verify these publications, or reproduce their numerical results:

- Julien Siems et al., *Complex KDA: Understanding and Enhancing the Expressivity of Kimi Delta Attention* (2026), arXiv `2609.24797v1`: upstream model and finite-group task context.
- Tao Zhang et al., *DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization* (2026), arXiv `2608.27513v1`: related recurrent-state precision allocation. The new comparator here uses the explicitly frozen retained CAL scores and actual serialized budget; it is not a reproduction of DAMP.
- Jiwan Chung, Heechan Choi and Seon Joo Kim, *Rethinking State Tracking in Recurrent Models Through Error Control Dynamics* (2026), arXiv `2605.07755v1`: related state tracking and error-control context.
- Ismail Erbas, Xavier Intes and Vikas Pandey, *When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference* (2026), arXiv `2609.04490v1`: related write-back and residual-memory baselines.

Residual transport identities, corrected-state feedback and byte packing are not claimed as new algorithms or theorems. This follow-up's empirical scope is determined by its own [frozen protocol](protocol_v2.json), retained observations, actual serialized bytes and stated audit boundaries.

## Reviewer availability and evidence boundaries

The exact requested files `CASE010_REVIEW_KO.md`, `diova_case010_independent_review_20260923.zip` and `probe_restart_failure.py` were not found in the accessible search scope. This does not assert that inaccessible or remote copies do not exist. The [availability receipt](results/historical-reanalysis/review_artifact_availability.json) records that limit.

The [synthetic failure probe](source/probe_v1_failure.py) was recreated from the request's described contract, and its [receipt](results/v1_failure_probe_from_request.json) is explicitly labelled `SYNTHETIC_CONTRACT_REPRODUCTION_FROM_REQUEST`. It was not the missing reviewer script; it does not verify unobserved reviewer numbers or constitute a trained-model score.

The v1 historical reanalysis reuses old TEST observations and is post-hoc. Fresh checkpoints are reported separately, without model-seed pooling. Precision/trace cohorts remain diagnostic. FP64 promotes fixed FP32 coefficients and readout parameters and is not exact arithmetic. Public receipts do not include private model weights, private resumable state bodies or full hidden-state/logit histories. See [REPRODUCTION](REPRODUCTION.md) for the distinct model-free, synthetic-test and actual-inference requirements.
