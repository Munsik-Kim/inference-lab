# Sources and review scope

Case014 implements an independent ridge/RTN diagnostic. The repository Apache-2.0 license applies to project code; no third-party source code was copied or relabeled.

Theory reference: Tianhao Guo, arXiv2610.16020v1. The official code link, pinned commit and license identity are recorded in [source audit](references/source_audit.json). The paper PDF/HTML and its figures are not redistributed here. The independent user-supplied four-code fixture uses different numbers from the paper's Theorem22.

External assets: Qwen/Qwen3-0.6B, PG19 and the Pile subset. Their model/data licenses remain applicable. No pretrained or quantized model weights, document text, activation traces or logits are included. `tests/fixtures/matrix_identity.json` is a tiny synthetic numerical/codec fixture, not model weights.

Publication and experiments are separate. P0 has actual CPU evidence. P1 is prepared but not run; its storage gate is described in the decision. P2/P3 and all remote publication actions were not run. Codex assisted the implementation and documentation; no independent human review is claimed.
