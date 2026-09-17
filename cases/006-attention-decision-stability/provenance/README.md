# Provenance timeline

The 2026-09-16 CPU preparation is retained in `environment.json`, `resource_gate.json`, `validation.json`, the old `cpu_tests.log`, `configs/design_draft.json`, and top-level empty result files. They are historical, not current execution status. The initial resource check was conservative; GPU process registration alone did not prove active competing work.

Current execution evidence is `execution_environment.json`, `integration_pass.json`, `prefill_decode_smoke.json`, the original `integration_head_shape_diagnostic.json`, and scalar records under `results/raw/`. Source and extension identities are in `source_fingerprints.json`; its source-inspection flags describe when that identity inventory was captured, while actual execution routes are in `integration_pass.json`.

`corrections.json` records pre-evaluation generator, resource-gate and native LM-head shape checks. `development_summary.json` and `development_cyclic_label_check.json` precede the design freeze. `input_audit.json` covers exact token replay and fact-derived gold. `study_scalar_audit.json` and `timing_audit.json` independently recalculate public scalars. `private_vector_audit.json` additionally checks complete local vocabulary vectors; those vectors are intentionally not redistributed.

`code_snapshot.json` is the final publication-build inventory. It is distinct from the immutable measurement-code hashes in the two freezes. `publication_validation.json` records the final tests, regeneration, hygiene and preservation checks. No original measurement or frozen experimental threshold is changed by publication formatting.
