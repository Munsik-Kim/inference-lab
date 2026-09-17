# Result inventory

`raw/paired/` contains measured scalar records for development, standard evaluation, selected boundary stress and dependent length support. `raw/pool/` preserves the BF16-only selection pool. `raw/secondary/` and `raw/timing/` keep separately scoped generation and model-cost measurements. `study/summary.json` and `study/pairs.json` are derived, CPU-recomputable outputs.

The top-level `records.json`, `summary.json`, `pairs.json`, `run_state.json` and any other preparation-state files predate GPU execution (2026-09-16). They are retained for the original CPU-harness audit, not used as current study results. The older `reproduce_analysis.py`, `verify_results.py` CLI and `build_demo.py` reproduce that empty preparation state. Current commands are `analyze_study.py`, `audit_study.py`, `audit_timing.py`, `plot_study.py` and `build_study_demo.py`.

Mock fixtures live only in tests. No full Q/K/V, hidden-state or vocabulary-vector dumps are included. Public sufficient statistics allow calculation checks; they do not independently reproduce a GPU kernel output.
