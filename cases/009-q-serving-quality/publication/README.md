# Case009 reviewed publication

[English report](../REPORT.md) · [한국어 보고서](../REPORT.ko.md) · [Current status](status.json) · [Review actions](review_actions.md)

This public tree preserves the original 57-file Case009 and eleven-file package source identity. Five explanatory case documents have recorded before/after hashes and restorable copies. Other historical case bytes, including RUN_STATE and SHA256SUMS, remain unchanged. The 0.1.0 wheel is retained only as historical identity; install the corrected 0.1.1 wheel.

`PUBLICATION_SHA256SUMS` covers the exact current case, package source and 0.1.1 wheel, excluding itself. `original_identity.json` records the original review archive and the anchored original source map. Rehashing altered raw data cannot satisfy the original-map checks. Download ZIP metadata stays outside the ZIP to avoid circular hashes.

From the repository or the standalone publication ZIP root, with Python3.12 and NumPy installed in a separate CPU environment:

```bash
OUT="$(mktemp -d)"
python -B cases/009-q-serving-quality/publication/verify_publication.py --root . --output "$OUT/current.json"
python -B cases/009-q-serving-quality/publication/restore_review.py --root . --output "$OUT/original"
python -B "$OUT/original/cases/009-q-serving-quality/scripts/verify_public.py" --case "$OUT/original/cases/009-q-serving-quality" --output "$OUT/original-scalars.json"
python -B -m unittest discover -s "$OUT/original/cases/009-q-serving-quality/tests" -v
python -B -m unittest discover -s cases/009-q-serving-quality/publication/tests -v
python -B cases/009-q-serving-quality/publication/posthoc/analyze.py --case cases/009-q-serving-quality --output "$OUT/posthoc.json"
```

The current tree must not be passed to the frozen exact-inventory verifier. Restore the original tree first; never rewrite original SHA256SUMS. NumPy is needed for original study tests; current identity, posthoc and the comparison core are standard-library paths.

Install `downloads/diova_compare-0.1.1-py3-none-any.whl` into a new venv with `pip install --no-index --no-deps`, then run `check_cli.py --root . --python /path/to/new-venv/bin/python --output /path/to/new-check`. It exports all official metric/filter files and verifies counts and failed source exits. CLI item-bootstrap intervals do not replace original subject-cluster intervals.

[Short lifecycle](../supplemental/lifecycle-v1/README.md) is a separate GPU diagnostic. Its 24 clean attempts did not reproduce the original error and do not change historical quality statuses. CPU CI checks its ledger only. Full GPU benchmark retries, new serving measurements, data-v2 model evaluation and full-vocabulary KL were not run.

The standalone ZIP includes this case, package source/tests/examples/license and corrected wheel. Historical Case008 adapters and full-site/previous-case checks need a full repository clone; their data are not duplicated in this ZIP. All independent raw prose, generated rationales, model weights, full tensors, environments and private logs are excluded.
