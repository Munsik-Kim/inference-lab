# Case017 — Stored-result replay and model re-execution

Use Python3.10+ standard library from the root of a normal repository clone. No dependency installation, model, tokenizer, GPU or personal path is required. The output directory must be new. Approximately2.83MB of compressed evidence expands to roughly250MB; allow1GB free space.

```bash
python3 -B cases/017-agent-contract-cost-bench/publication/run_checks.py --output /tmp/case017-public-check-new --tests
```

The command checks the public manifest and original bytes, then restores529 retained files plus five pinned Case016 imports to original-view/cases/017-agent-contract-cost-bench/. It reparses/replays64 P1 and192 P2 records into new fixtures, recalculates an independent endpoint and original10000-draw bootstrap, and compares display JSON and the review ledger. Original84 unit tests, eight postprocess tests and new publication tests are reported separately. verification.json and logs record results. This replays stored records, not GPU inference.

To regenerate only the display CSV/JSON, use the following command. It independently checks endpoints from events/final state rather than merely summing saved success flags. Full fixture replay is the first command.

```bash
python3 -B cases/017-agent-contract-cost-bench/publication/derive.py --output /tmp/case017-display-new
```

GPU repetition requires separate authorization and pinned snapshot/environment/private runtime. Actual historical commands/fields are preserved in the [original guide](publication/originals/pre-publication/REPRODUCE.md#gpu-reproduction-boundary). Completed roots reject overwrites. Do not rewrite old freezes for another machine. The public CPU command runs no new inference, server, retokenization, installation or P3, and does not authorize repeating the omitted6/8 expansion gate.

All six original root entry documents retain their bytes under originals. Old generated demo/browser files and regenerable replay fixture states are omitted. Original cells, inputs, schemas, sources and receipts necessary for all256 endpoint replays are retained. See the [full mapping and omission reasons](publication/provenance/original_mapping.json).
