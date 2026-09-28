# Review actions

| Request | Implementation/evidence | Scope |
|---|---|---|
| Verify the original archive and local case | `original_identity.json`, anchored 140-file inventory | Original ZIP SHA, size, CRC, paths, duplicates and local member bytes checked |
| Preserve historical sources and edited documents | `original_docs/`, `restore_original.py`, `integrity.py` | Original scientific source/data/status retained; document overlay restored separately |
| Recompute primary outcomes | Original `analysis/aggregate.py` plus independent `posthoc/analyze.py` | Same predictions, input pairing, original bootstrap seeds/level; no forward |
| Compare storage-dependent readout effects | `posthoc/data/interaction.csv` | Paired sequence interaction, post-hoc pointwise 95% interval |
| Inspect position bands and first-error movement | `posthoc/data/bands.csv`, `error_shifts.csv`, `items.json` | All inputs and exact token denominators; no favorable subset |
| Inspect conditional margins | `posthoc/data/summary.json` → `conditional_margin` | Unavailable: item-level margin/logit arrays were not recorded |
| Verify controls and solver status | `posthoc/data/summary.json`, `solver.csv`, `checkpoint_head_audit.json` | All18 fits reviewed, SHORT no-op and selected MIXED caps preserved |
| Make bilingual reading paths | Root/case README, REPORT, PORTFOLIO; generated EN/KO Case011 pages | One case, six visible navigation sections, nine report sections |
| Check original synthetic contracts | `run_checks.py --synthetic` | Runs restored originals with CPU Torch; learned checkpoint replay not repeated |
| Verify publication identity and prevent checksum substitution | `verify_publication.py`, publication tests | Fixed original anchors checked separately from current public manifest |
| Publish and deploy normally | Existing GitHub PR/CI and gated Pages workflow | Remote receipts kept outside the self-hashed package; no bypass or direct main push |

The recorded primary RMST0 decrease and the out-of-fitting-range CE decrease are
both visible in the main summary. Solver budget exhaustion is an interpretation
limit, not relabeled as convergence. Original research counts are not copied into
the current test count. The actual run receipts distinguish local, CI and live
HTTPS/browser checks.
