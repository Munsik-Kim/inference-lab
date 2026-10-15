# P2 local handoff

Status: `COMPLETED_P2_CORE`. All4 methods and320 fresh-document metric rows completed.
Primary PG19 QUARTER−RTN: LOWER_NLL. SHIFT QUARTER is inconclusive with a higher point
estimate; HALF is higher NLL. See REPORT.ko.md / REPORT.md and results/p2_r1/decision.json.

The original P0/P1/P1b files and historical decisions remain frozen. New material is confined
to followups/p2_core, results/p2_r1, figures/p2_r1 and the new followup navigation file.
Existing environments and WSL model snapshot were reused without installation or cleanup.

The private P2 root is labeled P2_PRIVATE in the review export. It contains actual model/Python
paths in identity.private.json, original command/log receipts, input token arrays,392 K matrices,
588 core INT4 files, teacher/current-path CAL captures and SMOKE arrays. Their locations and
hashes are recorded in the external private manifest; these files are excluded from the ZIP.

Use REPRODUCE.md for actual CLI boundaries. Running the GPU workflow again requires the
external model/inputs and an actual repository checkout; the original runner records Git HEAD.
The extracted review ZIP supports CPU tests/scalar reanalysis without those private materials.
Historical EVAL data are replay inputs, never a new fresh sample on rerun.

The ZIP and its actual review_verification.json/sha256 sidecar live in the separate local review
directory. Verification unpacks into a new folder and repeats CPU tests and raw-only reanalysis.
No remote publication, Git stage/commit, new selector, P3/4B/agent or packed serving was run.
