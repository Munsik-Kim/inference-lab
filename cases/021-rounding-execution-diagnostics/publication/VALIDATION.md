# Publication validation scope

The current entry point runs on Python3.11.15 / NumPy1.26.4 / Matplotlib3.10.9 in Linux/WSL, using an existing CPU environment read-only. No installation or research-environment upgrade is required by the publisher. Public consumers can use the dedicated venv recipe in REPRODUCE.

Local receipts are written outside the checkout. The executable commands are `publication/run_checks.py --mode quick --tests --output NEW_PATH` and `--mode full --output NEW_PATH`, invoked with isolated Python (`-I -B`), CUDA_VISIBLE_DEVICES empty and offline model flags. A clean copy includes only manifest-listed public files and no original research path in sys.path. Full checks regenerate the public figures; they do not recompute the excluded private logits/weights.

The Case021 workflow runs public quick checks, mutation tests, docs checks and the Case021 static display test. Portfolio CPU checks cover the integrated repository site. These are CPU record/display checks, not new GPU measurements. This file describes the validation contract; run outcomes, actual commands, head identities and wall/CPU costs are recorded in external delivery receipts and CI logs. Pending or failed CI is not represented as PASS here.

PNG images are inspected locally; SVG XML and generated HTML/link checks are static checks. No browser engine was available at initial inspection (Node/Playwright unavailable; browser tool failed on the WSL cwd), so browser rendering/keyboard/mobile interaction are NOT_RUN unless separately evidenced. Pages is manual and is not dispatched by this work.

Public quick+tests and full figure regeneration passed locally; the same commands passed from a clean manifest-only copy with cwd outside the research trees. The new overview PNG/SVG files were byte-identical in that same CPU environment and both PNGs were opened for axes/units/legend review. These are local self-checks.

The shared showcase entrypoint requires Python3.12 syntax and its established script-directory imports. It is run with `-B` (not `-I`), PYTHONPATH removed, using existing CPU Python3.12.12/NumPy2.4.6 locally; the original Portfolio workflow pins3.12.14/NumPy2.3.5. The Case021 CI uses Python3.12.14 and its declared small dependencies. No existing environment was upgraded. Initial publication check failures and exact fixes are retained in [attempts](attempts.json).

Worker accounting is independently re-summed from public scalar ledger extracts, including failed/wait records, rather than trusting saved aggregate PASS flags. The original large private raw audit itself is not rerun.
