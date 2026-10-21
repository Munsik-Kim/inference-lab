# Omitted expansion criterion and interpretation of P2

Protocol deviation: BF16 KV with free output completed 5/8 DEV tasks, below the original 6/8 expansion criterion. The local plan and execution gate omitted this success threshold and proceeded to P2 after checking completion, freezing and sealing. No separate waiver approval was found. We disclose this deviation and present P2 as an exploratory measurement on new synthetic tasks.

This explanation neither changes nor discards the measured scores. `COMPLETE` means the registered execution matrix completed, not that a readiness criterion passed or that P2 was a confirmation study. Measurement recalculation, evidence for sealing before recorded outputs, and gate compliance are separate judgments.

## Original intent versus local execution

| Item | Evidence and finding |
| --- | --- |
| Original intent | The newly supplied ChatGPT-side preserved excerpt specifies BF16 KV+C0 ≥6/8 and no automatic HOLDOUT execution when genuine model failures leave DEV below the criterion. |
| Local plan | Original `PLAN.ko.txt` lines6–8 say only sections0–5 were received. Line13 schedules P2 after P1 completion and source freeze. |
| Local registration | `specs/protocol.json`, version `v1.1-local-registration-r1`, contains no 6/8 success criterion. |
| Actual code | `src/agent_contract_cost/experiment.py` lines134–149, `check_p2_gate`, and lines417–439, `seal-holdout`, check64 completed cells, freeze and seal, without reading success counts. |
| Three DEV failures | `DEV-B00/B01/B02_K0_C0` generated two read JSON objects in the first response, rejected as PARSE_ERROR by the single-object contract. No events executed; state version0; not budget exhaustion. |
| Waiver evidence | No separate user waiver/decision was found in the inspected local plan, provenance or attempt records. Codex local continuation is not user approval. |
| Unknown | How much of the original prompt local Codex received before execution, and where the criterion was lost in delivery/registration, remain unknown. |

[Original local PLAN](PLAN.ko.txt) · [Frozen protocol](specs/protocol.json) · [Frozen execution code](src/agent_contract_cost/experiment.py) · [Earlier review copy](publication/reviews/CASE017_REVIEW_ADDENDUM_KO.sanitized.txt)

## Chronology and preservation

All times below are local self-records on 2026-10-20 UTC, not external timestamp attestation.

| Step | UTC | Record |
| --- | --- | --- |
| Protocol registration | 07:03:28.201933 | `specs/protocol.json` |
| P1 freeze | 07:21:27.507465 | `provenance/execution_freeze_p1.json` |
| P1 complete | 07:25:17.672620 | `results/p1_r1/completion.json` |
| HOLDOUT input manifest | 07:26:44.012959 | `provenance/holdout_input_manifest.json` |
| HOLDOUT seal, outputs_observed=0 | 07:26:44.040365 | `provenance/holdout_seal.json` |
| P2 freeze | 07:26:59.253593 | `provenance/execution_freeze_p2.json` |
| P2 worker start | 07:27:24.042065 | `provenance/worker_journal.jsonl` |
| First P2 cell start | 07:27:42.423255 | `results/p2_r1/attempts.jsonl` |
| First SSE created | 07:27:42Z, second resolution | First response of `HOLDOUT-B05_K1_N` |

The input-manifest timestamp is written after generation, not at generation start; SSE created is not exact client receipt time. Records place generation, sealing and P2 freeze before the first recorded P2 response. All **35 common P1/P2 frozen files are byte/hash identical**. P2 adds only two HOLDOUT inputs and a seal. The missing 6/8 condition has not been retroactively inserted into original source or protocol.

## Newly supplied original-intent reference

The reference is section7 of `Case017_Fresh_20261020.txt`, version1. The user reports ChatGPT-side creation at 2026-10-20 15:42:13 KST and rechecking/delivery on 2026-10-21. The [received excerpt](publication/reviews/gate_excerpt_received_20261021.ko.txt) is preserved separately; the full original file was not newly obtained locally.

It establishes the criterion in the original intent, not that local Codex received the whole prompt before execution. The earlier addendum's statement that the full text was unavailable describes its then-local investigation, so that original remains unchanged. The public review copy replaces only personal paths with placeholders; its [transformation receipt](publication/reviews/addendum_copy_receipt.json) separates original and public-copy hashes.

## Publication interpretation

- Retained: original scores, complete256-response replay, independent endpoint/paired arithmetic, recorded pre-output HOLDOUT seal and common source identity.
- Disclosed deviation: DEV5/8 below6/8, omitted local success gate, and no separate user waiver record found.
- Scope: exploratory comparison of policies and KV configurations on new synthetic tasks, not real-business confirmation, noninferiority or equivalence certification.

This explanation was added on 2026-10-21. It is not a backdated waiver or revised preregistration and requests no new experiment.
