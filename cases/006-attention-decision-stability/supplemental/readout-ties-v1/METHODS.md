# Supplementary methods

## Boundary and local freeze

This protocol was written after original results were known. Its SHA256 was recorded before viewing new FP32 head results. It is a local post-hoc freeze, not external preregistration or unseen confirmation. `protocol.json` fixes 238 existing IDs, token hashes, option IDs [32,33,34,35], gold, split, 4096 length, readouts, tolerances, budgets and statistical definitions. `readout_source_freeze.json` additionally hashes the executed runner before its first readout. Analysis and presentation code were completed afterward against these definitions.

Model: Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`. SageAttention: `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`. Original public adapters and source/binary/model hashes were checked, without upgrades. Layer 13 only, B1/Hq16/Hkv8/D128, square causal prompt prefill. A_PUBLIC uses explicit INT8-QK/FP8-PV, per_warp, smooth_k=True, smooth_v=False, fp32+fp16, V scale maximum 2.25. V4 uses explicit INT8-QK/FP16-PV, fp32, with native V conversion. Decode and all other layers are unchanged; no decode was run here.

Installed Transformers Qwen code applies `model.norm` after all blocks, then `lm_head(hidden_states[:, -1:, :])` when logits_to_keep=1. The head is bias-free, with no post-projection transform. Original hooks retained the last normalized vector (1,1024), promoted losslessly from BF16 through float32 to NumPy float64. These values round-trip exactly to BF16. They are not pre-normalization block outputs. Head input is reshaped to (1,1,1024); the full (151936,1024) head produces (1,1,151936) before selecting options.

## State integrity and private storage

Original public payloads were matched to their private ledger identities and full native vectors. B NPZ files retained actual final-normalized hidden states, layer13 Q/K/V hashes and common reference outputs. Candidate files retained only full logits; their hidden norms cannot reconstruct vectors. Thus 238 B states were reused and only the 476 A_PUBLIC/V4 cells were replayed, once each. Six fixed smoke forwards preceded core readouts. Every replay matched the entire original native vector bitwise, plus options, Q/K/V hashes, routing and full validity. No failed attempt, excluded core cell or retry occurred.

All 28 attention outputs, all 28 decoder blocks, final normalized hidden states and complete native vocabulary outputs were checked using the frozen diagnostic path. New shadow vectors were checked in full. Missing validity evidence blocks completion. The head's BF16 weight hash and dtype were unchanged after execution, including its shared embedding storage. Temporary tensors and full vectors remain private.

## Readouts and controls

H_NATIVE is the retained native BF16 final-position full-vocabulary vector. Reapplying the same-shaped native head to each exact h reproduces it bitwise. H_FP32 uses a detached cloned float32 copy of the same represented BF16 W and float32 promotion of the same h, on the same RTX 5080. It does not restore an unavailable training-precision checkpoint and is never fed into generation.

The installed Torch 2.13 legacy `allow_tf32=False` API was inspected and used consistently, with autocast disabled. No incompatible newer setter was mixed in. The previous flag is restored in `finally`; original replay settings remain unchanged. This is IEEE-FP32-intent GPU arithmetic, not a claim of exact dot products.

C1 widens native values to float32/float64 and checks exact equality. C2 rounds full shadow values through verified native BF16 storage, then compares full vectors and options. C3 computes four independent CPU NumPy float64 dot products on represented values for every cell; six smoke cells also use Python `math.fsum` products. FP32-versus-FP64 validation was fixed before core readouts to atol=1e-4, rtol=1e-5 for a 1024-term float32 reduction. These tolerances validate arithmetic; **exact ties and winners never use a tolerance**. All score/gap differences and winner disagreements remain recorded.

C2 option agreement supports a rounding contribution for these particular labels. Full-vector disagreement prevents claiming that only final storage rounding changed. C3 checks four rows, not a complete high-precision model or vocabulary oracle. Stored native values are BF16-representable; observed minimum positive gap 0.125 does not imply universal 0.125 spacing, since BF16 spacing depends on magnitude.

## Counts, scores and readout comparisons

The top set contains only entries exactly equal to the finite maximum. Earliest A/B/C/D index reproduces the original deterministic argmax. Equal lower-ranked values are not top ties. NaN/Inf are invalid, not ties. The four B/candidate strata are unique/unique, tied/unique, unique/tied and tied/tied; their counts reconstruct every original comparison. Gold membership of the top set is a secondary set-valued description and does not replace deterministic correctness.

Choice NLL and Brier use four-label conditional softmax in CPU float64; Brier is the sum over four options. Gold margin differs from winner gap. Full label mass uses the measured complete-vocabulary log normalizer. `NLL_full = NLL_choice - log(label_mass)`. Choice KL is four-option KL; the separate full-vocabulary shadow KL field was computed from complete private vectors, not top-k values.

Candidate versus B is always computed within one readout. Original native B tie groups are held fixed for one sensitivity analysis; current FP32 B tie groups are reported separately. No new stress selection is made. Flip persistence, disappearance and emergence match by exact scenario ID. The optional NLL interaction is a same-input task-balanced difference of differences, not a causal fraction of model error.

## Uncertainty and propagation

STANDARD has 64 scenarios per task, 192 total. The 46 selected stress IDs are separate baseline-conditioned observations. The 1428 native/shadow records represent 238 scenarios, not 1428 samples. Within-task paired scenario bootstrap uses seed617093 and 5000 draws, identical indices across arms/readouts, equal task weighting, and linear 2.5/97.5 percentiles. Conditional ratios use resampled numerator/denominator counts; zero-denominator draws are undefined and counted. These are post-hoc pointwise descriptive intervals. A zero-width interval from zero events is not zero population risk. The new seed yields slightly different native interval endpoints from the preserved original analysis; it does not overwrite original intervals.

Existing local scalars distinguish pooled32query×16head error against common FP32 from candidate-vs-native-BF16 error. Last-query metrics remain separate. `hidden_boundary_summary.csv` names each boundary and shows absolute RMS, B reference RMS and relative error; ratios of relative errors across boundaries do not measure absolute amplification. Spearman uses finite, nonconstant scenario-level ranks, without p-values. Sparse flips/regressions preclude a predictor comparison. R uses both outputs, is undefined at zero gap, and R>=1 does not imply a flip.

## Reproduction scope

CPU commands in README recompute all scores, strata, task summaries and intervals from lightweight measured records. The scalar checker uses a separate math.fsum/log/exp path, explicit first-max rule and resampling calculation. This is arithmetic consistency, not third-party kernel replication. Full-vector/hidden/weight checks need excluded private data.

To reproduce the bounded GPU supplement with those original cached assets, use a new private output directory outside the original case:

```bash
python -B scripts/run_readout.py --original-case <original-case-dir> --original-run <original-private-run> --snapshot <pinned-cached-snapshot> --work-dir <new-private-dir>
```

The runner checks original frozen source/model fingerprints, private cells, resource availability and the supplement hash. A mismatch or prior failure stops the path; no automatic retry or fallback is used. Successful exports in this review retain no full vectors, model weights or private paths. Reconstructing private hidden states from public scalar summaries is impossible; full independent GPU reproduction first requires the original capture/scoring procedure.
