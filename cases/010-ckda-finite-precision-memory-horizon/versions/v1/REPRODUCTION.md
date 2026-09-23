# Reproduction and audit paths

[English](README.md) · [한국어](README.ko.md) · [Methods](METHODS.md)

Run from this case directory. Every `--output` below must be a **new path outside the case tree**. Keep `PYTHONDONTWRITEBYTECODE=1`; no command below installs into an existing model environment. Source, NumPy, Torch and plotting versions are recorded in [environment provenance](provenance/environment.json).

## 1. Model-free CPU storage and scalar audit

The packing, integer-group and survival components require Python and NumPy. They neither load a checkpoint nor invoke CUDA. In an existing compatible CPU environment:

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
python -B -m unittest discover -s tests -p 'test_packed.py' -v
python -B -m unittest discover -s tests -p 'test_groups_survival.py' -v
python -B -m unittest discover -s tests -p 'test_online.py' -v
python -B -m unittest discover -s tests -p 'test_toy.py' -v
```

The online tests launch a fresh child process and compare continued packed state and reads. Temporary test outputs stay outside this case. `CKDA_AUDIT_OUTPUT` can name a private external audit directory.

The retained correctness bits allow CPU reconstruction of first failures, F(T), step accuracy and restricted mean. The independent auditor also checks binomial-bound endpoints, actual shared byte buffers and total-byte ledgers. Commands for the packaged result directories are listed with the final verification receipt. Auditing these recorded labels does not rerun CKDA or the GPU training.

For this review tree, the exact inventory and complete recorded-label audit are:

```bash
python -B scripts/verify_publication.py --root .
python -B scripts/audit_records.py \
  --toy-original results/toy-run --toy-repair results/toy-repair-v1b \
  --learned results/learned-seed0 --learned results/learned-seed1 \
  --learned results/learned-seed2 --output /NEW/EXTERNAL/scalar-audit.json
```

The audit can run without model weights or upstream code. It verifies recorded correctness, IDs, counts, byte buffers and identities; it cannot independently reconstruct model predictions from missing checkpoint weights. The retained checkpoint hashes identify the separate local models. Private tensor/model-forward verification is not silently substituted for this scalar audit.

Regenerate tables and figures in a fresh external directory with the recorded NumPy and Matplotlib versions:

```bash
python -B scripts/summarize_results.py \
  --toy-original results/toy-run --toy-repair results/toy-repair-v1b \
  --learned results/learned-seed0 --learned results/learned-seed1 \
  --learned results/learned-seed2 \
  --isolated-timing 0=results/timing-isolated/seed0.json \
  --isolated-timing 1=results/timing-isolated/seed1.json \
  --isolated-timing 2=results/timing-isolated/seed2.json \
  --output /NEW/EXTERNAL/tables-and-figures
```

CSV values and summary content are reproducible. PDF file timestamps can differ; byte-identical PDF regeneration is not required. The checksummed delivered figures are tied to the included summarizer source and recorded results.

## 2. Fresh numerical mechanism run

```bash
python -B scripts/run_toys.py --smoke --output /NEW/EXTERNAL/toy-smoke
python -B scripts/run_toys.py --output /NEW/EXTERNAL/toy-primary
```

The smoke is N8/T64 and cannot be reported as the N512/T2048 primary experiment. The full run freezes geometry, CAL bases, bit maps and input seeds before TEST. Current code includes documented executor and RNG repairs. The retained first v1 attempt and its seven explicitly versioned repairs remain separate; rerunning current code does not recreate old defective metadata or correlated stochastic streams.

Toy runtime fields include diagnostic calculations and are not a complete-call speed comparison.

## 3. Learned CKDA — pinned upstream required

The official trained S3 checkpoint is not assumed to be available. The local study trains its own small model. Obtain and verify the official source revision listed in [upstream identity](provenance/upstream.json); a different checkout is rejected. No large language-model weights are required.

Set `CKDA_UPSTREAM` to that clean local checkout. The learned adapter and evaluation tests additionally need the recorded compatible Torch/Transformers/FLA dependencies. Reuse a suitable environment read-only or create a separate one; do not upgrade an existing experiment environment to force a match. If continuing from the CPU-only examples, restore visibility of the authorized training GPU before using `--device cuda`.

```bash
export CKDA_UPSTREAM=/LOCAL/PINNED/ComplexKDA
export PYTHONDONTWRITEBYTECODE=1
python -B -m unittest discover -s tests -v
python -B scripts/train_small.py \
  --upstream "$CKDA_UPSTREAM" --protocol configs/protocol.json \
  --output /NEW/EXTERNAL/training-seed0 --device cuda --seed 0
```

The protocol limits updates, walltime, architecture and checkpoint selection. The runner saves `final.pt` only after the entire scheduled run; an interrupted checkpoint has a different name and status. Seed1/2 require the predefined seed0 DEV gate and remaining total budget; this command does not automatically start them. Training on CPU is supported by the runner but is not presented as equivalent GPU timing.

With a validated completed local checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=''
python -B scripts/evaluate_learned.py \
  --upstream "$CKDA_UPSTREAM" --checkpoint /LOCAL/training-seed0/final.pt \
  --protocol configs/protocol.json --runtime configs/evaluation_runtime.json \
  --output /NEW/EXTERNAL/evaluation-seed0
```

For the original 26-arm menu, the evaluator uses the recorded checkpoint for every arm, validates the native replay boundary and freezes codec/source identities before evaluating TEST. Those summaries retain the original 26×7×3=546 comparison family. All prefix lengths reuse the same fixed maximum-length sequences. It uses sequential reference computation and actual per-token write-back; it does not run a fused low-bit CUDA kernel. CAL/DEV/TEST IDs and numerical masks are recorded; no native-correct subset replaces the primary 512 inputs. The separate post-hoc budget supplement below adds four allocations on the reused TEST sample; it is not part of this original freeze.

The exact checkpoint plus packed cache can be tested in a new process using:

```bash
python -B scripts/check_restart.py \
  --upstream "$CKDA_UPSTREAM" --checkpoint /LOCAL/training-seed0/final.pt \
  --evaluation /LOCAL/evaluation-seed0 --output /NEW/EXTERNAL/restart-seed0
```

This uses a separate fixed DEV input and regenerates model projections from the checkpoint. It transfers no full-precision hidden state. Checkpoints remain private local review artifacts unless a later explicit publication decision includes them.

After **all three** primary indexes exist, the isolated timing wrapper invokes the unchanged frozen benchmark once per seed, serially in new processes. The paths below must point to the completed evaluations and their exact checkpoints:

```bash
python -B scripts/benchmark_isolated.py --upstream "$CKDA_UPSTREAM" \
  --evaluations /LOCAL/evaluation-seed0 /LOCAL/evaluation-seed1 /LOCAL/evaluation-seed2 \
  --checkpoints /LOCAL/training-seed0/final.pt /LOCAL/training-seed1/final.pt /LOCAL/training-seed2/final.pt \
  --output /NEW/EXTERNAL/isolated-timing
```

Timing uses CPU Torch with two threads and CUDA hidden. It includes the sequential call and packed cache work, with canonical coefficient-table gathering rather than newly projecting all model coefficients inside the timed loop. It excludes gold, diagnostic norms, file I/O and checkpoint loading. The original embedded timing records are preserved but can contain contention from the three parallel primary evaluations.

## 4. Calibration fitting cost

After all three isolated inference timing records are complete, run the separately frozen [`POST_HOC_COST_ONLY` measurement](configs/calibration_cost.json):

```bash
export CUDA_VISIBLE_DEVICES=''
python -B scripts/measure_calibration_cost.py --upstream "$CKDA_UPSTREAM" \
  --evaluations /LOCAL/evaluation-seed0 /LOCAL/evaluation-seed1 /LOCAL/evaluation-seed2 \
  --checkpoints /LOCAL/training-seed0/final.pt /LOCAL/training-seed1/final.pt /LOCAL/training-seed2/final.pt \
  --isolated-timing /NEW/EXTERNAL/isolated-timing \
  --config configs/calibration_cost.json \
  --output /NEW/EXTERNAL/calibration-cost
```

The wrapper checks primary and timing identities, then uses one fresh two-thread CPU process per seed, serially. It times exactly one unchanged `calibrate()` call on CAL 64×32, seed 1001: initial INT4 encoding, causal rollout, covariance, eigensolver and mixed-channel scoring. Model/input/table preparation, reference loading, comparisons, hashing and file I/O are outside the timer. Every returned array is compared with the retained `calibration.npz`; differences are reported, and the candidate calibration is never replaced. `returned_stats_array_nbytes` counts returned statistics storage, **not peak memory**. This phase generates no TEST inputs and makes no model or threshold selection.

## 5. Separate budget supplement on reused TEST

The frozen [`POST_HOC_BUDGET_AUDIT_SAME_TEST` configuration](configs/budget_supplement_v1.json) adds four CAL-ranked allocations: `MIXED_4_8_TOP2`, `MIXED_4_8_TOP5`, `MIXED_6_8_TOP1`, and `MIXED_6_8_TOP4`. Their byte feasibility and existing CAL ranks were frozen before low-rank TEST results existed, while other original TEST results already existed. The [budget ledger](configs/budget_supplement_v1.budgets.csv) records their exact target caps. This is a descriptive additional audit using the same 512×2048 TEST sample, not fresh independent confirmation or a global optimality search.

Run only after the original primary evaluations, isolated timings, and calibration-cost measurements have all finished. Each primary directory must retain its original `training-result.json` as well as the indexed evaluation artifacts. The supplement is pinned to the recorded checkpoint, CAL, table and source hashes; an arbitrary newly trained checkpoint will be rejected. The examples below therefore require those retained local artifacts. New training would require a separately declared study, without rewriting this freeze.

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for seed in 0 1 2; do
  python -B scripts/run_budget_supplement.py \
    --primary "/LOCAL/evaluation-seed${seed}" \
    --checkpoint "/LOCAL/training-seed${seed}/final.pt" \
    --upstream "$CKDA_UPSTREAM" \
    --isolated-timing /NEW/EXTERNAL/isolated-timing \
    --calibration-cost /NEW/EXTERNAL/calibration-cost \
    --config configs/budget_supplement_v1.json \
    --output "/NEW/EXTERNAL/budget-supplement-seed${seed}"
done
```

One fresh process evaluates all four additions for each seed. Allocation uses unchanged saved CAL scores; there is no new calibration fit or outcome-based allocation selection. Original 26-arm results and their 546-family bounds remain unchanged. Supplementary bounds use (26+4)×7×3=630 comparisons; joint-family statements require original bounds to be re-expressed at 630 in a separate derived view. Supplementary complete-call timing remains `NOT_MEASURED`; original-arm timings cannot stand in for these allocations. Failed attempts are retained without favorable retries.

The independent recorded-data audit needs neither checkpoints nor model inference:

```bash
python -B scripts/audit_budget_supplement.py \
  --supplement /NEW/EXTERNAL/budget-supplement-seed0 \
  --supplement /NEW/EXTERNAL/budget-supplement-seed1 \
  --supplement /NEW/EXTERNAL/budget-supplement-seed2 \
  --output /NEW/EXTERNAL/budget-supplement-audit.json
```

## 6. Identity and interpretation

For the delivered files, audit the three `results/budget-supplement-seed0` through `seed2` directories with the preceding command's `--supplement` arguments. Recreate the separate joint630-family tables without checkpoints or upstream code:

```bash
python -B scripts/summarize_budget_supplement.py \
  --learned results/learned-seed0 --learned results/learned-seed1 \
  --learned results/learned-seed2 \
  --supplement results/budget-supplement-seed0 \
  --supplement results/budget-supplement-seed1 \
  --supplement results/budget-supplement-seed2 \
  --output /NEW/EXTERNAL/joint-budget-summary
```

The [primary scalar receipt](provenance/combined-scalar-audit.json), [supplement receipt](provenance/budget-supplement-audit.json), [CPU test record](provenance/cpu-tests.json), and [checkpoint identities](provenance/checkpoints.json) describe distinct verification scopes. The review ZIP contains this case at its repository-relative path and the root license. It excludes checkpoint weights and the pinned upstream checkout: scalar audits work from the ZIP, while model replay and training require those separate artifacts and dependencies.

`configs/protocol.sha256` covers the exact learned-protocol file. The historical toy v1 protocol used a canonical-JSON digest without its final newline; the erratum records both that digest and the exact file SHA256. They are different identifiers, not contradictory experiments.

Current candidate checksums cover the review tree. They do not replace an earlier measurement identity. Frozen source copies explain the development versions used by retained attempts. State restart tests, original upstream replay tests, model-free scalar audits and GPU training are separate checks; a CPU audit does not claim a new GPU reproduction.
