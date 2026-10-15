# Public reproduction / 공개 검산

## A. Model-free records → tables → intervals

Run from the repository root. Use a new external output directory. The first command uses only Python stdlib;
the full command needs NumPy2.3.5 and Torch2.13.0 CPU operations. A hosted CI uses the pinned CPU-only requirements
in publication/requirements-cpu.txt. Existing research environments were not installed into or upgraded.

```bash
OUT=$(mktemp -d)
python3 -B cases/014-quantization-compensation/publication/scripts/derive.py --output "$OUT/display.json"
python3 -B cases/014-quantization-compensation/publication/verify.py --output "$OUT/publication.json"
```

With an existing suitable CPU Python selected as CPU_PY:

```bash
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT=$(mktemp -d)
"$CPU_PY" -B cases/014-quantization-compensation/publication/scripts/run_checks.py --output "$OUT/case014" --tests
```

This restores closed P0/P1/P1b/P2 views and checks 2880/3360/320 raw rows, original CAL/SELECT choices,
policy aliases, NLL/PPL/KL/flip aggregates,120 P1 intervals,462 P1b intervals and 20 P2 NLL/PPL interval fields
with the original bootstrap seeds. Existing15+10+9 tests are rerun; publication tests are a separate suite.
P0 dither SSE/SE are recalculated from saved noise outcomes; the2000-seed simulation itself is not rerun.
The public bundle identifies external weights/artifacts by hash; it does not decode private payloads or rerun GPU logits.

## B. Historical GPU experiment replay

This publication performs no model forward, K fit or re-quantization. The original model is
Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca; RTX5080/SM120,
Torch2.13.0+cu130/Transformers5.17.0. Full local shard SHA256 is recorded in the model manifests.
The WSL read-only snapshot exception was authorized; D-origin comparison was not required.

Private source text/token arrays, saved K/INT4 weights/large X/Z/logits are not in this repository or ZIP.
Hashes alone cannot reconstruct them. Follow the stage-specific requirements:

- [P1 original reproduction](publication/originals/pre_publication/REPRODUCE.md)
- [P1b read-only-artifact reproduction](followups/p1b_selection/REPRODUCE.md)
- [P2 own-trajectory reproduction](followups/p2_core/REPRODUCE.md)

Those original commands require their external materials and, where the runner records Git HEAD, a repository checkout.
Replaying already-observed EVAL is not a new fresh test. CPU publication checks do not invoke these commands.

## C. Static site and archive

```bash
OUT=$(mktemp -d)
"$CPU_PY" -B tools/showcase/run_checks.py --output "$OUT/portfolio"
```

The same built site is checked and deployed by the existing manual portfolio-pages workflow. No new framework is needed.
The unified download contains this Case014 tree once, with a separate content manifest; historical ZIPs are not nested.
Completed archive SHA/bytes and publication/merge/deploy revisions are separate receipts, avoiding self-hash cycles.
