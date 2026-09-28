# Case 011 methods — frozen-state readout adaptation

[한국어 보고서](REPORT.ko.md) · [English report](REPORT.md) · [Protocol](configs/protocol.json) · [Current adapter](source/adapter.py)

## Question and intervention

The experiment asks whether changing only the last linear classifier can increase the number of consecutive correct symbolic-state predictions from a fixed recurrent computation. It reuses the three S3 checkpoints and the physical-coordinate FP32 execution path from [Case 010 v2](../010-ckda-finite-precision-memory-horizon/versions/v2/README.md). Native and INT8 storage are the original failure-aware codecs. No transition, embedding, token projection, normalization, hidden MLP, quantization rule, or checkpoint selection is fitted again.

The original classifier has the structure `LayerNorm → Linear(48,192) → GELU → Linear(192,6)`. The feature `phi_t` is the existing GELU output, before the final Linear. Its upstream path remains the original query/state read, RMS normalization, sigmoid output gate, output projection and embedding residual. [The adapter](source/adapter.py) calls the same modules in the same FP32 order. The only fitted tensors are `mlp.2.weight` with shape `[6,192]` and `mlp.2.bias` with shape `[6]`: 1,158 parameter values.

For each checkpoint and storage, one recurrent rollout produces `phi_t`; ORIGINAL, SHORT_REFIT and MIXED_REFIT all read that same feature. Predictions never feed back into the supplied input sequence, coefficient table, codec, RNG, terminal state or cursor. The evaluator alone receives gold labels. Thus the 18 logical storage/readout/checkpoint combinations require six primary recurrent rollouts, not 18 independently sampled datasets.

## Frozen model and execution identity

[Reference identities](provenance/adapter_reference_identity.json) connect the adapter to the retained Case 010 snapshot, checkpoint SHA256 values, original codec inputs and source files. The upstream revision is `ef9d108d1692387cae37f5b2d539a71826a127c1`. The original FP32 token coefficients are loaded from Case 010 and compared with the coefficients recomputed from the verified checkpoint before use. No gate is recomputed in FP64 for this experiment.

[Preflight](source/preflight.py) checks the original and split feature/head paths on the frozen eight-input, 32-token SMOKE cohort. The receipt records bitwise logits, labels, final persistent bytes and split/resume equivalence separately. It is an execution-boundary check, not a quality estimate or a criterion for selecting a model seed. Synthetic fault-injection tests are also separate from the learned-checkpoint measurements.

## Data and independent symbolic target

Inputs are independent uniform draws from all six S3 group elements. Lexicographic permutation IDs, left cumulative multiplication `g_t = x_t ... x_1`, and BOS at write zero follow the original task. BOS is not a fitted or scored group token. [The new integer evaluator](source/data.py) constructs group products without reading model outputs; exhaustive short products are cross-checked with the frozen Case 010 multiplication table.

| Role | Generator seed | Base sequences | Group tokens per sequence | Use |
|---|---:|---:|---:|---|
| SMOKE | 61601 | 8 | 32 | Implementation and reload checks |
| FIT | 61101 | 512 | 256 | Native features and supervised fitting |
| DEV | 61201 | 128 | 256 | FP32 candidate selection |
| FRESH_TEST | 61301 | 1,024 | 2,048 | Paired final comparison |

[The input manifest](inputs/manifest.json) records actual token/gold bytes, sequence IDs, split tags, seeds and duplicate checks. The same maximum-length base sequence supplies all prefix summaries. A sequence evaluated by several heads or at several horizons remains one paired sampling unit.

SHORT_REFIT uses every position 1–32 of each FIT sequence, giving 16,384 feature/label rows. MIXED_REFIT uses eight predetermined positions per sequence in each of `[1,32]`, `[33,64]`, `[65,128]` and `[129,256]`, also 16,384 rows. The position RNG seed is 61501; choices are fixed before outputs are inspected. Both sets are extracted from a shared Native trajectory. The extraction receipt therefore reports the actual full 512×256 recurrence plus BOS work, rather than claiming that equal supervised row counts imply equal independently executed training cost. Full FIT feature arrays remain private and are excluded from the review ZIP.

## Head fitting and selection

[The fitting code](source/fitting.py) starts at the original `W0,b0` and minimizes

```text
mean cross_entropy(W phi + b, gold)
    + lambda * (sum((W-W0)^2) + sum((b-b0)^2)).
```

The fixed lambda menu is `1e-4, 1e-2, 1`. CPU FP64 full-batch `torch.optim.LBFGS` uses strong-Wolfe search, at most 200 optimizer iterations and at most 1,000 objective evaluations. The stored solver log includes its implementation/version, objective trace, stopping status and actual evaluations. A finite accepted iterate that reaches its iteration limit is explicitly marked not converged; an interrupted objective-budget line search is rejected and its initial head restored. There is no extra search after TEST.

Each result is converted to FP32 before DEV evaluation. DEV CE is the mean of four equally weighted position-band means, not a row-count-weighted average dominated by the longest band. Exact ties prefer larger lambda. One SHORT and one MIXED head are frozen for each checkpoint, giving six selected patches from a maximum of 18 fits. INT8 receives these Native-fitted heads without refitting. Feature scaling is not added.

## Controls and interpretation

The input-only control fits add-one class counts from FIT using only the current group token and one of the four fixed position bands. At positions beyond 256 it reuses the final band. Its count table is a limited diagnostic, not a capacity-matched neural classifier.

The state-shuffle diagnostic permutes features among different base sequences with the same current token and scored position. A random bucket ordering followed by a cyclic shift avoids fixed points when a bucket has at least two members. Singleton buckets remain unchanged and are counted. Its RNG uses seed 61401, position and current-token ID; labels are not inputs. Because the final Linear is row-wise, the runner applies the corresponding permutation to head outputs. Terminal destinations remain invalid. Controls do not modify recurrent state or choose fitting settings.

Control tasks follow the interpretive motivation of Hewitt and Liang (2019): prediction performance needs controls for what can be learned or exploited independently of the representation of interest. These two diagnostics are this study's design, not a reproduction of their experimental control task. Success would show recoverable predictive information for this feature/head family; failure would not prove information-theoretic loss under every possible decoder.

## Primary and secondary measurements

For every scored group token, a wrong label or `INVALID=-1` can establish the first error `tau`. A later correct prediction does not erase it. A stream with no observed error contributes the horizon limit, 2,048, to `RMST0`:

```text
RMST0 = mean(min(tau - 1, Tmax)), with a censored stream contributing Tmax.
```

The primary comparison is INT8/MIXED_REFIT minus INT8/ORIGINAL, separately for each of the three retained checkpoints. Paired base-sequence bootstrap uses 5,000 draws, seed `63001 + checkpoint_seed`, and linear quantiles. Each interval has confidence `1 - 0.05/3`, a Bonferroni family target across the three primary contrasts. These are approximate percentile-bootstrap intervals, not exact-coverage intervals or a population claim over all possible training seeds.

Secondary results include Native MIXED−ORIGINAL, INT8 MIXED−SHORT, storage transfer with the same readout, token accuracy, gold CE, margins, recovery after the first error and summaries at 32/128/256/512/1024/2048. Positions 512–2048 exceed the maximum fitting position 256. `T0.05` is the largest observed token position with an empirical first-error fraction at most 5%; no new simultaneous confidence-supported horizon is introduced. Average lifetime, early accuracy and late accuracy remain separate columns.

## Failures, saved evaluation state and restart

The original v2 execution contract is unchanged. A finite wrong answer continues normally. Nonfinite readout from a finite state yields `-1` while state updates continue. A numerical state failure becomes terminal, preserves its last committed body/RNG and first failed write, and consumes future inputs through cursor-only no-ops. It never becomes active because its scratch array happens to be finite.

The evaluation runner saves byte state plus completed prediction/score prefixes to a temporary directory, verifies checksums, then renames it. This evaluator history is not part of the model cache budget. [The independent execution-checkpoint checker](source/check_execution_checkpoint.py) additionally validates exact inventory, current frozen identities, prefix array shapes, score denominators, absorbing terminal outputs and `cursor = scored_offset + 1` for BOS. It must pass before a future resume. Resource or process errors remain execution failures; they are not converted into algorithmic terminal events.

## Artifacts and cost boundaries

[Two-tensor patches](source/head_patch.py) include the required base-checkpoint hash, tensor names/shapes/dtypes, FIT/DEV identities, solver/lambda, feature boundary and source identity. They are not standalone models. The loader copies values into the existing final parameters and verifies that other model tensors and Parameter objects are unchanged. [Fresh-process checks](source/reload_check.py) compare actual patched-model execution with detached-head execution on SMOKE, including all cache boundaries.

The persistent stream formats remain 12,305 bytes for Native and 3,137 bytes for INT8. Same-shape head replacement adds no recurrent cache tensor or additional inference matrix multiplication. Patch headers and tensor files, model parameter values, shared coefficient/config bytes, transient readout features and private fitting arrays are reported in distinct ledgers. Serialized bytes are not a process RAM or GPU VRAM measurement.

[Timing](source/timing.py) measures one selected head per complete call using two CPU threads, the historical timing-only 16×128 inputs, and three fixed repetitions. State initialization and BOS work are included; loading, patch assignment, gold/control/shadow work and file I/O are outside the timer. The denominator is 16×128 group tokens. The multi-head evaluation runner is not used to claim single-head latency. Timing is a CPU prototype measurement, not a GPU or individual-request speed claim.

## Reanalysis and attribution

The independent scalar audit reads saved predictions, gold and frozen identities without requiring private checkpoints. Actual model reruns and head fitting require the pinned upstream and the original local checkpoint files. See [REPRODUCTION](REPRODUCTION.md) for the supported commands and file boundaries.

The experiment reuses the original ComplexKDA model/task, Case 010 recurrence and failure-aware codecs, and NumPy/PyTorch numerical routines. Case 011 adds the frozen feature boundary, bounded final-head fitting, paired evaluation/control integration, two-tensor patch format, restart checks and reporting. It does not introduce a new probing principle, recurrent architecture or codec.
