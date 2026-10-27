# Case021 — Single terms agreed; combinations of actual values differed

Can the same INT4 weights produce different results? In this final diagnostic, **D1 and Marlin agreed when a row contained one nonzero value, but differed when several actual values were combined.** We used two saved q_proj projections, at layers0 and25, from the existing Qwen3-0.6B study.

D1 first reconstructs a BF16 weight from each integer code and scale. P supplies the same codes and scales to Marlin. BF16 represents a limited set of numbers, so the location of rounding can matter even when the mathematical expression looks the same.

- **Single terms:** D1/P agreed across10 input conditions containing fixed or actual coefficients. At 406,626 output coordinates where arithmetic candidates A and B differed, both implementations matched A, which rounds the weight first. Coordinates where A and B agreed provide no discrimination.
- **Combined values:** all four layer/document combinations agreed at K=1 and2. The first observed difference was at K=8 for one combination and K=64 for three. K is a mask budget; actual nonzero counts and input norms also change.
- **Cancellation:** the34 existing EVAL local-gap increases split into13/17/4 cases according to the changes in noise energy and the signed cross term. More accurate individual calculations need not make two executions agree more closely.

![Fixed sparse replay](figures/03_sparse_replay.png)

The panels show mismatched coordinates, BF16 representable-step distance and input norm. K uses logarithmic spacing. Points represent fixed conditions, not independent user workloads.

| Comparison | Observed scope |
|---|---|
| Single terms | Two projections×5 fixtures,512 rows each; output equality across3 process repeats |
| Actual-value combinations | Two projections×two DEV documents×six K values;512 rows and3 process repeats |
| Original full-model verdict | **INCONCLUSIVE retained**;0 new full-model forwards |

The result narrows the difference to multi-term execution in this input grid. It does not isolate internal accumulation order, partial sums or scale placement. The original full-model interval still includes zero; this posthoc diagnostic establishes no model-quality, RL-stability or serving-speed improvement.

[P5 report](P5_REPORT.md) · [P0–P5 explanation](CONSOLIDATED_REPORT.md) · [CPU reproduction](REPRODUCE.md) · [한국어](README_PUBLIC.ko.md)

This is a **local publication candidate**. Actual activations and single-term raw outputs that can expose weights are private. The public CPU path checks aggregate tables, figures and small synthetic formulas. Full private-data recalculation is a separate scope.
