# Case019 — Learning Before the Loss Moves

A CPU reproduction of alignment–amplitude time separation with checkpoint interventions

A model’s learned directions can approach a target while its loss barely moves. We independently implemented a two-layer linear network on CPU, reproduced this time separation, and changed directions at saved checkpoints while preserving prediction size to observe subsequent loss paths.

These are finite-dimensional FP64 gradient-flow experiments on a whitened two-layer linear network. Direction means relation to the target matrix’s singular subspaces; size mainly means singular values of Z=UVᵀ; loss is squared matrix error.

1. Smaller initialization separated mean-alignment and visible-loss times. With larger initialization, the loss moved first.
2. A narrow boundary left a slow direction that the average partly hid. Separate calculations confirmed R2’s long tail and mode re-exit.
3. Directions changed subsequent paths at the same prediction size. Some early gains shrank, and some perturbed paths later overtook the original.

The aligned control starts in the target directions and still shows an amplitude-growth delay at small initialization. Alignment and size growth are measured separately.

![Initialization, direction and loss](publication/figures/p1-01_rho_direction_modes_loss.png)

This is our figure from stored finite-dimensional experiments, not a copy of an author figure. Pointwise median curves differ from statistics of event times measured per trajectory.

Implemented here: the ODE and initialization, rank-aware direction/size metrics, oracle/rotation controls, checkpoint branches, J/G/A analysis, separate numerical checks and replay tools. NumPy supplies linear algebra, SciPy adaptive integration and Matplotlib figures.

Quick checks use public source and the declared small raw slice in the pinned CPU environment. Complete recorded replay also requires the separate evidence asset.

```sh
python3 -I -B cases/019-silent-alignment-dynamics/publication/run_checks.py --output /tmp/case019-quick-new --mode quick --tests
```

[Results and methods](REPORT.md) · [Environment and quick/full replay](REPRODUCE.md) · [Source map](SOURCE_MAP.md) · [Paper v1](https://arxiv.org/abs/2610.29864v1)

See [CPU replay and full evidence](REPRODUCE.md) for quick/full scope and the dedicated Release asset identity. The [P3 local-preparation record](publication/history/P3_LOCAL_PREPARATION.md) remains historical.
