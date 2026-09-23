# Attribution and scope

This local DIOVA research candidate implements a packed-state storage adapter, exact group-label and first-failure audits, byte-budget accounting, bounded CKDA training/evaluation and review tooling. Codex assisted implementation, source inspection, testing and writing. Independent calculation paths in this repository are not external independent GPU reproductions.

Project code follows the repository [LICENSE](../../LICENSE). The upstream model implementation and cited papers retain their own authorship and terms.

Complex KDA and its task, model, optimizers and native recurrence are upstream work. The upstream code is consumed from an explicit pinned checkout rather than copied into this candidate. Any separate distribution of upstream software must retain its license and notices. See [upstream MIT license](https://github.com/OpenEuroLLM/ComplexKDA/blob/ef9d108d1692387cae37f5b2d539a71826a127c1/LICENSE) and [source hashes](provenance/upstream.json).

## Related work

- Julien Siems et al., **Complex KDA: Understanding and Enhancing the Expressivity of Kimi Delta Attention** (2026), [arXiv2609.24797v1](https://arxiv.org/html/2609.24797v1). Signed gates, extended beta, finite-group tasks and phase-control experiments motivate the chosen model. Exact algebraic expressivity is distinct from floating-point robustness.
- Tao Zhang et al., **DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization** (2026), [arXiv2608.27513v1](https://arxiv.org/html/2608.27513v1). Error energy and decay persistence motivate an explicitly generic CAL mixed-precision comparator. This study does not reproduce DAMP's precision allocation, Hadamard blocks or published performance.
- Jiwan Chung, Heechan Choi and Seon Joo Kim, **Rethinking State Tracking in Recurrent Models Through Error Control Dynamics** (2026), [arXiv2605.07755v1](https://arxiv.org/html/2605.07755v1). Neutral state-separating subspaces and readability horizons motivate examining first failure; their assumptions do not turn the present MLP readout into a certified decoder.
- Ismail Erbas, Xavier Intes and Vikas Pandey, **When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference** (2026), [arXiv2609.04490v1](https://arxiv.org/html/2609.04490v1). SupplementS8's write-back residual and direction-memory baselines are related prior work. Their GRU/LSTM update, clipping and residual-grid conventions differ from corrected-input affine feedback here.

Full v1 text was inspected for all four papers. Their numerical results were not reproduced. Corrected-state feedback and the identity transporting an affine residual are not claimed as new algorithms. New empirical claims must come from this candidate's own frozen inputs, stored bytes and observed first failures.
