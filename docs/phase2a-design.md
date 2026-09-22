# Future option: multi-layer structural reconstruction

**NOT RUN. Requires a separate user choice and a new frozen protocol.** The current implementation and results cover Qwen3 and one smaller MLP layer.

Start with one model and uniform 25% intermediate-width reduction across its MLP layers. Uniform widths simplify configuration; they do not demonstrate runtime compatibility. A 50% variant requires another approved design.

Calibration fixes one removal set per layer, shared by native/prune-only/fixed-deletion mean-output bias compensation/ridge repair comparisons. Bias-only compensation is FLAP-inspired; it does not reproduce FLAP's fluctuation selector and global allocation. Define calibration/DEV/heldout, fitting order, ridge eta selection and extra parameter bytes before evaluation.

Sequential fitting should use the input produced by already compressed student layers. Specify whether the local teacher is the unpruned current layer applied to that same student input or a separate full teacher trajectory. These are different targets and must not be mixed.

Before a full run, inspect the chosen Qwen runtime's gate/up/down bias support and loading code. A saved compensation bias that the runtime ignores invalidates the experiment. Use the same supported runtime adapter for all candidates; do not compare PyTorch repair timing against a vLLM baseline as a repair-only cost. Strictly validate tensor names, layer dimensions, tied weights and buffers on reload.

Plan sufficient-statistic accumulation layer by layer: bound activation storage, Gram/B cross-product memory, condition diagnostics and FP64 solves. Ridge stabilizes the system; small solve residuals do not guarantee held-out transfer. QR/SVD are possible future solver comparisons, not silent replacements after results.

Measure standard quality and complete-model cost for each candidate. The single-layer local SSE reductions in Case008 do not predict multi-layer quality. [Related work](related-work/README.md) · [Existing R loader](../tools/modelpack/artifact.py).
