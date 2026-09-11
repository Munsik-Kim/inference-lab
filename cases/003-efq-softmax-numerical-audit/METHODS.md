# Methods and reproduction

## Scope and paper-to-code audit

This is an independent equation implementation of [Han et al., EFQ-Softmax v1](https://arxiv.org/html/2609.09721v1). Probability operands are decoded to FP32 and multiplied by a common FP32 V. The original authors' packed kernel and hardware baseline were not recovered. “Paper-exact” below means the finite-input mathematical rule matches v1, not that the authors' code or bitwise execution was reproduced.

| Item | Classification | Local implementation / paper reference |
|---|---|---|
| Nonnegative E2M1 codebook | paper-exact | `[0, .5, 1, 1.5, 2, 3, 4, 6]`; section II-B; `numerics.E2M1` |
| EFQ exponent scale | paper-exact | Equation 8 in `numerics.operand`, before the explicit finite-range policy |
| Residual log normalization | paper-exact | Equation 10 in `numerics.operand` |
| Affine floor, then +1, then clip | paper-exact | Equation 11 in `numerics.affine_codes` |
| MMLU, Mean, Balance parameters | paper-exact | Section IV-A values in `numerics.PUBLIC`; their original task objectives differ |
| LUT index and 16→8 folding | paper-exact | Section II-D in `affine_codes`; formal definition takes precedence over the loose “16-level E2M1” description in IV-A |
| Mask before maxima/code generation | paper-exact | Section III-D; causal `-Inf` scores in `audit.load_trace` before `online` |
| Online maximum and historical rescaling | paper-exact | Equations 4–7 / Algorithm 2; both accumulated numerator and denominator use `exp(old_max-new_max)` |
| Shared operand and final normalization | paper-exact | The same decoded `p` updates PV and row sum in `numerics.online`; final A/l |
| Decoded FP32 probability/PV simulation | reasonable independent implementation choice | Functional numerical model only; no packed FP4 execution |
| Tile 128; per-row contiguous block 32 | reasonable independent implementation choice | Blocks never cross query rows or heads; reset at each tile. Sizes 16/64 are sensitivity analyses only |
| Headroom nearest scale and tie handling | reasonable independent implementation choice | Explicit scale rule below; nearest midpoint ties choose even code index |
| Finite exponent / masks / final partial block | reasonable independent implementation choice | Clamp exponents to [-127,127]; masked padding is zero; all-masked row policy below |
| Authors' exact operand layout and MXFP4 baseline details | unresolved ambiguity | v1 allows a layout-compatible block partition but does not specify enough to establish equivalence to these local baselines |

No author repository was identified in the recorded v1 paper/abstract check. That is not a claim that no code exists elsewhere. The equations, parameters and method are the authors' contribution; this project supplies numerical simulation and evidence.

## Exact local probability rules

For each tile, set `new_max=max(old_max, max(masked_scores_in_tile))`, then `x=score-new_max`. For one microscaling block let `M=max(x)` and let `C=[0,.5,1,1.5,2,3,4,6]`.

- **EFQ scale (equation 8):** `k=floor((M+ln(2/9))/ln(2))`.
- **Headroom scale (independent reference):** `k=ceil((M-ln(6))/ln(2))`. Within the represented exponent range this places the exact local maximum at or below `6 * 2**k`.
- **Finite representation:** clamp k to `[-127,127]`, then `s=2**k`. Codes use this represented scale; below-range blocks and represented zero scales are counted.
- **Nearest mapping:** calculate `exp(x)/s` and choose the nearest C value. Boundaries are `[.25,.75,1.25,1.75,2.5,3.5,5]`; an exact midpoint chooses the even code index. Both nearest references use this mapping.
- **EFQ mapping:** `z=x-k*ln(2)-ln(6)`; `c=clip(floor((z-tau)*h)+1,0,7)`; the decoded operand is `s*C[c]`.
- **LUT mapping:** `j=clip(floor((z-ln(1/24))*(14/ln(24)))+1,0,15)`; fold using `[0,1,1,1,1,1,2,2,3,3,4,5,5,6,7,7]` and decode `s*C[fold[j]]`.

| Operating point | tau | h |
|---|---:|---:|
| EFQ-MMLU | -2.90 | 2.00 |
| EFQ-Mean | -3.06 | 2.30 |
| EFQ-Balance | -2.10 | 2.70 |

The headroom reference is an independent MXFP4 numerical reference, not a verified implementation of the paper's MXFP4 baseline. The same-scale nearest reference controls scale choice while testing the affine mapping.

Block padding uses `-Inf` and produces zero codes. An all-masked block contributes nothing; an all-masked row returns zero output and `valid=false`. Valid rows with denominator zero or nonfinite output remain failures. NaN and positive-infinite scores are rejected. No epsilon repairs an invalid denominator.

Online A and l both rescale by `exp(old_max-new_max)` and consume the same tile operand. Diagnostic probabilities multiply each tile operand by `exp(tile_max-final_max)` before global normalization. Their PV reconstruction was checked against online output at relative tolerance 5e-5. JS therefore uses the effective online distribution, not a separate dense approximation. Initial maximum updates from `-Inf` are excluded from the finite max-jump statistic.

## Trace capture and sampling

The official BF16 checkpoint is `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`. [Model provenance](provenance/model.json) records file hashes and BF16 tensors. Transformers' `AutoModel` backbone does not use the checkpoint's output head. No text generation or task scoring is performed.

A temporary process-local wrapper intercepts the actual Transformers SDPA inputs after QK RMSNorm and RoPE, then restores the original function. Batch size is one, inference mode is enabled, dropout is zero and cache reuse is disabled. Layers 0/13/27 and query heads 0/5/10/15 are sampled. Two query heads share each KV head, yielding KV heads 0/2/5/7.

At sequence length N in `[512,2048,4096]`, query positions are `floor(i*(N-1)/15)`, i=0..15. Every key from 0 through that query position is used. Selected Q/K/V are stored privately on CPU. `audit.load_trace` forms the score tensor once per trace, applies the causal mask, and supplies that same score tensor and V to every method. Original capture/source hashes, per-row common reference norms and metadata support this correspondence; row summaries alone do not contain the full vectors.

A separate 128-token development check compared reconstructed attention with native BF16 output. The same SDPA calculation matched exactly; FP32 reconstruction differed by 0.142–0.186%, within the recorded 0.02 tolerance. The BF16 same-backend tolerance was 0.01. The small original development fixture remains in [tests/fixtures](tests/fixtures).

## Inputs, calibration and freeze

[Inputs](inputs/manifest.json) contain exact tokenizer IDs and hashes for eight development documents (3 English / 3 Korean / 2 code) and sixteen evaluation documents (5 / 5 / 6). They are self-authored synthetic texts with unique scenario IDs, text hashes and 512-token prefixes across document IDs. Longer prefixes of the same document overlap. Shared generation rules limit population interpretation. Input is plain text without a chat template or added special tokens.

The calibration grid is tau `[-3.6,-3.3,-3.06,-2.9,-2.6,-2.3,-2.1]` × h `[1.7,2,2.3,2.5,2.7,3,3.3]`. Each of the 49 candidates uses all eight development documents at length 2048, layer 13, heads 0/10 and the fixed sixteen queries. The objective is `sqrt(sum squared output differences / sum squared reference outputs)` across that subset. The first grid minimum wins; no refinement follows. The selected point equals EFQ-MMLU.

The recorded [specification](configs/experiment_spec.json) and its 50 frozen input/code/provenance hashes precede evaluation extraction. Its SHA256 remains `75b3d0e9ed46a22425bbf4b3d8c96affcd8e8faa5229a4f3f331eb84da0fcfdf`. This is a recorded before-evaluation specification, not external preregistration. A [development-only JS correction](provenance/development_corrections.json) explains calibration's earlier plan hash: only diagnostic metric text/code changed; the grid, attention output objective and selected setting were unchanged. The original unit records were byte-identical across that correction.

## Metrics and aggregation

The primary unit is document × length × layer × head, pooling sixteen query outputs. Its error is `norm(O-Oref)_F / norm(Oref)_F`. Reference RMS <=1e-6 makes relative error null; absolute error and the count remain available. No epsilon is added. Main tables summarize 192 repeated units per length using median, linear-interpolated p95 and maximum; there are 16 independent document IDs, not 192 independent samples.

Rows retain absolute error, relative error, cosine, JS, top-8 overlap, ties, zero-code fraction, removed reference mass, score statistics and denominator status. JS is in nats, with both effective distributions normalized in FP64. Its stable log1p/even-series diagnostic does not change FP32 attention arithmetic. Top-k uses min(8, valid keys) and ascending key index for ties. The tie fraction counts equal adjacent effective probabilities after sorting, including zeros. E2M1 zero fraction is measured before historical rescaling; removed mass uses the dense reference probability at zero-code keys. Undefined gap/cosine fields remain missing.

The prespecified screen compares calibrated EFQ with **nearest under the same EFQ scale**: median <=1.10× and p95 <=1.25× at each length, zero valid-row numerical failures and no layer×length median >2×. Baseline error <=1e-5 holds a ratio; near-zero/missing cases cannot silently pass. The original document-paired bootstrap describes mean per-document error differences with 2,000 draws. It is retained unchanged.

## Publication audit and counterexample

The [post-hoc plan](configs/publication_audit_plan.json) defines an independent row-norm reconstruction, comparison decomposition and 5,000-draw document-cluster bootstrap. This new bootstrap recalculates medians for all layer/head units in each resampled document and uses the same draws at all lengths. It reports median differences and ratios, not an independent-row significance test. The screen remains unchanged. Exploratory Spearman correlations also use dependent rows; no p-values are reported.

Diagnostic flags for high-error rows are defined before their post-hoc tally. They are overlapping descriptions, not assignments of a dominant cause. Global nonzero-code-boundary attribution is unsupported by the row summaries. A representative case is checked separately using a [171-score fixture](tests/fixtures/efq_mean_counterexample.json). It contains only one query's valid scores, reconstructed with CPU FP64 dot products from archived BF16 Q/K. The [scalar checker](scripts/check_counterexample.py) independently reconstructs probabilities, rescaling and code boundaries and compares them with original FP32 key records at absolute probability tolerance 2e-6. No new model inference is involved.

## Reproduction and verification commands

Use the pinned environment in [provenance](provenance/REPRODUCE.md); it was tested locally, not on a freshly installed third-party machine. No new packages are required for this audit. From this case directory:

```bash
# Supplied numerical evidence: CPU reanalysis, no model download.
python scripts/publication_audit.py --output-dir /path/to/new-derived-output
python scripts/check_counterexample.py --output /path/to/new-counterexample-check.json
python tests/test_numerics.py

# Recreate the original four figures in a scratch COPY of the case.
python scripts/analyze.py

# Full original workflow only when intentionally starting a separate experiment.
python scripts/reproduce.py --mode full --model-cache /path/to/model-cache --work-dir /path/to/new-private-work --output-dir /path/to/new-case-output
```

The publication script imports neither the original analysis nor its numeric summary functions. It calculates from compressed row records before reading unit/aggregate files as comparators. It writes only new derived outputs to the requested directory. Both analysis paths were checked in scratch copies so the supplied original measurements and plots remain unchanged. `write_report.py` is the initial narrative renderer; the current README and ANALYSIS are generated by `publication_audit.py`.

[Publication audit records](provenance/publication_audit.json) distinguish current checks from the initial numerical run. [Package hashes](SHA256SUMS) cover publication files; the fixed experiment spec separately covers its original frozen inputs/code. Model weights, full activations, environments and private logs are excluded.
