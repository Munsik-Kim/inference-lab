# Methods and reproduction

## What this experiment changes

The audit implements EFQ-Softmax from [Han et al., v1](https://arxiv.org/html/2609.09721v1), independently of the authors' code. It changes the unnormalized probability operand inside tiled attention. Every method receives the same FP32 score tensor, value tensor and causal mask. QK is calculated once from recorded BF16 Q/K promoted to FP32; V is also promoted to FP32. TF32 is disabled. The E2M1 operand is decoded to FP32 before multiplication and summation. This is not a packed FP4 kernel, a model-quality evaluation, or an A5 performance reproduction.

The formal definitions in sections II–III take precedence over loose descriptions of a “16-level” LUT elsewhere in the paper. The LUT has 16 intermediate indices and eight final E2M1 values. The paper's Balance operating point comes from vision-language evaluation; using it here is a transfer check on a smaller text model. No author repository was identified in the v1 body/abstract or the recorded title search. See [paper audit](provenance/paper_audit.json) for scope and source hashes. The complete paper is not redistributed.

## Probability methods

The nonnegative E2M1 values are `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`. For shifted block maximum `M`, EFQ selects `k=floor((M+ln(2/9))/ln(2))`, uses scale `2**k`, and computes `z=x-k*ln(2)-ln(6)`. Its code is `clip(floor((z-tau)*h)+1, 0, 7)`.

| Method | Rule |
|---|---|
| FP32 dense | Stable softmax over all valid keys, then PV |
| FP32 online | Tiled stable attention without quantization |
| Nearest, headroom scale | Explicit exp, `k=ceil((M-ln(6))/ln(2))`, nearest E2M1 |
| Nearest, EFQ scale | Explicit exp and nearest E2M1 using exactly EFQ's scale |
| EFQ-MMLU | tau=-2.90, h=2.00 |
| EFQ-Mean | tau=-3.06, h=2.30 |
| EFQ-Balance | tau=-2.10, h=2.70 |
| EFQ-LUT | tau=ln(1/24), h=14/ln(24); clip intermediate code to 0–15, then fold |
| EFQ calibrated | One candidate chosen only from the fixed development grid |

The LUT folding table is `[0,1,1,1,1,1,2,2,3,3,4,5,5,6,7,7]`. Nearest-rounding midpoint ties choose the even code index. Affine thresholds use floor directly; there is no nearest-rounding tie rule on that path.

The headroom-scale method is an **MXFP4 numerical reference**, not a verified copy of the paper's MXFP4 baseline. Its scale rule differs from EFQ, so the same-scale nearest method is the primary control and engineering-screen baseline. It separates scale selection from the affine mapping effect.

Each attention tile has 128 keys. Each microscaling block covers 32 consecutive keys **within one query row and head**, resetting at the tile boundary. [MXFP4's block-of-32 convention](https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/) motivates the primary size. Blocks of 16 and 64 are sensitivity experiments on the numerical mapping, not claims about matching a particular hardware operand layout. Only nearest/EFQ-scale, EFQ-Mean and the calibrated setting are included in that sensitivity analysis.

The paper does not completely fix layout or finite-range edge policies. This implementation clamps scale exponents to [-127,127], then calculates codes using the represented scale. It counts below-range blocks and represented zero scales. Incomplete blocks are padded with masked entries. Masked elements always produce zero. An entirely masked block makes no contribution; an entirely masked row returns a zero output with `valid=false`. A valid row with denominator zero or nonfinite output remains a recorded failure rather than being repaired. NaN or positive-infinite input scores are rejected. These are explicit independent implementation choices.

## Online state and diagnostic probabilities

At each tile, both accumulators are rescaled by `exp(old_row_max-new_row_max)`. The very same reconstructed tile operand contributes to the numerator PV and denominator sum. The output is the accumulated numerator divided by the accumulated denominator. No high-precision substitute denominator is used.

For JS and top-k diagnostics, each recorded tile operand is multiplied by `exp(tile_row_max-final_row_max)`. Those effective weights are concatenated and normalized. Their PV reconstruction is checked against the online output, with relative tolerance 5e-5. This is an online-path diagnostic, not a separate dense EFQ approximation. The row-max jump statistic excludes the initial update from negative infinity.

## Model and inputs

The model is `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`. All checkpoint tensors were BF16. Transformers loads the backbone through `AutoModel`; the checkpoint's language-model output-head tensor is unused. No tokens are generated and no task score is measured. The snapshot's tokenizer converts plain text without a chat template or extra special tokens. Exact token IDs and prefix hashes are preserved in [input manifest](inputs/manifest.json).

There are eight development documents and sixteen evaluation documents, created before execution from independent scenario IDs and seeds. They include English prose, Korean prose and code. All documents are self-authored synthetic material, with shared generation rules; they do not represent general natural-language or code distributions. Both text hashes and 512-token prefixes are unique across document IDs. Prefixes of length 512, 2048 and 4096 from the same document overlap and are **not independent samples**.

Batch size is one, inference mode is enabled, dropout and cache reuse are disabled. The native backbone uses Transformers' SDPA path. A temporary in-process wrapper intercepts its actual query/key/value arguments after QK RMSNorm and RoPE. No installed source is patched. Layers are 0, 13 and 27; query heads are 0, 5, 10 and 15. Qwen uses two query heads per KV head, so the selected KV heads are 0, 2, 5 and 7.

For each length N, query positions are `floor(i*(N-1)/15)` for i=0..15. Each selected query uses **every key from 0 through its position**. Only these sixteen rows per head are evaluated. The full sequence still passes through the original backbone. Q/K/V for selected layers/heads are copied to CPU files, one layer at a time; full attention matrices for all layers are never retained.

A separate 128-token development input checks captured Q/K/V against the actual native attention output. Reapplying the same BF16 SDPA backend must have relative error <=0.01; FP32 dense reconstruction must have error <=0.02 against native BF16 output. These tolerances were specified before the check. The fixture is a small subset of that development trace, not an evaluation example.

## Calibration and freeze

The 49-candidate grid is the Cartesian product of tau `[-3.6,-3.3,-3.06,-2.9,-2.6,-2.3,-2.1]` and h `[1.7,2.0,2.3,2.5,2.7,3.0,3.3]`. Calibration uses all eight development documents at length 2048, layer 13, heads 0 and 10, and the fixed sixteen query rows. Its objective is the relative Frobenius error of the concatenated outputs: square root of total squared error divided by total squared reference norm. All candidates use the same subset. The first minimum in listed grid order wins; no refinement follows.

Published parameters remain separate methods. EFQ-Mean and the development-selected setting are the prespecified primary comparisons. The matched-subset development/evaluation objective uses the same length/layer/head definition, changing only document split. It is distinct from the all-head/length median table.

After tests, trace validation, calibration and development measurements, [experiment_spec.json](configs/experiment_spec.json) freezes the plan, selected parameters and input/code/provenance hashes. Evaluation trace extraction starts only afterward. A development-only JS precision correction is documented in [development corrections](provenance/development_corrections.json); prior files were preserved privately, and QK/PV/output errors and parameter selection were unchanged.

## Measurements and decision rule

The main unit is one document × length × layer × head, containing sixteen query outputs. Relative error is `||O-Oref||_F / ||Oref||_F`; absolute Frobenius error and both squared norms are also saved. When reference RMS <=1e-6, relative error is null and the unit is counted separately. No epsilon is added to disguise a small denominator.

Row records include absolute/relative output error, cosine similarity, normalized JS in nats, top-8 overlap, zero-code fraction and the reference mass removed at zero-code keys. Top-k uses k=min(8, valid keys), breaking probability ties by ascending key index. The tie fraction is the number of equal adjacent effective probabilities after sorting divided by valid_keys-1; zero ties are included. This measures ties in the effective quantized distribution, not equal codes across unrelated scales. The FP32 references have no E2M1 zero-code statistic.

JS diagnostics renormalize probabilities in FP64. A stable log1p formulation uses its even series through r^8 for |r|<.01, with remainder below 1.2e-22. No negative-divergence clipping is used. All primary attention arithmetic remains FP32. Cosine is undefined for zero vectors and is retained as null. Missing/nonfinite metrics and failed rows are counted, not silently removed from the screen.

Tables report median, p95 (linear interpolation) and maximum across head units, including layer×length and head×length breakdowns. Bootstrap draws 16 paired document IDs with replacement 2,000 times, retaining every head/layer and overlapping length prefix of each drawn document. The interval describes the mean per-document error difference, calibrated minus same-scale nearest, separately by length. Row correlations are exploratory pooled Spearman associations without independent-sample p-values or causal interpretation.

The engineering screen requires calibrated EFQ to have median error <=1.10× and p95 <=1.25× the same-scale nearest reference at each length, no valid-row numerical failures, and no layer×length median >2× baseline. Ratios are held if baseline error <=1e-5; near-zero/missing cases cannot silently pass. Passing only motivates consideration of limited kernel feasibility. It does not establish model quality, speedup or packed-FP4 correctness.

Synthetic stress uses independent development/evaluation seeds, nine families and two replicas per family at each length. Families cover near-uniform scores, three Gaussian spreads, a single peak, several close peaks, clipped heavy tails, causal masking and large late row-max jumps. Values are fixed random FP32 vectors shared by methods. These tests are reported separately from model traces.

## Commands

Use Python 3.12 with the recorded PyTorch CUDA build, Transformers, NumPy, safetensors and matplotlib versions in [environment](provenance/environment.json). The actual environment reused CASE 002 packages read-only through a separate venv and added plotting dependencies there. [Package versions](provenance/package_versions.txt) are provenance, not a claim of a clean-machine install test. vLLM is not used by this audit.

From this case directory, set `MODEL_CACHE` and `WORK` to your own directories. The latter holds model-derived traces and private logs, outside Git. Set `PYTHON` to your prepared environment's Python.

```bash
hf download Qwen/Qwen3-0.6B config.json generation_config.json tokenizer_config.json tokenizer.json vocab.json merges.txt model.safetensors README.md LICENSE --revision c1899de289a04d12100db370d81485cdf75e47ca --cache-dir "$MODEL_CACHE"
mkdir -p "$WORK/traces/dev" "$WORK/traces/eval"
"$PYTHON" scripts/prepare.py --model-cache "$MODEL_CACHE"
"$PYTHON" scripts/extract_traces.py --stage validate --model-cache "$MODEL_CACHE" --work-dir "$WORK"
"$PYTHON" scripts/record_tests.py
"$PYTHON" scripts/extract_traces.py --stage dev --model-cache "$MODEL_CACHE" --work-dir "$WORK"
"$PYTHON" scripts/audit.py --stage calibrate --work-dir "$WORK"
"$PYTHON" scripts/audit.py --stage dev --work-dir "$WORK"
"$PYTHON" scripts/audit.py --stage synthetic-dev --work-dir "$WORK"
"$PYTHON" scripts/audit.py --stage freeze --work-dir "$WORK"
```

Those preparation commands belong in a **new reproduction working copy**: the supplied case already contains a frozen specification and result files, and measurement commands refuse to overwrite existing results. The recorded numerical-test JSON must be regenerated by the included test-recording command before making a new freeze. Keep prior runs in a different directory rather than deleting their evidence.

To reproduce the recorded evaluation with the supplied immutable inputs/specification, use a separate output copy and keep the original as an audit reference. `extract_traces.py --stage eval` verifies the frozen file hashes; it needs the supplied frozen code/input files unchanged. Full clean-copy orchestration is described in [reproduction notes](provenance/REPRODUCE.md).

```bash
"$PYTHON" scripts/extract_traces.py --stage eval --model-cache "$MODEL_CACHE" --work-dir "$WORK"
"$PYTHON" scripts/audit.py --stage eval --work-dir "$WORK"
"$PYTHON" scripts/audit.py --stage synthetic-eval --work-dir "$WORK"
"$PYTHON" scripts/analyze.py
```

Tables and the four PNG figures can be regenerated without a GPU from the saved unit JSONL and compressed row CSV files by running only `scripts/analyze.py` with the analysis dependencies available. Model weights and full activations are deliberately excluded.

## Verify the supplied evidence

Run `python scripts/verify_artifacts.py` for a static check of the recorded package. It checks frozen hashes, document/prefix identities, trace metadata, row/unit correspondence, summary statistics, the engineering screen, local links and fixture CRC. It uses only the standard library and does not run the GPU. Its output is [artifact_validation.json](provenance/artifact_validation.json). The checks target the supplied experiment and its recorded counts, not arbitrary future experiments.

`inspect_failure.py`, `write_report.py` and `verify_artifacts.py` were added after the evaluation to inspect or present saved results. They did not change the frozen arithmetic, inputs, parameters or evaluation criteria. The representative-case inspection checked its recomputed units against the saved unit errors. The English report renderer contains this run's interpretation; it is not an automatic conclusion generator for a different result.

The reproduction wrapper leaves the original narrative and supplemental publication records in the copied directory as historical context. New measurements are the trace manifests and files produced by `extract_traces.py`, `audit.py` and `analyze.py`. Copied `evaluation_execution.json`, artifact-validation records, package checksums and narrative do not attest to a new run. Preserve your own command/output logs and label new measurements separately when sharing a reproduction. The individual stages ran locally; the wrapper was statically inspected and was not used to repeat the completed experiment.
