# Methods — finite-precision recurrent memory

[English introduction](README.md) · [한국어 소개](README.ko.md) · [Frozen protocol](configs/protocol.json)

## Question and units

Can a fixed recurrent model retain its exact symbolic state longer at the same **total persistent byte cap**? The independent observation is an input sequence. State MSE, last-token accuracy, throughput, and whole-model memory answer other questions.

For each sequence, `tau` is the first wrong group label at a scored position, starting at 1. A later correct prediction cannot erase that event. `null` means no observed failure through the maximum length, not infinite memory. We report `F(T)=Pr(tau<=T)`, current-step accuracy, and restricted mean failure-free length `mean(min(tau-1,Tmax))`, with censored observations contributing Tmax. The largest qualifying grid point is an empirical grid estimate. A confidence-supported lower bound uses exact one-sided Clopper–Pearson upper failure bounds with Bonferroni correction over the frozen arm/horizon/condition/possible-seed family. With 512 sequences, zero observed failures does not establish arbitrarily small population risk.

## Exact task and read boundary

An independent integer multiplication table supplies `g_t = x_t ... x_1`. Learned S3 inputs are arbitrary group elements, sampled uniformly from all six elements. CAL, DEV, and TEST generators are separate; each longest TEST sequence supplies all its shorter prefixes. The model does not receive gold labels during candidate evaluation.

In the learned experiment, a single BOS is processed and quantized like any write, but BOS prediction is reported separately and excluded from tau. The zero initial matrix itself is not decoded into a scored group label. Readout occurs **after each write-back** from the represented state. Ties choose the lowest class index. Entire outputs must be finite. The original MLP decoder remains fixed; no new decoder is fitted.

An invalid logit vector counts as a wrong read at that position. A nonfinite state terminates that stream: its later predictions remain wrong, while other streams continue. Zero placeholders permit a vectorized batch to continue, but are evaluator bookkeeping rather than a recovered model cache. Aggregate diagnostic norms/MSE after such termination include these placeholders; they must not be interpreted as a faithful finite trajectory or a successful repair. First-failure and failure counts remain valid and include every original sequence.

Official upstream evaluation reports final-quarter, chance-scaled token accuracy. This study adds all-prefix first-failure survival. The two metrics are not interchangeable.

## Model and physical-coordinate recurrence

The source is [OpenEuroLLM/ComplexKDA at the pinned revision](https://github.com/OpenEuroLLM/ComplexKDA/tree/ef9d108d1692387cae37f5b2d539a71826a127c1). Architecture: one layer, width48, 12 heads, head key/value dimension16, no short convolution, and a 48→192→6 MLP decoder. The official S3/a11_b02 parameterization extends signed channel gates and beta. We use normalized queries/keys, the actual gate function, RMS output gating/projection, residual embedding, layer normalization, and original classifier. Weights and recurrence computation are FP32 throughout learned comparisons.

For each head, `A=(I-beta*k*k.T)Diag(alpha)`, `B=beta*k*v.T`, `S_next=A*S+B`, and the attention read is `q.T*S_next/sqrt(16)`. The reference implementation uses a cumulative sign gauge internally. Our explicit physical-coordinate path has no persistent gauge; the official gauged and physical paths are compared at fixed atol2e-5/rtol2e-4, including nonzero state and chunk handoff. This is a sequential reference experiment, not a fused INT4 CUDA kernel.

The seven token IDs (six S3 elements plus BOS) are projected separately with the official `B=1,T=1` operation. Those canonical FP32 coefficients form a shared table, charged to every arm. Transition arithmetic and the attention read reduction use Torch and the official `einsum` order. A bulk-sequence projection or NumPy transition has different rounding; the initial NumPy prototype failed the fixed long-DEV replay tolerance and is recorded as a development failure. The final path passes length2048 with zero and nonzero initial state; its final physical state equals the official one-token cache bit for bit. This check establishes the chosen sequential boundary, not equality with every GEMM batch shape.

A bounded local training run is a new small-training experiment, not an official released checkpoint or paper reproduction. The frozen budget is 20,000 updates, batch128, Muon/AdamW and curriculum4/6/8/16/32, with the final scheduled update selected. Seed0 admits seed1/2 only through the predefined DEV32 gate and remaining walltime. TEST never selects a checkpoint. The official paper uses a larger schedule; native extrapolation is shown for each executed seed before interpreting codec effects.

## Causal stored-state adapter

Each stream stores only bytes representing `Z` and optional residual `Rhat`:

```
U = A*(Decode(Z) + Decode(Rhat)) + B
Z_next = Encode_state(U)
E = U - Decode(Z_next)
Rhat_next = Encode_residual(E)
read = Decode(Z_next) + Decode(Rhat_next)
```

`E` is the candidate's own rounding residual, not `S_native-Z`. Decoded arrays are temporary workspaces and never become an undeclared persistent cache. A full floating residual is an accuracy positive control with its full byte cost. A rank-limited residual uses CAL-fixed orthonormal per-head bases and coefficients. The complete transition transports the reconstructed residual before the next projection. Projection leakage is reported; no rank2 sufficiency is presumed.

The untransported ablation instead updates `A*Z+B+Rhat`. It isolates write placement in this implementation and does not stand for every published error-feedback method. The named Erbas baseline has its own clipping, residual-grid and read conventions; we do not claim to reproduce it. The mixed-precision arm ranks CAL channel error weighted by transition persistence. It is a generic CAL baseline, not a DAMP reproduction.

## Packing and equal-byte comparisons

Uniform widths2–16 are implemented as real LSB-first packed unsigned byte payloads. Signed quantized values use an offset code with zero reserved. Per-head max-absolute scales are stored as little-endian FP32; deterministic rounding is ties-to-even. Mixed channels use explicit serialized bit maps. Stochastic rounding stores its 64-bit seed and next64-bit counter in each stream. An8-byte cursor is charged to every arm. Invalid codes, trailing bytes, nonzero padding and nonfinite scales are rejected.

`Total(N) = shared_bytes + N*per_stream_persistent_bytes`, for N=1,16,128. Shared configuration and fixed bases are serialized and counted. Toy transition/readout tables are separately counted common shared data. Fixed learned weights are a common checkpoint, reported separately from changing state-cache bytes. No artificial baseline padding is used. At each candidate cap, all feasible uniform and mixed arms remain available; comparisons may favor a higher-bit uniform state. Serialized bytes and actual payload-array storage are checked separately from temporary arrays, Python object overhead and whole-device memory.

The learned token table occupies29,100 shared bytes, including its serialization descriptor. It is an implementation cost of the canonical replay path, not a proposed replacement for all model weights. The common model has56,530 parameters (226,120 FP32 value bytes); its checkpoint file also contains serialization metadata. Bases are stored as FP32 and ranks1/2/4 operate on the key dimension of every head. No per-sequence moving basis is fitted in the learned primary experiment. The separate moving-basis test verifies coordinate conversion and its metadata cost.

## Mechanism controls and theory boundary

Toy controls initialize the known identity prototype and score input group tokens without BOS. They separate lattice-compatible signed permutations, nontrivial rotations, noncommuting transitions, and moving/switching planes. These are general affine geometry controls; not every matrix is asserted to be a single parameterized CKDA step. Coefficient quantization is a separate toy-only secondary condition. FP64 is toy arithmetic control. An exact group-ID tracker with a multiplication table is a task-specific capacity control: finite groups do not imply a universal log(sequence-length) memory lower bound.

In exact arithmetic, full residual reconstruction recovers the same affine update. Define `e_t` as represented candidate state minus reference state. With exact affine computation and remaining storage error eta, `e_t=A_t*e_(t-1)+eta_t`. Actual FP32 paths additionally have arithmetic roundoff: `e_t=A_t*e_(t-1)+eta_storage,t+rho_candidate,t-rho_reference,t`. Using the same dtype does not make the two roundoff terms cancel when their input states differ. If normalized k, beta in[0,2], and each |alpha|<=1 hold, `||A||_2<=1`; the accumulated norm bound uses the sum of all those error terms, not storage error alone. Changed coefficients require further terms. These are accounting and norm bounds, not new theorems. For a linear classifier, the appropriate pairwise score margin and perturbation bound give a sufficient stable-label condition. The actual learned decoder is an MLP: a local Jacobian or average margin is not a global survival guarantee.

## Timing and uncertainty

Learned-prototype complete-call timing includes state allocation, coefficient gathering, decode/transition/encode, residual projection and the original readout. Gold calculation, confidence intervals and file I/O are excluded. Toy runtime records include readout and error/norm/phase diagnostics and exclude initial encoding; they are diagnostic-loop durations, not complete-call benchmarks. CPU NumPy/PyTorch timing describes this prototype only. No whole-model VRAM or GPU speedup is inferred. Input uncertainty is reported separately for each training seed; three models are not thousands of independent model seeds. No TEST-fitted long-horizon predictor or new threshold is introduced.

The primary CPU evaluations run in three separate processes, with two threads each. Their embedded timing records remain available but may contain contention. The frozen [execution policy](configs/execution_policy.json) calls for one separate serial timing run per checkpoint after all primary evaluations finish:16 fixed DEV sequences, length128, three measured repeats per arm after warmup. Only this isolated run supplies the complete-call cost comparison. Its three repeats describe local variability, not an independently replicated GPU speed claim.

All frozen arms are evaluated. A best-feasible baseline on a plotted byte frontier is a descriptive selection from this same TEST family, not a DEV-selected deployable codec. Simultaneous bounds cover the whole arm/horizon/seed family. A longer point estimate alone is not a paired superiority test. At epsilon1%,512 samples cannot give a qualifying simultaneous upper risk bound even with zero observed failures under this family size; the empirical and confidence-supported columns deliberately differ.

## Separate budget-menu audit

The original learned menu remains26 arms and its confidence family remains546. During execution, a byte-ledger review found four additional feasible mixed allocations that could spend more of some low-rank candidates' caps. The [supplement policy](configs/budget_supplement_v1.json) froze those allocations before any low-rank TEST result existed:4/8 bits with2 or5 promoted key channels per head, and6/8 bits with1 or4 promoted channels. The original CAL ranking determines the promoted channels; there is no new fit, checkpoint selection or candidate tuning.

This separately versioned audit reuses the original512 TEST sequences and is labeled `POST_HOC_BUDGET_AUDIT_SAME_TEST`. It is not an independent confirmation set. All four allocations and all three seeds are retained. Its joint view recomputes bounds for30×7×3=630 arm/horizon/seed comparisons while retaining the original546-family summaries unchanged. The four extra arms have no new timing measurement. These additions strengthen the explicitly tested baseline menu; they do not establish an optimum over every possible bit allocation.
