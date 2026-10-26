# Case020 P4 — Measurement and validation record

P4 preserves the historical memory result and adds shared-context latency, decoder-width and actual cumulative-call measurements. The [integrated report](INTEGRATED_REPORT.md) explains execution choices; this report provides numerical and contract detail.

## Identity

- Case/worktree: `/home/mskim/Codex_main/inference-lab-case020-optic-decode-20261026-1215/cases/020-optic-on-demand-decode`
- Branch: `research/case020-optic-on-demand-decode-20261026-1215`
- Git base/HEAD: `3e6f2c6b36fb2a3853cc3677bee84904d28567cf`
- Upstream: `344c712aeb6620e39eb78930fb922447f0e123f3`
- P4 run: `p4-20261026-1451`
- Uncommitted source/config/input bundle: `e841f818bf81263fcd78f5513244feabc79a73bc8dc07d68a53fcf1d5cfcd25e`
- PROTOCOL SHA256: `31c29c6724c221c72d7a8f43fb1e8d5098c19b3871e375f043aa3664054b939f`


Original final run `20261026-1235`; original order-defect diagnostic run remains separately preserved. The initial P4 source/config/input freeze is `2026-10-26T06:03:17.512180+00:00`; main freeze is `2026-10-26T06:04:16.519482+00:00`. This is a follow-up designed after inspecting P2. No source, grid, seeds, thresholds or numerical tolerance changed after P4 GPU observations. A CPU-auditor draft variable-shadowing error was fixed before final aggregation and logged as `CPU_AUDITOR_DRAFT_FIX`; it did not alter GPU source/raw or incur a worker retry.

## Environment / operator contract

Python3.12.14, NumPy2.3.5, Torch2.13.0+cu130, CUDA runtime13.0, driver621.01, NVIDIA RTX5080 SM12.0, WSL2 Ubuntu / Linux5.15.167.4. Actual import paths, affinity16 logical CPUs and telemetry are stored per worker. Runtime reuse was read-only, not an independent clean-environment validation. Plotting used Python3.10.19 / NumPy2.2.6 / Matplotlib3.10.8 from the existing `ai_lab` runtime. No package was installed.

Eager, inference_mode, default stream; raw unscaled BF16 flattened `[M,4096]` equivalent to `[M,16,256]` in layer-major order. GPU IDs are int64; output stays on GPU while timed. No scale from `install_compact_ple` is applied. TF32=false; FP16/BF16 reduced-precision reduction=false; FP16 accumulation=false. Upstream runtime defaults had reduction flags=true; P4 preserves the original Case's explicit FP32-accumulation policy. Threads OMP/MKL/OPENBLAS/Torch=1; CPU affinity was observed, not pinned to one core. GPU clock/power settings were not modified. Allocator/backend/interop threads and all observed values remain in raw.

A uses the official decoder in fixed256-row chunks to build the full BF16 table. In latency-only workers A and one compact module for B/C/S coexist. A's temporary build decoder is removed before loading the shared compact module. This state is shared by every method and is not used as method-isolated memory evidence. C computes unique/inverse every call. S sorts IDs, decodes all M rows, and restores order with an inverse permutation; only Z32/M2048/U2048 has S.

## Fixtures and preflight

| Z | groups | serialized NPZ bytes | array payload bytes | module tensor bytes incl offsets |
|---|---|---|---|---|
| 32 | 8 | 550998 | 548864 | 548928 |
| 128 | 32 | 2173022 | 2170880 | 2171136 |
| 644 | 161 | 10891362 | 10889216 | 10890504 |


All shapes use V32768,L16,H256,K256,group_width4. Z32 and all three ID files reuse original bytes exactly; new widths use `SeedSequence([61026020,4,Z])` with the original FP16 generation/scale/bias rule. Array hashes, finite/min/max and metadata are in `provenance/artifacts.json`. Input file/payload hashes and exact M/U are in `provenance/inputs.json`. No trained model artifact was used. Each width has one artifact, not independent replicated models.

Independent NumPy FP64 uses the stored FP16 values, concatenated group codebook lookups and `latent @ W.T + bias` in128-row chunks. The scalar loop/vector spot check passed rtol1e-10/atol1e-12 for each width. Original P1's full257-token, all-layer, seed0/1/2, boundary/order/zero/bias checks remain in historical evidence. New dedicated GPU preflight checks every requested output across all16 layers and every A/B/C/S pair at unchanged overall/per-layer relative_L2<=1e-2, finite required; exact-zero control max_abs<=1e-6. Maximum layer relative_L2 across new method/reference checks was0.0017823104599562255. CPU audit re-derives full-output relative gates from error/reference sums and independently recomputes the saved BF16 sample outputs from NPZ arrays. Full large GPU outputs are not repeatedly copied during timing.

| Z | input | method | overall relative L2 | max layer relative L2 | output norm | output min | output max |
|---|---|---|---|---|---|---|---|
| 128 | unique2048 | A | 0.00165918 | 0.00166416 | 6334.241 | -9.875 | 13.188 |
| 128 | unique2048 | B | 0.00165918 | 0.00166416 | 6334.241 | -9.875 | 13.188 |
| 128 | unique2048 | C | 0.00165918 | 0.00166416 | 6334.241 | -9.875 | 13.188 |
| 128 | repeat2048 | A | 0.00165988 | 0.00166662 | 6173.745 | -9.250 | 12.438 |
| 128 | repeat2048 | B | 0.00165988 | 0.00166662 | 6173.745 | -9.250 | 12.438 |
| 128 | repeat2048 | C | 0.00165988 | 0.00166662 | 6173.745 | -9.250 | 12.438 |
| 32 | m1 | A | 0.00163684 | 0.00178231 | 81.479 | -3.047 | 4.156 |
| 32 | m1 | B | 0.00163684 | 0.00178231 | 81.479 | -3.047 | 4.156 |
| 32 | m1 | C | 0.00163684 | 0.00178231 | 81.479 | -3.047 | 4.156 |
| 32 | unique2048 | A | 0.00165701 | 0.00166427 | 3970.490 | -4.719 | 7.906 |
| 32 | unique2048 | B | 0.00165701 | 0.00166427 | 3970.490 | -4.719 | 7.906 |
| 32 | unique2048 | C | 0.00165701 | 0.00166427 | 3970.490 | -4.719 | 7.906 |
| 32 | unique2048 | S | 0.00165701 | 0.00166427 | 3970.490 | -4.719 | 7.906 |
| 32 | repeat2048 | A | 0.00165722 | 0.00166779 | 3971.639 | -4.438 | 7.500 |
| 32 | repeat2048 | B | 0.00165722 | 0.00166779 | 3971.639 | -4.438 | 7.500 |
| 32 | repeat2048 | C | 0.00165722 | 0.00166779 | 3971.639 | -4.438 | 7.500 |
| 644 | unique2048 | A | 0.00166111 | 0.00166586 | 12918.337 | -25.125 | 25.875 |
| 644 | unique2048 | B | 0.00166111 | 0.00166586 | 12918.337 | -25.125 | 25.875 |
| 644 | unique2048 | C | 0.00166111 | 0.00166586 | 12918.337 | -25.125 | 25.875 |
| 644 | repeat2048 | A | 0.00165960 | 0.00166471 | 13015.935 | -22.250 | 27.000 |
| 644 | repeat2048 | B | 0.00165960 | 0.00166471 | 13015.935 | -22.250 | 27.000 |
| 644 | repeat2048 | C | 0.00165960 | 0.00166471 | 13015.935 | -22.250 | 27.000 |


All64 workers ran a small FP16 GEMM with actual FP32 out_dtype and finite output as common initialization. Three dedicated preflight workers are included in64. Frozen source/config/input and preflight hash binding are required before non-preflight workers. The Z644 smoke was separate (6 H,6 D,6 E brackets), with no shape amendment. Additional allocated peak stayed below2GiB for every worker; exact values are in raw. Context/library32MiB baseline, reserved bytes and device-wide samples are separate measures.

## H / D / E measurement design

Core:3 inputs ×6 fresh processes=18 workers,60 method-blocks including S. Width:2 widths ×2 inputs ×3 processes=12 workers,36 method-blocks. Total30 workers /96 method-blocks. Each method has initial20 warm calls plus20 at every pass start. Six core pass permutations and cyclic Latin width orders balance pass order. Every main pass uses full balanced6-method-order permutations for ABC or24 for ABCS, with fixed shuffled cycles/starts.

H:120 rounds/method, K1, pre-sync→host start→fn→device synchronize→host end. D:48 rounds/method, K1, host start→event start→fn→event end→end-event synchronize→host end. Its host includes event records/wait. E:24 brackets/method, common K16; stream elapsed/16. Outputs are released rather than accumulated. No profiler, I/O, logging, telemetry, gc or empty_cache runs inside timed functions. H/D/E occur in separate passes. D stores both clocks on the same row. No timer-agreement success criterion exists.

Quantiles use linear interpolation. No trimming/winsorizing or slow-sample removal. Raw includes round/rank/order/start/end. Main raw observations:11520 H +4608 D +2304 E=18432 rows; the4608 D rows have two clocks, not additional independent workloads. Rates in CSV are positions/s and layer-rows/s; event-bracket and synchronized-host rate scopes remain separate. They are not LLM tokens/s. See `round_samples.csv`, `same_call_dual_timer.csv`, `paired_round_comparison.csv` and `paired_block_comparison.csv`.

| Z | M / U | blocks | C/B H median ratio* | block ratio range | C−B median difference* (us) | C faster |
|---|---|---|---|---|---|---|
| 32 | 1 / 1 | 6 | 2.333 | 1.742–2.595 | +138.17 | 0/6 |
| 32 | 2048 / 2048 | 6 | 1.936 | 1.864–2.155 | +140.92 | 0/6 |
| 32 | 2048 / 512 | 6 | 1.777 | 1.511–1.920 | +114.27 | 0/6 |
| 128 | 2048 / 2048 | 3 | 1.942 | 1.799–1.943 | +265.92 | 0/3 |
| 128 | 2048 / 512 | 3 | 1.800 | 1.745–1.817 | +119.73 | 0/3 |
| 644 | 2048 / 2048 | 3 | 1.409 | 1.350–1.470 | +130.53 | 0/3 |
| 644 | 2048 / 512 | 3 | 1.142 | 1.115–1.306 | +56.13 | 0/3 |


The descriptive C/B benefit cutoff is a median paired H ratio<=0.90 with>=5/6 core wins or3/3 width wins. Width success would still be exploratory. Differences within(0.90,1.10) are SMALL_DIFFERENCE. No condition met a benefit rule; every H C/B block ratio was>1. These rules are not significance tests or implementation PASS gates.

## Primary H: all96 method-block origins

Each line describes120 calls within one process/method. These are not pooled across blocks. P95s are within-block synchronized-call p95s, not population confidence intervals. D host/event and E stream statistics, each with mean/median/p95/min/IQR, are separately available in [controlled_latency.csv](summary/controlled_latency.csv).

| Z | M/U | block | method | H mean us | H median us | H p95 us | H min us | H IQR us |
|---|---|---|---|---|---|---|---|---|
| 32 | 1/1 | 0 | A | 23.937 | 21.426 | 37.557 | 14.531 | 7.025 |
| 32 | 1/1 | 0 | B | 96.295 | 82.268 | 170.141 | 54.732 | 31.820 |
| 32 | 1/1 | 0 | C | 226.888 | 186.996 | 364.750 | 124.959 | 105.960 |
| 32 | 1/1 | 1 | A | 35.415 | 26.578 | 98.051 | 15.019 | 16.453 |
| 32 | 1/1 | 1 | B | 160.814 | 98.334 | 482.822 | 55.697 | 49.005 |
| 32 | 1/1 | 1 | C | 334.676 | 238.496 | 835.514 | 133.757 | 118.701 |
| 32 | 1/1 | 2 | A | 30.775 | 29.039 | 48.088 | 17.215 | 8.927 |
| 32 | 1/1 | 2 | B | 131.598 | 105.342 | 334.050 | 60.461 | 37.538 |
| 32 | 1/1 | 2 | C | 301.983 | 252.142 | 710.909 | 141.827 | 57.219 |
| 32 | 1/1 | 3 | A | 26.384 | 24.697 | 45.436 | 14.649 | 11.905 |
| 32 | 1/1 | 3 | B | 93.357 | 85.361 | 130.228 | 52.375 | 36.921 |
| 32 | 1/1 | 3 | C | 231.295 | 221.541 | 359.712 | 125.510 | 105.637 |
| 32 | 1/1 | 4 | A | 85.271 | 95.947 | 103.639 | 17.022 | 8.540 |
| 32 | 1/1 | 4 | B | 473.994 | 480.792 | 624.356 | 54.805 | 39.552 |
| 32 | 1/1 | 4 | C | 823.249 | 837.357 | 945.779 | 144.305 | 48.661 |
| 32 | 1/1 | 5 | A | 46.394 | 26.420 | 97.396 | 14.846 | 70.041 |
| 32 | 1/1 | 5 | B | 254.894 | 98.172 | 598.122 | 54.324 | 402.536 |
| 32 | 1/1 | 5 | C | 463.805 | 198.352 | 887.456 | 124.657 | 667.977 |
| 32 | 2048/512 | 0 | A | 67.752 | 60.562 | 115.546 | 25.820 | 76.935 |
| 32 | 2048/512 | 0 | B | 332.867 | 395.249 | 545.412 | 101.568 | 374.451 |
| 32 | 2048/512 | 0 | C | 583.328 | 725.240 | 967.865 | 143.042 | 707.494 |
| 32 | 2048/512 | 1 | A | 54.120 | 41.499 | 110.697 | 24.105 | 36.807 |
| 32 | 2048/512 | 1 | B | 279.325 | 169.097 | 536.095 | 100.585 | 350.678 |
| 32 | 2048/512 | 1 | C | 489.203 | 271.621 | 957.685 | 143.130 | 634.399 |
| 32 | 2048/512 | 2 | A | 44.944 | 38.953 | 70.937 | 26.604 | 12.447 |
| 32 | 2048/512 | 2 | B | 167.111 | 133.946 | 469.942 | 105.494 | 19.174 |
| 32 | 2048/512 | 2 | C | 310.626 | 257.215 | 816.199 | 152.871 | 74.365 |
| 32 | 2048/512 | 3 | A | 32.197 | 29.072 | 43.906 | 23.363 | 8.309 |
| 32 | 2048/512 | 3 | B | 132.380 | 130.173 | 186.472 | 101.363 | 15.008 |
| 32 | 2048/512 | 3 | C | 237.888 | 235.451 | 337.058 | 144.373 | 65.346 |
| 32 | 2048/512 | 4 | A | 30.242 | 27.038 | 45.583 | 22.705 | 3.088 |
| 32 | 2048/512 | 4 | B | 124.116 | 119.790 | 176.253 | 100.139 | 20.576 |
| 32 | 2048/512 | 4 | C | 199.965 | 181.060 | 274.646 | 145.981 | 43.391 |
| 32 | 2048/512 | 5 | A | 75.430 | 66.128 | 116.190 | 26.542 | 64.322 |
| 32 | 2048/512 | 5 | B | 375.729 | 468.276 | 571.071 | 103.603 | 387.941 |
| 32 | 2048/512 | 5 | C | 665.526 | 817.638 | 1025.397 | 169.566 | 649.784 |
| 32 | 2048/2048 | 0 | A | 52.362 | 49.018 | 76.063 | 31.905 | 11.777 |
| 32 | 2048/2048 | 0 | B | 140.586 | 123.843 | 217.709 | 98.761 | 42.037 |
| 32 | 2048/2048 | 0 | C | 257.355 | 241.131 | 343.177 | 189.498 | 65.261 |
| 32 | 2048/2048 | 0 | S | 228.620 | 215.712 | 303.849 | 177.140 | 48.126 |
| 32 | 2048/2048 | 1 | A | 64.155 | 59.763 | 113.335 | 29.726 | 25.297 |
| 32 | 2048/2048 | 1 | B | 270.638 | 180.601 | 531.206 | 114.036 | 297.892 |
| 32 | 2048/2048 | 1 | C | 502.721 | 349.171 | 878.489 | 189.672 | 483.352 |
| 32 | 2048/2048 | 1 | S | 548.297 | 334.920 | 1025.775 | 177.073 | 619.252 |
| 32 | 2048/2048 | 2 | A | 62.247 | 55.440 | 114.687 | 24.568 | 34.065 |
| 32 | 2048/2048 | 2 | B | 289.811 | 180.416 | 544.840 | 100.112 | 374.099 |
| 32 | 2048/2048 | 2 | C | 517.575 | 336.289 | 914.204 | 186.496 | 555.329 |
| 32 | 2048/2048 | 2 | S | 539.674 | 289.480 | 1061.514 | 177.045 | 697.782 |
| 32 | 2048/2048 | 3 | A | 44.502 | 35.390 | 76.266 | 24.000 | 16.322 |
| 32 | 2048/2048 | 3 | B | 147.846 | 113.516 | 364.489 | 99.818 | 26.922 |
| 32 | 2048/2048 | 3 | C | 281.566 | 218.772 | 709.733 | 174.012 | 74.889 |
| 32 | 2048/2048 | 3 | S | 296.328 | 202.500 | 927.653 | 173.381 | 47.848 |
| 32 | 2048/2048 | 4 | A | 59.817 | 53.153 | 80.586 | 35.117 | 22.099 |
| 32 | 2048/2048 | 4 | B | 190.213 | 136.959 | 440.390 | 104.666 | 87.262 |
| 32 | 2048/2048 | 4 | C | 343.002 | 265.377 | 769.681 | 191.338 | 134.960 |
| 32 | 2048/2048 | 4 | S | 344.069 | 224.731 | 916.207 | 187.287 | 64.451 |
| 32 | 2048/2048 | 5 | A | 49.934 | 47.405 | 71.902 | 23.746 | 28.676 |
| 32 | 2048/2048 | 5 | B | 200.158 | 132.841 | 452.134 | 100.243 | 114.936 |
| 32 | 2048/2048 | 5 | C | 384.625 | 286.267 | 791.823 | 180.454 | 240.258 |
| 32 | 2048/2048 | 5 | S | 395.288 | 249.373 | 934.444 | 176.396 | 345.944 |
| 128 | 2048/512 | 0 | A | 37.324 | 34.521 | 52.835 | 26.240 | 10.610 |
| 128 | 2048/512 | 0 | B | 151.924 | 146.690 | 197.481 | 115.787 | 17.400 |
| 128 | 2048/512 | 0 | C | 264.479 | 255.933 | 361.299 | 151.134 | 55.043 |
| 128 | 2048/512 | 1 | A | 39.874 | 35.980 | 63.515 | 25.720 | 13.975 |
| 128 | 2048/512 | 1 | B | 182.080 | 149.728 | 393.296 | 119.982 | 33.651 |
| 128 | 2048/512 | 1 | C | 311.731 | 269.461 | 685.072 | 156.784 | 47.254 |
| 128 | 2048/512 | 2 | A | 65.900 | 60.373 | 110.280 | 24.591 | 19.185 |
| 128 | 2048/512 | 2 | B | 372.333 | 400.183 | 562.398 | 131.688 | 173.106 |
| 128 | 2048/512 | 2 | C | 692.094 | 727.327 | 946.310 | 234.771 | 301.169 |
| 128 | 2048/2048 | 0 | A | 49.227 | 48.568 | 69.168 | 33.941 | 18.184 |
| 128 | 2048/2048 | 0 | B | 198.043 | 163.990 | 371.780 | 120.463 | 54.921 |
| 128 | 2048/2048 | 0 | C | 360.319 | 318.504 | 639.818 | 208.067 | 81.614 |
| 128 | 2048/2048 | 1 | A | 56.614 | 53.615 | 73.479 | 34.541 | 18.011 |
| 128 | 2048/2048 | 1 | B | 272.258 | 281.911 | 399.486 | 135.865 | 160.712 |
| 128 | 2048/2048 | 1 | C | 521.346 | 547.830 | 725.139 | 242.638 | 245.639 |
| 128 | 2048/2048 | 2 | A | 99.784 | 107.383 | 137.503 | 35.929 | 43.061 |
| 128 | 2048/2048 | 2 | B | 453.127 | 471.451 | 579.109 | 120.884 | 51.700 |
| 128 | 2048/2048 | 2 | C | 811.626 | 848.228 | 986.850 | 208.127 | 82.333 |
| 644 | 2048/512 | 0 | A | 70.940 | 65.529 | 113.123 | 28.586 | 34.530 |
| 644 | 2048/512 | 0 | B | 392.137 | 395.954 | 605.678 | 241.222 | 245.931 |
| 644 | 2048/512 | 0 | C | 567.424 | 452.084 | 1014.484 | 199.455 | 594.282 |
| 644 | 2048/512 | 1 | A | 55.441 | 53.233 | 75.469 | 27.616 | 17.539 |
| 644 | 2048/512 | 1 | B | 360.651 | 330.870 | 555.537 | 244.115 | 121.752 |
| 644 | 2048/512 | 1 | C | 476.776 | 431.952 | 850.125 | 181.705 | 318.577 |
| 644 | 2048/512 | 2 | A | 52.842 | 54.040 | 71.173 | 27.047 | 16.290 |
| 644 | 2048/512 | 2 | B | 306.821 | 284.314 | 460.296 | 243.054 | 54.399 |
| 644 | 2048/512 | 2 | C | 374.391 | 316.994 | 669.425 | 206.169 | 102.529 |
| 644 | 2048/2048 | 0 | A | 67.132 | 66.886 | 82.310 | 32.495 | 6.982 |
| 644 | 2048/2048 | 0 | B | 330.847 | 319.488 | 464.310 | 274.054 | 37.023 |
| 644 | 2048/2048 | 0 | C | 489.400 | 450.020 | 735.426 | 381.921 | 77.590 |
| 644 | 2048/2048 | 1 | A | 70.082 | 69.768 | 90.341 | 26.545 | 10.653 |
| 644 | 2048/2048 | 1 | B | 390.101 | 320.581 | 575.736 | 242.700 | 234.409 |
| 644 | 2048/2048 | 1 | C | 588.704 | 471.351 | 886.284 | 332.140 | 391.495 |
| 644 | 2048/2048 | 2 | A | 63.706 | 66.655 | 85.255 | 27.255 | 4.893 |
| 644 | 2048/2048 | 2 | B | 319.236 | 308.576 | 420.407 | 277.025 | 24.408 |
| 644 | 2048/2048 | 2 | C | 450.269 | 416.706 | 727.017 | 356.828 | 62.410 |


## Matched mean/median comparisons by timer

Starred columns first pair the same statistic inside a block, then take the median of block ratios. Mean/mean and median/median are not mixed. H/D/E retain different intervals. Absolute per-clock means and medians are in controlled_latency.csv.

| Z | input | pass/clock | block count | median/median C/B* | mean/mean C/B* |
|---|---|---|---|---|---|
| 32 | m1 | H/host | 6 | 2.333 | 2.188 |
| 32 | m1 | D/host | 6 | 1.970 | 1.977 |
| 32 | m1 | D/event | 6 | 2.171 | 2.223 |
| 32 | m1 | E/event | 6 | 2.284 | 2.089 |
| 32 | unique2048 | H/host | 6 | 1.936 | 1.844 |
| 32 | unique2048 | D/host | 6 | 1.661 | 1.619 |
| 32 | unique2048 | D/event | 6 | 1.734 | 1.700 |
| 32 | unique2048 | E/event | 6 | 2.395 | 1.987 |
| 32 | repeat2048 | H/host | 6 | 1.777 | 1.762 |
| 32 | repeat2048 | D/host | 6 | 1.552 | 1.630 |
| 32 | repeat2048 | D/event | 6 | 1.668 | 1.703 |
| 32 | repeat2048 | E/event | 6 | 2.157 | 1.804 |
| 128 | unique2048 | H/host | 3 | 1.942 | 1.819 |
| 128 | unique2048 | D/host | 3 | 1.519 | 1.754 |
| 128 | unique2048 | D/event | 3 | 1.586 | 1.822 |
| 128 | unique2048 | E/event | 3 | 2.504 | 2.341 |
| 128 | repeat2048 | H/host | 3 | 1.800 | 1.741 |
| 128 | repeat2048 | D/host | 3 | 1.700 | 1.735 |
| 128 | repeat2048 | D/event | 3 | 1.817 | 1.760 |
| 128 | repeat2048 | E/event | 3 | 1.994 | 1.905 |
| 644 | unique2048 | H/host | 3 | 1.409 | 1.479 |
| 644 | unique2048 | D/host | 3 | 1.518 | 1.536 |
| 644 | unique2048 | D/event | 3 | 1.568 | 1.600 |
| 644 | unique2048 | E/event | 3 | 1.728 | 1.911 |
| 644 | repeat2048 | H/host | 3 | 1.142 | 1.322 |
| 644 | repeat2048 | D/host | 3 | 1.011 | 1.130 |
| 644 | repeat2048 | D/event | 3 | 1.019 | 1.138 |
| 644 | repeat2048 | E/event | 3 | 1.125 | 1.610 |

## Actual lifecycle / intervals

27 fresh process workers = A/B/C ×N128/1024/4096 ×3 blocks. No warm-up or correctness call occurs in the lifecycle worker before the real loop. Session starts before artifact read and ends after final owned tensor cleanup+sync. ID read/H2D belongs to setup. A dense build and compact removal belong to setup. First call is part of total_loop, never added twice. Setup+total_loop+final_cleanup=session is checked from raw ns clocks. Parent spawn-to-exit includes imports/context/JSON/exit; component intervals are not added to their enclosing intervals. Process user/system CPU and peak RSS are sampled at final reporting before JSON/exit, rather than claimed as complete parent-side CPU accounting.

| N | method | session s: b0 / b1 / b2 | median session s | parent worker s: b0 / b1 / b2 |
|---|---|---|---|---|
| 128 | A | 0.196851 / 0.169624 / 0.196438 | 0.196438 | 2.072030 / 1.898246 / 1.904284 |
| 128 | B | 0.168025 / 0.190640 / 0.185433 | 0.185433 | 2.009308 / 1.854379 / 1.887632 |
| 128 | C | 0.224380 / 0.229143 / 0.245484 | 0.229143 | 2.086732 / 2.058524 / 2.039747 |
| 1024 | A | 0.252050 / 0.223652 / 0.195350 | 0.223652 | 2.060568 / 2.041846 / 1.878002 |
| 1024 | B | 0.410242 / 0.410329 / 0.318430 | 0.410242 | 2.180281 / 2.144023 / 2.216770 |
| 1024 | C | 0.507840 / 0.655918 / 0.632942 | 0.632942 | 2.318672 / 2.321995 / 2.431254 |
| 4096 | A | 0.442087 / 0.530962 / 0.342683 | 0.442087 | 2.117734 / 2.402362 / 2.118920 |
| 4096 | B | 0.941224 / 1.514844 / 0.832234 | 0.941224 | 2.781958 / 3.278795 / 2.605198 |
| 4096 | C | 1.234773 / 1.125446 / 1.362115 | 1.234773 | 3.017688 / 2.867949 / 3.179739 |


All detailed import/init/artifact/ID/dense/first/loop/cleanup/session values, process CPU/RSS, allocated/reserved peaks and session/N are in [lifecycle.csv](summary/lifecycle.csv). Session/N is amortized average cost, not p95. File caches may already be warm; no drop_caches was performed. Each N has independent workers; N4096 prefixes are not counted as three lifecycle sessions. Historical A/B linear scenarios stay at their original setup+N*warm median, displayed as dashed lines and never refitted to the new data.

## P4A historical reanalysis / memory stages

Recomputed from all90 final raw files: host-median C wins5/30; event-mean4/30; common0/30; disagreement9/30; host wins U=M3; all-three-host-win conditions0/10. Matched median/median gives5/4/1/7 and mean/mean8/4/1/10. Both measurements retain their original separate-pass K32/8/1 contracts. `legacy_statistics.csv` has180 clock records including all100 samples' statistics; raw samples remain in original files. P2's denominator90 is never added to P4's96 method-blocks.

Historical baseline already32MiB in all90 workers; warm−baseline−module−IDs residual0–952 bytes. Separate after-load/output-release endpoints were not saved. New memory workers trace A/B/C before/after library GEMM, artifact, first output, output release, warm20 and module/ID removal. Their active32MiB blocks after removal have setWorkspaceForHandle/getCurrentCUDABlasHandle C++ stack frames, identifying Torch cuBLAS handle workspace in the new replay. New stage endpoints and reserved history are diagnostic, not replacements for historical timed peaks. No Python-tensor enumeration alone was used to assign ownership. No profile worker was needed after stage/stack evidence answered this attribution question; no internal unique/decode/inverse pure-cost claim is made.

## Budget, completion and preservation

| worker kind | count | parent wall s |
|---|---|---|
| preflight | 3 | 31.343979899 |
| smoke | 1 | 2.267547397 |
| timing | 30 | 161.496339344 |
| lifecycle | 27 | 61.774636383 |
| memory | 3 | 8.893412413 |


Original196 workers: 457.424303870002s; P4 cap900s, spent265.775915435997s, remaining634.224084564003s. Global260 workers: 723.200219305999s /1800s; remaining1076.799780694001s. Wall includes startup, preflight CPU checks, context/setup, execution, failure/retry allowance and teardown. Actual failures/retries/probe-only/profile workers=0. The64 common probes are inside existing workers. Budget records are unique and raw-bound; costs may be summed, performance samples may not.

Reconstructed work accounting:108553 actual PLE method calls (including warm-up/smoke/preflight/memory/lifecycle);60672 belong to main H/D/E workers including their warm-up,47232 to lifecycle. Full-table setup made5632 official256-row decoder calls. Nominal GEMM FLOPs=2×decoded_rows×4096×Z; A request lookup has0 decoder FLOPs but A setup does not. No FLOP count is presented as measured hardware operations, total cost, energy or money.

All747 original Case files and18 external original files retain bytes/SHA256; only `followup/p4_speed_cost/` is added. Git HEAD/branch/status/staged are compared in preservation receipts. The historical PUBLIC_MANIFEST remains unchanged. Its unchanged verifier passed on a temporary747-file original-scope copy; no coverage expectation was loosened. Strict local audit verifies external originals; portable mode explicitly reports unavailable outside-case originals. Local postprocessing and validation are identified by the final P4 manifest separately from the pre-GPU bundle.

## Scope

This is an eager operator study on one RTX 5080. Identical synthetic artifacts and GPU ID arrays are reused in warm-cache calls returning all16 layers as raw, unscaled BF16. Trained OPTIC quality, LLM inference, TTFT/TPOT, KV redistribution and server concurrency were not measured. Z644 matches only the E4B latent width, not its V/L/H or quality. OS file caches may be warm and GPU/CPU clocks were not fixed. WSL device samples are neither continuous peaks nor complete per-process attribution. P2 isolated workers and P4 shared-context workers retain separate contracts and denominators. Process blocks are the repetition unit; inner calls are not independent models or user tasks. Power samples are not converted to energy or monetary cost.

[CPU reproduction](REPRODUCE.md) · [SOURCE_MAP](SOURCE_MAP.md) · [REUSE_MAP](REUSE_MAP.md) · [P4 manifest](PUBLIC_MANIFEST_P4.json)

Three initial P4A decision/paired/memory-stage snapshots are preserved byte-identically in `results/p4a_initial/`. External CPU replay exposed a scope mismatch between those snapshots and generated summary files; no numerical disagreement or GPU rerun occurred. Original paths, the observed error and preserved hashes are in `provenance/p4a_snapshot_preservation.json`.
