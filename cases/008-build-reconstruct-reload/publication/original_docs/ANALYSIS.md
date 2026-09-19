# Results and interpretation

Q and R use separate models, input manifests and runtimes. All intervals below are pointwise descriptive intervals from 5,000 frozen resamples; no deployment quality tolerance was specified.

## Q artifact and quality

- Native safetensors: 8,044,982,000 bytes. W4 safetensors: 2,651,839,568 bytes; reduction 67.037%.
- Conversion elapsed 1712.458 seconds; calibration 256 prompts / 30,011 tokens. This elapsed time includes capture/compression/save work and is not an isolated conversion benchmark.

| Arm / 192 scenarios | Correct | Δ full gold NLL (nats; 95% CI) |
| --- | ---: | --- |
| Q-BF16 | 137/192 | +0.000000 [+0.000000, +0.000000] |
| Q-W4 | 137/192 | -0.591295 [-0.723442, -0.470060] |

## R model scores

| Arm / 192 scenarios | Correct | Δ full gold NLL (nats; 95% CI) |
| --- | ---: | --- |
| I25 | 128/192 | -0.583823 [-0.660297, -0.505045] |
| I25-R | 124/192 | -0.058966 [-0.082720, -0.034713] |
| P25 | 130/192 | +0.774289 [+0.727695, +0.820419] |
| P25-R | 123/192 | -0.134708 [-0.155610, -0.113348] |
| R-B | 124/192 | +0.000000 [+0.000000, +0.000000] |
| S50 | 107/192 | +2.267479 [+2.169380, +2.371095] |
| S50-R | 122/192 | -0.139956 [-0.173521, -0.106027] |

## Reconstruction transfer

| Fixed structure | Pooled recovery (95% CI) | Improved prompts |
| --- | ---: | ---: |
| I25 | 95.40% [95.25, 95.55] | 192/192 |
| P25 | 94.96% [94.79, 95.13] | 192/192 |
| S50 | 94.09% [93.90, 94.28] | 192/192 |

## Q paired outcomes and baseline task performance

- **Q-BF16**: flips 0/192; regressions 0/137 baseline-correct; gains 0/55 baseline-wrong; wrong-to-different-wrong 0/192. Mean allowed-label mass 0.666963; conditional choice NLL change +0.000000 [+0.000000, +0.000000] nats; top ties 0/192; KL(B || candidate) 0.000000.
  code: 16/64 correct, comparison: 58/64 correct, retrieval: 63/64 correct
- **Q-W4**: flips 2/192; regressions 1/137 baseline-correct; gains 1/55 baseline-wrong; wrong-to-different-wrong 0/192. Mean allowed-label mass 0.673368; conditional choice NLL change +0.379465 [+0.248205, +0.497079] nats; top ties 1/192; KL(B || candidate) 0.135071.
  code: 16/64 correct, comparison: 57/64 correct, retrieval: 64/64 correct

Conditional denominators differ from the overall 192-item denominator. Low baseline accuracy limits claims about useful task capability. Zero regressions do not prove zero risk.

## Q cost

### fixed_8_token_request

| Arm | Mean wall ms | Speedup [95% CI] |
| --- | ---: | --- |
| Q-BF16 | 164.7106 | 1.0000 [1.0000, 1.0000] |
| Q-W4 | 150.9686 | 1.0910 [0.8790, 1.2899] |

### one_token_request

| Arm | Mean wall ms | Speedup [95% CI] |
| --- | ---: | --- |
| Q-BF16 | 21.6376 | 1.0000 [1.0000, 1.0000] |
| Q-W4 | 23.3290 | 0.9275 [0.8474, 1.0095] |

## R paired outcomes and baseline task performance

- **I25**: flips 8/192; regressions 2/124 baseline-correct; gains 6/68 baseline-wrong; wrong-to-different-wrong 0/192. Mean allowed-label mass 0.479167; conditional choice NLL change -0.336605 [-0.410648, -0.258953] nats; top ties 0/192; KL(B || candidate) 0.122761.
  code: 16/64 correct, comparison: 50/64 correct, retrieval: 62/64 correct
  Local relative error: mean 39.6698%, median 39.7500%, p95 41.5073%. Worst prompt `c008-r-heldout-comparison-007`.
- **I25-R**: flips 5/192; regressions 2/124 baseline-correct; gains 2/68 baseline-wrong; wrong-to-different-wrong 1/192. Mean allowed-label mass 0.441140; conditional choice NLL change -0.062951 [-0.086391, -0.039724] nats; top ties 1/192; KL(B || candidate) 0.008785.
  code: 16/64 correct, comparison: 48/64 correct, retrieval: 60/64 correct
  Local relative error: mean 8.1835%, median 8.6310%, p95 10.8643%. Worst prompt `c008-r-heldout-retrieval-002`.
- **P25**: flips 10/192; regressions 2/124 baseline-correct; gains 8/68 baseline-wrong; wrong-to-different-wrong 0/192. Mean allowed-label mass 0.316446; conditional choice NLL change -0.087308 [-0.134529, -0.040644] nats; top ties 0/192; KL(B || candidate) 0.182441.
  code: 16/64 correct, comparison: 51/64 correct, retrieval: 63/64 correct
  Local relative error: mean 39.4471%, median 39.3732%, p95 40.8853%. Worst prompt `c008-r-heldout-comparison-052`.
- **P25-R**: flips 5/192; regressions 3/124 baseline-correct; gains 2/68 baseline-wrong; wrong-to-different-wrong 0/192. Mean allowed-label mass 0.464507; conditional choice NLL change -0.026047 [-0.045053, -0.007373] nats; top ties 2/192; KL(B || candidate) 0.019861.
  code: 16/64 correct, comparison: 47/64 correct, retrieval: 60/64 correct
  Local relative error: mean 8.4832%, median 8.9495%, p95 11.4404%. Worst prompt `c008-r-heldout-retrieval-002`.
- **R-B**: flips 0/192; regressions 0/124 baseline-correct; gains 0/68 baseline-wrong; wrong-to-different-wrong 0/192. Mean allowed-label mass 0.443230; conditional choice NLL change +0.000000 [+0.000000, +0.000000] nats; top ties 3/192; KL(B || candidate) 0.000000.
  code: 16/64 correct, comparison: 49/64 correct, retrieval: 59/64 correct
- **S50**: flips 24/192; regressions 20/124 baseline-correct; gains 3/68 baseline-wrong; wrong-to-different-wrong 1/192. Mean allowed-label mass 0.080457; conditional choice NLL change +0.178372 [+0.108593, +0.248319] nats; top ties 4/192; KL(B || candidate) 0.966793.
  code: 16/64 correct, comparison: 35/64 correct, retrieval: 56/64 correct
  Local relative error: mean 58.8122%, median 58.9535%, p95 61.0358%. Worst prompt `c008-r-heldout-retrieval-028`.
- **S50-R**: flips 5/192; regressions 3/124 baseline-correct; gains 1/68 baseline-wrong; wrong-to-different-wrong 1/192. Mean allowed-label mass 0.452981; conditional choice NLL change -0.081570 [-0.112689, -0.049763] nats; top ties 2/192; KL(B || candidate) 0.016958.
  code: 16/64 correct, comparison: 47/64 correct, retrieval: 59/64 correct
  Local relative error: mean 13.7042%, median 14.5365%, p95 18.5131%. Worst prompt `c008-r-heldout-retrieval-002`.

Conditional denominators differ from the overall 192-item denominator. Low baseline accuracy limits claims about useful task capability. Zero regressions do not prove zero risk.

## R cost

### fixed_8_token_request

| Arm | Mean wall ms | Speedup [95% CI] |
| --- | ---: | --- |
| I25 | 169.6314 | 1.0052 [0.9542, 1.0583] |
| I25-R | 164.7643 | 1.0349 [0.9837, 1.0956] |
| P25 | 169.5877 | 1.0054 [0.9547, 1.0519] |
| P25-R | 168.1158 | 1.0142 [0.9625, 1.0703] |
| R-B | 170.5068 | 1.0000 [1.0000, 1.0000] |
| S50 | 173.7106 | 0.9816 [0.9348, 1.0217] |
| S50-R | 174.0325 | 0.9797 [0.9418, 1.0156] |

### prefill_and_argmax

| Arm | Mean wall ms | Speedup [95% CI] |
| --- | ---: | --- |
| I25 | 24.4411 | 1.0092 [0.8946, 1.1336] |
| I25-R | 25.2555 | 0.9767 [0.8713, 1.1010] |
| P25 | 25.9966 | 0.9488 [0.8325, 1.0881] |
| P25-R | 24.8760 | 0.9916 [0.8800, 1.1224] |
| R-B | 24.6662 | 1.0000 [1.0000, 1.0000] |
| S50 | 24.9655 | 0.9880 [0.8530, 1.1230] |
| S50-R | 26.8074 | 0.9201 [0.8144, 1.0416] |

## Reconstruction across splits

The common DEV-selected eta is 0.01; lambda is eta × trace(G)/retained width. Calibration uses 128 prompts / 4,096 dependent sampled vectors, DEV 48 prompts, and held-out 192 prompts. Selection used only DEV, with no new deletion search.

| Structure | Calibration / DEV / held-out recovery | Independent prompts |
| --- | ---: | ---: |
| I25 | 98.941% / 95.698% / 95.402% | 128 / 48 / 192 |
| P25 | 98.856% / 95.443% / 94.961% | 128 / 48 / 192 |
| S50 | 97.973% / 94.699% / 94.092% | 128 / 48 / 192 |

## Local error levels and tails

Each value summarizes one prompt over its same 32 fixed positions; percentages use that prompt’s native teacher-output norm. Calibration and DEV are fitting/selection evidence, while held-out is the transfer comparison.

| Split / arm | Mean relative error | Prompt p95 |
| --- | ---: | ---: |
| calibration / I25 | 39.4537% | 41.5846% |
| calibration / I25-R | 3.9606% | 4.9109% |
| calibration / P25 | 39.2846% | 40.8468% |
| calibration / P25-R | 4.0947% | 4.9826% |
| calibration / S50 | 58.7138% | 61.0211% |
| calibration / S50-R | 8.0797% | 10.4574% |
| development / I25 | 39.2180% | 41.6533% |
| development / I25-R | 7.6741% | 10.6652% |
| development / P25 | 39.1457% | 40.6709% |
| development / P25-R | 7.8672% | 11.3608% |
| development / S50 | 58.4276% | 60.4625% |
| development / S50-R | 12.7215% | 17.3165% |
| heldout / I25 | 39.6698% | 41.5073% |
| heldout / I25-R | 8.1835% | 10.8643% |
| heldout / P25 | 39.4471% | 40.8853% |
| heldout / P25-R | 8.4832% | 11.4404% |
| heldout / S50 | 58.8122% | 61.0358% |
| heldout / S50-R | 13.7042% | 18.5131% |

The repaired prompt-error mean and p95 are lower for all three structures in each split. Worst prompt IDs and absolute RMS values remain in [local tables](results/derived/local_tables.json). These are reconstruction diagnostics, separate from gold-token scores.

## Same-size repair cost

| Same-size repair pair | Prefill + argmax ratio [95% CI] | Fixed 8-token ratio [95% CI] |
| --- | ---: | --- |
| I25 | 0.9678 [0.8668, 1.0718] | 1.0295 [0.9727, 1.0905] |
| P25 | 1.0450 [0.8942, 1.2265] | 1.0088 [0.9609, 1.0598] |
| S50 | 0.9313 [0.8329, 1.0374] | 0.9981 [0.9631, 1.0328] |

The numerator is the uncorrected arm at the same width; the denominator is its repaired arm. These comparisons do not change the matrix count or width.

## Structure and resident memory

| Layer-13 structure | Intermediate width / MLP parameters | BF16 MLP parameter bytes |
| --- | ---: | ---: |
| Native | 3072 / 9,437,184 | 18,874,368 |
| I25 / P25, before and after repair | 2304 / 7,077,888 | 14,155,776 |
| S50, before and after repair | 1536 / 4,718,592 | 9,437,184 |

These are one MLP, not total-model compression fractions. Exported complete-model parameter bytes are recorded below; tied embeddings are stored once.

| R arm | Complete-model parameter bytes | Stored tensors |
| --- | ---: | ---: |
| I25 | 1,187,381,248 | 310 |
| I25-R | 1,187,381,248 | 310 |
| P25 | 1,187,381,248 | 310 |
| P25-R | 1,187,381,248 | 310 |
| R-B | 1,192,099,840 | 310 |
| S50 | 1,182,662,656 | 310 |
| S50-R | 1,182,662,656 | 310 |

### Q memory by process round

| Arm | Round | Bytes: allocated / reserved |
| --- | ---: | --- |
| Q-BF16 | 1 | resident named parameter bytes 8,044,936,192; allocator peaks 11,260,111,360 / 11,383,341,056 |
| Q-W4 | 1 | resident named parameter bytes 2,651,735,728; allocator peaks 5,795,133,952 / 6,025,117,696 |
| Q-BF16 | 2 | resident named parameter bytes 8,044,936,192; allocator peaks 11,260,111,360 / 11,383,341,056 |
| Q-W4 | 2 | resident named parameter bytes 2,651,735,728; allocator peaks 5,795,133,952 / 6,025,117,696 |
| Q-BF16 | 3 | resident named parameter bytes 8,044,936,192; allocator peaks 11,260,111,360 / 11,383,341,056 |
| Q-W4 | 3 | resident named parameter bytes 2,651,735,728; allocator peaks 5,795,133,952 / 6,025,117,696 |

### R memory by process round

| Arm | Round | Bytes: allocated / reserved |
| --- | ---: | --- |
| R-B | 1 | allocator peaks 1,245,358,080 / 1,270,874,112 |
| I25 | 1 | allocator peaks 1,241,163,776 / 1,270,874,112 |
| I25-R | 1 | allocator peaks 1,241,163,776 / 1,270,874,112 |
| P25 | 1 | allocator peaks 1,241,163,776 / 1,270,874,112 |
| P25-R | 1 | allocator peaks 1,241,163,776 / 1,270,874,112 |
| S50 | 1 | allocator peaks 1,236,864,512 / 1,249,902,592 |
| S50-R | 1 | allocator peaks 1,236,864,512 / 1,249,902,592 |
| I25 | 2 | allocator peaks 1,241,163,776 / 1,270,874,112 |
| I25-R | 2 | allocator peaks 1,241,163,776 / 1,270,874,112 |
| P25 | 2 | allocator peaks 1,241,163,776 / 1,270,874,112 |
| P25-R | 2 | allocator peaks 1,241,163,776 / 1,270,874,112 |
| S50 | 2 | allocator peaks 1,236,864,512 / 1,249,902,592 |
| S50-R | 2 | allocator peaks 1,236,864,512 / 1,249,902,592 |
| R-B | 2 | allocator peaks 1,245,358,080 / 1,270,874,112 |
| I25-R | 3 | allocator peaks 1,241,163,776 / 1,270,874,112 |
| P25 | 3 | allocator peaks 1,241,163,776 / 1,270,874,112 |
| P25-R | 3 | allocator peaks 1,241,163,776 / 1,270,874,112 |
| S50 | 3 | allocator peaks 1,236,864,512 / 1,249,902,592 |
| S50-R | 3 | allocator peaks 1,236,864,512 / 1,249,902,592 |
| R-B | 3 | allocator peaks 1,245,358,080 / 1,270,874,112 |
| I25 | 3 | allocator peaks 1,241,163,776 / 1,270,874,112 |

Q build allocator peaks: 10,270,743,552 allocated / 10,720,641,024 reserved bytes. Q runtime memory includes its fixed 1 GiB KV-cache policy and initialization workspace; R memory is one complete model and the fixed workload. These are allocator observations, not a sampled whole-device or empty-desktop peak.

## Interpretation and evidence limits

R recovery describes native teacher-output error energy. Gold NLL can move in a different direction. Q and R speedups have different runtimes and boundaries and are never pooled. Timing has 12 fixed prompts, three process rounds and dependent warmup/block repetitions; it is not 192 independent timing samples. No CUDA-event or sampled whole-device peak is reported.

The first 4B smoke attempt stopped because the model-wrapper output hook was bypassed. The correction checks the actual final norm without changing weights or recipe; both later reloads passed. Earlier tiny-fixture initialization/RPC failures are in provenance. No held-out outcomes were used to select precision, deletion sets, task difficulty or retry parameters.

[Machine-readable summary](results/derived/summary.json) · [Methods](METHODS.md) · [Limitations](LIMITATIONS.md)
