# Method and execution contract / 방법·실행 계약

이번 결과의 이름은 **PAPER_INT4_REFERENCE_VARIANT**입니다. 논문·고정 공개 코드와 같은 부분, 실제로 다르게 실행한 부분을 구분합니다. 아래 source와 byte 장부가 계약의 근거이며 새로운 알고리즘을 주장하지 않습니다.

## Paper, public wrapper and this study

| Item | Paper v1 | Pinned public wrapper | This study |
|---|---|---|---|
| Scale source | Weight reconstruction objective | Original weight tile | Original weight only; no calibration |
| INT4 inverse scales | BF16 admissible set in §4.2 | Alpha inverse projection selected by `scale_format`; `none` exists | FP32 a/b/γ; `scale_format=none` |
| Inner quantizer | Format-specific Q | Element-only switch; reviewed config default false | Element-only round/clip, then final group RTN |
| Format | INT4 among several formats | Element INT4 [-8,7]; final absmax/7, min1e−8 | Signed [-8,7], ties-even; group64, explicit γ |
| Initialization/iterations | Parameters supplied to Algorithm1; a universal default is UNVERIFIED | Configurable max_abs row/L1 column; EPS1e−12 | max_abs/L1;15; output-row split1 |
| Update order | Row → fresh Q → column | Same BCD ordering | Same ordering; all output rows in each column reduction |
| Zero/tiny values | Degenerate-row convention discussed | Native final clamp; detailed zero handling differs by branch | Unit scale for structural zeros; reject invalid nonzero scale |
| Input beta transport | Rescale X by inverse column scale | Rescale or fold branch | FP32 X·b, then BF16; no double application |
| Packed deployment | Separate author implementation | Private block-format dependency in complete program | Main quality is dense BF16; public Marlin probe remains blocked at full-model gate |

Primary sources: [paper §4.2–4.4/Algorithm1](https://arxiv.org/html/2610.13767v1), [public wrapper at reviewed commit](https://github.com/MahsaMoonlike/dualquant/blob/d84b7e9dff653320e47c06a365eb8295c1825bc3/layer_wrapper_dualquant.py). Local source locators are in [paper/code mapping](records/paper_code_mapping.txt); the unlicensed vendor tree is not redistributed.

## Scale directions, shapes and actual bytes

For W=[O,I], G=I/64: codes=[O,I], a=`alpha_inv`=[O,G], γ=`gamma`=[O,G], b=`beta_inv`=[I]. The shared-column update spans all O rows inside each input tile. Gamma and a broadcast along the64 columns; b broadcasts along output rows. The forward scales named alpha/beta by the wrapper are reciprocals of the stored a/b.

```text
W_tilde = alpha_inverse * gamma * codes
W_effective = W_tilde * beta_inverse
D16 = linear(BF16(FP32(X) * beta_inverse), BF16(W_tilde), bias)
D4 = linear(BF16(Q_activation(FP32(X) * beta_inverse)), BF16(W_tilde), bias)
```

D4 uses dynamic token/group64 absmax INT4 scale and reconstructs the activation before BF16 matrix multiplication. It is W4A4_REFERENCE_VARIANT, not a native packed W4A4 kernel. [Transport negative control and serialization tests](synthetic/test_int4.py) and [actual-matrix algebra diagnostics](records/diagnostics/linear_diagnostics.json) detect beta omission/double transport. BF16 folded and transported expressions can differ through rounding.

| Condition | Stored projection artifact | GPU weight / arithmetic | Activation treatment | KV |
|---|---|---|---|---|
| BF16 | N/A: no separate INT4 artifact | Original BF16 | Original BF16 path | BF16 |
| RTN | Nibble codes + FP32 γ +24B header | Reconstructed BF16, BF16 matmul | No nontrivial b | BF16 |
| D16 | Codes + FP32 γ/a/b +24B header | Reconstructed BF16, BF16 matmul | FP32 rescale →BF16 | BF16 |
| D4 | Reuses the exact D16 artifact | Reconstructed BF16, BF16 matmul | Rescale →reference INT4/reconstruction →BF16 | BF16 |
| Marlin probe | INT4 codes repacked on GPU; BF16 combined a·γ | Fused unpack with BF16 tensor-core MMA | FP32 rescale →BF16 | Original path |

Per projection: code=O·I/2 bytes; γ=4·O·G; D16 additionally a=4·O·G and b=4·I. RTN omits identity a/b. [4B D16 metadata](records/artifact_metadata/4b-D16.json) contains actual layer shapes, per-field ledger and external artifact hashes. Payload includes converted projections only, not embedding/head/norm or a standalone model. The raw BF16 artifact field0 means no INT4 artifact, never a zero-byte model.

## Scope and weight error

Apply q/k/v/o/gate/up/down in all blocks:196 projections in0.6B,252 in4B. [Runner](source/paper_model_runner.py) records unchanged norm hashes and embedding/head tie relationships. Embedding/lm_head/norm/attention/RoPE/KV remain on original paths.

The normalized weight error is `sum_layer ||float32(W) − W_effective||² / sum_layer ||float32(W)||²`, with squared differences/energies reduced in FP64. It is **energy-weighted**, not the unweighted mean of layer ratios, nor mean squared error per scalar. [Artifact metadata](records/artifact_metadata/4b-RTN.json) permits independent sum/byte aggregation; private weights are not required for that aggregation and are not included. D4 reuses D16 weights, so their weight error and artifact bytes match.

RTN W4A4 and stock GPTQ in the same runner are NOT_RUN. Same4-bit codes do not imply equal bytes: D16 stores extra scales. [Results](REPORT.md#results) are not an equal-byte superiority claim.
