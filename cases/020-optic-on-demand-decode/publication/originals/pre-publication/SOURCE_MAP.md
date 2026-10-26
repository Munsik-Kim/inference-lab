# Source map — pinned OPTIC to this operator case

| Source | Verified meaning | Local use / boundary |
|---|---|---|
| [Paper v1 §2.2/§2.3](https://arxiv.org/html/2610.31309v1) | Token-shared discrete codes, layer-specific affine decoders; output-path fitting | Context and affine form only; no fitting or model-quality reproduction |
| [Paper §6](https://arxiv.org/html/2610.31309v1) | Compact inference background | Paper RTX4090/full-model measurements are not Case020 numbers |
| [compact_adapter.py@344c712aeb6620e39eb78930fb922447f0e123f3](https://github.com/VILA-Lab/optic/blob/344c712aeb6620e39eb78930fb922447f0e123f3/src/ple_eval/compact_adapter.py) | load_compact_ple selects optic_npz; discrete,8bits,K256; Z=G*D; exact geometry | Vendored unchanged. uint8 token codes; FP16 codebooks/weights/bias; metadata and bytes in source/fixture manifests |
| CompactOptic.forward | Flatten IDs→code index_select→group offsets/embedding→latent→stacked GEMM→FP32 bias→BF16 [*ids.shape,L*H] | B calls original forward. All layers. No built-in torch.unique; C is an explicit wrapper |
| _matmul_fp32 | CUDA torch.mm(FP16,FP16,out_dtype=FP32); CPU casts operands toFP32 | Actual sm_120 probe, no fallback. CPU and GPU arithmetic paths differ |
| compact_resident_bytes | Counts persistent tensors, excludes nonpersistent buffers | Preserved comparison column; main accounting adds group_offsets and deduplicates underlying storage |
| install_compact_ple | Wraps decoder output with sqrt(H) scaling | Not called. A/B/C use raw unscaled load_compact_ple output |
| [CPU example](https://github.com/VILA-Lab/optic/blob/344c712aeb6620e39eb78930fb922447f0e123f3/examples/decode_optic.py) | V32,L3,H16,G2,D4,Z8 synthetic; BF16 CPU output | Executed first; original stdout/receipt preserved. P1 extends toV257/nonzero bias/three seeds |
| [deployment.md](https://github.com/VILA-Lab/optic/blob/344c712aeb6620e39eb78930fb922447f0e123f3/docs/deployment.md) | Python3.12,24GiB full-model example; vLLM0.19.1 serving | Background only; not required for this operator and not substituted for RTX5080 results |
| pyproject.toml / src/ple_eval/__init__.py | Python≥3.12, empty project dependencies, minimal package init | No project install; module import from pinned vendor path; NumPy/Torch suffice |
| LICENSE | Apache-2.0 | Original bytes/hash retained; notices below |

Independent reference axes: codes[V,G]; codebooks[G,K,D]; z[m,g*D+d]=codebooks[g,codes[ids[m],g],d]; weight[L,H,Z]; bias[L,H]; y[m,l,h]=sum_z z[m,z]*weight[l,h,z]+bias[l,h]. Stored FP16 values are promoted toFP64 before the sum. Layers remain in source order. The scalar loop, einsum and independent per-layer matrix products use the same documented axes but do not call the Torch decoder for reference arithmetic.

Original source identity: `344c712aeb6620e39eb78930fb922447f0e123f3`. Exact URLs/bytes/SHA256 for 8 files: [source_manifest.json](provenance/source_manifest.json). No floating upstream main, package init workaround, .git copy, model or unused benchmark dataset was vendored.
