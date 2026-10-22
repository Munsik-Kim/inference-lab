# Sources, licenses and assistance

Original Case018 code, synthetic fixtures, SVG and authored documentation use the repository Apache-2.0 license. No model weights, environments, KV tensors, full papers, private credentials or third-party source distributions are included.

Qwen/Qwen3-4B-Instruct-2507, vLLM0.32.0, FlashInfer and xgrammar are upstream model/engine/kernel/grammar implementations; we did not develop their algorithms or kernels. Historical runtime versions and observed limitations are preserved in archived receipts.

ReTrace / Lost in Recovery, arXiv:2610.27954v1 §§5.1–5.2 motivated checking behavior after recovery. ACE-Compression, arXiv:2610.27913v1 §4.1/Appendix B.1 motivated separating component and simultaneous success. See SOURCE_MAP.md for actual access and differences. No full-paper reproduction, copied author implementation or general future-state equivalence claim.

Codex assisted implementation, CPU replay, analysis and bilingual presentation. Independent refers to separate local ReferenceState/SSE implementations, not human peer review or another-machine/GPU reproduction. Shared parser, serialization, public builder and wire rendering are disclosed.

The quantity-read analysis is posthoc and observational. Original synthetic DEV/HOLDOUT results, failures, missing engine-token costs and shutdown warnings remain unchanged. New publication GPU/model requests:0.
