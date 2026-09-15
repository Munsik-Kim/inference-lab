# Attribution

SageAttention and its CUDA/quantization kernels are the work of the SageAttention authors and contributors at [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention). This case uses their unmodified source at commit `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`. Its [Apache-2.0 license](provenance/SageAttention_LICENSE) is retained. The full source and binary wheel are obtained separately; this case does not claim their algorithm or kernels as new work.

The full-cost comparison was motivated by [Attention Quantization for Tabular Foundation Models, v1](https://arxiv.org/html/2609.13031v1). The INT8-QK/FP8-PV implementation measured here is different from that paper's pure-FP8 Q/K/V path. No TabPFN result or published crossover is reproduced.

Qwen3-0.6B is distributed by the Qwen team. PyTorch provides the BF16 baseline and Transformers the capture interface. Case 003 supplied its pinned model provenance, capture design and eight development texts; its original data and findings are unchanged.

This project's contribution is the adapter harness, CPU/GPU checks, full-operator measurements, local attention-output analysis and reproducible evidence. OpenAI Codex assisted with synthetic inputs, implementation, local execution, analysis and documentation. These checks are not described as independent human review. No new attention kernel, production router or model-quality guarantee is claimed.
