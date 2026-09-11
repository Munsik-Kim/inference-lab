# Attribution and scope

EFQ-Softmax, its affine mapping, scale rule, LUT and published parameter settings are the work of Haohui Han and coauthors, “EFQ-Softmax: Exp-Free Quantization for Softmax,” arXiv:2609.09721v1 (2026). This case independently implements the mathematical definitions and states additional numerical policies where v1 is not specific. It does not reproduce or claim authorship of their hardware kernel.

Qwen supplies the model checkpoint, tokenizer, architecture and their license. Transformers and PyTorch supply the native inference and numerical operators; NumPy and matplotlib support analysis. Upstream licenses remain applicable. The Qwen license is included under `provenance/model_LICENSE`.

The contribution here is a local numerical comparison, self-authored synthetic inputs, capture validation, a small development fixture, independent scalar-reference tests, fixed evaluation and inspectable evidence. Full model weights and large activation files are excluded.

OpenAI Codex assisted with the inputs, implementation, analysis and documentation. There was no independent human annotation review, third-party reproduction, new quantization-method invention, model-quality evaluation or hardware-acceleration claim. Project code and documents use the included Apache-2.0 license.
