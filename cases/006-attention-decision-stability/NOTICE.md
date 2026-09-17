# Attribution and provenance

OpenAI Codex assisted the code, local execution, analysis, tests and documentation of this study. The contribution is a controlled paired evaluation, explicit validity checks, gold-grounded synthetic tasks, retained measurements and an offline evidence explorer. It is not a new quantizer or attention kernel. No human or third-party review is claimed.

- **Qwen team:** [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), pinned model/tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca`. No weights are redistributed.
- **SageAttention authors:** [pinned source](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/core.py), revision `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`. Installed source and extension hashes are recorded; no upstream source bundle or binary is redistributed.
- **Hugging Face Transformers:** installed 5.17.0 Qwen causal-LM, tokenizer and SDPA interfaces were inspected. [Official Qwen3 API documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3) is background context; local inspected source governs this study.
- **Inference Lab Cases [004](https://github.com/Munsik-Kim/inference-lab/tree/main/cases/004-low-precision-attention-break-even) and [005](https://github.com/Munsik-Kim/inference-lab/tree/main/cases/005-attention-precision-pareto):** explicit adapter settings, capture location, GQA/reference conventions and wall timing boundaries informed the new code. [The reuse ledger](provenance/reuse_ledger.json) records files/hashes and changed validity handling. Earlier cases and measurements are unchanged.

These public URLs were consulted on 2026-09-16. Repository code retains Apache-2.0 licensing in [LICENSE](LICENSE). Margin and softmax identities are elementary diagnostics, not claims of invention. Public task text is self-written synthetic input.
