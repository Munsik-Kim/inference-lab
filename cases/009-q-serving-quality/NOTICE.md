# Sources and licensing

Project code follows the repository [Apache-2.0 license](../../LICENSE). Qwen/Transformers provide the architecture and tokenizer; GPTQ, LLM Compressor and compressed-tensors produced the existing W4 checkpoint. vLLM and its integrated Marlin kernels execute it. This case contributes workload freezing, matched process orchestration, request validation, compact scalar export, paired analysis and presentation. Codex assisted implementation, execution, testing and writing.

Official benchmark definitions and metrics come from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520). ARC, MMLU, GSM8K and WikiText retain their source licenses and attribution. This package stores task/source identities, hashes and measured scalars; it does not bundle benchmark prose, model weights or generated rationales. The synthetic serving workload and data-v2 fixtures are project-generated.

See [related work](../../docs/related-work/README.md) for MARLIN, Dutta et al., He et al. and FLAP, with the reused mechanisms and differences. Neither a new quantizer nor a reproduction of all methods in those papers is claimed. Case008's original eager short-request measurements and R reconstruction remain separate historical evidence.
