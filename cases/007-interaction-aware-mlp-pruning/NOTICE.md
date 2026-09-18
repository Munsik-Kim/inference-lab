# Attribution

Qwen supplies Qwen3-0.6B and its model/tokenizer assets; the snapshot is pinned to `c1899de289a04d12100db370d81485cdf75e47ca`. Hugging Face Transformers supplies the Qwen implementation and PyTorch supplies the tensor, linear and attention operations. Model assets are not redistributed. [Pinned Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) and [Transformers Qwen3 documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3) identify upstream components; recorded installed-source hashes control this experiment.

Alex M. Tseng, Prannay Kaul, Luca Zancato, Wei Xia and Stefano Soatto's [HOPE: Higher-order pruning of experts in mixture-of-experts language models, v1](https://arxiv.org/abs/2609.18916v1) motivates considering interactions between deletions. HOPE concerns MoE expert pruning. This dense grouped-SwiGLU accounting identity is not a reproduction of its objective or experiments.

Xinyin Ma, Gongfan Fang and Xinchao Wang's [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627v3) is relevant structural-pruning prior work. This case does not implement its gradient selection or recovery training. References were checked on 2026-09-18; upstream numerical claims are not local measurements.

The bounded integer AST interpreter is reused from [Case006](../006-attention-decision-stability/src/tasks.py). Its source hash and exact portion are in [the reuse ledger](provenance/reuse.json). Case004–006 inform measurement boundaries and scoring conventions. Their inputs and measured results are not reused as Case007 observations.

This project's work is the fresh-input evaluation harness, group accounting, exhaustive comparisons, structural matrix slicing, validity gates, scalar audits, timing and evidence presentation. No new pruning-algorithm or theorem claim is made. OpenAI Codex assisted implementation, local execution, checks, analysis and writing. A separate CPU calculation path is not third-party GPU replication. Repository code uses [Apache-2.0](LICENSE); upstream model terms remain applicable.
