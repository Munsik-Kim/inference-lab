# Attribution and scope

Qwen provides the two official model checkpoints and their model cards/licenses, preserved under `provenance/bf16/` and `provenance/fp8/`. The vLLM, PyTorch, Triton, FlashAttention and related projects provide the inference software. Their original authors receive credit for those models and implementations.

This case contributes a local RTX 5080 comparison, deterministic synthetic Korean document/gold data, a small streaming measurement runner, strict scoring, and inspectable execution evidence. It does not introduce a quantization method or GPU kernel.

OpenAI Codex assisted with synthetic data generation, code, local execution, evidence checks, analysis and documentation. Gold values were calculated from structured facts and confirmed events, not from model predictions or an LLM judge. Codex checks are not described as human review. No independent reproduction, external adoption, or general performance/quality guarantee is claimed.

Project code and documentation use the included Apache-2.0 license. Model weights are not included. Refer to the upstream model and software licenses for their respective artifacts.
