# Related work and implementation boundaries

[한국어](README.ko.md) · [Home](../../README.md)

These sources explain the mechanisms reused in DIOVA. Existing case reports stay as historical snapshots. New measurements belong to Case 009.

| Source | Shared idea and difference | Project implementation |
|---|---|---|
| He, Zhang and Sun, *Channel Pruning for Accelerating Very Deep Neural Networks*, ICCV 2017, arXiv:1707.06168v2 | CNN channel selection using LASSO, followed by least-squares reconstruction. R fixes an already selected dense LLM MLP structure and fits a regularized down projection; it is not a reproduction of the CNN selector. | [Fixed-structure ridge solver](../../tools/modelpack/numerics.py), [shape-aware checkpoint loader](../../tools/modelpack/artifact.py). |
| An, Zhao, Yu, Tang and Wang, *Fluctuation-based Adaptive Structured Pruning for Large Language Models*, AAAI 2024, arXiv:2312.11983v1 | FLAP combines fluctuation importance, global structure allocation and output-bias compensation. R uses a ridge fit in existing smaller weights and adds no inference bias. A future bias-only control would be FLAP-inspired, not a FLAP reproduction. | [Future multi-layer design](../phase2a-design.md); not executed in this study. |
| Frantar, Castro, Chen, Hoefler and Alistarh, *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models*, arXiv:2408.11743v1 (2024), PPoPP 2025 | Quantized linear-kernel design for batched autoregressive execution. Paper speedups are conditional on its devices, shapes, dtypes and baselines. Client concurrency here is not the Linear token-row dimension. | Existing GPTQ artifacts and vLLM's integrated Marlin route; [new request runner](../../cases/009-q-serving-quality/scripts/serving.py) measures the complete serving workload. |
| Dutta, Krishnan, Kwatra and Ramjee, *Accuracy is Not All You Need*, NeurIPS 2024, arXiv:2407.09141v1 | Compression can preserve accuracy while changing correctness on individual questions. Their flip definition excludes wrong-to-wrong transitions. | [Paired comparison core](../../packages/diova-compare/src/diova_compare/core.py) keeps correctness flips and all-answer disagreement as different fields. |

For question i, let bᵢ and cᵢ be correctness indicators and let aᵢᴮ/aᵢᶜ be the extracted answers. Dutta-style correctness-flip fraction is Σ1[bᵢ≠cᵢ]/N = (correct_to_wrong + wrong_to_correct)/N. Our all-answer disagreement is Σ1[aᵢᴮ≠aᵢᶜ]/N. With one accepted answer these differ by wrong-to-wrong changes; with multiple accepted answers, two correct answers can also differ. Choice-score KL, normalized over task alternatives, is not full-vocabulary next-token KL. Dutta's paper averages full-vocabulary KL across tokens of answer options; these scopes are not interchangeable.

R's represented BF16 corrections, artifact checks, strict meta-device reload, runner integration and evidence UI are project implementation work. GPTQ, Qwen architectures, Transformers, safetensors, vLLM and their CUDA kernels are upstream components. Codex assisted implementation, execution, testing and writing; original notices and licenses remain linked from each case.

## Primary sources

- He et al.: [CVF proceedings](https://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html), [author paper on arXiv](https://arxiv.org/abs/1707.06168). CVF returned HTTP 403 during this audit; the arXiv record was available.
- FLAP: [paper](https://arxiv.org/abs/2312.11983), [official code](https://github.com/CASIA-LMC-Lab/FLAP).
- MARLIN: [paper](https://arxiv.org/abs/2408.11743), [PPoPP 2025](https://ppopp25.sigplan.org/details/PPoPP-2025-Main-Conference-1/22/MARLIN-Mixed-Precision-Auto-Regressive-Parallel-Inference-on-Large-Language-Models), [official code](https://github.com/IST-DASLab/marlin).
- Dutta et al.: [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e0e956681b04ac126679e8c7dd706b2e-Abstract-Conference.html), [§2 definitions](https://arxiv.org/html/2407.09141v1#S2).
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520): official task construction, multi-token likelihood, answer extraction and aggregation. The new study pins this source revision.
- vLLM: [serving benchmark](https://docs.vllm.ai/en/stable/cli/bench/serve/), [CUDA Graph design](https://docs.vllm.ai/en/stable/design/cuda_graphs/). Execution flags and timing definitions are checked against installed **0.29.0**, not assumed from moving `stable` pages.

Source records checked 2026-09-22. No kernel speedup from these papers is used as an acceptance threshold for RTX 5080 results.
