# Sources, attribution and current publication scope

Project code follows the unchanged repository/Case014 Apache-2.0 LICENSE. No third-party source was copied or relicensed.
This is an independent ridge/RTN reference study, not the author's complete QEP reproduction or a new quantizer.

- Tianhao Guo, *When Does Compensating Inherited Quantization Error Help?*, arXiv2610.16020v1:
  [paper](https://arxiv.org/html/2610.16020v1), [historical source audit](references/source_audit.json).
- [Author code identity](https://github.com/tellsgo/quantization-compensation/tree/1e5dea93dfd9466ce0569fc8dde55ef8c2fcb9e0): identified historically, not vendored/executed as official full reproduction.
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), Transformers/PyTorch native model execution and NumPy arithmetic are upstream.
- Project contributions: actual-path capture/fit orchestration, signed INT4 codec/independent decoder,
  decomposition/measurement, independent scalar auditors and traceable bilingual publication.

External model/book/article rights remain applicable. Model/INT4 weights/K/activation/logits/source text are excluded.
Historical raw reports/decisions/failed attempts remain preserved; their old nonpublication scope is historical.
Current publication approval covers PR, normal main merge and the existing manual Pages workflow. No new GPU inference
is performed in publication. Codex assisted this work; no independent human review, external GPU rerun or official author approval is claimed.
