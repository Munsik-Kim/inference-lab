# Sources and reuse

This follow-up adds a tool-name enum decoder, registry-aware feedback, a bounded runner and CPU event replay. It reuses DIOVA's existing independent INT4 reference and synthetic local-tool harness as external dependencies. DualQuant is the upstream work of Mahsa Salmani, Sayeh Sharify and Hesham Mostafa; Qwen supplies the model/tokenizer; PyTorch and Transformers supply inference. The vendor tree, private block-format library and model/INT4 weights are not redistributed.

See the preserved [Case 013 notice](https://github.com/Munsik-Kim/inference-lab/blob/27294fc5cfde575cbed638c17b9a547e126a2f01/cases/013-dualquant-reference-quality/NOTICE.md) for original attribution and external license boundaries. The repository's existing Apache-2.0 LICENSE is included unchanged; third-party rights are not relicensed. Synthetic task fixtures are experiment-owned. Codex assisted implementation and reporting.
