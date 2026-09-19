# Case 008 — Build, Reconstruct, Reload

[English](README.md) | [한국어](README.ko.md)

Built two Qwen checkpoint workflows on RTX 5080: **Q** converts a 4B model to GPTQ W4A16, checks packed storage and runs it in fresh vLLM processes; **R** repairs the output weights of a fixed smaller MLP in a 0.6B model and restores it with a per-layer structure loader. The two tracks use different models and runtimes.

**Start with the [structured report](REPORT.md):** background → hypotheses → theory → methods → experiments → results → analysis → conclusions → references. It defines the settings before their scores and explains what the data can show.

| Track | Measured result | Interpretation |
|---|---|---|
| Q: weight representation and reload | Safetensors: 8,044,982,000 → 2,651,839,568 bytes; both arms correct on 137/192 | Smaller files and recorded resident weights; one regression and one gain. Full and conditional-choice NLL change in opposite directions. |
| R: fixed smaller MLP and repair | 94.1–95.4% of deletion-induced squared output error recovered across three structures | Same shapes before/after repair, on 192 short synthetic prompts; gold-score changes are mixed. |

Request-speed intervals did not establish a speedup. R's recovery is squared local error reduction, not accuracy. Both tracks' code task has constant option predictions; [the analysis](REPORT.md#analysis) shows label mass, task results and generator cues. Execution is `COMPLETED`; deployment is `NOT_ASSESSED`.

## Inspect and reproduce

- [Full report](REPORT.md) · [Original methods](METHODS.md) · [Interpretation](ANALYSIS.md).
- [Code and CPU/GPU reproduction boundaries](REPRODUCTION.md) · [Modelpack source](../../tools/modelpack/) · [Portfolio/code tour](PORTFOLIO.md).
- [Recorded-results explorer](demo/index.html): download and open locally; English UI, no model execution. GitHub's HTML source view is not a running demo.
- [Code, recipes and measured evidence ZIP](../../downloads/case008_build_reconstruct_reload_reviewed_publication_v2.zip) · [ZIP metadata/checksum](../../downloads/case008_build_reconstruct_reload_reviewed_publication_v2.json). Large model weights and private activation/logit vectors are excluded.
- [Q raw scores](results/raw/Q/model_records.json) · [R raw scores](results/raw/R/model_records.json) · [Original summary](results/derived/summary.json) · [Post-hoc same-record diagnostics](publication/posthoc/metrics.json).

The [publication verifier and historical restore path](publication/README.md) preserve the original experiment separately from edited explanations. Source and input freezes, scientific code, original figures, HTML and decisions retain their original bytes. Existing `RUN_STATE.json` describes the historical review snapshot; [publication status](publication/status.json) concerns this package.

## Attribution

Qwen/Transformers provide the models, GPTQ/LLM Compressor provide quantization, and compressed-tensors/vLLM/Marlin provide storage and execution. The project adds validation, per-layer restoration, fixed-set ridge reconstruction and paired evidence tools. Codex assisted the work. [NOTICE](NOTICE.md) · [License](LICENSE).
