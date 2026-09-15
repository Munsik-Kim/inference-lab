# Attribution

SageAttention and its INT8/FP8/FP16 kernels are the work of the thu-ml/SageAttention authors. This case uses official revision d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 and preserves the upstream [Apache-2.0 license](provenance/SageAttention_LICENSE). It does not introduce an attention or quantization algorithm.

The project contributes a bounded source/runtime audit, precision-setting adapters, paired complete-call measurements, local error analysis and reproducible evidence. The [reuse manifest](provenance/reuse.json) records code copied or adapted from Case004; existing Case004 results and interpretations remain unchanged.

OpenAI Codex assisted with source inspection, code, local GPU execution, analysis, tests and documentation. Inputs are self-authored synthetic documents. No independent human review, third-party GPU reproduction, upstream endorsement or downstream task-quality validation is claimed.

The publication-only post-hoc review compares official papers and test definitions; it does not incorporate their kernels or claim their experimental results as this project's measurements. Source revisions and provenance are in [posthoc_sources.json](provenance/posthoc_sources.json). Codex also assisted with this source review, paired scalar recalculation and publication documentation. The original protocol and STOP_DEV_SCREEN decision were preserved.

This standalone case package retains the repository’s [Apache-2.0 license](LICENSE). The separate upstream license above preserves SageAttention attribution.
