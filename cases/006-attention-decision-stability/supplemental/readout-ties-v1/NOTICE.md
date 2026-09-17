# Attribution and reuse

This is an additive, post-hoc supplement to Case006. It preserves all original files and decisions. Original `src/model_runtime.py`, `intervention.py`, `reference.py`, `metrics.py`, `validity.py` and `scripts/measure_scores.py` are reused read-only; their hashes are frozen in `protocol.json`. Their model-norm hook and native full-output checks control the replay boundary. No Case005 frozen code or STOP_DEV_SCREEN decision was changed.

Qwen model authors supplied Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`: https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca

SageAttention authors supplied the existing attention implementations at `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`: https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/core.py

This supplement adds the tie/readout audit, CPU scalar cross-checks and offline evidence presentation, with OpenAI Codex assistance in implementation, local execution, analysis and writing. No new quantizer, upstream patch, human review or independent third-party GPU replication is claimed. Apache-2.0 text is retained from the original case. No upstream source is vendored unnecessarily.

Current PyTorch docs are general numerical references, not the installed version used here: https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html and https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.linear.html . Installed Torch2.13 source/API was inspected; it was not upgraded.
