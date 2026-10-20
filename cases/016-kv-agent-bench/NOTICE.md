# Sources and scope

This case uses the existing Qwen/Qwen3-4B-Instruct-2507 snapshot (revision `cdbee75f17c01a7cc42f958dc650907174af0554`) read-only. No model weights are included in this case or its review archive. The model's upstream license continues to apply.

vLLM0.32.0 is used as an installed binary distribution. Official tag commit: `5df63a7203cdb9328d824a27b682e77c7b9928be` (Apache-2.0). The small `kernel_check.py` follows the structure of that release's `test_flashinfer_nvfp4_kv_cache_update`; the upstream test is Copyright contributors to the vLLM project, Apache-2.0. See `provenance/source_audit.json` and the accompanying upstream license. FlashInfer, PyTorch, Transformers, Tokenizers and their dependencies retain their own licenses.

The synthetic business records and tools are local research fixtures, not customer records or a connection to a business service. ToolBottleneckBench, LURCH and FEB-30 are motivations supplied in the user's design; their complete benchmarks or training methods are not reproduced here.

Implementation and reporting used Codex assistance. The independent scalar audit is a separate calculation path, not human peer review or an independent reproduction of GPU inference.
