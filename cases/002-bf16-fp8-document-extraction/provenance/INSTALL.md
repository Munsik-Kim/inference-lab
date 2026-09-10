# Tested environment and model provenance

The actual run used a new Conda prefix cloned from the preserved baseline, with Python 3.12.14, official vLLM 0.29.0, PyTorch 2.13.0+cu130 and Triton 3.7.1. The clone initially restored Conda setuptools 84.0.0; only the new prefix was returned to baseline setuptools 80.10.2. All 198 package versions then matched the baseline. vLLM source hashes match; no CASE 001 INT8 guard patch was applied. See environment.json and environment_integrity.json for observed values and wheel origin.

Actual setup, using local path variables:

```bash
conda create --prefix "$CASE_ENV" --clone "$BASELINE_ENV" --offline --yes
"$CASE_ENV/bin/python" -m pip install --no-deps setuptools==80.10.2
```

For a separate reproduction machine, prepare a dedicated Python 3.12 environment and install `pip-freeze.txt`. Its official wheel URLs and hashes record the tested package artifacts. A new installation on another machine was not tested; this file is an environment record, not a guarantee of hardware or OS compatibility. Driver 610.47 was observed; `nvidia-smi` CUDA support is not an installed toolkit version. PyTorch's CUDA build was 13.0. No system driver/toolkit or existing environment was changed.

```bash
python3.12 -m venv "$CASE_ENV"
"$CASE_ENV/bin/python" -m pip install -r provenance/pip-freeze.txt
```

`run.py` uses the current interpreter prefix, prepends its bin directory to the child server PATH, selects the bundled NVIDIA cu13 compiler location, and isolates compilation caches. Only the recorded common VLLM overrides are inherited. FlashInfer sampling is disabled after the recorded development JIT link failure; FlashAttention 2 remains enabled. Normal engine multiprocessing and CUDA graphs are retained.

Both checkpoints are pinned in models.json. Configurations differ only by the FP8 quantization configuration. The fast tokenizer JSON, vocabulary, tokenizer config and chat template are identical; FP8 merges.txt adds only a version header. All 120 final requests were checked for identical token IDs. Generation configs are semantically equal, but both runs override their sampling defaults through `--generation-config vllm` and the common request settings.

The FP8 config uses dynamic activation quantization, e4m3, and 128x128 weight blocks, with lm_head and normalization exclusions. Actual safetensors headers show 252 F8_E4M3 tensors and 399 F16 tensors in the FP8 distribution, versus 398 BF16 tensors in the BF16 distribution. Thus this is a comparison of the two official deployment artifacts, not a controlled re-quantization of verified identical original tensors. The engine dtype and unquantized KV cache dtype are both bfloat16. Exact upstream tensor conversion history was not independently established. No new quantization or training was performed.

local_model_manifest.json records actual file sizes, SHA256 values checked against published LFS hashes, and storage dtype counts. File bytes, engine model-loading memory, and whole-device NVML occupancy are different measurements.

Model cards and licenses in bf16/ and fp8/ are copied from the pinned official repositories. Their benchmark claims are not results of this case. Official references: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 and https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8 . CLI/cache semantics were checked against installed vLLM 0.29.0 source and local help, with https://docs.vllm.ai/en/latest/configuration/engine_args/ and https://docs.vllm.ai/en/latest/cli/bench/serve/ used only as references. Explicit kv_cache_memory_bytes overrides automatic cache sizing from gpu_memory_utilization.
