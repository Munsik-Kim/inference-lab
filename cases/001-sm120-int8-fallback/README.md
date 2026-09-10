# RTX 5080 / SM120 INT8 fallback reproduction

In vLLM 0.29.0, the INT8 kernel selector accepts CUTLASS on the RTX 5080 (SM120), even though the CUTLASS W8A8 dispatcher does not support INT8 on that architecture. The engine then fails during initialization with `Int8 not supported on SM120`.

[PR #54316 by hclsys](https://github.com/vllm-project/vllm/pull/54316) makes CUTLASS decline these GPUs so the selector can use the existing Triton implementation. This bundle records the failure before that change and successful text generation afterward on an RTX 5080. See [issue #54311](https://github.com/vllm-project/vllm/issues/54311).

**Tested change:** the exact Python runtime hunk from PR head `4878fe1154e1bccf70a18d170f5fa093f1190734`, applied to the official **vLLM 0.29.0 wheel** (release commit `98dff2a81d747d1dba01a47f939f48c3526d4206`). The complete PR checkout was not tested. The fix and existing branch tests are the PR author's work; this bundle supplies GPU validation and a reproducer.

## Tested environment

WSL2 Ubuntu 24.04.4, Linux x86_64; RTX 5080 16303 MiB, SM120; driver 610.47; Python 3.12.14; torch 2.13.0+cu130 (CUDA build 13.0); Triton 3.7.1; vLLM 0.29.0; compressed-tensors 0.17.0; Transformers 5.17.0.

[environment.json](environment.json) records all 198 baseline package versions, the official wheel URL/hash, patch provenance and installation history. The historical candidate also contained pytest 9.0.2, pluggy 1.6.0 and iniconfig 2.3.0; inference package versions were identical. Those test-only extras are not required here. Conda's pip/packaging build-time paths were non-editable metadata, not dependencies on the original machine.

## Environment reconstruction

Use the two already tested environments if available. Otherwise the following reconstructs new environments from the recorded Conda packages and complete 195-wheel URL/SHA256 lock. The original installation used the same wheel lock after dependency resolution, followed by `pip check`. The recorded replay used the existing environments on the same machine. A fresh installation on another host has not been tested; it also depends on a compatible driver, OS ABI and package availability.

Run from this extracted directory, with Conda and Git available:

```bash
BUNDLE="$PWD"
BEFORE_ENV="$PWD/env-before"
AFTER_ENV="$PWD/env-after"

CONDA_REGISTER_ENVS=false conda create --prefix "$BEFORE_ENV" --copy --file conda-explicit.txt --yes
CONDA_REGISTER_ENVS=false conda create --prefix "$AFTER_ENV" --copy --file conda-explicit.txt --yes

"$BEFORE_ENV/bin/python" -I -m pip --isolated install --only-binary=:all: --require-hashes --no-deps -r requirements-baseline.lock
"$AFTER_ENV/bin/python" -I -m pip --isolated install --only-binary=:all: --require-hashes --no-deps -r requirements-baseline.lock
"$BEFORE_ENV/bin/python" -I -m pip check
"$AFTER_ENV/bin/python" -I -m pip check
```

The lock pins setuptools 80.10.2 even though the Conda record contains 84.0.0. `--no-deps` uses the recorded complete resolver output. Both environments must pass `pip check` before proceeding.

Apply the supplied patch **only to the after environment**:

```bash
AFTER_SITE="$("$AFTER_ENV/bin/python" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
git -C "$AFTER_SITE" apply --check "$BUNDLE/pr54316-vllm-0.29.0.patch"
git -C "$AFTER_SITE" apply "$BUNDLE/pr54316-vllm-0.29.0.patch"
```

The patch is byte-identical to the runtime diff tested locally. `smoke.py` checks the complete installed `cutlass.py` SHA256 for the selected phase before running. Applying it to another vLLM version is outside this recorded comparison. The vLLM package version remains 0.29.0 after this local file modification.

## Model preparation

Use [RedHatAI/Qwen2.5-0.5B-Instruct-quantized.w8a8](https://huggingface.co/RedHatAI/Qwen2.5-0.5B-Instruct-quantized.w8a8/tree/2379f83cfb42ec0905ac46b68b234d36013c843d), revision **`2379f83cfb42ec0905ac46b68b234d36013c843d`** (Apache-2.0). This is a public prequantized checkpoint: symmetric INT8 weights per output channel and dynamic symmetric INT8 activations per token, in compressed-tensors format. No local calibration or quantization was performed. Inference seed is 0; provider calibration seeds/tools were not independently established.

```bash
"$AFTER_ENV/bin/python" -I -B prepare_model.py --cache-dir "$PWD/model-cache"
MODEL_DIR="$PWD/model-cache/models--RedHatAI--Qwen2.5-0.5B-Instruct-quantized.w8a8/snapshots/2379f83cfb42ec0905ac46b68b234d36013c843d"
```

Or supply an existing copy without downloading:

```bash
MODEL_DIR="/path/to/the/existing/snapshot"
"$AFTER_ENV/bin/python" -I -B prepare_model.py --model-dir "$MODEL_DIR"
```

The downloader requests only the 11 pinned files in [model-files.json](model-files.json), about 919 MB in total, and verifies their recorded SHA256 values. These hashes were previously checked against Hub metadata. Each smoke child also verifies them before loading. The weight file is 903,168,128 bytes, SHA256 `5e8edffe394b0fef5422540357eedaa30e4b072060a728ce2a344402da8d24e2`. [model-quantization.json](model-quantization.json) records the actual schema. **No weights are redistributed in this bundle.** The replay used the existing verified checkpoint.

## Run the pair

If reusing existing environments, set `BEFORE_ENV`, `AFTER_ENV` and `MODEL_DIR` to their locations first. `python3` below only launches the selected interpreter and requires no ML packages itself. The bundle can be elsewhere than the working directory.

```bash
python3 "$BUNDLE/run.py" before --python "$BEFORE_ENV/bin/python" --model-dir "$MODEL_DIR" --work-dir "$PWD/repro-work"
python3 "$BUNDLE/run.py" after --python "$AFTER_ENV/bin/python" --model-dir "$MODEL_DIR" --work-dir "$PWD/repro-work"
```

Run the commands sequentially, **not joined with `&&`**: the before command is expected to return **1**, not 0. Every command launches a fresh subprocess, creates a unique run directory and preserves its raw child exit code in `exit_code.txt` and `status.json`. Only the actual recorded SM120 CUTLASS runtime exception during engine initialization qualifies as `EXPECTED_FAILURE`. Missing files/packages, OOM, wrong source hashes, arbitrary exceptions and timeouts do not qualify. After must report `PASS`, actually generate nonempty text, observe the expected backend information, and exit 0. Matching a particular sentence, including “Paris”, is not a pass condition.

No public serving port is opened. Inference uses the local snapshot, offline Hub mode and `trust_remote_code=False`. The wrapper keeps caches in `--work-dir`, uses a private short temporary directory for IPC, and clears explicit kernel overrides. It stops before launch if available VRAM is below 5500 MiB or utilization is at least 80%; it never stops another GPU workload. A 600-second timeout terminates only its own child process group. The status file distinguishes `GPU_BUSY` and access failures from model results.

## Conditions and observations

Both phases use [config.json](config.json): TP=1, one request, BF16 compute, explicit compressed-tensors quantization, length/batched tokens 1024, max_num_seqs=1, generation limit 32, memory utilization 0.2, eager mode, seed/temperature 0, no CPU offload or prefix cache. Both use `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`, and `VLLM_ALLOW_INSECURE_SERIALIZATION=0`. CUDA_HOME is derived from the chosen environment's NVIDIA cu13 wheel. The nvidia-smi CUDA display is not used as evidence of a system toolkit version.

| Evidence | Before | After |
|---|---|---|
| Original run | `20260910T034745.187033Z_step03_model_before`, exit 1 | `20260910T034826.798225Z_step03_model_after`, exit 0 |
| Reproducer replay | `20260910T043844.464249Z_before`, EXPECTED_FAILURE, exit 1 | `20260910T043959.054768Z_after`, PASS, exit 0 |
| Behavior | Weights load; `profile_run → _dummy_run → qkv_proj` raises SM120 error | Initializes and generates `The capital of France is Paris.`; 8 tokens including EOS |

The replay was performed once per phase from a candidate ZIP extracted into an unrelated temporary directory, using the existing tested interpreters and checkpoint. Executable files in the final bundle match those rerun files. Original and replay records are separate in [evidence/original.json](evidence/original.json) and [evidence/replay.json](evidence/replay.json); corresponding before/after logs retain the traceback and generated output with private paths replaced by labels.

`LLM.apply_model()` read the loaded module state: 96 `CompressedTensorsW8A8Int8` modules used `TritonInt8ScaledMMLinearKernel`, with int8 weights on CUDA. Counts are 24 qkv_proj, 24 o_proj, 24 gate_up_proj and 24 down_proj; 48 are fused modules. **This is module-class/device inspection accompanying real generation, not GPU kernel profiling.** No Nsight/CUPTI kernel-count or timing claim is made. Full per-module records are in the evidence JSON. A Triton JIT message and NCCL process-exit warning remain in the raw logs.

## Scope

This is one small model in single-process/eager mode on one SM120 GPU. It is not the reporter's exact 3B checkpoint, a full PR-head build, a default multiprocessing generation test, CUDA graph or V2-runner validation, a quality evaluation, a long-running stability test, or an SM100/SM121 result. Before cannot generate, so no before/after speed ratio is defined. Cold JIT timings are not a benchmark.

[evidence/upstream_status.json](evidence/upstream_status.json) separates the latest observed PR head from the tested head; at the recorded check both were `4878fe1154e1bccf70a18d170f5fa093f1190734`, open and unmerged. This file records the PR status at the time of validation.

AI assistance: OpenAI Codex helped adapt the scripts, execute the local checks and draft the documentation. See [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE) for attribution and license information.
