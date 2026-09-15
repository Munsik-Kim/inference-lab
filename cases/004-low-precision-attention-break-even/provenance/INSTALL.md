# Tested environment and installation

The recorded system is WSL2 Ubuntu 24.04.4, RTX 5080 SM120, driver 610.47. GPU execution required access outside the agent's initial restricted sandbox. A restricted `nvidia-smi` failure was an access limitation, not a driver incompatibility result.

Python 3.12.14, PyTorch 2.13.0+cu130, Triton 3.7.1 and Transformers 5.17.0 were inherited read-only from an existing project prefix. A new case-specific venv installed only SageAttention 2.2.0. [dependencies.json](dependencies.json) records versions and inherited packages; it is an inventory, not a claim that `pip freeze` alone guarantees fresh installation. vLLM and other inherited packages were not used as attention backends. Analysis/plots used an existing read-only CPU environment with NumPy 2.3.5 and Matplotlib 3.10.8.

Set these user-local paths explicitly. Do not install into the existing prefix:

```bash
BASE_PYTHON=/path/to/existing/compatible/prefix/bin/python
CASE004_WORK=/path/to/private/case004-work
mkdir -p "$CASE004_WORK"
"$BASE_PYTHON" -m venv --system-site-packages "$CASE004_WORK/env"
CASE004_PYTHON="$CASE004_WORK/env/bin/python"
git clone https://github.com/thu-ml/SageAttention.git "$CASE004_WORK/SageAttention"
git -C "$CASE004_WORK/SageAttention" checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
```

The tested base is a standalone project Python prefix, so its site-packages are visible to the new venv. Check `sys.base_prefix`, actual import locations and versions on your installation; a nested venv need not inherit another venv's packages in the same way. The execution fingerprint must match for direct confirmation replay. A different environment needs its own development run and locally frozen protocol, without replacing the published records.

The build used an existing NVIDIA CUDA toolkit wheel: NVCC 13.0.88, matching Torch's CUDA 13.0 build. System NVCC 12.8.93 was inspected but not used. Set `CASE004_CUDA_HOME` to the directory containing that toolkit's `bin/nvcc`, `include/` and `lib/`. `nvidia-smi`'s CUDA support display is not the installed toolkit version.

```bash
CASE004_CUDA_HOME=/path/to/existing/site-packages/nvidia/cu13
mkdir -p "$CASE004_WORK/wheels" "$CASE004_WORK/pip-cache"
env CUDA_HOME="$CASE004_CUDA_HOME" TORCH_CUDA_ARCH_LIST=12.0 \
  MAX_JOBS=4 EXT_PARALLEL=1 NVCC_APPEND_FLAGS=--threads=1 \
  PIP_CACHE_DIR="$CASE004_WORK/pip-cache" \
  "$CASE004_PYTHON" -m pip wheel --no-build-isolation --no-deps \
  "$CASE004_WORK/SageAttention" --wheel-dir "$CASE004_WORK/wheels"
```

Attempt 1 compiled the CUDA sources but failed at final linking: `cannot find -lcudart`. The existing toolkit wheel supplied `libcudart.so.13`, not an unversioned linker symlink. Attempt 2 added only a private linker directory:

```bash
mkdir -p "$CASE004_WORK/cuda-link"
ln -s "$CASE004_CUDA_HOME/lib/libcudart.so.13" "$CASE004_WORK/cuda-link/libcudart.so"
env CUDA_HOME="$CASE004_CUDA_HOME" TORCH_CUDA_ARCH_LIST=12.0 \
  MAX_JOBS=4 EXT_PARALLEL=1 NVCC_APPEND_FLAGS=--threads=1 \
  LIBRARY_PATH="$CASE004_WORK/cuda-link" PIP_CACHE_DIR="$CASE004_WORK/pip-cache" \
  "$CASE004_PYTHON" -m pip wheel --no-build-isolation --no-deps \
  "$CASE004_WORK/SageAttention" --wheel-dir "$CASE004_WORK/wheels"
"$CASE004_PYTHON" -m pip install --no-deps --no-index \
  "$CASE004_WORK/wheels/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
```

There were two build attempts, with maximum four build jobs; no third retry, source patch, global linker change, driver replacement or existing-environment installation. The first and second builds took approximately 514 and 466 seconds. These are installation costs, not attention latency. The exact built-wheel size/hash and outcomes are in [installation.json](installation.json). Compiled binary hashes and runtime profiler evidence are also recorded. Rebuilding may produce a different wheel hash; do not silently call a new binary the tested artifact.

Official API controls were checked in the installed Torch source/docstrings as well as documentation; see [installed_api.json](installed_api.json). GPU probes are separate from CPU tests. The full private build and disassembly dumps, existing environments, wheel and model weights are not in the publication bundle.
