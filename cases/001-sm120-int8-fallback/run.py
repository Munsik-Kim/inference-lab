"""Launch exactly one fresh offline child; preserve its original exit code."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

BUNDLE = Path(__file__).resolve().parent


def classify(phase, rc, runtime, error, result):
    # Do not classify dependency, input, OOM, timeout or arbitrary exceptions
    # as reproduction success, even if the exception mentions an error string.
    if phase == "before" and rc == 1 and runtime and error:
        exact_failure = (
            error.get("stage") == "engine_initialization"
            and error.get("exception_type") == "RuntimeError"
            and "dispatch_scaled_mm," in error.get("exception", "")
            and "Int8 not supported on SM120." in error.get("exception", "")
            and "cutlass_scaled_mm" in error.get("traceback", "")
            and runtime.get("compute_capability") == [12, 0]
            and runtime.get("cutlass_support_sm120") == [True, None]
            and runtime.get("source_and_environment_verified") is True
        )
        if exact_failure:
            return "EXPECTED_FAILURE"
    if phase == "after" and rc == 0 and runtime and result:
        if (result.get("status") == "PASS_W8A8_MODEL_SMOKE"
                and result.get("generated_text", "").strip()
                and result.get("generated_tokens", 0) > 0
                and result.get("kernel_counts") == {"TritonInt8ScaledMMLinearKernel": 96}
                and runtime.get("source_and_environment_verified") is True):
            return "PASS"
    return "UNEXPECTED_FAILURE"


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["before", "after"])
    parser.add_argument("--python", type=Path, required=True, help="Interpreter from the chosen isolated environment")
    parser.add_argument("--model-dir", type=Path, required=True, help="Pinned local snapshot; verified before loading")
    parser.add_argument("--work-dir", type=Path, required=True, help="New logs and compilation caches go here")
    args = parser.parse_args()
    # Do not resolve an interpreter symlink: that would bypass a standard venv.
    python = args.python.absolute()
    model = args.model_dir.resolve()
    work = args.work_dir.resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = work / "runs" / (stamp + "_" + args.phase)
    run.mkdir(parents=True)
    print("RUN_DIR=" + str(run), flush=True)
    for name in ["run.py", "smoke.py", "prepare_model.py", "config.json", "model-files.json"]:
        shutil.copy2(BUNDLE / name, run / name)
    env = os.environ.copy()
    removed = ["PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "VLLM_DISABLED_KERNELS",
               "VLLM_LINEAR_BACKEND", "TRITON_INTERPRET", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
    for key in removed:
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    info = subprocess.run([str(python), "-I", "-B", "-c",
        "import json,sys,sysconfig; print(json.dumps({'prefix':sys.prefix,'site':sysconfig.get_path('purelib')}))"],
        env=env, capture_output=True, text=True)
    if info.returncode:
        (run / "preflight.log").write_text(info.stdout + info.stderr)
        (run / "status.json").write_text(json.dumps({"status": "ENVIRONMENT_ERROR", "preflight_exit_code": info.returncode}))
        return info.returncode
    paths = json.loads(info.stdout)
    cache = work / "cache"
    # A private, short temporary directory avoids the 107-byte IPC path limit.
    # Only files owned by this invocation are placed here.
    with tempfile.TemporaryDirectory(prefix="sm120-", dir="/tmp") as temp:
        overrides = {
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0", "CUDA_HOME": str(Path(paths["site"]) / "nvidia/cu13"),
            "VLLM_HOST_IP": "127.0.0.1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HOME": str(cache / "huggingface"),
            "HF_HUB_CACHE": str(cache / "huggingface/hub"), "XDG_CACHE_HOME": str(cache),
            "VLLM_CACHE_ROOT": str(cache / "vllm"), "VLLM_CONFIG_ROOT": str(cache / "vllm-config"),
            "TORCH_HOME": str(cache / "torch"), "TRITON_CACHE_DIR": str(cache / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"), "CUDA_CACHE_PATH": str(cache / "cuda"),
            "NUMBA_CACHE_DIR": str(cache / "numba"), "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
            "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_ALLOW_INSECURE_SERIALIZATION": "0",
            "OMP_NUM_THREADS": "2", "TMPDIR": temp,
        }
        env.update(overrides)
        cmd = [str(python), "-I", "-B", "-u", str(BUNDLE / "smoke.py"), "--config", str(BUNDLE / "config.json"),
               "--phase", args.phase, "--model-dir", str(model), "--expected-prefix", paths["prefix"], "--run-dir", str(run)]
        (run / "command.txt").write_text("env " + " ".join("-u " + k for k in removed) + " " +
            " ".join(k + "=" + shlex.quote(v) for k, v in overrides.items()) + " " + shlex.join(cmd) + "\n")
        (run / "environment_overrides.json").write_text(json.dumps(overrides, indent=2) + "\n")
        smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else None)
        if not smi:
            (run / "status.json").write_text(json.dumps({"status": "BLOCKED_ACCESS", "reason": "nvidia-smi unavailable"}))
            return 77
        query = [smi, "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
        gpu = subprocess.run(query, env=env, capture_output=True, text=True)
        (run / "gpu_before.log").write_text(gpu.stdout + gpu.stderr)
        if gpu.returncode:
            (run / "status.json").write_text(json.dumps({"status": "BLOCKED_ACCESS", "reason": "nvidia-smi failed"}))
            return 77
        first = gpu.stdout.strip().splitlines()[0].split(",")
        if float(first[-2]) < 5500 or float(first[-1]) >= 80:
            (run / "status.json").write_text(json.dumps({"status": "GPU_BUSY", "free_mib": float(first[-2]), "utilization_percent": float(first[-1])}))
            return 75
        started = time.monotonic()
        timed_out = False
        with (run / "stdout.log").open("w") as out, (run / "stderr.log").open("w") as err:
            child = subprocess.Popen(cmd, env=env, cwd=work, stdout=out, stderr=err, start_new_session=True)
            try:
                rc = child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    rc = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    rc = child.wait()
        (run / "exit_code.txt").write_text(str(rc) + "\n")
        status = "TIMEOUT" if timed_out else classify(args.phase, rc, read_json(run / "runtime.json"), read_json(run / "exception.json"), read_json(run / "result.json"))
        record = {"status": status, "phase": args.phase, "run_id": run.name,
                  "child_exit_code": rc, "timed_out": timed_out, "elapsed_seconds": time.monotonic() - started,
                  "script_sha256": hashlib.sha256((BUNDLE / "smoke.py").read_bytes()).hexdigest()}
        (run / "status.json").write_text(json.dumps(record, indent=2) + "\n")
        gpu = subprocess.run(query, env=env, capture_output=True, text=True)
        (run / "gpu_after.log").write_text(gpu.stdout + gpu.stderr)
        print(json.dumps(record), flush=True)
        # Preserve the subprocess return code; EXPECTED_FAILURE remains nonzero.
        return rc


if __name__ == "__main__":
    sys.exit(main())
