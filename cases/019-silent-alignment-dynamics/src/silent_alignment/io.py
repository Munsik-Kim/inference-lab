"""Small, explicit provenance and output contracts (no model/SDK imports)."""

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def fresh_output(path):
    path = Path(path).absolute()
    path.mkdir(parents=True, exist_ok=False)
    return path


def entry(path, root):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": digest(path.read_bytes())}


def source_paths(root):
    root = Path(root)
    return sorted(set([*root.glob("src/**/*.py"), *root.glob("tests/*.py"),
                       *root.glob("configs/*.json"), *root.glob("specs/*.json")]))


def bundle(root):
    rows = [entry(p, Path(root)) for p in source_paths(root)]
    return {"files": rows, "bundle_sha256": digest(canonical(rows))}


def assert_bundle(root, frozen):
    actual = bundle(root)
    if actual != {k: frozen[k] for k in ("files", "bundle_sha256")}:
        raise ValueError("SOURCE_OR_CONFIG_CHANGED")
    return actual


def seal_files(root, exclude=()):
    root = Path(root)
    return [entry(p, root) for p in sorted(root.rglob("*"))
            if p.is_file() and str(p.relative_to(root)) not in exclude]


def check_files(root, rows):
    root = Path(root)
    for row in rows:
        p = Path(row["path"])
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("UNSAFE_MANIFEST_PATH")
        f = root / p
        if not f.is_file() or f.stat().st_size != row["bytes"] or digest(f.read_bytes()) != row["sha256"]:
            raise ValueError("BYTES_CHANGED: " + row["path"])


class Meter:
    def __init__(self):
        self.wall = time.monotonic()
        self.cpu = time.process_time()

    def receipt(self):
        return {"wall_seconds": time.monotonic() - self.wall,
                "process_cpu_seconds": time.process_time() - self.cpu,
                "peak_rss_kib_process_lifetime": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "scope": "this Python process; RSS is process-lifetime high-water mark on Linux"}


class Budget:
    def __init__(self, seconds):
        self.start = time.monotonic()
        self.seconds = seconds

    def check(self):
        if time.monotonic() - self.start >= self.seconds:
            raise TimeoutError("PILOT_REFERENCE_WORKER_WALL_LIMIT")


def environment():
    import numpy as np
    import scipy
    import matplotlib
    cpu = "NOT_MEASURED"
    p = Path("/proc/cpuinfo")
    if p.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in p.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    import io
    from contextlib import redirect_stdout
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.show_config()
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "cpu": cpu,
            "logical_cpu_count": os.cpu_count(), "numpy": np.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "blas_configuration": stream.getvalue(),
            "child_environment": {k: os.environ.get(k) for k in
                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONDONTWRITEBYTECODE",
                                   "MPLCONFIGDIR", "MPLBACKEND")},
            "gpu_queries": 0, "model_requests": 0, "model_tokens": "NOT_APPLICABLE",
            "servers": 0, "installs_or_upgrades": 0}


def git_identity(root):
    def g(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"base_git_commit": g("rev-parse", "HEAD"), "branch": g("branch", "--show-current"),
            "working_source_uncommitted": True, "git_status": g("status", "--porcelain=v1"),
            "staged": g("diff", "--cached", "--name-only")}
