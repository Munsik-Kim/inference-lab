"""Case018-owned server boundary, adapting unchanged Case017 lifecycle helpers."""
import json
import os
import time
import urllib.request
from pathlib import Path
from .common import CASE_ROOT, write_new
from .live_client import references


def create_server(runtime, kv, name):
    references(runtime)
    from agent_contract_cost.server_manager import OwnedServer
    class RecoveryServer(OwnedServer):
        def command(self):
            cmd = super().command()
            return [{"case017-qwen3-4b": "case018-qwen3-4b",
                     "agent_contract_cost.p0_inspector.ContractInspector": "agent_state_recovery.inspector.RecoveryInspector"}.get(v, v) for v in cmd]

        def environment(self):
            env = super().environment()
            env["PYTHONPATH"] = os.pathsep.join([str(CASE_ROOT / "src"), *references(runtime)])
            env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
            return env
    return RecoveryServer(runtime, kv, name, mode="GRAPH")


def rpc(base, method):
    from agent_contract_cost.server_manager import rpc as old_rpc
    return old_rpc(base, method)


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        return response.read().decode()


def activity_gauges(raw):
    gauges = {}
    for line in raw.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            gauges[name] = gauges.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return gauges


def drain(base, *, seconds=10):
    """Observe two successive zero pending/running gauges; no fictional abort API."""
    started, rows, zero = time.monotonic(), [], 0
    while time.monotonic() - started < seconds:
        raw = metrics(base)
        gauges = activity_gauges(raw)
        idle = len(gauges) == 2 and all(v == 0 for v in gauges.values())
        rows.append({"elapsed_seconds": time.monotonic() - started, "gauges": gauges, "idle": idle})
        zero = zero + 1 if idle else 0
        if zero >= 2:
            return {"status": "DRAINED", "rows": rows, "scope": "owned engine running/waiting gauges after transport close; no abort API called"}
        time.sleep(0.1)
    return {"status": "NOT_CONFIRMED", "rows": rows}
