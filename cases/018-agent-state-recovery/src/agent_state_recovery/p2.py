"""Case018 P2 orchestration. Reuses unchanged P1 runner/scorer/transport."""
import argparse
import asyncio
import copy
import datetime
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time
from .common import CASE_ROOT, canonical, digest, read_json, write_new
from .live_client import LiveClient, references, request_body, tokenize_only
from .p1 import identity, technical_input, cache_gate, _journal
from .p2_contract import file_row, check_files, p1_preconditions, parent_debit, remaining_budget, request_ceiling, validate_runtime
from .public import grammar
from .runner import run_episode
from .server import create_server, rpc, drain


class P2LiveClient(LiveClient):
    """Only phase-level journal ceilings differ from P1's DEV384/TECH32.

    prepare/body/chat/raw-response checks are reused unchanged; no policy or
    per-episode limit changes. This small outer adapter records P2 categories.
    """
    def respond(self, body, timeout_seconds, *, cap, request_id, prepared):
        submitted = [json.loads(line) for line in self.journal.read_text().splitlines() if line.strip()]
        request_ceiling(submitted, self.category)
        if self.remaining_seconds() <= 0:
            raise TimeoutError("global worker deadline before submission")
        timeout = min(timeout_seconds, self.remaining_seconds())
        payload = request_body(prepared["messages"], self.schema, cap, request_id)
        if self.prepare(body) != prepared:
            raise ValueError("prepared input changed before submission")
        _journal(self.journal, {"type": "REQUEST_SUBMITTED", "category": self.category, "request_id": request_id,
            "request_body_sha256": digest(payload), "input_token_sha256": prepared["input_token_sha256"],
            "monotonic": time.monotonic(), "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        self.attempts += 1
        response = asyncio.run(self.chat(self.base, payload, timeout_seconds=timeout, tokenizer=self.tokenizer))
        self.responses += response["model_responses_received"]
        if response.get("status") == "OK":
            if (response["prompt_token_ids"] != prepared["token_ids"] or response["prompt_text"] != prepared["prompt_text"]
                    or response["input_token_sha256"] != prepared["input_token_sha256"]):
                response["status"] = "INPUT_IDENTITY_MISMATCH"
            elif len(response["output_token_ids"]) > cap:
                response["status"] = "OUTPUT_CAP_VIOLATION"
        response["local_input"] = copy.deepcopy(prepared)
        response["category"] = self.category
        write_new(self.output / (request_id + ".json"), response)
        _journal(self.journal, {"type": "RESPONSE_CLOSED", "category": self.category, "request_id": request_id,
            "status": response["status"], "received": response["model_responses_received"],
            "observed_output_ids": len(response.get("output_token_ids") or []), "monotonic": time.monotonic()})
        return response


def verify_seal(root, runtime):
    root = Path(root)
    before = read_json(root / "pre_generation_freeze.json")
    candidate = identity(runtime)
    if any(before[k] != candidate[k] for k in candidate):
        raise ValueError("P2 source/runtime changed since before generation")
    seal = read_json(root / "input_seal.json")
    for row in (seal["input_manifest"], seal["grammar_compilation"], seal["plan"]):
        check_files(root, [row])
    check_files(CASE_ROOT, [seal["data_manifest"]])
    manifest = read_json(root / "input_manifest.json")
    check_files(root, manifest["files"] + [manifest["initial_inputs"], manifest["pre_generation_freeze"]])
    check_files(CASE_ROOT, read_json(CASE_ROOT / "data/holdout/MANIFEST.json")["files"])
    if seal["holdout_outputs_observed"] != 0 or seal["generator_calls"] != 1:
        raise ValueError("invalid HOLDOUT seal")
    return candidate


def execute(output, runtime_path, evidence, parent):
    root, evidence, parent = Path(output), Path(evidence), Path(parent)
    runtime = read_json(runtime_path)
    validate_runtime(runtime, read_json(parent / "runtime.json"))
    p1_preconditions(parent)
    candidate = verify_seal(root, runtime)
    # An existing attempt is never silently restarted or its cells regenerated.
    for name in ("episodes", "fixtures"):
        (root / name).mkdir(exist_ok=False)
    references(runtime)
    tokenizer = tokenize_only(runtime)
    engine_schema = grammar()
    planned = read_json(root / "plan.json")["planned"]
    tasks = {p.stem: read_json(p) for p in sorted((root / "inputs").glob('*.json'))}
    initial = read_json(root / "initial_inputs.json")
    journal, ledger = evidence / "request_journal.jsonl", evidence / "worker_ledger.jsonl"
    journal.touch(exist_ok=False)
    if ledger.exists():
        raise FileExistsError("existing P2 worker attempt; no automatic retry")
    debit = parent_debit(parent)
    began = time.monotonic()
    deadline = began + remaining_budget(debit)
    _journal(ledger, {"type": "WORKER_STARTED", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "monotonic": began, "parent_worker_debit_seconds": debit, "source": candidate["execution_source_sha256"]})
    servers, active, gates, entries, ended, started, errors = [], None, {}, [], [], [], []
    technical_count = 0
    parameter_signature = None
    def launch(kv, label):
        if deadline - time.monotonic() < 120:
            raise TimeoutError("worker cleanup reserve before load")
        gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], capture_output=True, text=True)
        if gpu.returncode != 0 or gpu.stdout.strip():
            raise RuntimeError("external GPU compute PID present; no shared service change")
        server = create_server(runtime, kv, root.name + "_" + label)
        servers.append(server)
        server.start(ready_seconds=min(600, max(1, deadline - time.monotonic() - 60)))
        write_new(server.root / "owned_identity.json", {"pid": server.proc.pid,
            "start_time_ticks": Path(f"/proc/{server.proc.pid}/stat").read_text().split()[21],
            "command": server.command(), "port": server.port, "server_code": candidate["execution_source_sha256"]})
        return server
    def client(server, tag, category):
        return P2LiveClient(runtime, server.base, tokenizer, engine_schema, root / tag, deadline, journal, category, max_requests=1544)
    def technical(server, kv, tag):
        nonlocal technical_count, parameter_signature
        observed = rpc(server.base, "recovery_inspect")
        write_new(root / (tag + "_cache.json"), observed)
        gate = cache_gate(observed, kv, runtime)
        if parameter_signature is None:
            parameter_signature = gate["model_parameter_sample_sha256"]
        if gate["model_parameter_sample_sha256"] != parameter_signature:
            raise ValueError("BF16 parameter sample mismatch across workers")
        finite = rpc(server.base, "recovery_direct_finite")
        write_new(root / (tag + "_finite.json"), finite)
        if not all(r["all36_finite"] and r["finite_returned_hidden_states"] for r in finite["rows"]):
            raise ValueError("actual all36 finite attention gate")
        rpc(server.base, "recovery_observer_start")
        rpc(server.base, "case016_profile_start")
        adapter = client(server, tag + "_transport", "TECH")
        body = technical_input()
        response = adapter.respond(body, min(30, adapter.remaining_seconds()), cap=64,
                                   request_id=tag + "_normal", prepared=adapter.prepare(body))
        technical_count += 1
        profile = rpc(server.base, "case016_profile_stop")
        write_new(root / (tag + "_profile.json"), profile)
        mask = rpc(server.base, "recovery_observer_state")
        write_new(root / (tag + "_mask.json"), mask)
        if (response["status"] != "OK" or not mask["mask_calls"]
                or not any(r["negative_inf_after"] > r["negative_inf_before"] and r["remaining_finite"] > 0 and not r["NaN_after"] for r in mask["mask_rows"])):
            raise ValueError("fresh raw/IDs/usage/actual grammar mask gate")
        if kv == "K1" and not any("fp4" in row["kernel"].lower() for row in profile["gpu_kernels"]):
            raise ValueError("fresh NVFP4 write/read path not observed")
        drained = drain(server.base)
        write_new(root / (tag + "_drain.json"), drained)
        if drained["status"] != "DRAINED":
            raise RuntimeError("technical drain unconfirmed")
        return {**gate, "G1_model_input": "PASS", "G2_actual_mask": "PASS", "G3_actual_KV_finite": "PASS",
            "G4_transport_budget_drain": "PASS", "technical_API_requests": 1, "direct_forward_probes": 2,
            "timeout_path": "P1 current-identical transport and runtime receipt reused; no new cancellation needed"}
    try:
        for kv in ("K1", "K0"):
            active = launch(kv, "technical_" + kv)
            gates[kv] = technical(active, kv, "TECH_" + kv)
            if kv == "K1":
                active.stop()
                active = None
        verify_seal(root, runtime)
        freeze = {**candidate, "holdout_outputs_observed": 0, "prior_DEV_episodes": 24, "prior_DEV_responses": 180,
            "prior_TECH_attempts": 6, "new_TECH_attempts_observed": technical_count,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "input_seal": file_row(root / "input_seal.json", root), "grammar_schema": engine_schema,
            "planned_keys": planned, "initial_inputs_sha256": digest(initial), "technical_gates": gates}
        write_new(root / "execution_freeze.json", freeze)
        _journal(root / "episode_journal.jsonl", {"type": "EXECUTION_FROZEN", "utc": freeze["created_utc"],
            "holdout_outputs_observed": 0, "execution_freeze_sha256": file_row(root / "execution_freeze.json")["sha256"]})
        for kv in ("K0", "K1"):
            if kv == "K1":
                active = launch(kv, "HOLDOUT_" + kv)
                reloaded = technical(active, kv, "RELOAD_" + kv)
                write_new(root / ("HOLDOUT_" + kv + "_fresh_gate.json"), reloaded)
            adapter = client(active, "HOLDOUT_" + kv + "_transport", "HOLDOUT")
            for cell in (c for c in planned if c["kv"] == kv):
                if deadline - time.monotonic() < 60:
                    raise TimeoutError("worker cleanup reserve before next episode")
                _journal(root / "episode_journal.jsonl", {"type": "STARTED", **cell,
                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "monotonic": time.monotonic()})
                started.append(cell["episode_id"])
                row = run_episode(tasks[cell["task_id"]], adapter, root / "fixtures" / cell["episode_id"],
                                  policy=cell["policy"], episode_id=cell["episode_id"])
                row.update(phase="P2", kv=kv, engine_schema=engine_schema, execution_source_sha256=candidate["execution_source_sha256"])
                if row["requests"] and row["requests"][0]["input_token_ids"] != initial[cell["task_id"]]["token_ids"]:
                    raise ValueError("first actual HOLDOUT IDs mismatch")
                path = root / "episodes" / (cell["episode_id"] + ".json")
                write_new(path, row)
                entries.append({**cell, **file_row(path, root), "canonical_sha256": digest(row)})
                ended.append(cell["episode_id"])
                _journal(root / "episode_journal.jsonl", {"type": "ENDED", **cell, "monotonic": time.monotonic(),
                    "terminal_reason": row["terminal_reason"]})
                if row["terminal_reason"] in ("ADAPTER_ERROR", "REQUEST_TIMEOUT", "WORKER_DEADLINE"):
                    drained = drain(active.base)
                    write_new(root / (cell["episode_id"] + "_drain.json"), drained)
                    if drained["status"] != "DRAINED" or row["terminal_reason"] != "REQUEST_TIMEOUT":
                        raise RuntimeError("infrastructure/observability failure; stop without retry")
            after = rpc(active.base, "recovery_inspect")
            write_new(root / ("HOLDOUT_" + kv + "_after_cache.json"), after)
            if after["parameter_sample_sha256"] != parameter_signature:
                raise ValueError("parameter sample changed during policies")
            write_new(root / ("HOLDOUT_" + kv + "_mask.json"), rpc(active.base, "recovery_observer_state"))
            d = drain(active.base)
            write_new(root / ("HOLDOUT_" + kv + "_drain.json"), d)
            if d["status"] != "DRAINED":
                raise RuntimeError("block drain unconfirmed")
            active.stop()
            active = None
    except Exception as error:
        errors.append({"type": type(error).__name__, "message": str(error), "ended_HOLDOUT_cells": len(ended)})
        write_new(root / "blocking_error.json", errors[-1])
    finally:
        for server in servers:
            if server.proc is not None and server.proc.poll() is None:
                try:
                    write_new(server.root / "last_drain.json", drain(server.base))
                except Exception as e:
                    errors.append({"stage": "drain cleanup", "type": type(e).__name__, "message": str(e)})
                try:
                    server.stop()
                except Exception as e:
                    errors.append({"stage": "owned stop", "type": type(e).__name__, "message": str(e)})
    elapsed = time.monotonic() - began
    _journal(ledger, {"type": "WORKER_STOPPED", "elapsed_seconds": elapsed, "monotonic": time.monotonic(),
        "parent_worker_debit_seconds": debit, "cumulative_worker_seconds": debit + elapsed,
        "remaining_seconds": remaining_budget(debit, elapsed), "overrun_seconds": max(0, debit + elapsed - 3600)})
    missing = [c for c in planned if c["episode_id"] not in ended]
    submitted = [json.loads(s) for s in journal.read_text().splitlines() if s.strip()]
    technical_count = sum(r["type"] == "REQUEST_SUBMITTED" and r["category"] == "TECH" for r in submitted)
    manifest = {"phase": "P2_HOLDOUT", "planned": planned, "planned_count": 96, "started": started,
        "ended": ended, "missing": missing, "episodes": entries, "execution_source_sha256": candidate["execution_source_sha256"],
        "execution_status": "COMPLETE" if not missing and not errors else "PARTIAL", "technical_gates": gates,
        "errors": errors, "model_success_threshold": None, "worker_wall_seconds": elapsed,
        "parent_worker_seconds": debit, "cumulative_worker_seconds": debit + elapsed,
        "remaining_3600_budget_seconds": remaining_budget(debit, elapsed), "technical_generation_requests": technical_count,
        "P3": "NOT_RUN", "remote_publication": "NOT_RUN"}
    write_new(root / "run_manifest.json", manifest)
    write_new(root / "worker_cost.json", {"worker_wall_seconds": elapsed, "parent_worker_seconds": debit,
        "cumulative_worker_seconds": debit + elapsed, "remaining_3600_seconds": remaining_budget(debit, elapsed),
        "peak_RSS_KiB_controller": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "P2 first launch through final owned shutdown; includes load/compile/TECH/failure/I/O; overlaps episode/continuation wall"})
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        q = sub.add_parser(name)
        for arg in ("output", "runtime", "evidence", "parent"):
            q.add_argument("--" + arg, required=True)
    for name in ("audit", "analyze", "tokencheck"):
        q = sub.add_parser(name)
        q.add_argument("--run", required=True)
        q.add_argument("--output", required=True)
        if name == "analyze": q.add_argument("--audit", required=True)
        if name == "tokencheck": q.add_argument("--runtime", required=True)
    a = p.parse_args()
    if a.command == "prepare":
        from .p2_prepare import prepare
        result = prepare(a.output, a.runtime, a.evidence, a.parent)
    elif a.command == "run":
        result = execute(a.output, a.runtime, a.evidence, a.parent)
    elif a.command == "audit":
        from .p2_audit import audit_run
        result = audit_run(a.run, a.output)
    elif a.command == "tokencheck":
        from .p2_audit import tokencheck
        result = tokencheck(a.run, a.runtime, a.output)
    else:
        from .p2_analysis import analyze
        result = analyze(a.run, a.audit, a.output)
    print(json.dumps({k: v for k, v in result.items() if k in ("status", "execution_status", "episodes", "requests", "tasks", "totals", "new_model_requests")}, ensure_ascii=False))
    return 0 if result.get("status", result.get("execution_status")) in ("PASS", "COMPLETE", "SEALED_CPU_READY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
