"""Registered P1 technical gates and DEV24 only; no HOLDOUT generation path."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from .common import CASE_ROOT, canonical, digest, fresh_dir, read_json, source_manifest, write_new
from .generator import dev_tasks
from .public import definitions, grammar
from .live_client import LiveClient, references, tokenize_only
from .runner import run_episode
from .server import create_server, rpc, drain, metrics


def identity(runtime):
    own = source_manifest()
    dependency_paths = [CASE_ROOT.parent / "017-agent-contract-cost-bench/src/agent_contract_cost" / name
                        for name in ("__init__.py", "client.py", "server_manager.py")]
    c16 = Path(runtime["original_case"])
    dependency_paths += [c16 / rel for rel in ("src/kv_agent_bench/__init__.py", "src/kv_agent_bench/worker_inspection.py",
        "src/kv_agent_bench/profiler_extension.py", "followups/nvfp4-technical-r1/src/nvfp4_technical/__init__.py",
        "followups/nvfp4-technical-r1/src/nvfp4_technical/worker.py")]
    dependencies = [{"path": str(p), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in dependency_paths]
    body = {"own": own, "readonly_dependencies": dependencies, "runtime_configuration_sha256": digest(runtime)}
    return {**body, "execution_source_sha256": digest(body)}


def technical_input(long=False):
    prompt = ("Return final for task_id TECH, recovered false, validation_event_id TECH, and twelve quantities records. "
              "Use distinct 90-character arbitrary record_id strings and nonnegative quantities." if long else
              "Return a read_history call with empty arguments. This is only a transport/cache probe; no tool will execute.")
    return {"messages": [{"role": "system", "content": "Return one JSON object using the public API. Technical probe only."},
                         {"role": "user", "content": prompt}], "tools": definitions(), "output_schema": grammar()}


def cache_gate(r, kv, runtime):
    if r["cuda_capability"] != [12, 0] or len(r["layers"]) != 36:
        raise ValueError("SM120/all36 layer gate")
    if r["model_dtype"] != "torch.bfloat16" or r["model_path"] != runtime["snapshot"] or r["tokenizer_path"] != runtime["snapshot"]:
        raise ValueError("actual model path/dtype gate")
    if r["structured_outputs_backend"] != "xgrammar" or r["cache_config"]["enable_prefix_caching"]:
        raise ValueError("grammar/prefix gate")
    if r["tf32"] is not False or r["bf16_reduced_precision_reduction"] is not True or r["cache_config"].get("kv_cache_dtype_skip_layers"):
        raise ValueError("common numerical policy/skip layer gate")
    if r["max_model_len"] != 28672 or r["scheduler"] != {"max_num_seqs": 4, "max_num_batched_tokens": 2048, "enable_chunked_prefill": True, "async_scheduling": False}:
        raise ValueError("common context/scheduler gate")
    if "NONE" in r["cudagraph_policy"] or "NO" == r["cudagraph_policy"]:
        raise ValueError("common GRAPH gate")
    dtype = "torch.bfloat16" if kv == "K0" else "torch.uint8"
    if len(r["cache_tensors"]) != 36 or any(t["dtype"] != dtype for t in r["cache_tensors"]):
        raise ValueError("actual all36 cache dtype gate")
    if kv == "K1":
        if len(r["nvfp4_views"]) != 36 or not all(layer["is_kvcache_nvfp4"] for layer in r["layers"]):
            raise ValueError("all36 native NVFP4 reader path gate")
        if not all(v["data_dtype"] == ["torch.uint8"] * 2 and v["scale_dtype"] == ["torch.float8_e4m3fn"] * 2
                   and v["data_scale_storage_shared"] for v in r["nvfp4_views"]):
            raise ValueError("packed data/FP8 block scales gate")
    bpt = 147456 if kv == "K0" else 41472
    size = r["actual_unique_storage_bytes"]
    if size % bpt or not 4.2e9 < size <= 4294967296:
        raise ValueError("actual common pool arithmetic gate")
    return {"status": "PASS", "bytes_per_token": bpt, "actual_pool_bytes": size, "token_slots": size // bpt,
            "blocks": r["cache_config"]["num_gpu_blocks"], "block_size": r["cache_config"]["block_size"],
            "model_parameter_sample_sha256": r["parameter_sample_sha256"]}


def _journal(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(canonical(row) + "\n")
        stream.flush()


def execute(output, runtime_path, preparation, evidence):
    root = fresh_dir(output)
    runtime = read_json(runtime_path)
    prep, evidence = Path(preparation), Path(evidence)
    references(runtime)
    tokenizer = tokenize_only(runtime)
    engine_schema = grammar()
    planned = read_json(prep / "plan.json")["planned"]
    candidate = identity(runtime)
    write_new(root / "pre_technical_source.json", candidate)
    (root / "inputs").mkdir()
    for task in dev_tasks():
        write_new(root / "inputs" / (task["task_id"] + ".json"), task)
    (root / "episodes").mkdir()
    (root / "fixtures").mkdir()
    for name in ("grammar_cpu.json", "initial_inputs.json", "script_token_feasibility.json", "plan.json"):
        write_new(root / name, read_json(prep / name))
    journal = evidence / "request_journal.jsonl"
    if not journal.exists():
        journal.touch(exist_ok=False)
    ledger = evidence / "worker_ledger.jsonl"
    previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    debit = sum(r["elapsed_seconds"] for r in previous if r["type"] == "WORKER_STOPPED")
    if sum(r["type"] == "WORKER_STARTED" for r in previous) != sum(r["type"] == "WORKER_STOPPED" for r in previous):
        raise RuntimeError("unclean previous worker ledger; no automatic retry")
    worker_start = time.monotonic()
    deadline = worker_start + max(0, 3600 - debit)
    _journal(ledger, {"type": "WORKER_STARTED", "monotonic": worker_start, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "source": candidate["execution_source_sha256"], "previous_debit_seconds": debit})
    gates, servers, entries, ended, errors, started_cells = {}, [], [], [], [], []
    active = None
    frozen = None
    technical_count = 0
    initial_inputs = read_json(root / "initial_inputs.json")
    tasks = {t["task_id"]: t for t in dev_tasks()}
    def launch(kv, label):
        if deadline - time.monotonic() < 120:
            raise TimeoutError("worker budget cleanup reserve")
        gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], capture_output=True, text=True)
        if gpu.returncode != 0 or gpu.stdout.strip():
            raise RuntimeError("external GPU compute PID present; no service termination")
        server = create_server(runtime, kv, root.name + "_" + label)
        servers.append(server)
        server.start(ready_seconds=min(600, max(1, deadline - time.monotonic() - 60)))
        write_new(server.root / "owned_identity.json", {"pid": server.proc.pid,
            "start_time_ticks": Path(f"/proc/{server.proc.pid}/stat").read_text().split()[21],
            "command": server.command(), "port": server.port, "server_code": candidate["execution_source_sha256"]})
        return server
    def client(server, name, category):
        return LiveClient(runtime, server.base, tokenizer, engine_schema, root / name, deadline, journal, category)
    def verify_loaded(server, kv, tag):
        observed = rpc(server.base, "recovery_inspect")
        write_new(root / (tag + "_cache.json"), observed)
        result = cache_gate(observed, kv, runtime)
        rpc(server.base, "recovery_observer_start")
        return result
    def technical(server, kv):
        nonlocal technical_count
        gate = verify_loaded(server, kv, "TECH_" + kv)
        finite = rpc(server.base, "recovery_direct_finite")
        write_new(root / ("TECH_" + kv + "_finite.json"), finite)
        if not all(r["all36_finite"] and r["finite_returned_hidden_states"] for r in finite["rows"]):
            raise ValueError("actual attention prefill/decode finite gate")
        adapter = client(server, "TECH_" + kv + "_transport", "TECH")
        rpc(server.base, "case016_profile_start")
        body = technical_input()
        response = adapter.respond(body, min(30, adapter.remaining_seconds()), cap=64, request_id="TECH_" + kv + "_normal", prepared=adapter.prepare(body))
        technical_count += 1
        profile = rpc(server.base, "case016_profile_stop")
        write_new(root / ("TECH_" + kv + "_profile.json"), profile)
        if response["status"] != "OK":
            raise ValueError("raw transport identity/usage gate:" + response["status"])
        observer = rpc(server.base, "recovery_observer_state")
        write_new(root / ("TECH_" + kv + "_mask.json"), observer)
        if not observer["mask_calls"] or not any(r["negative_inf_after"] > r["negative_inf_before"] and r["remaining_finite"] > 0 and not r["NaN_after"] for r in observer["mask_rows"]):
            raise ValueError("actual grammar mask application gate")
        if kv == "K1" and not any("fp4" in r["kernel"].lower() for r in profile["gpu_kernels"]):
            raise ValueError("NVFP4 kernel path not observed")
        before_cancel = drain(server.base)
        if before_cancel["status"] != "DRAINED":
            raise RuntimeError("owned server not drained before cancellation probe")
        body = technical_input(long=True)
        response = adapter.respond(body, 0.05, cap=512, request_id="TECH_" + kv + "_cancel", prepared=adapter.prepare(body))
        technical_count += 1
        drained = drain(server.base)
        write_new(root / ("TECH_" + kv + "_cancel_drain.json"), {"transport_status": response["status"],
            "partial_SSE_lines": len(response["raw_sse_lines"]), "partial_output_ids": len(response["output_token_ids"]),
            "drain": drained, "abort_API_calls": 0})
        if drained["status"] != "DRAINED":
            raise RuntimeError("cancellation drain unconfirmed; stop before DEV")
        return {**gate, "G1_model_input": "PASS", "G2_actual_mask": "PASS", "G3_actual_KV_finite": "PASS",
                "G4_transport_budget_drain": "PASS", "technical_API_requests": 2,
                "direct_forward_probes": 2, "cancel_observed": response["status"] == "REQUEST_TIMEOUT"}
    try:
        for kv in ("K1", "K0"):
            active = launch(kv, "technical_" + kv)
            gates[kv] = technical(active, kv)
            if kv == "K1":
                active.stop()
                active = None
        if gates["K0"]["model_parameter_sample_sha256"] != gates["K1"]["model_parameter_sample_sha256"]:
            raise ValueError("BF16 parameter signature differs across KV conditions")
        if identity(runtime) != candidate:
            raise ValueError("source changed during technical gate")
        all_submitted = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        prior_and_current_technical = sum(r["type"] == "REQUEST_SUBMITTED" and r["category"] == "TECH" for r in all_submitted)
        frozen = {**candidate, "dev_outputs_observed": 0, "technical_API_requests_observed": prior_and_current_technical,
                  "current_attempt_technical_API_requests_observed": technical_count,
                  "grammar_schema": engine_schema, "planned_keys": planned,
                  "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "initial_inputs_sha256": digest(initial_inputs), "technical_gates": gates}
        write_new(root / "execution_freeze.json", frozen)
        for kv in ("K0", "K1"):
            if kv == "K1":
                active = launch(kv, "DEV_" + kv)
                loaded = verify_loaded(active, kv, "DEV_" + kv)
                if loaded["model_parameter_sample_sha256"] != gates[kv]["model_parameter_sample_sha256"]:
                    raise ValueError("reloaded parameter signature mismatch")
            adapter = client(active, "DEV_" + kv + "_transport", "DEV")
            parameter_before = rpc(active.base, "recovery_inspect")["parameter_sample_sha256"]
            for cell in (c for c in planned if c["kv"] == kv):
                if deadline - time.monotonic() < 60:
                    raise TimeoutError("worker cleanup reserve reached")
                _journal(root / "episode_journal.jsonl", {"type": "STARTED", **cell, "monotonic": time.monotonic()})
                started_cells.append(cell["episode_id"])
                task = tasks[cell["task_id"]]
                row = run_episode(task, adapter, root / "fixtures" / cell["episode_id"], policy=cell["policy"], episode_id=cell["episode_id"])
                row.update(phase="P1", kv=kv, engine_schema=engine_schema,
                           execution_source_sha256=candidate["execution_source_sha256"])
                if row["requests"] and row["requests"][0]["input_token_ids"] != initial_inputs[task["task_id"]]["token_ids"]:
                    raise ValueError("first actual DEV input differs from frozen task input")
                path = root / "episodes" / (cell["episode_id"] + ".json")
                write_new(path, row)
                entries.append({**cell, "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "canonical_sha256": digest(row)})
                ended.append(cell["episode_id"])
                _journal(root / "episode_journal.jsonl", {"type": "ENDED", **cell, "terminal_reason": row["terminal_reason"], "monotonic": time.monotonic()})
                if row["terminal_reason"] in ("ADAPTER_ERROR", "REQUEST_TIMEOUT", "WORKER_DEADLINE"):
                    drained = drain(active.base)
                    write_new(root / (cell["episode_id"] + "_drain.json"), drained)
                    if drained["status"] != "DRAINED" or row["terminal_reason"] != "REQUEST_TIMEOUT":
                        raise RuntimeError("infrastructure/observability failure; no retry or next cell")
            after = rpc(active.base, "recovery_inspect")
            write_new(root / ("DEV_" + kv + "_after_cache.json"), after)
            if after["parameter_sample_sha256"] != parameter_before:
                raise ValueError("model parameters changed during policy branches")
            write_new(root / ("DEV_" + kv + "_mask.json"), rpc(active.base, "recovery_observer_state"))
            write_new(root / ("DEV_" + kv + "_drain.json"), drain(active.base))
            active.stop()
            active = None
    except Exception as error:
        errors.append({"type": type(error).__name__, "message": str(error), "technical_API_requests_observed": technical_count,
                       "ended_DEV_cells": len(ended)})
        write_new(root / "blocking_error.json", errors[-1])
    finally:
        if active is not None:
            try:
                if active.proc and active.proc.poll() is None:
                    write_new(root / "final_drain.json", drain(active.base))
                active.stop()
            except Exception as error:
                errors.append({"type": type(error).__name__, "message": str(error), "stage": "owned cleanup"})
        # A startup failure can occur before the active assignment completes.
        for server in servers:
            if server.proc is not None and server.proc.poll() is None:
                try:
                    server.stop()
                except Exception as error:
                    errors.append({"type": type(error).__name__, "message": str(error), "stage": "owned startup cleanup"})
    elapsed = time.monotonic() - worker_start
    _journal(ledger, {"type": "WORKER_STOPPED", "elapsed_seconds": elapsed, "monotonic": time.monotonic(),
                      "remaining_seconds": max(0, 3600 - debit - elapsed), "normal_budget_overrun_seconds": max(0, elapsed + debit - 3600)})
    missing = [c for c in planned if c["episode_id"] not in ended]
    submitted = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    technical_count = sum(r["type"] == "REQUEST_SUBMITTED" and r["category"] == "TECH" for r in submitted)
    manifest = {"phase": "P1", "planned": planned, "planned_count": 24, "started": started_cells,
                "ended": ended, "missing": missing, "episodes": entries, "execution_source_sha256": candidate["execution_source_sha256"],
                "execution_status": "COMPLETE" if not missing and not errors else "PARTIAL",
                "technical_gates": gates, "errors": errors, "model_success_threshold": None,
                "worker_wall_seconds": elapsed, "remaining_3600_budget_seconds": max(0, 3600 - debit - elapsed),
                "technical_generation_requests": technical_count, "P2": "NOT_RUN", "HOLDOUT": "NOT_CREATED"}
    write_new(root / "run_manifest.json", manifest)
    write_new(root / "worker_cost.json", {"worker_wall_seconds": elapsed,
                 "peak_RSS_KiB_controller": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 "scope": "first owned model launch through final owned shutdown; includes TECH/load/compile/failure/I/O, overlaps episode and continuation costs",
                 "remaining_3600_seconds": manifest["remaining_3600_budget_seconds"]})
    return manifest
