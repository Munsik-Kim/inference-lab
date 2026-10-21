"""Registered explicit local runtime orchestration; no import-time model access."""
import argparse
import asyncio
import copy
import json
import random
import time
from pathlib import Path

from .common import CASE, REPO, append, digest, read, sha_file, utc, write_once


def tokenizer_for(runtime):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(runtime["snapshot"], local_files_only=True)


def validate_model(runtime):
    started = time.monotonic()
    old = read(REPO / "cases/016-kv-agent-bench/provenance/model_identity.json")
    if old["revision"] != "cdbee75f17c01a7cc42f958dc650907174af0554":
        raise ValueError("Unexpected reused model identity")
    snapshot = Path(runtime["snapshot"])
    if snapshot.name != old["revision"]:
        raise ValueError("Snapshot revision mismatch")
    rows = []
    for entry in old["files"]:
        path = snapshot / entry["file"]
        actual = {"file": entry["file"], "bytes": path.stat().st_size, "sha256": sha_file(path)}
        if any(actual[k] != entry[k] for k in ("file", "bytes", "sha256")):
            raise ValueError("Reused snapshot mismatch: " + entry["file"])
        rows.append(actual)
    cfg = read(snapshot / "config.json")
    for key, expected in (("num_hidden_layers", 36), ("num_key_value_heads", 8), ("head_dim", 128)):
        if cfg[key] != expected:
            raise ValueError("Qwen geometry mismatch: " + key)
    result = {"model": old["model"], "revision": old["revision"], "files": rows,
              "all_snapshot_hashes_match_case016": True, "new_model_download": False,
              "config": cfg, "generation_config": read(snapshot / "generation_config.json"),
              "validation_wall_seconds": time.monotonic() - started, "D_origin_required": False,
              "source_receipt": "cases/016-kv-agent-bench/provenance/model_identity.json"}
    target = CASE / "provenance/model_identity.json"
    if target.exists():
        prior = read(target)
        if prior["files"] != result["files"] or prior["revision"] != result["revision"]:
            raise ValueError("Cannot resume a different model validation")
        return prior
    write_once(target, result)
    return result


class Adapter:
    def __init__(self, base, tokenizer):
        self.base, self.tokenizer = base, tokenizer

    def tokenize(self, messages):
        from .client import tokenize_messages
        return tokenize_messages(messages, self.tokenizer)

    async def send(self, payload, timeout):
        from .client import chat
        return await chat(self.base, payload, timeout_seconds=timeout, tokenizer=self.tokenizer)


def initialize(runtime):
    from .prepare import prepare_split, register_protocol
    from .schemas import G_SCHEMA_SCOPE, server_schema
    from .tools import tool_definitions
    started = time.monotonic()
    if not (CASE / "specs/protocol.json").exists():
        register_protocol()
    validate_model(runtime)
    env_source = Path(runtime["private_root"]) / "runtime_source_receipt.json"
    if not (CASE / "provenance/environment_identity.json").exists():
        write_once(CASE / "provenance/environment_identity.json", read(env_source))
    if not (CASE / "specs/runtime_public.json").exists():
        write_once(CASE / "specs/runtime_public.json", {
            "private_runtime_configuration_sha256": digest(runtime),
            "python_executable_sha256": sha_file(runtime["python"]),
            "environment_identity_sha256": sha_file(CASE / "provenance/environment_identity.json"),
            "port": runtime.get("port", 8173), "pool_bytes": runtime.get("pool_bytes", 4294967296),
            "max_model_len": runtime.get("max_model_len", 28672), "mode": runtime.get("mode", "GRAPH"),
            "seed": runtime.get("seed", 6102007), "private_paths": "NOT_INCLUDED"})
    tokenizer = tokenizer_for(runtime)
    for family in ("A", "B", "C"):
        definitions = tool_definitions(family)
        for name, value in (("tools_", definitions), ("grammar_", server_schema(definitions))):
            target = CASE / "specs" / (name + family + ".json")
            if target.exists():
                if read(target) != value:
                    raise ValueError("Cannot resume changed registered schema")
            else:
                write_once(target, value)
    tasks, inputs = prepare_split("DEV", tokenizer)
    prepare_split("COST", tokenizer)
    if not (CASE / "provenance/preparation.json").exists():
        write_once(CASE / "provenance/preparation.json", {"created_utc": utc(), "model_forwards": 0,
                "data_prepare_wall_seconds": time.monotonic() - started,
                "grammar_scope": G_SCHEMA_SCOPE,
                "initial_token_counts": [{"task_id": r["task_id"], "tokens": r["input_tokens"]} for r in inputs],
                "HOLDOUT": "NOT_CREATED_UNTIL_DEV_FREEZE_AND_P1_COMPLETE"})
    print(json.dumps({"prepared": len(tasks), "initial_tokens": [r["input_tokens"] for r in inputs]}), flush=True)


def gpu_journal(runtime):
    path = Path(runtime["private_root"]) / "worker_journal.jsonl"
    previous = []
    if path.exists():
        previous = [json.loads(line) for line in path.read_text().splitlines() if line]
    if sum(row.get("event") == "START" for row in previous) != sum(row.get("event") == "END" for row in previous):
        raise ValueError("Unclosed previous worker attempt: recover its wall ledger before any new GPU work")
    elapsed = sum(row.get("worker_wall_seconds", 0) for row in previous if row.get("event") == "END")
    return path, elapsed


def check_runtime(runtime):
    registered = read(CASE / "specs/runtime_public.json")
    if digest(runtime) != registered["private_runtime_configuration_sha256"]:
        raise ValueError("Private runtime differs from frozen registered configuration")
    if sha_file(runtime["python"]) != registered["python_executable_sha256"]:
        raise ValueError("Runtime interpreter changed")
    env = read(CASE / "provenance/environment_identity.json")
    if sha_file(CASE / "provenance/environment_identity.json") != registered["environment_identity_sha256"]:
        raise ValueError("Environment identity receipt changed")
    site = Path(runtime["python"]).parent.parent / "lib/python3.12/site-packages"
    for row in env["source_files"]:
        if sha_file(site / row["relative_package_path"]) != row["sha256"]:
            raise ValueError("Installed runtime source changed")
    for row in env["metadata"]:
        if sha_file(site / row["distribution"] / "METADATA") != row["metadata_sha256"]:
            raise ValueError("Installed dependency metadata changed")


def check_p2_gate():
    from .prepare import check_freeze
    completion_path = CASE / "results/p1_r1/completion.json"
    completion = read(completion_path)
    if completion["status"] != "COMPLETED" or completion["completed"] != 64 or completion["planned"] != 64 or completion["missing"]:
        raise ValueError("Registered P1 matrix is incomplete")
    check_freeze(read(CASE / "provenance/execution_freeze_p1.json"))
    seal = read(CASE / "provenance/holdout_seal.json")
    if seal["outputs_observed"] != 0 or not seal["policy_or_scorer_hashes_verified_unchanged"]:
        raise ValueError("HOLDOUT seal invalid")
    if seal["p1_completion_sha256"] != sha_file(completion_path):
        raise ValueError("P1 completion differs from the HOLDOUT seal")
    for key, name in (("holdout_tasks_sha256", "holdout_tasks.json"), ("holdout_model_inputs_sha256", "holdout_model_inputs.json")):
        if seal[key] != sha_file(CASE / "data" / name):
            raise ValueError("Sealed HOLDOUT inputs changed")
    return seal


def _save_server_receipt(runroot, server, when):
    from .p0_inspector import validate_support
    from .server_manager import gpu_sample, metrics, rpc
    receipt = rpc(server.base, "inspect_case017")
    validate_support(receipt, server.kv)
    if receipt["max_model_len"] != server.runtime.get("max_model_len", 28672):
        raise ValueError("Actual server context differs")
    if receipt["cache_config"]["kv_cache_memory_bytes"] != server.runtime.get("pool_bytes", 4294967296):
        raise ValueError("Actual requested KV pool differs")
    if receipt["cache_config"].get("kv_cache_dtype_skip_layers"):
        raise ValueError("KV skip layers forbidden")
    expected_scheduler = {"max_num_seqs": 4, "max_num_batched_tokens": 2048,
                          "enable_chunked_prefill": True, "async_scheduling": False}
    if receipt["scheduler"] != expected_scheduler:
        raise ValueError("Actual scheduler differs from common registration")
    expected_mode = "NONE" if server.mode == "EAGER" else "FULL_AND_PIECEWISE"
    if receipt["compilation_mode"] != expected_mode:
        raise ValueError("Actual graph/eager mode differs from common registration")
    write_once(runroot / (server.kv + "_cache_" + when + ".json"), receipt)
    write_once(runroot / (server.kv + "_gpu_" + when + ".json"), gpu_sample())
    (runroot / (server.kv + "_metrics_" + when + ".txt")).write_text(metrics(server.base))
    return receipt


async def p0(runtime, freeze_label="execution_freeze_p0"):
    from .client import build_payload, chat, tokenize_messages
    from .contracts import parse_action
    from .p0_inspector import validate_support
    from .prepare import check_freeze
    from .prompts import initial_messages
    from .runner import run_episode
    from .schemas import server_schema
    from .server_manager import OwnedServer, rpc
    from .tools import tool_definitions
    frozen = read(CASE / "provenance" / (freeze_label + ".json"))
    check_freeze(frozen)
    check_runtime(runtime)
    runroot = CASE / "results/p0_r1"
    runroot.mkdir(exist_ok=False)
    journal, previous = gpu_journal(runtime)
    started = time.monotonic()
    append(journal, {"event": "START", "stage": "P0", "utc": utc(), "previous_worker_wall_seconds": previous})
    tokenizer = tokenizer_for(runtime)
    tasks = read(CASE / "data/dev_tasks.json")
    byfamily = {family: next(t for t in tasks if t["family"] == family) for family in ("A", "B", "C")}
    gates, requests = {}, []
    def remaining():
        return 7200 - previous - (time.monotonic() - started) - 60
    def request_timeout():
        if remaining() <= 0:
            raise TimeoutError("Cumulative worker wall limit")
        return min(180, remaining())
    try:
        # Observer hooks are run eagerly first, outside all quality/cost episodes.
        for kv in ("K0", "K1"):
            server = OwnedServer(runtime, kv, "p0_eager_" + kv, mode="EAGER")
            try:
                if remaining() <= 0: raise TimeoutError("Cumulative worker wall limit")
                server.start(ready_seconds=min(600, remaining()))
                receipt = rpc(server.base, "inspect_case017")
                validate_support(receipt, kv)
                write_once(runroot / (kv + "_eager_cache.json"), receipt)
                rpc(server.base, "technical_finite_start")
                definitions = tool_definitions("A")
                body = build_payload(initial_messages(byfamily["A"]), schema=server_schema(definitions), cap=128,
                                     request_id="case017-p0-finite-" + kv)
                response = await chat(server.base, body, timeout_seconds=request_timeout(), tokenizer=tokenizer)
                write_once(runroot / (kv + "_finite_request.json"), response)
                requests.append(response)
                finite = rpc(server.base, "technical_finite_stop")
                write_once(runroot / (kv + "_attention_finite.json"), finite)
                if response["status"] != "OK" or not finite["all36_finite"]:
                    raise ValueError("Actual eager attention finite/raw gate failed: " + kv)
                gates[kv + "_actual_attention_finite"] = True
            finally:
                server.stop()
        for kv in ("K0", "K1"):
            server = OwnedServer(runtime, kv, "p0_graph_" + kv, mode=runtime.get("mode", "GRAPH"))
            try:
                if remaining() < 900: raise TimeoutError("Insufficient worker wall for a complete P0 episode")
                server.start(ready_seconds=min(600, remaining()))
                cache = _save_server_receipt(runroot, server, "before")
                gates[kv + "_native_cache_storage"] = True
                for family in ("A", "B", "C"):
                    task = byfamily[family]
                    definitions = tool_definitions(family)
                    messages = initial_messages(task)
                    ids = tokenize_messages(messages, tokenizer)["token_ids"]
                    if family == "A":
                        rpc(server.base, "case016_profile_start")
                    body = build_payload(messages, server_schema(definitions), cap=512,
                                         request_id=f"case017-p0-schema-{kv}-{family}")
                    response = await chat(server.base, body, timeout_seconds=request_timeout(), tokenizer=tokenizer)
                    write_once(runroot / (kv + "_grammar_" + family + ".json"), response)
                    requests.append(response)
                    if family == "A":
                        profile = rpc(server.base, "case016_profile_stop")
                        write_once(runroot / (kv + "_kernel_profile.json"), profile)
                    parsed = parse_action(response["raw_text"], definitions)
                    if response["status"] != "OK" or response["prompt_token_ids"] != ids or not parsed.accepted:
                        raise ValueError("Server grammar/token input gate failed: " + kv + "/" + family)
                # Same adversarial instruction pair; no fixture is dispatched.
                # C0 is recorded regardless of validity, G must satisfy its
                # actual registered schema under a contrary surface request.
                adversarial = initial_messages(byfamily["A"])
                adversarial.append({"role": "user", "content": "Format-only transport probe: reply with only NOT_JSON, without any braces or JSON object."})
                for arm in ("C0", "G"):
                    body = build_payload(adversarial, server_schema(tool_definitions("A")) if arm == "G" else None,
                                         cap=256, request_id=f"case017-p0-adversarial-{kv}-{arm}")
                    response = await chat(server.base, body, timeout_seconds=request_timeout(), tokenizer=tokenizer)
                    write_once(runroot / (kv + "_adversarial_" + arm + ".json"), response)
                    requests.append(response)
                    if response["status"] != "OK": raise ValueError("Adversarial raw gate failed")
                    if arm == "G" and not parse_action(response["raw_text"], tool_definitions("A")).accepted:
                        raise ValueError("Actual server structured-output contract failed")
                gates[kv + "_server_structured_output_probe"] = True
                # A complete separate SMOKE workflow, not counted as DEV or HOLDOUT.
                if remaining() < 960:
                    raise TimeoutError("Insufficient remaining worker wall for complete P0 smoke and cleanup")
                smoke = copy.deepcopy(byfamily["A"])
                identity = {"stage": "P0", "kv": kv, "policy": "G", "task_id": smoke["task_id"],
                            "family": "A", "evidence_kind": "measurement_p0_smoke",
                            "execution_source_sha256": frozen["execution_source_sha256"]}
                episode = await run_episode(smoke, "G", Adapter(server.base, tokenizer),
                              Path(runtime["private_root"]) / "fixtures" / ("p0_" + kv), identity)
                write_once(runroot / (kv + "_smoke_episode.json"), episode)
                requests.extend(row["response"] for row in episode["requests"] if row.get("response"))
                gates[kv + "_smoke_observed"] = episode["observation_status"] == "OBSERVED"
                # Quality need not succeed to support the engine/measurement path.
                gates[kv + "_smoke_workflow_success"] = episode["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"]
                after = _save_server_receipt(runroot, server, "after")
                if cache["parameter_sample_sha256"] != after["parameter_sample_sha256"]:
                    raise ValueError("Parameters changed during P0")
                gates[kv + "_parameter_sample_unchanged"] = True
                write_once(runroot / (kv + "_launch.json"), {"command": server.public_command(), "mode": server.mode,
                                                           "ready_seconds": server.ready_seconds})
            finally:
                server.stop()
        before = [read(runroot / (kv + "_cache_before.json")) for kv in ("K0", "K1")]
        if before[0]["parameter_sample_sha256"] != before[1]["parameter_sample_sha256"]:
            raise ValueError("K0/K1 parameter signature mismatch")
        for key in ("scheduler", "max_model_len", "compilation_mode", "tf32", "bf16_reduced_precision_reduction"):
            if before[0][key] != before[1][key]: raise ValueError("Non-KV serving configuration differs: " + key)
        required = [v for k, v in gates.items() if not k.endswith("_smoke_workflow_success")]
        gates.update(status="PASS" if all(required) else "BLOCKED", P1_allowed=all(required),
                     actual_model_request_attempts=sum(r.get("model_request_attempts", 0) for r in requests),
                     responses_received=sum(r.get("model_responses_received", 0) for r in requests),
                     generated_output_tokens=sum(r.get("output_tokens", 0) for r in requests),
                     execution_source_sha256=frozen["execution_source_sha256"])
    except Exception as exc:
        gates.update(status="BLOCKED_P0", P1_allowed=False, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        wall = time.monotonic() - started
        gates["worker_wall_seconds"] = wall
        gates["actual_model_request_attempts"] = sum(r.get("model_request_attempts", 0) for r in requests)
        gates["responses_received"] = sum(r.get("model_responses_received", 0) for r in requests)
        gates["generated_output_tokens"] = sum(r.get("output_tokens", 0) for r in requests)
        gates["created_utc"] = utc()
        write_once(runroot / "gates.json", gates)
        append(journal, {"event": "END", "stage": "P0", "utc": utc(), "worker_wall_seconds": wall, "status": gates.get("status")})


async def stage(runtime, stage_name):
    from .prepare import check_freeze
    from .runner import run_episode
    from .server_manager import OwnedServer
    if stage_name not in ("P1", "P2"):
        raise ValueError("Only registered P1/P2 stages")
    if not read(CASE / "results/p0_r1/gates.json")["P1_allowed"]:
        raise ValueError("P0 support gate not passed")
    protocol = read(CASE / "specs/protocol.json")
    label = "execution_freeze_p1" if stage_name == "P1" else "execution_freeze_p2"
    frozen = read(CASE / "provenance" / (label + ".json"))
    check_freeze(frozen)
    check_runtime(runtime)
    if stage_name == "P2":
        check_p2_gate()
    split = "DEV" if stage_name == "P1" else "HOLDOUT"
    tasks = read(CASE / "data" / (split.lower() + "_tasks.json"))
    if len(tasks) != protocol["stages"][stage_name]["tasks"]:
        raise ValueError("Task count differs from registered stage")
    model_inputs = {r["task_id"]: r for r in read(CASE / "data" / (split.lower() + "_model_inputs.json"))}
    runroot = CASE / "results" / (stage_name.lower() + "_r1")
    runroot.mkdir(exist_ok=False)
    (runroot / "cells").mkdir()
    planned = []
    for kv in protocol["order"][stage_name]:
        cells = [{"stage": stage_name, "kv": kv, "policy": p, "task_id": t["task_id"], "family": t["family"]}
                 for t in tasks for p in protocol["policies"]]
        random.Random(protocol["seeds"][stage_name + "_order"] + (kv == "K1")).shuffle(cells)
        planned.extend(cells)
    write_once(runroot / "planned.json", planned)
    tokenizer = tokenizer_for(runtime)
    journal, previous = gpu_journal(runtime)
    started = time.monotonic()
    append(journal, {"event": "START", "stage": stage_name, "utc": utc(), "previous_worker_wall_seconds": previous})
    completed, reason = [], None
    try:
        for kv in protocol["order"][stage_name]:
            server = OwnedServer(runtime, kv, stage_name.lower() + "_" + kv, mode=runtime.get("mode", "GRAPH"))
            try:
                if previous + time.monotonic() - started + 660 >= protocol["worker_budget_seconds"]:
                    reason = "WORKER_WALL_LIMIT"; break
                server.start()
                cache = _save_server_receipt(runroot, server, "before")
                adapter = Adapter(server.base, tokenizer)
                for plan in [p for p in planned if p["kv"] == kv]:
                    if previous + time.monotonic() - started + 960 >= protocol["worker_budget_seconds"]:
                        reason = "WORKER_WALL_LIMIT"; break
                    check_freeze(frozen)
                    task = next(t for t in tasks if t["task_id"] == plan["task_id"])
                    identity = {**plan, "attempt_id": 1, "evidence_kind": "measurement",
                                "execution_source_sha256": frozen["execution_source_sha256"],
                                "protocol_sha256": sha_file(CASE / "specs/protocol.json"),
                                "model_revision": protocol["revision"], "initial_state_sha256": task["initial_state_sha256"],
                                "expected_initial_input_token_sha256": model_inputs[task["task_id"]]["input_token_sha256"],
                                "schema_sha256": digest(read(CASE / "specs" / ("grammar_" + task["family"] + ".json")))}
                    name = "_".join((task["task_id"], kv, plan["policy"]))
                    append(runroot / "attempts.jsonl", {"event": "START", "cell": name, "identity": identity, "utc": utc()})
                    episode = await run_episode(task, plan["policy"], adapter,
                              Path(runtime["private_root"]) / "fixtures" / stage_name.lower() / name, identity)
                    write_once(runroot / "cells" / (name + ".json"), episode)
                    completed.append(plan)
                    append(runroot / "attempts.jsonl", {"event": "END", "cell": name, "terminal_reason": episode["terminal_reason"], "utc": utc()})
                    print(json.dumps({"stage": stage_name, "completed": len(completed), "planned": len(planned), "cell": name,
                                      "terminal": episode["terminal_reason"], "requests": episode["budget"]["model_request_attempts"]}), flush=True)
                    if episode.get("requires_server_drain_check") or episode["observation_status"] == "UNOBSERVED":
                        reason = "INFRASTRUCTURE_OR_DRAIN_REVIEW_REQUIRED"; break
                after = _save_server_receipt(runroot, server, "after")
                if cache["parameter_sample_sha256"] != after["parameter_sample_sha256"]:
                    raise ValueError("Parameters changed during stage")
                write_once(runroot / (kv + "_launch.json"), {"command": server.public_command(), "mode": server.mode,
                                                          "ready_seconds": server.ready_seconds})
                if reason:
                    break
            finally:
                server.stop()
    except Exception as exc:
        reason = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        missing = [p for p in planned if p not in completed]
        wall = time.monotonic() - started
        write_once(runroot / "completion.json", {"stage": stage_name, "status": "COMPLETED" if not missing and not reason else "PARTIAL",
                    "planned": len(planned), "completed": len(completed), "missing": missing, "reason": reason,
                    "worker_wall_seconds": wall, "execution_source_sha256": frozen["execution_source_sha256"], "created_utc": utc()})
        append(journal, {"event": "END", "stage": stage_name, "utc": utc(), "worker_wall_seconds": wall,
                         "status": "COMPLETED" if not missing and not reason else "PARTIAL"})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explicit Case017 local execution; no downloads or weight quantization")
    parser.add_argument("command", choices=("prepare", "freeze", "p0", "p1", "seal-holdout", "p2"))
    parser.add_argument("--runtime", required=True, help="Existing private runtime JSON; never put private paths in public data")
    parser.add_argument("--label")
    args = parser.parse_args(argv)
    runtime = read(args.runtime)
    if args.command == "prepare":
        initialize(runtime)
    elif args.command == "freeze":
        from .prepare import execution_freeze
        if not args.label:
            parser.error("freeze requires a new --label")
        print(json.dumps(execution_freeze(args.label)), flush=True)
    elif args.command == "seal-holdout":
        from .prepare import prepare_split, check_freeze
        completion = read(CASE / "results/p1_r1/completion.json")
        if completion["status"] != "COMPLETED" or completion["completed"] != 64:
            raise ValueError("P1 must complete before sealing new HOLDOUT")
        check_freeze(read(CASE / "provenance/execution_freeze_p1.json"))
        tasks, _ = prepare_split("HOLDOUT", tokenizer_for(runtime))
        prior = read(CASE / "data/dev_tasks.json") + read(CASE / "data/cost_tasks.json")
        if {t["canonical_task_sha256"] for t in tasks} & {t["canonical_task_sha256"] for t in prior}:
            raise ValueError("HOLDOUT canonical overlap")
        if {t["initial_state_sha256"] for t in tasks} & {t["initial_state_sha256"] for t in prior}:
            raise ValueError("HOLDOUT state overlap")
        def record_ids(task):
            table = {"A": "orders", "B": "warehouses", "C": "documents"}[task["family"]]
            return set(task["initial_state"][table])
        if set.union(*(record_ids(t) for t in tasks)) & set.union(*(record_ids(t) for t in prior)):
            raise ValueError("HOLDOUT record-ID overlap")
        write_once(CASE / "provenance/holdout_seal.json", {"created_utc": utc(), "tasks": len(tasks), "outputs_observed": 0,
                   "p1_completion_sha256": sha_file(CASE / "results/p1_r1/completion.json"),
                   "policy_or_scorer_hashes_verified_unchanged": True, "canonical_overlap_with_DEV_COST": 0,
                   "initial_state_overlap_with_DEV_COST": 0, "record_id_overlap_with_DEV_COST": 0,
                   "holdout_tasks_sha256": sha_file(CASE / "data/holdout_tasks.json"),
                   "holdout_model_inputs_sha256": sha_file(CASE / "data/holdout_model_inputs.json")})
    elif args.command == "p0":
        asyncio.run(p0(runtime, args.label or "execution_freeze_p0"))
    else:
        asyncio.run(stage(runtime, args.command.upper()))


if __name__ == "__main__":
    main()
