"""CPU input preparation and immutable execution registration."""
import copy
import json
import subprocess
import tempfile
import time
from pathlib import Path

from .common import CASE, REPO, digest, read, sha_file, utc, write_once
from .fixtures import GENERATOR_VERSION, generate_tasks, gold_actions
from .schemas import G_SCHEMA_SCOPE, server_schema
from .tools import tool_definitions


def register_protocol():
    protocol = {
        "schema": 1, "case": "017", "version": "v1.1-local-registration-r1",
        "registered_utc": utc(), "base_commit": "740a26f013c4a516b8322c0c07e077a7667bd8f1",
        "user_instruction_extent": "Supplied instructions sections 0 through 5; later-stage details are new local preregistration, not attributed to the user.",
        "model": "Qwen/Qwen3-4B-Instruct-2507", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "weights": "BF16", "engine": "vLLM 0.32.0", "kv": {"K0": "auto verified BF16", "K1": "native nvfp4"},
        "endpoint": "/v1/chat/completions", "native_tool_calls": False,
        "policies": ["C0", "G", "N", "E"], "grammar_scope": G_SCHEMA_SCOPE,
        "grammar_backend": "xgrammar", "host_only_constraints": ["array uniqueItems"],
        "grammar_no_gold": "Only public tool names, public fields/types/static bounds; no task IDs, state, amounts, oracle or action order.",
        "limits": {"max_new_tokens": 512, "model_request_attempts": 8, "parsed_tool_and_final_attempts": 6,
                   "generated_tokens": 4096, "retry_per_episode": 1, "request_seconds": 180, "episode_seconds": 900,
                   "client_context_tokens": 24576, "initial_prompt_target_tokens": "approximately 2048; actual counts and offsets recorded without truncation"},
        "sixth_action_policy": "Final consumes an attempt. Sixth final is admissible; sixth nonfinal executes then TOOL_CALL_LIMIT. No seventh generation because no registered action remains. Record final_opportunity_blocked_by_registered_attempt_budget.",
        "sampling": {"temperature": 0, "top_p": 1, "thinking": False, "ignore_eos": False, "n": 1},
        "serving": {"episode_concurrency": 1, "tensor_parallel": 1, "pool_bytes_target": 4294967296,
                    "server_max_model_len": 28672, "max_num_seqs": 4, "max_num_batched_tokens": 2048,
                    "prefix_cache": False, "async_scheduling": False, "mode": "GRAPH if both P0 gates pass; otherwise a common pre-P1 amendment"},
        "stages": {"P0": {"purpose": "technical support, raw observability, schema enforcement, separate synthetic smoke only",
                              "planned_model_request_ceiling": 32},
                   "P1": {"split": "DEV", "tasks": 8, "family_counts": {"A": 3, "B": 3, "C": 2}, "episodes": 64},
                   "P2": {"split": "HOLDOUT", "tasks": 24, "family_counts": {"A": 8, "B": 8, "C": 8}, "episodes": 192},
                   "P3": {"split": "COST", "tasks": 6, "gpu_status": "NOT_RUN_DESIGN_ONLY"}},
        "seeds": {"DEV": 6102001, "HOLDOUT": 6102002, "COST": 6102003, "P1_order": 6102011,
                  "P2_order": 6102012, "bootstrap": 6102013, "engine": 6102007},
        "order": {"P1": ["K0", "K1"], "P2": ["K1", "K0"],
                  "within_kv": "fixed seeded shuffle of task/policy cells, no metric-based exclusions"},
        "holdout_seal": "Generate only after DEV policy/scorer freeze and P1 completion; generator and fixed seed are frozen before P1. No policy/scorer tuning from HOLDOUT outputs.",
        "worker_budget_seconds": 7200,
        "budget_scope": "Self-imposed operational ceiling for all P0/P1/P2 GPU worker attempts including load, compile, failed attempts, metrics and I/O; CPU implementation/data preparation separately timed. It is not a predicted completion time.",
        "infrastructure_policy": "No automatic cell regeneration. Unobserved transport/raw/server/OOM failures remain missing; normal timeout/budget exhaustion is observed functional failure. Stop if server drain is uncertain.",
        "primary_endpoint": "VERIFIED_WORKFLOW_SUCCESS from actual state, obligations, pre-generation observations, latest validation and strict final",
        "analysis": {"status": "exploratory paired engineering evaluation; no confirmatory p-values or equivalence claim",
                     "policy_pairs_within_kv": ["G-C0", "N-C0", "E-C0", "E-N"],
                     "kv_pair_within_policy": "K1-K0", "bootstrap_replicates": 10000,
                     "bootstrap_unit": "task, stratified by family, all eight cells share sampled task indices",
                     "interval": "pointwise percentile 95%; missing counts, loss/recovery and bounds separate",
                     "N_vs_E": "independent free runs; not isolated causal effect of feedback information"},
        "publication": {"stage": False, "commit": False, "push": False, "PR": False, "merge": False, "Pages": False},
    }
    write_once(CASE / "specs/protocol.json", protocol)
    return protocol


def prepare_split(split, tokenizer=None):
    from .prompts import initial_messages
    from .client import tokenize_messages
    path = CASE / "data" / (split.lower() + "_tasks.json")
    tasks = generate_tasks(split)
    records = []
    for task in tasks:
        messages = initial_messages(task)
        model_input = {"task_id": task["task_id"], "messages": messages,
                       "messages_sha256": digest(messages), "schema_sha256": digest(server_schema(tool_definitions(task["family"]))) }
        if tokenizer is not None:
            model_input.update(tokenize_messages(messages, tokenizer))
            model_input["input_tokens"] = len(model_input["token_ids"])
        records.append(model_input)
        if len(gold_actions(task)) > 6:
            raise ValueError("Contract-impossible fixture")
    model_path = CASE / "data" / (split.lower() + "_model_inputs.json")
    if path.exists() or model_path.exists():
        if not path.exists() or not model_path.exists() or read(path) != tasks or read(model_path) != records:
            raise ValueError("Cannot resume changed or incomplete input registration")
        return tasks, records
    write_once(path, tasks)
    write_once(CASE / "data" / (split.lower() + "_model_inputs.json"), records)
    write_once(CASE / "provenance" / (split.lower() + "_input_manifest.json"), {
        "split": split, "generator_version": GENERATOR_VERSION, "tasks": len(tasks), "created_utc": utc(),
        "tasks_sha256": sha_file(path), "model_inputs_sha256": sha_file(CASE / "data" / (split.lower() + "_model_inputs.json")),
        "task_records": [{"task_id": t["task_id"], "family": t["family"], "template_id": t["template_id"],
                          "initial_state_sha256": t["initial_state_sha256"], "oracle_sha256": t["oracle_sha256"]} for t in tasks],
        "model_input_allowlist": ["task_id", "prompt", "public tool definitions"],
        "oracle_initial_state_goal_excluded_from_model_payload": True,
    })
    return tasks, records


def execution_freeze(label):
    names = ["__init__.py", "common.py", "contracts.py", "policies.py", "fixtures.py", "tools.py", "scoring.py",
             "schemas.py", "prompts.py", "runner.py", "client.py", "server_manager.py", "p0_inspector.py", "prepare.py", "experiment.py"]
    files = [{"path": "src/agent_contract_cost/" + name,
              "sha256": sha_file(CASE / "src/agent_contract_cost" / name),
              "bytes": (CASE / "src/agent_contract_cost" / name).stat().st_size} for name in names]
    for folder in ("specs", "data"):
        for path in sorted((CASE / folder).glob("*.json")):
            files.append({"path": str(path.relative_to(CASE)), "sha256": sha_file(path), "bytes": path.stat().st_size})
    originals = ["followups/nvfp4-technical-r1/src/nvfp4_technical/__init__.py",
                 "followups/nvfp4-technical-r1/src/nvfp4_technical/worker.py",
                 "src/kv_agent_bench/__init__.py", "src/kv_agent_bench/profiler_extension.py",
                 "src/kv_agent_bench/worker_inspection.py"]
    for name in originals:
        rel = "cases/016-kv-agent-bench/" + name
        path = REPO / rel
        files.append({"root_path": rel, "sha256": sha_file(path), "bytes": path.stat().st_size,
                      "role": "READ_ONLY_ORIGINAL_IMPORT"})
    for name in ("model_identity.json", "environment_identity.json", "grammar_cpu_validation_r2.json"):
        path = CASE / "provenance" / name
        files.append({"path": str(path.relative_to(CASE)), "sha256": sha_file(path), "bytes": path.stat().st_size,
                      "role": "PINNED_IDENTITY_RECEIPT"})
    seal = CASE / "provenance/holdout_seal.json"
    if seal.exists():
        files.append({"path": str(seal.relative_to(CASE)), "sha256": sha_file(seal), "bytes": seal.stat().st_size,
                      "role": "SEALED_HOLDOUT_IDENTITY"})
    source_files = [entry for entry in files if entry.get("path", "").startswith("src/") or "root_path" in entry]
    identity = digest(source_files)
    record = {"label": label, "created_utc": utc(), "execution_source_sha256": identity,
              "registration_bundle_sha256": digest(files), "execution_source_files": source_files,
              "git_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
              "git_commit_is_base_not_execution_source": True, "git_commit_created": False, "files": files,
              "source_hash_and_model_revision_are_different_identifiers": True}
    write_once(CASE / "provenance" / (label + ".json"), record)
    return record


def check_freeze(freeze):
    for entry in freeze["files"]:
        path = REPO / entry["root_path"] if "root_path" in entry else CASE / entry["path"]
        if sha_file(path) != entry["sha256"]:
            raise ValueError("Frozen execution dependency changed: " + entry.get("root_path", entry.get("path")))
