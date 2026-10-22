"""One registered HOLDOUT generation and CPU controls/seal; no model loading."""
import datetime
from pathlib import Path
import time
from .common import CASE_ROOT, digest, fresh_dir, read_json, write_new
from .generator import generate, dev_tasks
from .public import initial_payload, grammar
from .client import FakeClient
from .runner import run_episode
from .audit import audit_episode
from .live_client import wire_messages, references
from .p1 import identity, _journal
from .p2_contract import file_row, p1_preconditions, parent_debit, remaining_budget, planned_keys, scenario_group, validate_runtime


def prepare(output, runtime_path, evidence, parent):
    root = fresh_dir(output)
    evidence, parent = Path(evidence), Path(parent)
    runtime = read_json(runtime_path)
    validate_runtime(runtime, read_json(parent / "runtime.json"))
    gate = p1_preconditions(parent)
    write_new(root / "P1_gate.json", gate)
    planned = planned_keys()
    write_new(root / "plan.json", {"phase": "P2_HOLDOUT", "seed": 61022012, "planned": planned,
        "planned_count": 96, "technical_server_order": ["K1", "K0"], "model_server_order": ["K0", "K1"],
        "model_success_threshold": None})
    candidate = identity(runtime)
    pre = {**candidate, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generation_calls_before": 0, "holdout_outputs_observed": 0, "prior_DEV_episodes": 24,
        "prior_DEV_responses": 180, "prior_TECH_attempts": 6,
        "plan_file": file_row(root / "plan.json"), "P1_gate_file": file_row(root / "P1_gate.json"),
        "registered_generator_seed": 61022002, "parent_worker_debit_seconds": parent_debit(parent),
        "remaining_worker_seconds_before_P2": remaining_budget(parent_debit(parent))}
    write_new(root / "pre_generation_freeze.json", pre)
    data = CASE_ROOT / "data/holdout"
    if data.exists():
        raise FileExistsError("HOLDOUT already exists; no regeneration")
    journal = evidence / "generation_journal.jsonl"
    if journal.exists():
        raise FileExistsError("generation already attempted; no regeneration")
    _journal(journal, {"type": "GENERATION_STARTED", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "source": candidate["execution_source_sha256"], "seed": 61022002, "call_number": 1})
    tasks = generate("HOLDOUT", gate)  # The sole actual HOLDOUT generator call.
    data.mkdir()
    (root / "inputs").mkdir()
    for task in tasks:
        write_new(data / (task["task_id"] + ".json"), task)
        write_new(root / "inputs" / (task["task_id"] + ".json"), task)
    _journal(journal, {"type": "GENERATED", "task_count": len(tasks), "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    expected = {f"HOLDOUT-{f}{i:02d}" for f in "ABC" for i in range(8)}
    groups = [scenario_group(t) for t in tasks]
    dev = {scenario_group(t) for t in dev_tasks()}
    input_checks = {"tasks24_unique": len(tasks) == 24 and {t["task_id"] for t in tasks} == expected,
                    "scenario_groups_unique": len(set(groups)) == 24, "DEV_groups_disjoint": not (set(groups) & dev),
                    "all_faults_change_state": all(t["identity"]["checkpoint"] != t["identity"]["error_state"] for t in tasks)}
    write_new(root / "input_checks.json", {"checks": input_checks, "groups": dict(zip([t["task_id"] for t in tasks], groups)),
        "scope": "Three registered family templates intentionally shared; distinct task/scenario instances, not independent companies or new held-out templates"})
    if not all(input_checks.values()):
        raise ValueError("registered generator produced duplicate/invalid tasks; raw retained, no regeneration")
    write_new(data / "MANIFEST.json", {"split": "HOLDOUT", "seed": 61022002, "generator_version": "inventory-recovery-v1",
        "files": [file_row(data / (t["task_id"] + ".json"), CASE_ROOT) for t in tasks]})
    controls = fresh_dir(root / "CPU_controls")
    control_rows, budget_rows, feasibility = [], [], []
    references(runtime)
    from transformers import AutoTokenizer
    import xgrammar as xgr
    tokenizer = AutoTokenizer.from_pretrained(runtime["snapshot"], local_files_only=True, trust_remote_code=False)
    config = read_json(Path(runtime["snapshot"]) / "config.json")
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config["vocab_size"]), max_threads=2).compile_json_schema(grammar())
    parent_grammar = read_json(parent / "preparation02/grammar_cpu.json")
    if str(compiled.grammar) != parent_grammar["compiled_grammar"]:
        raise ValueError("original ordered schema compiled grammar changed")
    write_new(root / "grammar_cpu.json", {"status": "PASS", "compiled_grammar": str(compiled.grammar),
        "original_schema_file": file_row(CASE_ROOT / "specs/grammar.json"), "P1_compiled_match": True,
        "boundary_tests_reused": file_row(parent / "preparation02/grammar_cpu.json"), "new_model_requests": 0})
    inputs = {}
    for task in tasks:
        body = initial_payload(task)
        wire = wire_messages(body)
        rendered = tokenizer.apply_chat_template(wire, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        inputs[task["task_id"]] = {"wire_messages": wire, "prompt_text": rendered, "token_ids": ids,
            "input_token_sha256": digest(ids), "public_payload_sha256": digest(body), **task["identity"]}
        for mode, start, policy in (("clean", "clean", "R0"), ("no_recovery", "error", "R0"),
                                    ("scripted_correct", "error", "R0"), ("guard_twice", "error", "R1")):
            ep = task["task_id"] + "_CPU_" + mode
            row = run_episode(task, FakeClient(mode), None, start=start, policy=policy, episode_id=ep)
            result = audit_episode(task, row)
            expected_q = mode != "no_recovery"
            if result["independently_recomputed"]["Q"] is not expected_q or result["independently_recomputed"]["W"] is not expected_q:
                raise ValueError("HOLDOUT control failed; no regeneration")
            write_new(controls / (ep + ".json"), row)
            entry = {"episode_id": ep, "mode": mode, "Q": expected_q, "W": expected_q,
                     "audit": "PASS", "requests_simulated": row["accounting"]["synthetic"]["model_request_attempts"]}
            (budget_rows if mode == "guard_twice" else control_rows).append(entry)
            for request in row["requests"]:
                text = tokenizer.apply_chat_template(wire_messages(request["model_payload"]), tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False)
                n = len(tokenizer.encode(text, add_special_tokens=False))
                out = len(tokenizer.encode(request["response"]["raw_text"], add_special_tokens=False)) + 1
                feasibility.append({"episode_id": ep, "request_id": request["request_id"], "input_tokens": n,
                                    "script_output_plus_EOS": out, "PASS": n + 512 <= 24576 and out <= 512})
    if not all(r["PASS"] for r in feasibility) or max(r["requests_simulated"] for r in budget_rows) > 16:
        raise ValueError("sealed task budget feasibility failed")
    write_new(root / "CPU_controls.json", {"evidence_kind": "SYNTHETIC_CONTRACT_TEST", "controls": control_rows,
        "controls_count": 72, "independent_audit": "PASS", "control_terminal_continuation_branches": 144,
        "control_checkpoint_reference_branches": 144, "separate_guard_budget_scripts": budget_rows,
        "separate_guard_budget_script_count": 24, "new_model_requests": 0})
    write_new(root / "script_token_feasibility.json", {"rows": feasibility,
        "max_input_tokens": max(r["input_tokens"] for r in feasibility),
        "max_script_output_plus_EOS": max(r["script_output_plus_EOS"] for r in feasibility),
        "max_correct_requests": max(r["requests_simulated"] for r in control_rows if r["mode"] == "scripted_correct"),
        "max_guard_correct_requests": max(r["requests_simulated"] for r in budget_rows), "status": "PASS",
        "scope": "CPU script feasibility, not model inference; fixed per-episode limits unchanged"})
    write_new(root / "initial_inputs.json", inputs)
    if identity(runtime) != candidate:
        raise ValueError("P2 source changed during generation/preparation")
    manifest = {"files": [file_row(p, root) for p in sorted((root / "inputs").glob('*.json'))],
        "task_count": 24, "per_family": 8, "initial_inputs": file_row(root / "initial_inputs.json", root),
        "pre_generation_freeze": file_row(root / "pre_generation_freeze.json", root)}
    write_new(root / "input_manifest.json", manifest)
    seal = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "holdout_outputs_observed": 0, "prior_DEV_episodes": 24, "prior_DEV_responses": 180, "prior_TECH_attempts": 6,
        "generator_calls": 1, "execution_source_sha256": candidate["execution_source_sha256"],
        "input_manifest": file_row(root / "input_manifest.json", root), "data_manifest": file_row(data / "MANIFEST.json", CASE_ROOT),
        "grammar_compilation": file_row(root / "grammar_cpu.json", root), "plan": file_row(root / "plan.json", root),
        "analysis_config": read_json(CASE_ROOT / "specs/protocol.json")["analysis"],
        "frozen_holdout_fields": read_json(CASE_ROOT / "specs/protocol.json")["holdout_frozen_fields"],
        "time_scope": "local self-record only, not external timestamp attestation"}
    write_new(root / "input_seal.json", seal)
    _journal(journal, {"type": "SEALED", "utc": seal["created_utc"], "seal_sha256": file_row(root / "input_seal.json")["sha256"],
                     "holdout_outputs_observed": 0})
    return {"status": "SEALED_CPU_READY", "tasks": 24, "planned_model_episodes": 96,
            "CPU_controls": 72, "additional_CPU_guard_budget_scripts": 24, "model_requests": 0}
