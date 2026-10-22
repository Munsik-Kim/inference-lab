"""P2 envelope around unchanged independent ReferenceState/SSE replay."""
import datetime
import hashlib
from pathlib import Path
import resource
import time
from .common import CASE_ROOT, digest, read_json, write_new, fresh_dir, source_manifest
from .audit import audit_episode, equal, AuditMismatch
from .p2_contract import check_files, file_row, planned_keys, complete_token_sum


def validate_coverage(planned, entries):
    expected = {r["episode_id"] for r in planned_keys()}
    equal(sorted(r["episode_id"] for r in planned), sorted(expected), "planned HOLDOUT96")
    if len(planned) != 96:
        raise AuditMismatch("invalid planned denominator")
    ids, paths = [r["episode_id"] for r in entries], [r["path"] for r in entries]
    if len(set(ids)) != len(ids) or len(set(paths)) != len(paths) or set(ids) - expected:
        raise AuditMismatch("duplicate/unknown episode")
    return expected


def audit_run(run, output):
    wall, cpu = time.perf_counter(), time.process_time()
    run, output = Path(run), Path(output)
    if output.exists(): raise FileExistsError(output)
    reconstructed = fresh_dir(output.parent / (output.stem + "-replayed-states"))
    m = read_json(run / "run_manifest.json")
    expected = validate_coverage(m["planned"], m["episodes"])
    freeze = read_json(run / "execution_freeze.json")
    equal(source_manifest(), freeze["own"], "current P2 full working source")
    for ref in freeze["readonly_dependencies"]: check_files("/", [ref])
    equal(digest({k: freeze[k] for k in ("own", "readonly_dependencies", "runtime_configuration_sha256")}),
          freeze["execution_source_sha256"], "P2 execution identity")
    equal(freeze["execution_source_sha256"], m["execution_source_sha256"], "run source")
    equal(freeze["holdout_outputs_observed"], 0, "freeze before HOLDOUT")
    check_files(run, [freeze["input_seal"]])
    seal = read_json(run / "input_seal.json")
    check_files(run, [seal["input_manifest"], seal["plan"], seal["grammar_compilation"]])
    input_manifest = read_json(run / "input_manifest.json")
    check_files(run, input_manifest["files"] + [input_manifest["initial_inputs"], input_manifest["pre_generation_freeze"]])
    initial = read_json(run / "initial_inputs.json")
    equal(digest(initial), freeze["initial_inputs_sha256"], "initial actual IDs/messages")
    equal(freeze["prior_DEV_episodes"], 24, "prior observation disclosure")
    from .generator import dev_tasks
    from .p2_contract import scenario_group
    tasks = [read_json(run / r["path"]) for r in input_manifest["files"]]
    if len(tasks) != 24 or len({scenario_group(t) for t in tasks}) != 24 or {scenario_group(t) for t in tasks} & {scenario_group(t) for t in dev_tasks()}:
        raise AuditMismatch("HOLDOUT scenario-instance coverage")
    rows, actual, costs, first_ids = [], set(), [], {}
    for entry in m["episodes"]:
        if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts:
            raise AuditMismatch("unsafe episode path")
        check_files(run, [entry])
        row = read_json(run / entry["path"])
        equal(digest(row), entry["canonical_sha256"], "canonical original row")
        equal(row["episode_id"], entry["episode_id"], "episode identity")
        equal(row["execution_source_sha256"], freeze["execution_source_sha256"], "row source")
        equal(row["engine_schema"], freeze["grammar_schema"], "row schema")
        task = read_json(run / "inputs" / (row["task_id"] + ".json"))
        result = audit_episode(task, row)
        write_new(reconstructed / (row["episode_id"] + ".json"), result.pop("reconstructed_terminal_snapshot"))
        if row["requests"]:
            ids = row["requests"][0]["input_token_ids"]
            equal(ids, initial[row["task_id"]]["token_ids"], "all-four initial actual token IDs")
            first_ids.setdefault(row["task_id"], []).append({"kv": row["kv"], "policy": row["policy"], "sha256": digest(ids)})
        costs.append(row["accounting"])
        actual.add(row["episode_id"])
        rows.append({**result, "kv": row["kv"], "policy": row["policy"], "task_id": row["task_id"], "family": row["family"]})
    missing = expected - actual
    equal(sorted(r["episode_id"] for r in m["missing"]), sorted(missing), "explicit missing keys")
    if m["execution_status"] == "COMPLETE" and (missing or m["errors"]):
        raise AuditMismatch("incomplete run labelled complete")
    # Local ordering evidence, not an external time authority.
    seal_time = datetime.datetime.fromisoformat(seal["created_utc"])
    freeze_time = datetime.datetime.fromisoformat(freeze["created_utc"])
    journal_path = read_json(CASE_ROOT / "specs/p2_execution.json")["evidence"]
    import json
    requests = [json.loads(s) for s in (Path(journal_path) / "request_journal.jsonl").read_text().splitlines()]
    submissions = [r for r in requests if r["type"] == "REQUEST_SUBMITTED" and r["category"] == "HOLDOUT"]
    if submissions:
        first = min(datetime.datetime.fromisoformat(r["utc"]) for r in submissions)
        if not seal_time < freeze_time <= first:
            raise AuditMismatch("seal/freeze/request local time ordering")
    first_response = None
    if submissions:
        request_id = min(submissions, key=lambda r: r["monotonic"])["request_id"]
        matches = list(run.glob('HOLDOUT_*_transport/' + request_id + '.json'))
        if len(matches) != 1: raise AuditMismatch("first raw response mapping")
        raw = read_json(matches[0])
        first_response = {"path": str(matches[0].relative_to(run)), "raw_sse_sha256": raw["raw_sse_sha256"],
            "first_role_chunk_seconds": raw["first_role_chunk_seconds"],
            "first_generated_token_id_event_seconds": raw["first_generated_token_id_event_seconds"],
            "HTTP_end_seconds": raw["http_end_seconds"],
            "scope": "SSE offsets relative to its HTTP client call; submission UTC and code order establish local seal-before-stream, not external timestamp attestation"}
    result = {"status": "PASS", "execution_status": m["execution_status"], "coverage": "COMPLETE" if not missing else "PARTIAL",
        "episodes": len(rows), "missing": sorted(missing), "endpoint_differences": 0, "raw_trace_differences": 0,
        "rows": rows, "model_request_attempts": sum(r["actual"]["model_request_attempts"] for r in costs),
        "responses_received": sum(r["actual"]["model_responses_received"] for r in costs),
        "actual_total_input_tokens": complete_token_sum([r["actual_input_tokens"] for r in costs]),
        "actual_total_output_tokens": complete_token_sum([r["actual_output_tokens"] for r in costs]),
        "observed_input_tokens": sum(r["observed_input_tokens"] for r in costs),
        "observed_output_tokens": sum(r["observed_output_tokens"] for r in costs),
        "first_input_pair_ids": first_ids, "local_seal_freeze_before_submission": True, "first_raw_stream": first_response,
        "source_sha256": freeze["execution_source_sha256"], "new_model_requests": 0,
        "shared": ["strict parser", "serialization", "public input/schema builder", "wire rendering"],
        "independent": ["ReferenceState transitions", "protection/freshness", "fixed continuation programs", "SSE assembler", "token/budget accounting"],
        "CPU_cost": {"wall_seconds": time.perf_counter() - wall, "process_CPU_seconds": time.process_time() - cpu,
            "peak_RSS_KiB_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "actual_tokenizer_reencoding": "separate tokencheck receipt; this audit verifies saved IDs/history/raw SSE"}
    write_new(output, result)
    return result


def tokencheck(run, runtime_path, output):
    # Adaptation of the preserved P1 r2 checker: original schema bytes/order and
    # the run's pre-output compiled grammar, not sorted schema from freeze JSON.
    run, output = Path(run), Path(output)
    if output.exists(): raise FileExistsError(output)
    from transformers import AutoTokenizer
    import xgrammar as xgr
    runtime = read_json(runtime_path)
    tokenizer = AutoTokenizer.from_pretrained(runtime["snapshot"], local_files_only=True, trust_remote_code=False)
    config = read_json(Path(runtime["snapshot"]) / "config.json")
    frozen = read_json(run / "execution_freeze.json")
    schema_path = CASE_ROOT / "specs/grammar.json"
    source = next(r for r in frozen["own"]["files"] if r["path"] == "specs/grammar.json")
    check_files(CASE_ROOT, [source])
    schema = read_json(schema_path)
    equal(schema, frozen["grammar_schema"], "schema semantic identity")
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config["vocab_size"]), max_threads=2).compile_json_schema(schema)
    equal(str(compiled.grammar), read_json(run / "grammar_cpu.json")["compiled_grammar"], "original ordered compiled grammar")
    wall, cpu = time.perf_counter(), time.process_time()
    rows = []
    for path in sorted((run / "episodes").glob('*.json')):
        row = read_json(path)
        for req in row["requests"]:
            response = req["response"]
            if response["status"] != "OK":
                rows.append({"episode_id": row["episode_id"], "request_id": req["request_id"], "status": "NOT_FULLY_OBSERVABLE"})
                continue
            text = tokenizer.apply_chat_template(req["wire_messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(response["output_token_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            matcher = xgr.GrammarMatcher(compiled, override_stop_tokens=[tokenizer.eos_token_id])
            prefix = all(matcher.accept_token(i) for i in response["output_token_ids"])
            match = ids == req["input_token_ids"] == response["prompt_token_ids"] and text == req["rendered_prompt"] == response["prompt_text"]
            decode_match = decoded == response["raw_token_text"]
            rows.append({"episode_id": row["episode_id"], "request_id": req["request_id"],
                "status": "PASS" if prefix and match and decode_match else "FAIL", "actual_reencoding": match,
                "special_token_decode": decode_match, "all_original_prefixes_EOS": prefix})
    result = {"status": "PASS" if rows and all(r["status"] == "PASS" for r in rows) else "FAIL_OR_PARTIAL", "rows": rows,
        "requests": len(rows), "new_model_requests": 0, "original_ordered_schema_file": file_row(schema_path),
        "pre_output_compiled_grammar_match": True, "CPU_cost": {"wall_seconds": time.perf_counter() - wall,
            "process_CPU_seconds": time.process_time() - cpu, "peak_RSS_KiB_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "scope": "Existing local tokenizer/xgrammar CPU only; no model load/forward or external GPU reproduction"}
    write_new(output, result)
    return result
