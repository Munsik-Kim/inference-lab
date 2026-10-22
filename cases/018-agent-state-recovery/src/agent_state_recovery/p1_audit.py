"""CPU-only P1 raw replay; no live endpoint, model, tokenizer or GPU queries."""
import hashlib
from pathlib import Path
from .common import CASE_ROOT, canonical, digest, read_json, write_new, fresh_dir, source_manifest
from .audit import audit_episode, equal, AuditMismatch


def audit_run(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    reconstructed = fresh_dir(output.parent / (output.stem + "-replayed-states"))
    m = read_json(run / "run_manifest.json")
    planned = m["planned"]
    expected = {f"DEV-{family}{i:02d}_{kv}_{policy}" for family in "ABC" for i in range(2)
                for kv in ("K0", "K1") for policy in ("R0", "R1")}
    equal(sorted(r["episode_id"] for r in planned), sorted(expected), "planned DEV24 identity")
    if len(planned) != 24 or len({r["episode_id"] for r in m["episodes"]}) != len(m["episodes"]):
        raise AuditMismatch("duplicate/missing planned key")
    freeze = read_json(run / "execution_freeze.json")
    equal(source_manifest(), freeze["own"], "current P1 scientific/adapter/input source")
    for ref in freeze["readonly_dependencies"]:
        p = Path(ref["path"])
        equal(hashlib.sha256(p.read_bytes()).hexdigest(), ref["sha256"], "readonly helper source")
    identity_body = {k: freeze[k] for k in ("own", "readonly_dependencies", "runtime_configuration_sha256")}
    equal(digest(identity_body), freeze["execution_source_sha256"], "execution identity")
    equal(freeze["execution_source_sha256"], m["execution_source_sha256"], "run/source identity")
    equal(freeze["dev_outputs_observed"], 0, "freeze before DEV")
    initial = read_json(run / "initial_inputs.json")
    equal(digest(initial), freeze["initial_inputs_sha256"], "frozen initial IDs/messages")
    rows, actual_ids, requests, responses, inputs, outputs = [], set(), 0, 0, 0, 0
    for e in m["episodes"]:
        if e["episode_id"] not in expected or ".." in Path(e["path"]).parts:
            raise AuditMismatch("unknown/unsafe cell path")
        path = run / e["path"]
        equal(hashlib.sha256(path.read_bytes()).hexdigest(), e["sha256"], "raw episode file bytes")
        row = read_json(path)
        equal(digest(row), e["canonical_sha256"], "canonical episode identity")
        equal(row["episode_id"], e["episode_id"], "episode path mapping")
        equal(row["execution_source_sha256"], freeze["execution_source_sha256"], "per-episode source")
        equal(row["engine_schema"], freeze["grammar_schema"], "per-episode grammar")
        task = read_json(run / "inputs" / (row["task_id"] + ".json"))
        result = audit_episode(task, row)
        write_new(reconstructed / (row["episode_id"] + ".json"), result.pop("reconstructed_terminal_snapshot"))
        if row["requests"]:
            equal(row["requests"][0]["input_token_ids"], initial[row["task_id"]]["token_ids"], "actual initial pair IDs")
        count = row["accounting"]["actual"]
        requests += count["model_request_attempts"]
        responses += count["model_responses_received"]
        inputs += row["accounting"]["actual_input_tokens"] or 0
        outputs += row["accounting"]["actual_output_tokens"] or 0
        actual_ids.add(row["episode_id"])
        rows.append({**result, "kv": row["kv"], "policy": row["policy"], "task_id": row["task_id"], "family": row["family"]})
    missing = expected - actual_ids
    equal(sorted(c["episode_id"] for c in m["missing"]), sorted(missing), "explicit missing coverage")
    if m["execution_status"] == "COMPLETE" and missing:
        raise AuditMismatch("incomplete run labelled complete")
    receipt = {"status": "PASS", "execution_status": m["execution_status"], "coverage": "COMPLETE" if not missing else "PARTIAL",
        "episodes": len(rows), "missing": sorted(missing), "endpoint_differences": 0, "raw_trace_differences": 0,
        "rows": rows, "model_request_attempts": requests, "responses_received": responses,
        "observed_complete_input_tokens": inputs, "observed_complete_output_tokens": outputs,
        "new_model_requests": 0, "actual_tokenizer_reencoding": "separate existing-runtime CPU receipt; this command validates saved IDs/history/SSE",
        "shared": ["strict parser", "serialization", "public schema/input builder", "wire public-declaration rendering"],
        "independent": ["ReferenceState transitions", "business/protection/freshness", "fixed continuation programs", "raw SSE assembler", "actual token/budget accounting"],
        "source_sha256": freeze["execution_source_sha256"]}
    write_new(output, receipt)
    return receipt
