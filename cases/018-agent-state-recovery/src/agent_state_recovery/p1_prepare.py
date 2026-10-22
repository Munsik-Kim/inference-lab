"""Local-only tokenizer/grammar preparation; no model load or GPU forward."""
import copy
import hashlib
import json
import random
from pathlib import Path
from .common import CASE_ROOT, canonical, digest, fresh_dir, read_json, write_new
from .generator import dev_tasks
from .public import initial_payload, grammar, definitions
from .live_client import wire_messages, references


def prepare(runtime, output, p0_original):
    references(runtime)
    from transformers import AutoTokenizer
    import xgrammar as xgr
    from agent_contract_cost.client import tokenize_messages
    root = fresh_dir(output)
    tokenizer = AutoTokenizer.from_pretrained(runtime["snapshot"], local_files_only=True, trust_remote_code=False)
    config = read_json(Path(runtime["snapshot"]) / "config.json")
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config["vocab_size"])
    compiler = xgr.GrammarCompiler(info, max_threads=2, cache_enabled=False)
    engine_schema = grammar()
    compiled = compiler.compile_json_schema(engine_schema)
    eos = tokenizer.eos_token_id
    probes = []
    def probe(name, value, expected):
        text = value if type(value) is str else json.dumps(value, separators=(",", ":"))
        matcher = xgr.GrammarMatcher(compiled, override_stop_tokens=[eos])
        ids = tokenizer.encode(text, add_special_tokens=False)
        prefix = all(matcher.accept_token(i) for i in ids)
        complete = matcher.accept_token(eos) if prefix else False
        probes.append({"name": name, "input_token_ids": ids, "expected": expected,
                       "observed": bool(prefix and complete), "PASS": bool(prefix and complete) == expected})
    def value(schema):
        kind = schema["type"]
        if kind == "object": return {k: value(v) for k, v in schema["properties"].items()}
        if kind == "array": return [value(schema["items"]) for _ in range(max(1, schema.get("minItems", 1)))]
        if kind == "integer": return max(schema.get("minimum", 0), 3)
        if kind == "boolean": return False
        return "ARBITRARY-WRONG-ID"
    for tool in definitions():
        probe("valid_public_envelope:" + tool["name"], {"tool": tool["name"], "arguments": value(tool["parameters"])}, True)
    for name, raw in (("multiple_objects", '{"tool":"read_history","arguments":{}} {}'),
                      ("trailing_prose", '{"tool":"read_history","arguments":{}} text'),
                      ("missing_arguments", '{"tool":"read_history"}'),
                      ("unknown_name", '{"tool":"not_registered","arguments":{}}'),
                      ("numeric_string", {"tool": "release_operation", "arguments": {"operation_id": "wrong", "expected_revision": "1"}}),
                      ("boolean_integer", {"tool": "release_operation", "arguments": {"operation_id": "wrong", "expected_revision": True}})):
        probe(name, raw, False)
    for qty in (9, 10, 11):
        probe("free_integer:" + str(qty), {"tool": "compensate_stock", "arguments": {"operation_id": "arbitrary-id", "source_restore": qty,
                                                    "destination_remove": 0, "expected_revision": 99}}, True)
    for label, op, revision, expected in (("empty_string", "", 1, False),
                                         ("long_string", "x" * 129, 1, False),
                                         ("negative_integer", "arbitrary", -1, False),
                                         ("over_max_integer", "arbitrary", 1000001, False)):
        probe(label, {"tool": "release_operation", "arguments": {"operation_id": op, "expected_revision": revision}}, expected)
    probe("over_max_array", {"tool": "read_records", "arguments": {"record_ids": ["arbitrary"] * 13}}, False)
    premature = xgr.GrammarMatcher(compiled, override_stop_tokens=[eos]).accept_token(eos)
    probes.append({"name": "premature_EOS", "expected": False, "observed": premature, "PASS": not premature})
    write_new(root / "grammar_cpu.json", {"status": "PASS" if all(p["PASS"] for p in probes) else "FAIL",
                 "probes": probes, "probe_count": len(probes), "schema_sha256": digest(engine_schema),
                 "host_only_constraints": [], "eos_token_id": eos, "compiled_grammar": str(compiled.grammar),
                 "scope": "actual local tokenizer and xgrammar token matcher; server mask not yet observed"})
    if not all(p["PASS"] for p in probes):
        raise ValueError("grammar token compatibility gate failed; receipts preserved")
    inputs = {}
    for task in dev_tasks():
        body = initial_payload(task)
        actual = tokenize_messages(wire_messages(body), tokenizer)
        if tokenizer.encode(actual["prompt_text"], add_special_tokens=False) != actual["token_ids"]:
            raise ValueError("independent rendered input reencoding mismatch")
        inputs[task["task_id"]] = {**actual, "wire_messages": wire_messages(body),
            "public_payload_sha256": digest(body), **copy.deepcopy(task["identity"])}
    write_new(root / "initial_inputs.json", inputs)
    bounds = []
    p0 = Path(p0_original) / "results/p0_r2"
    for path in sorted((p0 / "episodes").glob("*.json")):
        row = read_json(path)
        if "scripted_correct" not in row["episode_id"] and "guard_twice" not in row["episode_id"]:
            continue
        for request in row["requests"]:
            ids = tokenize_messages(wire_messages(request["model_payload"]), tokenizer)["token_ids"]
            output_ids = tokenizer.encode(request["response"]["raw_text"], add_special_tokens=False)
            # Allow one EOS in feasibility arithmetic; it is NOT a model measurement.
            bounds.append({"episode_id": row["episode_id"], "request_id": request["request_id"],
                           "input_tokens": len(ids), "scripted_output_tokens_plus_EOS": len(output_ids) + 1})
    write_new(root / "script_token_feasibility.json", {"evidence_kind": "CPU_SCRIPT_TOKENIZATION_NOT_MODEL_INFERENCE",
         "rows": bounds, "max_input_tokens": max(r["input_tokens"] for r in bounds),
         "max_scripted_output_plus_EOS": max(r["scripted_output_tokens_plus_EOS"] for r in bounds),
         "PASS": all(r["scripted_output_tokens_plus_EOS"] <= 512 and r["input_tokens"] + 512 <= 24576 for r in bounds)})
    if not all(r["scripted_output_tokens_plus_EOS"] <= 512 and r["input_tokens"] + 512 <= 24576 for r in bounds):
        raise ValueError("actual tokenizer feasibility exceeds frozen budget")
    rng = random.Random(61022011)
    planned = []
    for kv in ("K0", "K1"):
        block = [{"task_id": t["task_id"], "kv": kv, "policy": p, "episode_id": t["task_id"] + "_" + kv + "_" + p}
                 for t in dev_tasks() for p in ("R0", "R1")]
        rng.shuffle(block)
        planned.extend(block)
    write_new(root / "plan.json", {"seed": 61022011, "technical_server_order": ["K1", "K0"],
              "DEV_server_order": ["K0", "K1"], "planned": planned, "planned_count": 24,
              "HOLDOUT": "NOT_CREATED", "model_success_threshold": None,
              "actual_initial_token_ids_all_four_conditions": "same stored task input; verified again on actual requests"})
    return {"status": "PASS", "inputs": len(inputs), "grammar_probes": len(probes), "model_forward": False}
