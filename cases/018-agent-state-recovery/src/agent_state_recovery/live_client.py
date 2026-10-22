"""P1-only tokenizer/HTTP adapter. Heavy imports are explicit and lazy.

Case017's immutable content/SSE transport is a read-only dependency, not its
runner/scorer. Public definitions are rendered into system text, never native
tool_calls. All four conditions use the same serialization and schema.
"""
import asyncio
import copy
import json
import sys
import time
from pathlib import Path
from .common import canonical, digest, text_hash, write_new, CASE_ROOT


def references(runtime):
    paths = [str(CASE_ROOT.parent / "017-agent-contract-cost-bench" / "src"),
             str(Path(runtime["original_case"]) / "followups/nvfp4-technical-r1/src"),
             str(Path(runtime["original_case"]) / "src")]
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)
    return paths


def wire_messages(body):
    messages = copy.deepcopy(body["messages"])
    messages[0]["content"] += "\nPublic API declarations (arguments are JSON objects):\n" + canonical(body["tools"])
    return messages


def request_body(messages, schema, cap, request_id):
    if type(cap) is not int or not 1 <= cap <= 512:
        raise ValueError("invalid output cap")
    return {"model": "case018-qwen3-4b", "messages": copy.deepcopy(messages),
            "temperature": 0, "top_p": 1, "n": 1, "seed": 61022007,
            "max_tokens": cap, "ignore_eos": False, "stream": True, "stream_interval": 1,
            "stream_options": {"include_usage": True}, "return_token_ids": True, "return_prompt_text": True,
            "skip_special_tokens": True, "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"json": copy.deepcopy(schema)}, "request_id": request_id}


class LiveClient:
    is_live = True

    def __init__(self, runtime, base, tokenizer, schema, output, deadline, journal, category="DEV", max_requests=416):
        references(runtime)
        from agent_contract_cost.client import chat, tokenize_messages
        self.chat, self.tokenize = chat, tokenize_messages
        self.runtime, self.base, self.tokenizer, self.schema = runtime, base, tokenizer, schema
        self.output, self.deadline, self.journal = Path(output), deadline, Path(journal)
        self.category, self.max_requests = category, max_requests
        self.output.mkdir(parents=True, exist_ok=False)
        self.attempts = self.responses = 0

    def remaining_seconds(self):
        return max(0.0, self.deadline - time.monotonic())

    def prepare(self, body):
        messages = wire_messages(body)
        r = self.tokenize(messages, self.tokenizer)
        if self.tokenizer.encode(r["prompt_text"], add_special_tokens=False) != r["token_ids"]:
            raise ValueError("chat template rendered-text/ID mismatch")
        return {**r, "messages": messages}

    def respond(self, body, timeout_seconds, *, cap, request_id, prepared):
        with self.journal.open("r", encoding="utf-8") as stream:
            submitted = [json.loads(line) for line in stream if line.strip()]
        total = sum(r.get("type") == "REQUEST_SUBMITTED" for r in submitted)
        category_count = sum(r.get("type") == "REQUEST_SUBMITTED" and r.get("category") == self.category for r in submitted)
        if category_count >= (32 if self.category == "TECH" else 384):
            raise RuntimeError("registered category request ceiling")
        if total >= self.max_requests or self.remaining_seconds() <= 0:
            raise RuntimeError("global request/wall budget exhausted before submission")
        timeout = min(timeout_seconds, self.remaining_seconds())
        payload = request_body(prepared["messages"], self.schema, cap, request_id)
        if self.prepare(body) != prepared:
            raise ValueError("prepared input changed before submission")
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(canonical({"type": "REQUEST_SUBMITTED", "category": self.category,
                                    "request_id": request_id, "request_body_sha256": digest(payload),
                                    "input_token_sha256": prepared["input_token_sha256"], "monotonic": time.monotonic()}) + "\n")
            stream.flush()
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
        # Save transport raw before the harness can parse or execute any action.
        write_new(self.output / (request_id + ".json"), response)
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(canonical({"type": "RESPONSE_CLOSED", "category": self.category, "request_id": request_id,
                                    "status": response["status"], "received": response["model_responses_received"],
                                    "observed_output_ids": len(response.get("output_token_ids", [])),
                                    "monotonic": time.monotonic()}) + "\n")
            stream.flush()
        return response


def tokenize_only(runtime):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(runtime["snapshot"], local_files_only=True, trust_remote_code=False)
