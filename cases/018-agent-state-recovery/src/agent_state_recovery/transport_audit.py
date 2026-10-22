"""Separate stdlib SSE assembler and recorded-token integrity checks.

No import of the Case017 assembler/HTTP client or tokenizer. Actual re-tokenizing
and special-token decode are a separate existing-runtime CPU verification.
"""
import json
from .common import canonical, digest, text_hash


def assemble(lines):
    text, ids, prompt_ids, prompt_text, usage = "", [], None, None, None
    finish, stop, response_id, done = None, None, None, False
    for row in lines:
        line = row["line"] if type(row) is dict else row
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            if done:
                raise ValueError("duplicate DONE")
            done = True
            continue
        if done:
            raise ValueError("SSE data after DONE")
        event = json.loads(data)
        if "error" in event:
            raise ValueError("SSE error frame")
        if event.get("id"):
            if response_id not in (None, event["id"]):
                raise ValueError("response ID changed")
            response_id = event["id"]
        if event.get("prompt_token_ids") is not None:
            if prompt_ids is not None and prompt_ids != event["prompt_token_ids"]:
                raise ValueError("prompt IDs changed")
            prompt_ids = event["prompt_token_ids"]
        if event.get("prompt_text") is not None:
            if prompt_text is not None and prompt_text != event["prompt_text"]:
                raise ValueError("prompt text changed")
            prompt_text = event["prompt_text"]
        if event.get("usage") is not None:
            if usage is not None and usage != event["usage"]:
                raise ValueError("usage changed")
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("index") != 0:
                raise ValueError("unexpected choice")
            delta = choice.get("delta") or {}
            if any(delta.get(k) for k in ("tool_calls", "function_call", "reasoning", "reasoning_content")):
                raise ValueError("native tool or reasoning parser used")
            text += delta.get("content") or ""
            new_ids = choice.get("token_ids") or []
            if any(type(i) is not int for i in new_ids):
                raise ValueError("noninteger token")
            ids.extend(new_ids)
            if choice.get("finish_reason") is not None:
                finish, stop = choice["finish_reason"], choice.get("stop_reason")
    return {"content": text, "ids": ids, "prompt_ids": prompt_ids, "prompt_text": prompt_text,
            "usage": usage, "finish": finish, "stop": stop, "done": done}


def verify(response, request):
    r = assemble(response["raw_sse_lines"])
    if not r["done"] or r["usage"] is None or r["prompt_ids"] is None:
        raise ValueError("incomplete raw SSE/usage/IDs")
    checks = [(r["content"], response["raw_text"]), (r["ids"], response["output_token_ids"]),
              (r["prompt_ids"], request["input_token_ids"]), (r["prompt_text"], request["rendered_prompt"]),
              (r["finish"], response["server_finish_reason"]), (r["stop"], response["server_stop_reason"]),
              (r["usage"]["prompt_tokens"], len(r["prompt_ids"])),
              (r["usage"]["completion_tokens"], len(r["ids"])),
              (digest(r["prompt_ids"]), request["input_token_sha256"]),
              (text_hash(r["content"]), response["raw_utf8_sha256"]),
              (digest(r["ids"]), response["output_token_sha256"]),
              (digest([x["line"] for x in response["raw_sse_lines"]]), response["raw_sse_sha256"])]
    if any(canonical(a) != canonical(b) for a, b in checks):
        raise ValueError("raw SSE/token/usage consistency mismatch")
    if len(r["ids"]) > request["effective_output_cap"]:
        raise ValueError("output cap violation")
    body = response["request_body"]
    if (body["messages"] != request["wire_messages"] or body["structured_outputs"]["json"] != request["engine_schema"]
            or body["max_tokens"] != request["effective_output_cap"]
            or any(k in body for k in ("tools", "tool_choice", "guided_json", "allowed_token_ids"))):
        raise ValueError("wire contract differs")
    expected = {"temperature": 0, "top_p": 1, "n": 1, "seed": 61022007,
                "ignore_eos": False, "return_token_ids": True, "return_prompt_text": True,
                "chat_template_kwargs": {"enable_thinking": False}}
    if any(body.get(k) != value for k, value in expected.items()):
        raise ValueError("sampling configuration differs")
    return r
