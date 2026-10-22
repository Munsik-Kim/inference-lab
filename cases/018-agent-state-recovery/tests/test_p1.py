"""SYNTHETIC adapter/transport contract tests; no actual generation/GPU/network."""
import copy
import json
import time
import unittest
from agent_state_recovery.common import digest, read_json, CASE_ROOT, canonical
from agent_state_recovery.generator import dev_tasks
from agent_state_recovery.live_client import wire_messages, request_body
from agent_state_recovery.runner import run_episode
from agent_state_recovery.transport_audit import assemble


class P1AdapterContracts(unittest.TestCase):
    def test_drain_uses_exact_total_gauges_not_waiting_reason_prefix(self):
        from agent_state_recovery.server import activity_gauges
        text = 'vllm:num_requests_running{engine="0"} 0.0\nvllm:num_requests_waiting{engine="0"} 0.0\nvllm:num_requests_waiting_by_reason{reason="x"} 0.0\n'
        r = activity_gauges(text)
        self.assertEqual(len(r), 2)
        self.assertTrue(all(v == 0 for v in r.values()))
        self.assertEqual(activity_gauges(text.replace('running{engine="0"} 0.0', 'running{engine="0"} 1.0'))['vllm:num_requests_running'], 1.0)

    def test_impl_observer_counts_direct_calls_and_restores_dispatch(self):
        from types import SimpleNamespace
        from agent_state_recovery.inspector import attention_impl_observer
        class Impl:
            def forward(self, layer, query, key, value, cache, attn_metadata, output): return output
        impl = Impl()
        layer = SimpleNamespace(kv_cache_dtype="synthetic", impl=impl)
        model = SimpleNamespace(named_modules=lambda: [("layer0", layer)])
        with attention_impl_observer(model, lambda x: type(x) is int, lambda x: x == 7) as rows:
            self.assertEqual(impl.forward(layer, None, None, None, None, object(), 7), 7)
        self.assertEqual(rows["layer0"]["calls"], 1)
        self.assertEqual(rows["layer0"]["metadata_present_calls"], 1)
        self.assertTrue(rows["layer0"]["finite"])
        self.assertNotIn("forward", impl.__dict__)

    def test_technical_context_skip_compiled_restores_after_error(self):
        from contextlib import contextmanager
        from types import SimpleNamespace
        from agent_state_recovery.inspector import technical_uncompiled_context
        @contextmanager
        def original(**kwargs): yield kwargs
        module = SimpleNamespace(set_forward_context=original)
        with self.assertRaises(RuntimeError):
            with technical_uncompiled_context(module):
                with module.set_forward_context(skip_compiled=False) as args:
                    self.assertTrue(args["skip_compiled"])
                    raise RuntimeError("synthetic observer error")
        self.assertIs(module.set_forward_context, original)

    def test_public_wire_rendering_no_native_tools_or_hidden_state(self):
        from agent_state_recovery.public import initial_payload
        t = dev_tasks()[0]
        payload = initial_payload(t)
        text = canonical(wire_messages(payload))
        self.assertNotIn('"oracle"', text)
        self.assertNotIn('"checkpoint":', text)
        body = request_body(wire_messages(payload), payload["output_schema"], 512, "synthetic")
        self.assertNotIn("tools", body)
        self.assertNotIn("guided_json", body)
        self.assertEqual(body["structured_outputs"]["json"], payload["output_schema"])
        self.assertFalse(body["ignore_eos"])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_sse_role_usage_done_and_control_token_accounting(self):
        frames = [{"id": "x", "prompt_token_ids": [8, 9], "prompt_text": "p", "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                  {"id": "x", "choices": [{"index": 0, "delta": {"content": "{}"}, "token_ids": [11, 151645], "finish_reason": "stop"}]},
                  {"id": "x", "choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}}]
        rows = ["data: " + json.dumps(f) for f in frames] + ["data: [DONE]"]
        r = assemble(rows)
        self.assertEqual(r["content"], "{}")
        self.assertEqual(r["ids"], [11, 151645])
        self.assertEqual(r["usage"]["completion_tokens"], 2)
        self.assertTrue(r["done"])

    def test_partial_sse_is_not_a_complete_response(self):
        partial = assemble(['data: {"choices":[{"index":0,"delta":{"content":"{"},"token_ids":[8]}]}'])
        self.assertFalse(partial["done"])
        self.assertIsNone(partial["usage"])
        with self.assertRaises(ValueError):
            assemble(["data: [DONE]", "data: [DONE]"])

    def test_native_or_multiple_choice_sse_is_rejected(self):
        for choice in [{"index": 1, "delta": {}}, {"index": 0, "delta": {"tool_calls": [{"id": "wrong"}]}}]:
            with self.assertRaises(ValueError):
                assemble(["data: " + json.dumps({"choices": [choice]})])

    def test_live_accounting_uses_ids_not_raw_text_length_or_simulated_units(self):
        class SyntheticLive:
            is_live = True
            schema = {}
            def __init__(self): self.count = 0
            def remaining_seconds(self): return 500
            def prepare(self, body): return {"token_ids": [4, 5, 6], "input_token_sha256": digest([4, 5, 6]), "prompt_text": "fake", "messages": wire_messages(body)}
            def respond(self, body, timeout_seconds, **kwargs):
                self.count += 1
                if self.count > 1: raise StopIteration
                raw = canonical({"tool": "read_history", "arguments": {}})
                return {"status": "OK", "raw_text": raw, "output_token_ids": [20, 151645], "prompt_token_ids": [4, 5, 6],
                        "elapsed_seconds": 0.01, "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
        row = run_episode(dev_tasks()[0], SyntheticLive())
        self.assertEqual(row["accounting"]["actual_input_tokens"], 3)
        self.assertEqual(row["accounting"]["actual_output_tokens"], 2)
        self.assertIsNone(row["accounting"]["simulated_output_tokens"])

    def test_live_context_limit_prevents_submission_without_truncation(self):
        class NeverSubmit:
            is_live = True
            schema = {}
            def remaining_seconds(self): return 500
            def prepare(self, body): return {"token_ids": [1] * 24576, "input_token_sha256": "synthetic", "prompt_text": "unchanged", "messages": wire_messages(body)}
            def respond(self, *args, **kwargs): raise AssertionError("must not submit")
        row = run_episode(dev_tasks()[0], NeverSubmit())
        self.assertEqual(row["terminal_reason"], "CONTEXT_LIMIT")
        self.assertEqual(row["accounting"]["actual"]["model_request_attempts"], 0)

    def test_cap_and_deadline_contracts_fail_closed(self):
        for cap in (0, 513, True):
            with self.assertRaises(ValueError): request_body([], {}, cap, "synthetic")
        class AlreadyExpired:
            is_live = True
            def remaining_seconds(self): return 0
        row = run_episode(dev_tasks()[0], AlreadyExpired())
        self.assertEqual(row["terminal_reason"], "WORKER_DEADLINE")
        self.assertEqual(row["accounting"]["actual"]["model_request_attempts"], 0)

    def test_synthetic_raw_sse_real_accounting_path_has_independent_endpoint_replay(self):
        from agent_state_recovery.client import FakeClient
        from agent_state_recovery.public import grammar
        from agent_state_recovery.audit import audit_episode
        from agent_state_recovery.common import text_hash
        class SyntheticTransport:
            is_live = True
            schema = grammar()
            def __init__(self): self.fake = FakeClient("clean")
            def remaining_seconds(self): return 500
            def prepare(self, body): return {"token_ids": [4, 5, 6], "input_token_sha256": digest([4, 5, 6]), "prompt_text": "synthetic", "messages": wire_messages(body)}
            def respond(self, body, timeout_seconds, *, cap, request_id, prepared):
                raw = self.fake.respond(body, timeout_seconds)["raw_text"]
                ids = list(raw.encode()) + [151645]  # Synthetic IDs, not tokenizer measurements.
                usage = {"prompt_tokens": 3, "completion_tokens": len(ids)}
                frames = [{"id": "synthetic", "prompt_token_ids": [4, 5, 6], "prompt_text": "synthetic", "choices": []},
                          {"id": "synthetic", "choices": [{"index": 0, "delta": {"content": raw}, "token_ids": ids, "finish_reason": "stop", "stop_reason": 151645}], "usage": usage}]
                lines = [{"line": "data: " + json.dumps(f), "elapsed_seconds": 0.001} for f in frames] + [{"line": "data: [DONE]", "elapsed_seconds": 0.002}]
                return {"status": "OK", "raw_text": raw, "raw_utf8_sha256": text_hash(raw), "output_token_ids": ids,
                        "prompt_token_ids": [4, 5, 6], "prompt_text": "synthetic", "usage": usage,
                        "elapsed_seconds": 0.002, "server_finish_reason": "stop", "server_stop_reason": 151645,
                        "output_token_sha256": digest(ids), "raw_sse_lines": lines,
                        "raw_sse_sha256": digest([x["line"] for x in lines]),
                        "request_body": request_body(prepared["messages"], self.schema, cap, request_id)}
        t = dev_tasks()[0]
        row = run_episode(t, SyntheticTransport(), start="clean")
        row["engine_schema"] = grammar()
        self.assertTrue(row["reported_endpoint"]["W"])
        self.assertEqual(audit_episode(t, row)["status"], "PASS")
