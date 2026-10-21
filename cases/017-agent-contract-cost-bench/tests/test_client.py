"""Synthetic transport/config boundaries; these tests make no model request."""
import asyncio
import copy
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from agent_contract_cost.client import (StreamAssembly, build_payload, decoded_raw,
                                        loopback, parse_sse, request_cap, tokenize_messages, digest)
from agent_contract_cost.server_manager import OwnedServer
from agent_contract_cost.p0_inspector import validate_support


class ClientTests(unittest.TestCase):
    def test_chat_template_batchencoding_normalized_and_batch_rejected(self):
        class FakeTokenizer:
            def __init__(self, ids):
                self.ids = ids
                self.tokenize_options = None
            def apply_chat_template(self, messages, **kwargs):
                if kwargs['tokenize']:
                    self.tokenize_options = kwargs
                    return self.ids
                return 'rendered prompt'
        for ids in ([1, 2, 3], {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}):
            tokenizer = FakeTokenizer(ids)
            row = tokenize_messages([{'role': 'user', 'content': 'test'}], tokenizer)
            self.assertEqual(row['token_ids'], [1, 2, 3])
            self.assertEqual(row['input_token_sha256'], digest([1, 2, 3]))
            self.assertFalse(tokenizer.tokenize_options['return_dict'])
        for ids in ({'input_ids': [[1, 2]]}, {'attention_mask': [1]}, [1, True]):
            with self.assertRaises(ValueError):
                tokenize_messages([{'role': 'user', 'content': 'test'}], FakeTokenizer(ids))

    def test_shared_messages_schema_is_only_arm_field(self):
        messages = [{'role': 'system', 'content': 'Public contract'}, {'role': 'user', 'content': 'Task'}]
        schema = {'type': 'object', 'properties': {'tool': {'enum': ['read', 'final']}}}
        free, grammar = build_payload(messages), build_payload(messages, schema)
        self.assertEqual(free, {k: v for k, v in grammar.items() if k != 'structured_outputs'})
        self.assertEqual(grammar['structured_outputs'], {'json': schema})
        self.assertNotIn('tools', free)
        self.assertNotIn('guided_json', grammar)
        grammar['messages'][0]['content'] = 'changed'
        self.assertEqual(messages[0]['content'], 'Public contract')
        with self.assertRaises(ValueError):
            build_payload([{'role': 'user', 'content': 'x', 'oracle': 17}])

    def test_sse_ids_roles_usage_and_done_are_distinct(self):
        events = [
            {'id': 'r1', 'prompt_token_ids': [11, 12], 'prompt_text': 'rendered',
             'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'token_ids': []}]},
            {'id': 'r1', 'choices': [{'index': 0, 'delta': {'content': '{'}, 'token_ids': [31]}]},
            {'id': 'r1', 'choices': [{'index': 0, 'delta': {'content': '}'}, 'token_ids': [32, 151645]}]},
            {'id': 'r1', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop', 'stop_reason': 151645}]},
            {'id': 'r1', 'choices': [], 'usage': {'prompt_tokens': 2, 'completion_tokens': 3}},
        ]
        result = parse_sse([': keepalive', *['data: ' + json.dumps(e) for e in events], 'data: [DONE]'])
        self.assertEqual(result['output_token_ids'], [31, 32, 151645])
        self.assertEqual(result['prompt_token_ids'], [11, 12])
        self.assertEqual(result['content'], '{}')
        self.assertEqual(result['server_stop_reason'], 151645)
        self.assertEqual(len(result['token_events']), 2)
        self.assertTrue(result['stream_done'])

    def test_native_extraction_and_changed_prompt_rejected(self):
        assembly = StreamAssembly()
        with self.assertRaisesRegex(ValueError, 'Native'):
            assembly.add({'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0}]}}]}, 0)
        assembly.add({'prompt_token_ids': [1]}, 0)
        with self.assertRaisesRegex(ValueError, 'Prompt'):
            assembly.add({'prompt_token_ids': [2]}, 0)
        with self.assertRaisesRegex(ValueError, 'single choice'):
            assembly.add({'choices': [{'index': 1, 'delta': {}}]}, 0)

    def test_decode_retains_special_tokens_and_caps_fail_closed(self):
        class FakeTokenizer:
            def decode(self, ids, **kwargs):
                self.ids, self.kwargs = ids, kwargs
                return '{}<|im_end|>'
        tokenizer = FakeTokenizer()
        self.assertEqual(decoded_raw({'output_token_ids': [1, 2]}, tokenizer), '{}<|im_end|>')
        self.assertFalse(tokenizer.kwargs['skip_special_tokens'])
        self.assertEqual(request_cap(512, 50, 24576, 24526), (50,
            {'configured_cap': 512, 'remaining_episode_tokens': 50, 'remaining_context_tokens': 50},
            ['remaining_episode_tokens', 'remaining_context_tokens']))
        self.assertEqual(request_cap(512, 1, 24576, 24577)[0], -1)
        for base in ('https://127.0.0.1:8173', 'http://example.com', 'http://127.0.0.1:8173/other'):
            with self.assertRaises(ValueError):
                loopback(base)

    def test_launch_contract_common_dtype_only_and_owned_paths(self):
        runtime = {'python': '/existing/bin/python', 'snapshot': '/existing/model',
                   'case_root': '/case017', 'original_case': '/case016',
                   'private_root': '/private', 'port': 8173}
        b16, f4 = OwnedServer(runtime, 'K0', 'b16'), OwnedServer(runtime, 'K1', 'f4')
        a, b = b16.command(), f4.command()
        index = a.index('--kv-cache-dtype') + 1
        a[index] = b[index]
        self.assertEqual(a, b)
        self.assertNotIn('--tool-call-parser', a)
        self.assertNotIn('--enable-auto-tool-choice', a)
        self.assertIn('--structured-outputs-config', a)
        self.assertIn('--no-enable-prefix-caching', a)
        self.assertIn('<PINNED_LOCAL_SNAPSHOT>', b16.public_command())
        self.assertIn('/private/engine_cache', b16.environment()['VLLM_CACHE_ROOT'])
        with self.assertRaises(ValueError):
            OwnedServer({**runtime, 'port': 8152}, 'K0', 'conflict')
        with self.assertRaises(ValueError):
            OwnedServer(runtime, 'K0', '../outside')

    def test_cpu_imports_no_torch_or_transformers(self):
        source = pathlib.Path(__file__).resolve().parents[1] / 'src'
        code = """import sys
class Block:
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] in ('torch', 'transformers', 'vllm', 'httpx'):
   raise AssertionError('CPU import attempted ' + name)
sys.meta_path.insert(0, Block())
import agent_contract_cost.client
import agent_contract_cost.server_manager
import agent_contract_cost.p0_inspector
"""
        completed = subprocess.run([sys.executable, '-B', '-c', code],
            env={**os.environ, 'PYTHONPATH': str(source), 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1'},
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_actual_storage_gate_distinguishes_nvfp4_scales(self):
        receipt = {'cuda_capability': [12, 0], 'layers': [{'is_kvcache_nvfp4': True}] * 36,
                   'cache_config': {'enable_prefix_caching': False},
                   'cache_tensors': [{'dtype': 'torch.uint8'}], 'structured_outputs_backend': 'xgrammar',
                   'model_dtype': 'torch.bfloat16', 'nvfp4_views': [{'scale_dtype': ['torch.float8_e4m3fn'] * 2,
                   'data_scale_storage_shared': True}] * 36}
        self.assertTrue(validate_support(receipt, 'K1'))
        changed = copy.deepcopy(receipt)
        changed['cache_tensors'][0]['dtype'] = 'torch.bfloat16'
        with self.assertRaises(AssertionError):
            validate_support(changed, 'K1')
        changed = copy.deepcopy(receipt)
        changed['nvfp4_views'][0]['scale_dtype'] = ['torch.bfloat16'] * 2
        with self.assertRaises(AssertionError):
            validate_support(changed, 'K1')


if __name__ == '__main__':
    unittest.main()
