import asyncio
import importlib
import json
import sys
import unittest
from nvfp4_technical.client import compact_sse, expand_sse, parse_events, payload, request
from nvfp4_technical.metrics import metric_scalar, difference, interval_mean
from nvfp4_technical.common import token_hash


class Contracts(unittest.TestCase):
    def setUp(self):
        self.prompt = {'prompt_id': 'fixture', 'token_ids': [42, 17], 'input_token_sha256': token_hash([42, 17])}

    def test_sse_lossless_prompt_deduplication(self):
        line = 'data: {"choices":[{"index":0,"prompt_token_ids":[42, 17],"token_ids":[3]}]}'
        compact = compact_sse(line, self.prompt)
        self.assertNotIn('line', compact)
        self.assertEqual(line, expand_sse(compact, {'fixture': self.prompt}))

    def test_actual_prompt_mismatch_rejected(self):
        with self.assertRaises(AssertionError):
            compact_sse('data: {"prompt_token_ids": [42, 18]}', self.prompt)

    def test_token_events_exclude_empty_usage_chunks(self):
        lines = [(0, 'data: {"choices":[{"index":0,"token_ids":[],"prompt_token_ids":[42,17]}]}'),
                 (2, 'data: {"choices":[{"index":0,"text":"a","token_ids":[10]}]}'),
                 (3, 'data: {"choices":[{"index":0,"text":"b","token_ids":[11],"finish_reason":"length"}]}'),
                 (4, 'data: {"choices":[],"usage":{"completion_tokens":2,"prompt_tokens":2}}'),
                 (5, 'data: [DONE]')]
        row = parse_events(lines, [42, 17])
        self.assertEqual(row['output_token_ids'], [10, 11])
        self.assertEqual(row['text'], 'ab')
        self.assertEqual([x['elapsed'] for x in row['token_events']], [2, 3])
        self.assertTrue(row['done'] and row['prompt_ids_confirmed'])

    def test_unexpected_choice_not_merged(self):
        with self.assertRaises(AssertionError):
            parse_events([(0, 'data: {"choices":[{"index":1,"token_ids":[1]}]}')], [])

    def test_missing_prompt_is_not_confirmed(self):
        self.assertFalse(parse_events([(0, 'data: [DONE]')], [42, 17])['prompt_ids_confirmed'])

    def test_fixed_ids_and_natural_end_are_separate(self):
        timed, sanity = payload(self.prompt, 128, True), payload(self.prompt, 64, False)
        self.assertTrue(timed['ignore_eos'])
        self.assertFalse(sanity['ignore_eos'])
        self.assertFalse(timed['add_special_tokens'])
        self.assertEqual(timed['prompt'], self.prompt['token_ids'])

    def test_context_overflow_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            payload({'token_ids': [1] * 28600}, 128, True)

    def test_prometheus_missing_is_not_zero(self):
        self.assertIsNone(metric_scalar('unrelated 1', 'missing'))
        self.assertEqual(metric_scalar('q{r="a"} 2\nq{r="b"} 3', 'q'), 5)

    def test_interval_mean_uses_count_not_request_guess(self):
        before = 'decode_sum 10\ndecode_count 2'
        after = 'decode_sum 25\ndecode_count 5'
        self.assertEqual(interval_mean(before, after, 'decode'), 5)

    def test_counter_reset_is_not_a_speedup(self):
        with self.assertRaises(ValueError):
            difference('q 10', 'q 2', 'q')

    def test_timeout_keeps_attempt_separate_from_response(self):
        class Client:
            def stream(self, *args, **kwargs):
                class Stream:
                    async def __aenter__(self):
                        raise TimeoutError('synthetic transport timeout')
                    async def __aexit__(self, *args):
                        pass
                return Stream()
        async def example():
            import time
            return await request(Client(), 'http://127.0.0.1:1', self.prompt, {'evidence_kind': 'SYNTHETIC_CONTRACT_TEST'}, asyncio.Semaphore(1), time.monotonic())
        row = asyncio.run(example())
        self.assertEqual((row['model_request_attempts'], row['responses_received']), (1, 0))
        self.assertEqual(row['status'], 'TIMEOUT')

    def test_cpu_import_does_not_import_gpu_framework(self):
        before = set(sys.modules)
        for name in ('client', 'metrics', 'server', 'runner'):
            importlib.import_module('nvfp4_technical.' + name)
        self.assertFalse({'torch', 'transformers', 'vllm'} & (set(sys.modules) - before))


if __name__ == '__main__':
    unittest.main()
