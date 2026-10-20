import hashlib
import json
import pathlib
import tempfile
import unittest
from nvfp4_technical.audit import raw_request, counter_delta


class IndependentAudit(unittest.TestCase):
    def fixture(self, path):
        lines = ['data: {"choices":[{"index":0,"prompt_token_ids":[42,17],"token_ids":[9],"text":"a"}]}',
                 'data: {"choices":[{"index":0,"token_ids":[10],"text":"b","finish_reason":"length"}]}',
                 'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2}}', 'data: [DONE]']
        row = {'prompt_id': 'test', 'sse': [{'elapsed': i + 1, 'line': line} for i, line in enumerate(lines)],
               'raw_sse_sha256': hashlib.sha256(('\n'.join(lines) + '\n').encode()).hexdigest(),
               'ignore_eos': True, 'output_limit': 2, 'http_dispatch': 0.5, 'status': 'SUCCESS',
               'output_token_ids': [9, 10], 'text': 'ab'}
        path.write_text(json.dumps(row))
        return row, {'test': {'token_ids': [42, 17]}}

    def test_raw_endpoint_recomputed_and_client_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'request.json'
            _, prompts = self.fixture(path)
            actual = raw_request(path, prompts)
            self.assertTrue(actual['raw_success'])
            self.assertEqual((actual['ttft'], actual['decode_mean']), (0.5, 1))

    def test_saved_success_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'request.json'
            row, prompts = self.fixture(path)
            row['output_limit'] = 3
            path.write_text(json.dumps(row))
            with self.assertRaises(AssertionError):
                raw_request(path, prompts)

    def test_raw_alteration_fails_hash_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'request.json'
            row, prompts = self.fixture(path)
            row['sse'][0]['line'] = row['sse'][0]['line'].replace('"text":"a"', '"text":"x"')
            path.write_text(json.dumps(row))
            with self.assertRaises(AssertionError):
                raw_request(path, prompts)

    def test_missing_counter_remains_unobserved(self):
        self.assertIsNone(counter_delta('other 1', 'other 2', 'preemptions'))
        self.assertEqual(counter_delta('preemptions 1', 'preemptions 4', 'preemptions'), 3)


if __name__ == '__main__':
    unittest.main()
