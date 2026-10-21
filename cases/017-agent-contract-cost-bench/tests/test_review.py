"""Synthetic display fixtures only; no fixture below is a measured model run."""
import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from agent_contract_cost.review import digest, group_rows, intern_messages, load_review, page, render


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.script_ids = []
    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.script_ids.append(dict(attrs).get('id'))


def fixture_tree(root):
    """Unit-test records mimic storage schema, never leave the temporary root."""
    raw = {'task_id': 'TEST-A0', 'family': 'A', 'kv': 'K0', 'policy': 'C0',
           'episode_id': 'TEST-ONLY', 'requests': [], 'test_evidence_kind': 'SYNTHETIC_CONTRACT_TEST'}
    audited = {'task_id': raw['task_id'], 'family': 'A', 'stage': 'P1', 'kv': 'K0', 'policy': 'C0',
               'evidence_kind': 'measurement', 'audit_pass': True, 'observation_status': 'OBSERVED',
               'verified': False, 'state_correct': True, 'final_correct': False,
               'terminal_reason': 'CONTRACT_FAILURE', 'source_sha256': digest(raw),
               'counters': {'model_request_attempts': 3, 'model_responses_received': 3,
                            'generated_output_tokens': 87, 'contract_rejections': 1}, 'wall_seconds': 2.5}
    run = root / 'results/p1_r1'
    (run / 'audit').mkdir(parents=True)
    (run / 'cells').mkdir()
    (run / 'analysis').mkdir()
    (run / 'cells/cell.json').write_text(json.dumps(raw))
    (run / 'audit/audited_rows.json').write_text(json.dumps([audited]))
    summary = {'registered_tasks': 1, 'planned_cells': 8, 'status': 'PARTIAL',
               'evidence_scope': 'model_measurement_stored_trace_audit', 'groups': group_rows([audited], 1)}
    (run / 'analysis/summary.json').write_text(json.dumps(summary))
    return raw, audited, run


class ReviewTests(unittest.TestCase):
    def test_missing_results_not_run_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            case = root / 'case'
            case.mkdir()
            data = load_review(case)
            self.assertEqual([r['status'] for r in data['runs']], ['NOT_RUN', 'NOT_RUN'])
            self.assertEqual(data['episodes'], [])
            receipt = render(case, root / 'preview')
            self.assertEqual(receipt['new_model_requests'], 0)
            self.assertIn('NOT_RUN', (root / 'preview/ko.html').read_text())
            with self.assertRaises(FileExistsError):
                render(case, root / 'preview')

    def test_scalar_counts_from_audited_rows_and_stale_summary_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            raw, row, run = fixture_tree(root)
            data = load_review(root)
            group = data['runs'][0]['groups'][0]
            self.assertEqual((group['verified_success'], group['observed'], group['planned']), (0, 1, 1))
            self.assertEqual(group['model_request_attempts'], 3)
            self.assertEqual(group['generated_output_tokens'], 87)
            summary = json.loads((run / 'analysis/summary.json').read_text())
            summary['groups'][0]['verified_success'] = 1
            (run / 'analysis/summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'Summary disagrees'):
                load_review(root)

    def test_duplicate_and_changed_raw_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            raw, row, run = fixture_tree(root)
            (run / 'audit/audited_rows.json').write_text(json.dumps([row, row]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_review(root)
            (run / 'audit/audited_rows.json').write_text(json.dumps([row]))
            raw['policy'] = 'G'
            (run / 'cells/cell.json').write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, 'no independently audited'):
                load_review(root)

    def test_generated_script_injection_remains_data_and_core_table_is_static(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            fixture_tree(root)
            data = load_review(root)
            payload = '</script><script>alert("model output")</script><img src=x onerror=alert(1)>'
            data['episodes'][0]['raw']['final_state'] = {'generated_string': payload}
            for lang in ('ko', 'en'):
                rendered = page(data, lang)
                self.assertNotIn(payload, rendered)
                self.assertIn('\\u003c/script\\u003e', rendered)
                parser = Scripts()
                parser.feed(rendered)
                self.assertEqual(parser.script_ids, ['review-data', None])
                self.assertIn('<table>', rendered)
                self.assertIn('<noscript>', rendered)
                self.assertIn('textContent', rendered)

    def test_synthetic_rows_cannot_enter_measurement_pages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            raw, row, run = fixture_tree(root)
            row['evidence_kind'] = 'SYNTHETIC_CONTRACT_TEST'
            (run / 'audit/audited_rows.json').write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, 'Synthetic'):
                load_review(root)

    def test_compact_view_omits_transport_duplicates_preserves_original_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            raw, audited, run = fixture_tree(root)
            raw['requests'] = [{'generation_turn': 0, 'messages': [{'role': 'user', 'content': 'task'}],
                'available_result_event_ids': ['event-0000'], 'state_digest': 'before', 'budget_before': {'model_request_attempts': 2},
                'request_plan': {'max_tokens': 512}, 'input_token_ids': [999] * 100, 'input_token_sha256': 'input-hash',
                'parse': {'accepted': False}, 'feedback': 'retry', 'event': {'result': {'ok': False}},
                'response': {'raw_text': 'malformed', 'raw_token_text': 'malformed<|im_end|>',
                    'output_token_ids': [1, 151645], 'raw_utf8_sha256': 'raw-hash', 'usage': {'completion_tokens': 2},
                    'status': 'OK', 'finish_reason': 'stop', 'stop_reason': 151645, 'elapsed_seconds': .25,
                    'raw_sse_lines': ['large-transport-sentinel'] * 100, 'sse_events': ['large-events'],
                    'prompt_token_ids': [999] * 100, 'request_body': {'messages': ['large-request']}}}]
            path = run / 'cells/cell.json'
            path.write_text(json.dumps(raw))
            before = path.read_bytes()
            audited['source_sha256'] = digest(raw)
            (run / 'audit/audited_rows.json').write_text(json.dumps([audited]))
            data = load_review(root)
            row = data['episodes'][0]['raw']['requests'][0]
            # Display schema v2 interns full message objects. Reconstruct before
            # comparing; the original model input and raw file stay unchanged.
            reconstructed = [data['message_pool'][ref] for ref in row['message_refs']]
            self.assertEqual(reconstructed, raw['requests'][0]['messages'])
            self.assertEqual(row['messages_sha256'], digest(raw['requests'][0]['messages']))
            self.assertNotIn('messages', row)
            self.assertEqual(row['available_result_event_ids'], ['event-0000'])
            self.assertEqual(row['response']['raw_text'], 'malformed')
            self.assertEqual(row['response']['finish_reason'], 'stop')
            self.assertEqual(row['response']['stop_reason'], 151645)
            self.assertNotIn('input_token_ids', row)
            for key in ('raw_sse_lines', 'sse_events', 'prompt_token_ids', 'request_body'):
                self.assertNotIn(key, row['response'])
            self.assertEqual(before, path.read_bytes())
            self.assertFalse(data['episodes'][0]['raw']['display_projection']['lossless_raw'])
            receipt = render(root, root / 'preview')
            published = json.loads((root / 'preview/data.json').read_text())
            href = published['episodes'][0]['source_href']
            self.assertEqual(href, '../results/p1_r1/cells/cell.json')
            self.assertEqual((root / 'preview' / href).resolve(), path.resolve())
            self.assertEqual(receipt['episodes'], 1)

    def test_message_pool_deduplicates_exact_utf8_and_preserves_order(self):
        messages = [{'role': 'system', 'content': '한글\n<&> ☃'},
                    {'role': 'user', 'content': 'same'}, {'role': 'user', 'content': 'same'}]
        episodes = [{'raw': {'requests': [{'messages': copy.deepcopy(messages)},
                                         {'messages': list(reversed(copy.deepcopy(messages)))}]}}]
        before = copy.deepcopy(episodes)
        pool, receipt = intern_messages(episodes)
        self.assertEqual(len(pool), 2)
        self.assertEqual(receipt['pooled_request_count'], 2)
        self.assertLess(receipt['message_canonical_bytes_after'], receipt['message_canonical_bytes_before'])
        for old, compact in zip(before[0]['raw']['requests'], episodes[0]['raw']['requests']):
            reconstructed = [pool[ref] for ref in compact['message_refs']]
            self.assertEqual(reconstructed, old['messages'])
            self.assertEqual(digest(reconstructed), compact['messages_sha256'])
            for original, restored in zip(old['messages'], reconstructed):
                self.assertEqual(original['content'].encode(), restored['content'].encode())

    def test_review_cli_and_import_have_no_gpu_dependencies(self):
        source = pathlib.Path(__file__).resolve().parents[1] / 'src'
        script = """import sys
class Block:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','transformers','vllm','httpx','xgrammar'):
   raise AssertionError(name)
sys.meta_path.insert(0,Block())
from agent_contract_cost.review import main
try:main(['--help'])
except SystemExit as e:assert e.code==0
"""
        import os
        result = subprocess.run([sys.executable, '-B', '-c', script],
                                env={**os.environ, 'PYTHONPATH': str(source), 'CUDA_VISIBLE_DEVICES': ''},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
