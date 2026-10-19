"""Synthetic boundary tests; these are not new model recovery observations."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EXEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXEC))
from state import P2, MeasuredSession, digest, inputs, read, replay_cell
from json_recovery_p2.upstream import Registry, token_hash


class ExecutionContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='p2-gpu-contract-')
        self.root = Path(self.temp.name)
        points, snapshots, scenarios, _ = inputs()
        self.point = points[0]; self.snap = snapshots[self.point['checkpoint_id']]
        self.scenario = scenarios[self.point['scenario_id']]
        self.ids = next(x['input_token_ids'] for x in read(P2 / 'results/cpu_preparation/retokenization.json')['rows'] if x['checkpoint_id'] == self.point['checkpoint_id'])

    def tearDown(self):
        self.temp.cleanup()

    def session(self, name='base', arm='C0'):
        return MeasuredSession(self.point, self.snap, self.scenario, self.root / name, method='D16', condition=arm)

    def gpu(self, obj, text, request_id='synthetic-request-1'):
        payload, sidecar = obj.next_request()
        ids = self.ids if obj.budget.model_calls == 2 else [1, 2, 3]
        return {'raw_text': text, 'text': text, 'raw_utf8_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'output_token_ids': [151645], 'output_tokens': 1, 'input_token_ids': ids,
            'input_tokens': len(ids), 'input_token_sha256': token_hash(ids), 'messages_tools_hash': digest(payload),
            'grammar_applied': sidecar['grammar_active'], 'grammar_steps': [], 'parameters_unchanged': True,
            'request_id': request_id, 'method': 'D16', 'EOS': True, 'reached_token_cap': False,
            'seconds': 0.1, 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}

    def malformed(self, obj):
        return obj.consume_measured(self.gpu(obj, '<tool_call>{bad'), elapsed_worker_seconds=0.2)

    def test_gpu_grammar_function_AST_matches_registered_CPU(self):
        def functions(path):
            return [ast.dump(x, include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x, ast.FunctionDef)]
        self.assertEqual(functions(P2 / 'src/json_recovery_p2/grammar.py'), functions(EXEC / 'grammar_gpu.py'))

    def test_original_CPU_bytes_and_old_refusal_remain(self):
        for name, identity in read(EXEC / 'cpu_original_files.json')['files'].items():
            self.assertEqual(hashlib.sha256((P2 / name).read_bytes()).hexdigest(), identity['sha256'])
        self.assertIn("return 3", (P2 / 'src/json_recovery_p2/cli.py').read_text())

    def test_both_execution_help_paths_do_not_import_model(self):
        for name in ['run', 'gpu_worker', 'audit']:
            result = subprocess.run([sys.executable, '-B', str(EXEC / (name + '.py')), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)
        self.assertNotIn('torch', sys.modules); self.assertNotIn('transformers', sys.modules)

    def test_measurement_transition_preserves_prefix_and_no_fake_result(self):
        obj = self.session(); before = obj.fixture.snapshot()
        self.malformed(obj)
        self.assertEqual(obj.budget.model_calls, 3); self.assertEqual(obj.budget.tool_attempts, 3)
        self.assertEqual(len(obj.fixture.events), 2); self.assertEqual(obj.fixture.snapshot(), before)
        self.assertEqual(obj.responses[0]['available_result_event_ids'], self.point['available_result_event_ids'])
        self.assertEqual(obj.terminal, 'PARSE_ERROR')

    def test_actual_ids_hash_and_grammar_stage_rejected_before_execution(self):
        obj = self.session()
        bad = self.gpu(obj, '<tool_call>{bad'); bad['input_token_ids'] = [9]
        with self.assertRaisesRegex(ValueError, 'IDs/hash/length'):
            obj.consume_measured(bad, elapsed_worker_seconds=0.1)
        bad = self.gpu(obj, '<tool_call>{bad'); bad['grammar_applied'] = True
        with self.assertRaisesRegex(ValueError, 'grammar applied'):
            obj.consume_measured(bad, elapsed_worker_seconds=0.1)
        self.assertEqual(obj.budget.model_calls, 2)

    def test_N_E_share_actual_C0_and_only_feedback_differs(self):
        obj = self.session(); self.malformed(obj)
        n = obj.fork('N', self.root / 'n'); e = obj.fork('E', self.root / 'e')
        self.assertEqual(n.responses, e.responses)
        self.assertEqual(n.budget.__dict__, e.budget.__dict__)
        self.assertEqual(n.next_request()[0]['messages'][:-1], e.next_request()[0]['messages'][:-1])
        self.assertNotEqual(n.next_request()[0]['messages'][-1], e.next_request()[0]['messages'][-1])
        self.assertEqual(n.available, e.available); self.assertEqual(n.fixture.events, e.fixture.events)
        self.assertEqual(n.budget.model_calls, 3)

    def test_mixed_response_is_not_retry_and_G_is_focal_only(self):
        obj = self.session(arm='G')
        name = Registry('API_V1').exposed('order.read') if hasattr(Registry('API_V1'), 'exposed') else 'read_order'
        text = '<tool_call>' + json.dumps({'name': name, 'arguments': {'order_id': self.scenario['target_order_id']}}) + '</tool_call><tool_call>{bad'
        obj.consume_measured(self.gpu(obj, text), elapsed_worker_seconds=0.1)
        self.assertFalse(obj.next_request()[1]['grammar_active'])
        with self.assertRaises(ValueError):obj.fork('N', self.root / 'retry')

    def test_raw_CPU_replay_recomputes_endpoint_and_rejects_tampering(self):
        obj = self.session(); self.malformed(obj)
        identity = {'execution_source_commit': 'synthetic', 'execution_source_sha256': 'synthetic', 'generation_config_hash': 'synthetic'}
        row = obj.result(identity)
        self.assertFalse(replay_cell(row, self.root / 'audit')['VERIFIED_WORKFLOW_SUCCESS'])
        bad = copy.deepcopy(row); bad['VERIFIED_WORKFLOW_SUCCESS'] = True
        with self.assertRaisesRegex(ValueError, 'VERIFIED_WORKFLOW_SUCCESS'):
            replay_cell(bad, self.root / 'bad')
        bad = copy.deepcopy(row); bad['responses'][0]['raw_text'] += 'x'
        with self.assertRaisesRegex(ValueError, 'raw response hash'):
            replay_cell(bad, self.root / 'badraw')

    def test_retry_complete_workflow_requires_real_write_validation_receipt(self):
        obj = self.session(); self.malformed(obj)
        branch = obj.fork('N', self.root / 'retry')
        target = self.scenario['target_order_id']; order = self.scenario['orders'][target]
        total = order['quantity'] * order['unit_price_minor']
        calls = '<tool_call>' + json.dumps({'name': 'write_order_total', 'arguments': {'order_id': target, 'total_minor': total}}) + '</tool_call>'
        calls += '<tool_call>' + json.dumps({'name': 'validate_order', 'arguments': {'order_id': target}}) + '</tool_call>'
        branch.consume_measured(self.gpu(branch, calls, 'synthetic-request-2'), elapsed_worker_seconds=0.2)
        self.assertEqual(len(branch.fixture.events), 4)
        final = json.dumps({'order_id': target, 'total_minor': total, 'validated': True})
        branch.consume_measured(self.gpu(branch, final, 'synthetic-request-3'), elapsed_worker_seconds=0.2)
        row = branch.result({'execution_source_commit': 'synthetic', 'execution_source_sha256': 'synthetic', 'generation_config_hash': 'synthetic'})
        self.assertTrue(row['VERIFIED_WORKFLOW_SUCCESS'])
        self.assertEqual(row['budget']['model_calls'], 5)
        self.assertEqual(row['budget']['tool_attempts'], 5)
        self.assertTrue(replay_cell(row, self.root / 'audit')['VERIFIED_WORKFLOW_SUCCESS'])


if __name__ == '__main__':unittest.main()
