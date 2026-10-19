"""Scripted CPU boundary tests, never model-recovery measurements."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from json_recovery_p2.evidence import equivalent
from json_recovery_p2.io import read
from json_recovery_p2.plan import limits
from json_recovery_p2.requests import FEEDBACK, initial, retry, visible
from json_recovery_p2.session import Budget, Session
from json_recovery_p2.upstream import A1, Fixture, Registry, digest, extract, parse_slots


def call(name, arguments, closing=True):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + ('</tool_call>' if closing else '')


class CPUContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.counter = 0
        self.point = read(ROOT / 'data/checkpoints.json')[0]
        self.snapshot = read(ROOT / 'data/model_request_snapshots.json')[0]
        self.scenario = next(s for s in read(A1 / 'data/scenarios.json') if s['scenario_id'] == self.point['scenario_id'])
        self.oid = self.scenario['target_order_id']; row = self.scenario['orders'][self.oid]
        self.total = row['quantity'] * row['unit_price_minor']; self.names = Registry(self.point['api_version']).mapping

    def path(self):
        self.counter += 1
        return Path(self.tmp.name) / str(self.counter)

    def session(self, condition='C0'):
        return Session(self.point, self.snapshot, self.scenario, self.path(), condition=condition)

    def final(self):
        return json.dumps({'order_id': self.oid, 'total_minor': self.total, 'validated': True})

    def bad(self):
        return '<tool_call>{"name":"' + self.names['order.write'] + '","arguments":{"order_id":"' + self.oid + '",": "' + str(self.total) + '"}}<|im_end|>'

    def test_same_fixture_different_observation_ids_is_rejected(self):
        points = copy.deepcopy(self.point['per_method_identity']); points['RTN']['available_result_event_ids'] = ['other-read', 'other-product']
        with self.assertRaises(ValueError): equivalent(points)
        point = copy.deepcopy(self.point); point['available_result_event_ids'] = ['other-read', 'other-product']
        with self.assertRaises(ValueError): Session(point, self.snapshot, self.scenario, self.path())

    def test_allowlist_excludes_unseen_oracle_control_and_other_method_outputs(self):
        poisoned = copy.deepcopy(self.snapshot)
        for key in ['oracle', 'gold', 'control_order_id', 'semantic_tool_id', 'frozen_continuation']:
            poisoned[key] = 'DO_NOT_SEND_UNSEEN_AUDIT_DATA'
        request = visible(poisoned)
        self.assertEqual(set(request), {'messages', 'tools'}); self.assertEqual(request, visible(self.snapshot))
        self.assertNotIn('DO_NOT_SEND_UNSEEN_AUDIT_DATA', json.dumps(request))

    def test_key_serialization_order_is_bound_beside_canonical_hash(self):
        snapshot = copy.deepcopy(self.snapshot); fn = snapshot['tools'][0]['function']
        snapshot['tools'][0]['function'] = dict(reversed(list(fn.items())))
        self.assertEqual(snapshot['messages_tools_sha256'], digest({k: snapshot[k] for k in ['messages', 'tools']}))
        with self.assertRaises(ValueError): visible(snapshot)

    def test_C0_G_identical_payload_and_grammar_only_first_response(self):
        self.assertEqual(initial(self.snapshot, 'C0'), initial(self.snapshot, 'G'))
        s = self.session('G'); self.assertTrue(s.next_request()[1]['grammar_active'])
        s.consume(call(self.names['order.read'], {'order_id': self.oid}))
        self.assertFalse(s.next_request()[1]['grammar_active'])

    def test_N_E_share_new_C0_and_differ_only_in_exact_user_feedback(self):
        s = self.session(); s.consume(self.bad(), output_tokens=43, elapsed_seconds=0.25)
        n = s.fork_retry('N', self.path()); e = s.fork_retry('E', self.path())
        nr, er = n.next_request()[0], e.next_request()[0]
        self.assertEqual(nr['messages'][:-1], er['messages'][:-1]); self.assertEqual(nr['tools'], er['tools'])
        self.assertEqual(nr['messages'][-2], {'role': 'assistant', 'content': self.bad()})
        self.assertEqual(nr['messages'][-1], {'role': 'user', 'content': FEEDBACK['N']})
        self.assertEqual(er['messages'][-1], {'role': 'user', 'content': FEEDBACK['E']})
        self.assertNotEqual(digest(nr), digest(er))

    def test_historical_failure_is_not_new_C0(self):
        old = copy.deepcopy(read(ROOT / 'data/historical_continuations.json')[-1])
        old['parent_request_sha256'] = self.snapshot['messages_tools_sha256']
        with self.assertRaises(ValueError): retry(self.snapshot, 'N', old, model_calls_used=3, tool_attempts_used=3)

    def test_feedback_creates_no_tool_observation_and_cannot_repeat(self):
        s = self.session(); s.consume(self.bad()); n = s.fork_retry('N', self.path())
        self.assertEqual(n.available, self.point['available_result_event_ids']); self.assertEqual(len(n.fixture.events), 2)
        self.assertEqual(n.fixture.snapshot(), s.fixture.snapshot()); self.assertEqual(n.fixture.version, 0)
        n.consume(self.bad())
        with self.assertRaises(ValueError): n.fork_retry('E', self.path())

    def test_scripted_recovery_requires_feedback_request_and_actual_completed_chain(self):
        s = self.session(); s.consume(self.bad())
        for condition in ['N', 'E']:
            branch = s.fork_retry(condition, self.path())
            branch.consume(call(self.names['order.write'], {'order_id': self.oid, 'total_minor': self.total}) + call(self.names['order.validate'], {'order_id': self.oid}))
            self.assertEqual(branch.responses[1]['model_request']['messages'][-1], {'role': 'user', 'content': FEEDBACK[condition]})
            self.assertEqual(branch.responses[1]['available_result_event_ids'], self.point['available_result_event_ids'])
            branch.consume(self.final()); result = branch.verdict()
            self.assertTrue(result['VERIFIED_WORKFLOW_SUCCESS']); self.assertEqual(result['evidence_kind'], 'synthetic_contract_test')

    def test_mixed_valid_malformed_executes_valid_slot_without_retry(self):
        s = self.session(); s.consume(self.bad() + call(self.names['order.write'], {'order_id': self.oid, 'total_minor': self.total}))
        self.assertIsNone(s.c0_observation); self.assertEqual(s.budget.tool_attempts, 4); self.assertEqual(len(s.fixture.events), 3)
        self.assertEqual(s.fixture.snapshot()[self.oid]['total_minor'], self.total)
        with self.assertRaises(ValueError): s.fork_retry('N', self.path())
        s.consume(call(self.names['order.validate'], {'order_id': self.oid})); s.consume(self.final())
        self.assertTrue(s.verdict()['VERIFIED_WORKFLOW_SUCCESS'])

    def test_wrong_ID_key_string_and_bool_integer_follow_original_fixture(self):
        fixture = Fixture(self.scenario, Registry(self.point['api_version']), self.path()); before = fixture.snapshot()
        for args, expected in [({'order_id': '../outside', 'total_minor': self.total}, 'UNKNOWN_ORDER'),
                               ({'order_id': self.oid, 'total_minor': str(self.total)}, 'TYPE'),
                               ({'order_id': self.oid, 'total_minor': True}, 'TYPE'),
                               ({'order_id': self.oid, 'amount': self.total}, 'KEYS')]:
            with self.subTest(args=args):
                event = fixture.execute(self.names['order.write'], args, 2, 0, [])
                self.assertEqual(event['schema_result'], expected); self.assertEqual(fixture.snapshot(), before)

    def test_wrong_integer_is_saved_then_validation_false(self):
        for delta in [-1, 1]:
            s = self.session(); wrong = self.total + delta
            s.consume(call(self.names['order.write'], {'order_id': self.oid, 'total_minor': wrong}) + call(self.names['order.validate'], {'order_id': self.oid}))
            self.assertEqual(s.fixture.snapshot()[self.oid]['total_minor'], wrong)
            self.assertFalse(s.fixture.events[-1]['tool_result']['valid']); s.consume(self.final())
            self.assertFalse(s.verdict()['VERIFIED_WORKFLOW_SUCCESS'])

    def test_same_response_write_validate_then_observed_final_succeeds(self):
        s = self.session(); s.consume(call(self.names['order.write'], {'order_id': self.oid, 'total_minor': self.total}) + call(self.names['order.validate'], {'order_id': self.oid}))
        self.assertEqual(s.fixture.events[-2]['generated_after_event_ids'], s.fixture.events[-1]['generated_after_event_ids'])
        s.consume(self.final()); result = s.verdict()
        self.assertTrue(result['VERIFIED_WORKFLOW_SUCCESS']); self.assertTrue(result['independent_endpoint'])
        self.assertEqual(result['evidence_kind'], 'synthetic_contract_test')

    def test_final_answer_alone_cannot_replace_execution(self):
        s = self.session(); s.consume(self.final()); v = s.verdict()
        self.assertTrue(v['FINAL_ANSWER_CORRECT']); self.assertFalse(v['VERIFIED_WORKFLOW_SUCCESS'])

    def test_stale_validation_rejected_even_when_final_digest_returns(self):
        s = self.session(); s.consume(call(self.names['order.write'], {'order_id': self.oid, 'total_minor': self.total}) + call(self.names['order.validate'], {'order_id': self.oid}))
        s.consume(call(self.names['order.write'], {'order_id': self.oid, 'total_minor': self.total + 1}) + call(self.names['order.write'], {'order_id': self.oid, 'total_minor': self.total}))
        s.consume(self.final()); result = s.verdict()
        self.assertTrue(result['STATE_ONLY_SUCCESS']); self.assertFalse(result['VERIFIED_WORKFLOW_SUCCESS'])

    def test_alternate_reads_swapped_multiply_and_duplicate_write_permitted(self):
        s = self.session(); row = self.scenario['orders'][self.oid]
        s.consume(call(self.names['order.read'], {'order_id': self.oid}))
        s.consume(call(self.names['amount.multiply'], {'a': row['unit_price_minor'], 'b': row['quantity']}))
        s.consume(call(self.names['order.write'], {'order_id': self.oid, 'total_minor': self.total}) + call(self.names['order.validate'], {'order_id': self.oid}))
        s.consume(self.final()); self.assertTrue(s.verdict()['VERIFIED_WORKFLOW_SUCCESS'])
        t = self.session(); t.consume(call(self.names['order.write'], {'order_id': self.oid, 'total_minor': self.total}) + call(self.names['order.validate'], {'order_id': self.oid}))
        t.consume(call(self.names['order.write'], {'order_id': self.oid, 'total_minor': self.total})); t.consume(self.final())
        self.assertTrue(t.verdict()['VERIFIED_WORKFLOW_SUCCESS']); self.assertEqual(t.verdict()['auxiliary']['duplicate_successful_write_events'], 1)

    def test_prefix_malformed_shared_C0_costs_are_never_reset(self):
        s = self.session(); s.consume(self.bad(), elapsed_seconds=0.75)
        for branch in [s.fork_retry('N', self.path()), s.fork_retry('E', self.path())]:
            self.assertEqual((branch.budget.model_calls, branch.budget.tool_attempts, branch.budget.retries), (3, 3, 1))
            self.assertAlmostEqual(branch.budget.elapsed_seconds, self.point['prefix_elapsed_seconds_debit'] + 0.75)
        full = self.session(); full.consume(self.bad() * 4); self.assertEqual(full.budget.tool_attempts, 6)
        with self.assertRaises(ValueError): full.fork_retry('E', self.path())

    def test_model_context_and_deadline_limits(self):
        budget = Budget()
        for _ in range(6): budget.debit_response(192, 1.0)
        with self.assertRaises(ValueError): budget.before_request()
        Budget().check_input_tokens(24384)
        with self.assertRaises(ValueError): Budget().check_input_tokens(24385)
        s = self.session(); s.consume(call(self.names['order.write'], {'order_id': self.oid, 'total_minor': self.total}), elapsed_seconds=900)
        self.assertEqual(s.terminal, 'TIMEOUT'); self.assertEqual(len(s.fixture.events), 2)

    def test_original_first_dict_and_optional_closing_marker(self):
        raw = call(self.names['order.write'], {'order_id': self.oid, 'total_minor': self.total}, closing=False)
        self.assertIsNotNone(parse_slots(raw)[0]['parsed']); self.assertEqual(parse_slots(raw + ' trailing broken "'), parse_slots(raw))
        self.assertIsNone(parse_slots(self.bad())[0]['parsed'])
        self.assertEqual(extract('bad { not json } then {"valid":true} trailing'), {'valid': True})

    def test_conditional_request_bounds_are_not_episode_counts(self):
        self.assertEqual(limits(8)['first_diagnostic_responses'], 64); self.assertEqual(limits(8)['later_continuation_requests_upper_bound'], 264)
        self.assertEqual(limits(24)['first_diagnostic_responses'], 96); self.assertEqual(limits(24)['total_new_model_requests_upper_bound'], 408)
        for f in [True, -1, 25]:
            with self.assertRaises(ValueError): limits(f)

    def test_CLI_help_and_run_refusal_have_no_model_dependencies(self):
        env = os.environ.copy(); env['PYTHONPATH'] = str(ROOT / 'src'); env['PYTHONDONTWRITEBYTECODE'] = '1'
        for args, expected in [(['--help'], 0), (['run'], 3)]:
            result = subprocess.run([sys.executable, '-B', '-m', 'json_recovery_p2.cli', *args], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, expected, result.stderr)
        from json_recovery_p2.cpu_only import FORBIDDEN
        self.assertFalse(FORBIDDEN.intersection(sys.modules))


if __name__ == '__main__': unittest.main()

