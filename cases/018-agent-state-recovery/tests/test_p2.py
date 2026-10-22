"""Synthetic CPU boundaries. Never generates actual HOLDOUT inputs or a model response."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from agent_state_recovery.common import CASE_ROOT
from agent_state_recovery.generator import dev_tasks
from agent_state_recovery.p2_contract import (validate_gate, planned_keys, check_files, file_row,
    remaining_budget, request_ceiling, complete_token_sum, validate_runtime, scenario_group)
from agent_state_recovery.p2_analysis import paired
from agent_state_recovery.p2_audit import validate_coverage


class P2BoundaryContracts(unittest.TestCase):
    def test_gate_requires_actual_prerequisites_but_no_success_threshold(self):
        gate = {k: True for k in ("DEV24_complete", "execution_source_frozen", "actual_KV_verified",
            "grammar_verified", "input_identity_verified", "independent_audit_verified", "P1_preservation_verified", "registered_generator_verified")}
        gate.update(completed_episode_count=24, execution_source_sha256="synthetic-source", successes=0)
        self.assertTrue(validate_gate(gate))
        for key in [k for k in gate if gate[k] is True]:
            bad = dict(gate);bad[key] = False
            with self.assertRaises(ValueError): validate_gate(bad)

    def test_registered_order_has_96_unique_four_condition_keys(self):
        p = planned_keys()
        self.assertEqual(len(p), 96)
        self.assertEqual(len({r['episode_id'] for r in p}), 96)
        self.assertEqual(p, planned_keys())
        self.assertEqual([r['kv'] for r in p[:48]], ['K0'] * 48)
        self.assertEqual(len({r['task_id'] for r in p}), 24)

    def test_source_or_seal_file_change_is_rejected_without_rehashing_expected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fixed.json';p.write_text('{"registered":true}')
            row = file_row(p, Path(d))
            self.assertTrue(check_files(d, [row]))
            p.write_text('{"registered":false}')
            with self.assertRaises(ValueError): check_files(d, [row])

    def test_common_worker_budget_keeps_parent_failure_debt(self):
        self.assertAlmostEqual(remaining_budget(439.692244164), 3160.307755836)
        self.assertAlmostEqual(remaining_budget(439.692244164, 100), 3060.307755836)
        self.assertEqual(remaining_budget(439.692244164, 4000), 0)
        with self.assertRaises(ValueError): remaining_budget(-1)

    def test_PHASE_request_ceilings_do_not_inherit_DEV384_or_reset(self):
        rows = [{'type': 'REQUEST_SUBMITTED', 'category': 'HOLDOUT'}] * 400
        self.assertTrue(request_ceiling(rows, 'HOLDOUT'))
        with self.assertRaises(ValueError): request_ceiling(rows * 4, 'HOLDOUT')
        with self.assertRaises(ValueError): request_ceiling([{'type':'REQUEST_SUBMITTED','category':'TECH'}] * 8, 'TECH')

    def test_unobserved_tokens_remain_null_not_zero_received_IDS(self):
        self.assertIsNone(complete_token_sum([10, None, 0]))
        self.assertIsNone(complete_token_sum([]))
        self.assertEqual(complete_token_sum([10, 0]), 10)

    def test_only_owned_runtime_path_and_port_can_change(self):
        original = {'snapshot':'fixed', 'max_model_len':28672, 'private_root':'old', 'port':8196}
        new = {**original, 'private_root':'new', 'port':8197}
        self.assertTrue(validate_runtime(new, original))
        new['max_model_len'] = 8192
        with self.assertRaises(ValueError): validate_runtime(new, original)

    def test_bootstrap_resamples_task_with_all_four_conditions(self):
        # Boolean synthetic contract matrix, not generated HOLDOUT tasks/gold.
        ledger = [{'task_id': f'HOLDOUT-{family}{i:02d}', 'kv': kv, 'policy': p,
                   'Q': int(p == 'R1'), 'W': 0}
                  for family in 'ABC' for i in range(8) for kv in ('K0','K1') for p in ('R0','R1')]
        result = paired(ledger, replicates=20, seed=1)
        r = result[0]
        self.assertEqual((r['pairs_observed'], r['recovery']), (24,24))
        self.assertEqual((r['difference_pp'],r['exploratory_pointwise_95_lower_pp'],r['exploratory_pointwise_95_upper_pp']), (100,100,100))
        self.assertEqual(result[4]['difference_pp'], 0)

    def test_instance_group_is_not_just_unique_ID_spelling(self):
        t = dev_tasks()[0];renamed = copy.deepcopy(t)
        renamed['task_id'] = 'SYNTHETIC_OTHER_LABEL'
        self.assertEqual(scenario_group(t), scenario_group(renamed))
        renamed['origin_state']['records'][renamed['public_request']['record_ids'][0]]['quantity'] += 1
        self.assertNotEqual(scenario_group(t), scenario_group(renamed))

    def test_coverage_rejects_duplicate_or_unknown_not_partial_as_complete(self):
        plan = planned_keys()
        entries = [{'episode_id': r['episode_id'], 'path': 'episodes/' + r['episode_id'] + '.json'} for r in plan]
        self.assertEqual(len(validate_coverage(plan, entries)), 96)
        with self.assertRaises(ValueError): validate_coverage(plan, entries + [entries[0]])
        bad = copy.deepcopy(entries);bad[0]['episode_id'] = 'DEV-A00_K0_R0'
        with self.assertRaises(ValueError): validate_coverage(plan, bad)

    def test_CPU_P2_help_does_not_import_model_SDKs_or_contact_GPU(self):
        code = '''import sys, importlib.abc, runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','transformers','xgrammar','vllm'}: raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
sys.argv=['p2','--help']
runpy.run_module('agent_state_recovery.p2',run_name='__main__')
'''
        r = subprocess.run([sys.executable, '-B', '-c', code], cwd=CASE_ROOT, text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('prepare',r.stdout)
