"""Synthetic regression fixtures specified in the external review; no model measurements."""
import copy
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from test_compare import row, dist
from diova_compare.core import compare, validate, kl

class ReviewEdges(unittest.TestCase):
    def pair(self,b,c): return compare(b,c,replicates=20)
    def test_metric_coverage_all_id_and_row_orders(self):
        for ids in [('a','z'),('z','a')]:
            rows=[row(ids[0]),row(ids[1])|{'scores':{'gold_nll':1.}}]
            for order in itertools.permutations(rows):
                with self.assertRaisesRegex(ValueError,r'task metric mismatch:.*version=.*sample_id=.*expected=.*actual='):
                    self.pair(list(order),copy.deepcopy(list(order)))
    def test_validate_rejects_coverage(self):
        with self.assertRaisesRegex(ValueError,'task metric mismatch'):
            validate([row('0'),row('1')|{'scores':{'brier':.5}}])
    def test_same_scores_and_key_order_preserved(self):
        b=[row('a')|{'scores':{'gold_nll':1.,'brier':.4}},row('b')|{'scores':{'brier':.5,'gold_nll':2.}}]
        c=copy.deepcopy(b)
        for r in c:r['scores']['gold_nll']+=.25
        q=self.pair(b,c)
        self.assertEqual(q['tasks']['fixture@1']['metrics']['delta_gold_nll']['mean'],.25)
    def test_distinct_tasks_and_versions_allow_contracts(self):
        a=row('a');b=row('b')|{'task':'other','scores':{'gold_nll':1.}}
        c=row('c')|{'task_version':'2','scores':{'brier':.2}}
        self.assertEqual(len(self.pair([a,b,c],copy.deepcopy([a,b,c]))['tasks']),3)
    def test_effective_distribution_and_explicit_scores(self):
        a=dist(row('a'));b=dist(row('b'))|{'scores':{'gold_nll':9.}}
        self.assertEqual(len(self.pair([a,b],copy.deepcopy([a,b]))['pairs']),2)
        # Same NLL/Brier availability but missing KL is still a contract mismatch.
        c=row('c')|{'scores':{'gold_nll':1.,'brier':.3}}
        with self.assertRaisesRegex(ValueError,'task metric mismatch'):validate([a,c])
    def test_distribution_kind_coverage(self):
        a=dist(row('a'));b=dist(row('b'))
        b['distribution'].update(kind='full_vocabulary',complete=True,vocabulary_size=3)
        with self.assertRaisesRegex(ValueError,'task metric mismatch'):validate([a,b])
    def test_prefix_varies_per_item(self):
        a=dist(row('a'));b=row('b');b['prompt_hash']='b'*64;b=dist(b)
        self.assertEqual(self.pair([a,b],copy.deepcopy([a,b]))['n_pairs'],2)
    def test_intersection_does_not_hide_bad_contract(self):
        b=[row('0'),row('1')|{'scores':{'gold_nll':1.}}]
        with self.assertRaisesRegex(ValueError,'task metric mismatch'):compare(b,[row('0')],intersection=True)
    def run_cli(self,b,c,path):
        path.mkdir();bp=path/'B.jsonl';cp=path/'C.jsonl';out=path/'report'
        for p,records in [(bp,b),(cp,c)]:p.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
        result=subprocess.run([sys.executable,'-B','-m','diova_compare.cli','compare','--baseline',str(bp),'--candidate',str(cp),'--output',str(out),'--replicates','20'],capture_output=True,text=True,cwd=path)
        return result,out
    def test_rejected_cli_has_no_report_directory(self):
        with tempfile.TemporaryDirectory() as d:
            b=[row('0'),row('1')|{'scores':{'gold_nll':1.}}]
            result,out=self.run_cli(b,copy.deepcopy(b),Path(d)/'run')
            self.assertEqual(result.returncode,2);self.assertIn('task metric mismatch',result.stderr)
            self.assertNotIn('Traceback',result.stderr);self.assertFalse(out.exists())
    def test_subnormal_finite_choice_and_full_cli(self):
        reference=446.0279640505967
        for kind in ['choice','full_vocabulary']:
            b=dist(row());c=dist(row(),(5e-324,.9,.1))
            if kind=='full_vocabulary':
                for x in [b,c]:x['distribution'].update(kind=kind,complete=True,vocabulary_size=3)
            with tempfile.TemporaryDirectory() as d:
                result,out=self.run_cli([b],[c],Path(d)/'run');self.assertEqual(result.returncode,0,result.stderr)
                data=json.loads((out/'report.json').read_text(),parse_constant=lambda v: self.fail(v))
                p=data['pairs'][0];key=kind+'_kl_B_to_C'
                self.assertTrue(math.isfinite(p[key]));self.assertFalse(p[key+'_infinite'])
                self.assertAlmostEqual(p[key],reference,places=6)
                self.assertTrue((out/'pairs.csv').exists());self.assertTrue((out/'report.md').exists())
    def test_exact_zero_mass_cli_contract(self):
        with tempfile.TemporaryDirectory() as d:
            result,out=self.run_cli([dist(row())],[dist(row(),(.7,.3,0))],Path(d)/'run')
            self.assertEqual(result.returncode,0,result.stderr)
            p=json.loads((out/'report.json').read_text())['pairs'][0]
            self.assertIsNone(p['choice_kl_B_to_C']);self.assertTrue(p['choice_kl_B_to_C_infinite'])
    def test_zeros_and_identity(self):
        self.assertEqual(kl([0.,0.,1.],[0.,0.,1.]),0.)
        self.assertEqual(kl([0.,0.,1.],[0.,.5,.5]),math.log(2))
        self.assertEqual(kl([.6,.3,.1],[.6,.3,.1]),0.)
    def test_distribution_pair_metadata_mismatch(self):
        for field,value in [('tokenizer_id','different'),('labels',['c','b','a']),('prefix_hash','b'*64)]:
            b=dist(row());c=dist(row());c['distribution'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.pair([b],[c])
