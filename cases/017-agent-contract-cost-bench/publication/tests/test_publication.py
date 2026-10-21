"""New packaging/display contracts, separate from frozen execution tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from derive import project, fresh
from verify import CASE, safe, sha

class Publication(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.d=project(CASE)

    def test_lossless_raw_and_new_ledger_have_same_unique_matrix(self):
        rows=self.d['rows']
        self.assertEqual(len(rows),len({(r['stage'],r['task_id'],r['kv'],r['policy']) for r in rows}))
        for stage in ('P1','P2'):
            for kv in ('K0','K1'):
                rr=[r for r in rows if r['stage']==stage and r['kv']==kv]
                counts={p:len([r for r in rr if r['policy']==p]) for p in ('C0','G','N','E')}
                self.assertEqual(len(set(counts.values())),1)
                for r in rr:self.assertEqual(len(r['source_sha256']),64)

    def test_cost_ratios_include_failed_episodes_and_distinct_input_output(self):
        for g in self.d['groups']:
            rr=[r for r in self.d['rows'] if (r['stage'],r['kv'],r['policy'])==(g['stage'],g['kv'],g['policy'])]
            for k in ('input_tokens','output_tokens','requests'):
                self.assertEqual(g[k],sum(r[k] for r in rr))
            self.assertAlmostEqual(g['output_tokens_per_verified'],sum(r['output_tokens'] for r in rr)/sum(r['verified'] for r in rr))
            self.assertNotEqual(g['input_tokens'],g['output_tokens'])

    def test_paired_categories_partition_actual_tasks(self):
        for r in self.d['comparisons']:
            self.assertEqual(r['n'],sum(r[k] for k in ('recovery','loss','both_success','both_fail')))
            self.assertAlmostEqual(r['difference_pp'],100*(r['recovery']-r['loss'])/r['n'])
            self.assertEqual(r['bootstrap']['seed'],6102013)
            self.assertTrue(r['bootstrap']['paired_shared_task_indices'])

    def test_fresh_is_not_a_substitute_for_requested_state(self):
        e=next(e for e in self.d['selected_episodes'] if e['episode_id']=='HOLDOUT-C00_K1_G')
        self.assertNotEqual(e['goal']['status'],next(iter(e['final_state']['documents'].values()))['status'])
        self.assertEqual((e['scores']['fresh_validation'],e['scores']['state_correct'],e['scores']['verified']),(1,0,0))

    def test_disclosure_is_next_to_first_results_in_both_languages(self):
        for suffix in ('.md','.ko.md'):
            for stem in ('README','REPORT'):
                text=(CASE/(stem+suffix)).read_text()
                self.assertIn('5/8',text);self.assertIn('6/8',text)
                self.assertLess(text.index('PROTOCOL_DEVIATION'+suffix),text.index('REPRODUCE'+suffix))
            text=(CASE/('REPORT'+suffix)).read_text()
            self.assertIn('Output tokens/verified job',text)
            self.assertIn('Fresh',text)

    def test_original_entry_documents_restore_to_original_hashes(self):
        mapping=json.loads((HERE/'provenance/original_mapping.json').read_text())
        for r in mapping['files']:
            if r['path'] in ('README.md','README.ko.md','REPORT.md','REPORT.ko.md','REPRODUCE.md','REPRODUCE.ko.md'):
                self.assertTrue(r['storage'].startswith('publication/originals/'))
                self.assertEqual(sha((CASE/r['storage']).read_bytes()),r['sha256'])
                self.assertNotEqual(sha((CASE/r['path']).read_bytes()),r['sha256'])

    def test_unsafe_restore_paths_are_rejected(self):
        for name in ('../a','/tmp/a','a/../b','a\\b','C:/a','a//b'):
            with self.assertRaises(ValueError):safe(name)

    def test_display_derived_from_events_escapes_model_raw(self):
        tools=CASE.parents[1]/'tools/showcase'
        sys.path.insert(0,str(tools))
        from case017 import render_case017
        data=copy.deepcopy(self.d)
        data['selected_episodes'][0]['turns'][0]['raw_text']='</pre><script>alert(1)</script>'
        for lang in ('ko','en'):
            html=render_case017(data,lang)
            self.assertIn('&lt;script&gt;',html)
            self.assertNotIn('<script>alert',html)
            self.assertIn('case017-original-evidence.tar.xz',html)

if __name__=='__main__':unittest.main()
