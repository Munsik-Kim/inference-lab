"""Display contracts for original endpoints, posthoc scope and raw-text safety."""
from pathlib import Path
import copy
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/showcase'))
from case018 import load_case018, render_case018, home_case018, SECTIONS

class Case018(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=load_case018(ROOT)
    def test_phase_denominators_and_unchanged_scores(self):
        self.assertEqual([p['episodes'] for p in self.data['phases']],[24,96])
        self.assertEqual([(g['Q'],g['W']) for g in self.data['phases'][1]['groups']],[(24,0),(24,0),(19,5),(19,5)])
    def test_posthoc_is_separate_and_visible(self):
        for lang in ['ko','en']:
            html=render_case018(self.data,lang)
            for anchor in SECTIONS:self.assertIn('id="'+anchor+'"',html)
            self.assertIn('76',html);self.assertIn('91',html);self.assertIn('NA',html)
            self.assertIn('사후' if lang=='ko' else 'Posthoc',html)
            self.assertIn('CPU',home_case018(self.data,lang))
    def test_selected_recorded_ids_and_escape(self):
        data=copy.deepcopy(self.data);data['selected_episodes'][0]['turns'][0]['raw_text']='<script>alert(1)</script>'
        html=render_case018(data,'en')
        self.assertNotIn('<script>alert(1)</script>',html);self.assertIn('&lt;script&gt;',html)
        self.assertEqual(html.count('class="case018-episode"'),6)
        self.assertIn('HOLDOUT-C01_K1_R1',html)
    def test_Q1_subset_not_used_as_main_denominator(self):
        g=self.data['phases'][1]['groups'][0]
        self.assertEqual(g['episodes'],24);self.assertEqual(g['conditional_Q1']['episodes'],24)
        g=self.data['phases'][1]['groups'][2]
        self.assertEqual(g['episodes'],24);self.assertEqual(g['conditional_Q1']['episodes'],19)

if __name__=='__main__':unittest.main()
