"""Display contracts for original endpoints, posthoc scope and raw-text safety."""
from pathlib import Path
import copy
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/showcase'))
from case018 import load_case018, render_case018, home_case018, display_facts, quantity_pair, SECTIONS

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
            home=home_case018(self.data,lang)
            self.assertIn('case018.html#reproduce',home)
            self.assertIn('모델 없이' if lang=='ko' else 'CPU',home)
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
    def test_first_table_has_four_quality_rows_and_no_cost_columns(self):
        for lang in ['ko','en']:
            html=render_case018(self.data,lang)
            first=html.split('class="case018-quality"',1)[1].split('</table>',1)[0]
            self.assertEqual(first.count('<th scope="col">'),4)
            self.assertEqual(first.count('<td>'),16)
            self.assertNotIn('Input tokens',first);self.assertNotIn('Episode s',first)
            for g in self.data['phases'][1]['groups']:
                self.assertIn(f"{g['Q']} / {g['episodes']}",first)
                self.assertIn(f"{g['W']} / {g['episodes']}",first)
            self.assertIn('검증·완료 보고' if lang=='ko' else 'Validation and correct report',first)
    def test_A00_three_quantities_visible_and_posthoc_denominators_separate(self):
        f=display_facts(self.data)
        for lang in ['ko','en']:
            html=render_case018(self.data,lang)
            visible_story=html.split('id="results"',1)[1].split('<details>',1)[0]
            e=next(x for x in self.data['selected_episodes'] if x['episode_id']=='HOLDOUT-A00_K0_R0')
            for values in [e['diagnostic']['first_read_quantities'],e['terminal_quantities'],e['diagnostic']['final_quantities']]:
                self.assertIn(quantity_pair(values,'xy'),visible_story)
            posthoc=html.split('id="posthoc"',1)[1].split('<details>',1)[0]
            self.assertIn(str(f['accepted']),posthoc)
            self.assertIn(str(f['wrong_recovery_reports']),posthoc)
            self.assertIn('NA',posthoc)
            self.assertIn('id="guard"',html)
            self.assertIn('actions/workflows/case018-checks.yml?query=branch%3Amain',html)

if __name__=='__main__':unittest.main()
