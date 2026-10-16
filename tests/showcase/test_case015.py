"""Source-backed counts, separate stages, receipt visibility and escaped output."""
import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/showcase'))
from common import ROOT,read
from case015 import load_case015,render_case015,home_case015,SECTIONS,table


class Case015Display(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=load_case015(ROOT)

    def test_both_matrices_have_distinct_raw_denominators(self):
        self.assertEqual(len(self.data['P1_episodes']),96);self.assertEqual(len(self.data['A1_episodes']),96)
        self.assertIn('8 previously',self.data['P1']['denominator']);self.assertIn('16 new',self.data['A1']['denominator'])
        self.assertNotEqual(self.data['P1']['endpoint'],self.data['A1']['endpoint'])

    def test_bilingual_static_endpoints_and_receipt_groups(self):
        for lang in ['ko','en']:
            body=render_case015(self.data,lang)
            for sid in SECTIONS:self.assertIn('id="'+sid+'"',body)
            self.assertEqual(body.count('class="a1-episode"'),96)
            self.assertEqual(body.count('class="case015-turn"'),sum(len(r['responses']) for r in self.data['A1_episodes']))
            self.assertIn('generated_after_event_ids',body);self.assertIn('state_before_execution',body)
            self.assertIn('observ',body.lower());self.assertIn('8.33',body);self.assertIn('-8.33',body)

    def test_home_values_come_from_raw_projection(self):
        d=copy.deepcopy(self.data);d['A1']['summary']['success_table'][0]['verified_success']=3
        self.assertIn('3/16',home_case015(d,'en'))
        self.assertNotEqual(home_case015(d,'en'),home_case015(self.data,'en'))

    def test_hostile_model_text_and_table_cells_are_escaped(self):
        d=copy.deepcopy(self.data);d['A1_episodes'][0]['responses'][0]['raw_text']='<img src=x onerror=alert(1)><script>alert(2)</script>'
        body=render_case015(d,'en');self.assertIn('&lt;img',body);self.assertNotIn('<img src=x',body);self.assertNotIn('<script>alert',body)
        self.assertNotIn('<script>',table(['x'],[['<script>']],'x'))

    def test_current_sources_have_no_alias_or_feedback(self):
        for r in self.data['A1_episodes']:
            self.assertFalse(r['enum_enabled']);self.assertFalse(r['enhanced_feedback']);self.assertFalse(r['aliases_enabled'])
            self.assertEqual(r['execution_source_sha256'],self.data['execution_source_sha256']['A1'])
        for x in self.data['A1']['branching']:self.assertFalse(x['same_model_visible_policy_input'])

    def test_early_parse_stop_shows_original_and_annotated_terminal(self):
        body=render_case015(self.data,'en');start=body.index('id="D16__API_V1__A1-00"');end=body.index('<details class="a1-episode"',start)
        episode=body[start:end]
        for text in ['raw_terminal_reason','BUDGET_EXHAUSTED','derived_terminal_reason','PARSE_ERROR','posthoc_metadata_note']:self.assertIn(text,episode)


if __name__=='__main__':unittest.main()
