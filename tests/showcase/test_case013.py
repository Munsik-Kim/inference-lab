"""Case013 display/evidence checks; no model imports or inference."""
import importlib.util, json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/showcase'))
from build import build
from check import check
from case013 import load_case013, render_case013, SECTIONS, C13

class Case013Display(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.site=Path(cls.temp.name)/'site'
        build(ROOT,cls.site,'/inference-lab/');cls.data=load_case013(ROOT)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_complete_artifact_and_source_check(self):
        self.assertEqual(check(ROOT,self.site)['status'],'PASS')
    def test_ppl_direction_and_unaccepted_packed_gate(self):
        c={x['model']:x for x in self.data['comparisons']}
        self.assertLess(c['06b']['ppl_change_percent'],0);self.assertGreater(c['4b']['ppl_change_percent'],0)
        self.assertEqual(self.data['packed']['status'],'BLOCKED')
    def test_two_models_and_correct_task_denominators(self):
        self.assertEqual(self.data['units']['models'],2)
        self.assertEqual(self.data['units']['agent_unique_tasks'],8)
        self.assertEqual(self.data['units']['agent_episode_conditions'],96)
        self.assertEqual(self.data['units']['valid_tokens_per_condition'],65504)
    def test_bilingual_direct_results_and_local_sections(self):
        for lang in ['en','ko']:
            html=(self.site/lang/'case013.html').read_text()
            for name in SECTIONS:self.assertIn('id="'+name+'"',html);self.assertIn('href="#'+name+'"',html)
            self.assertIn(render_case013(self.data,lang),html)
            self.assertIn('6/8',html);self.assertIn('0/8',html);self.assertIn('2.61%',html)
            self.assertIn('../'+('ko' if lang=='en' else 'en')+'/case013.html',html)
            self.assertIn('case013-report-'+lang+'.md.txt',html)
    def test_single_home_feature_and_archive_entry(self):
        for lang in ['en','ko']:
            html=(self.site/lang/'index.html').read_text()
            self.assertEqual(html.count('id="dualquant-study"'),1)
            self.assertEqual(html.count('class="archive-number">013'),1)
            self.assertEqual((ROOT/f'docs/{lang}/CASEBOOK.md').read_text().count('<a id="case-013"></a>'),1)
    def test_data_equals_source_projection(self):
        self.assertEqual(json.loads((self.site/'data/case013.json').read_text()),self.data)

if __name__=='__main__':unittest.main()
