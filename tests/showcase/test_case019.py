"""Saved-condition display and unpublished-asset boundaries."""
from pathlib import Path
import sys,unittest,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/showcase'))
from case019 import load_case019,render_case019,home_case019,SECTIONS

class Case019(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=load_case019(ROOT)
    def test_phase_denominators_and_size_definition(self):
        f=self.data['facts'];self.assertEqual((f['counts']['p0_primary'],f['counts']['p1_primary'],f['counts']['p2_continuations']),(17,81,128))
        self.assertEqual(f['counts']['R_repetitions'],8);self.assertEqual(f['counts']['p2_original_paths'],16)
        for lang in ['ko','en']:
            html=render_case019(self.data,lang)
            self.assertIn('singular',html);self.assertIn('gradient-flow',html)
    def test_first_entry_is_readable_and_links_same_language_route(self):
        for lang in ['ko','en']:
            home=home_case019(self.data,lang);html=render_case019(self.data,lang)
            self.assertIn('case019.html#overview',home)
            for key in SECTIONS:self.assertIn('id="'+key+'"',html)
            self.assertIn('two-layer' if lang=='en' else '두 층 선형망',home)
            self.assertNotIn('88af9a5',home)
    def test_scenarios_only_saved_values_and_R2_distinguished(self):
        p=self.data['projection'];self.assertEqual(set(map(float,p['rho'])),{.01,.0001,.000001})
        self.assertEqual(set(map(float,p['gap'])),{.4,.1,.025})
        self.assertEqual(len(p['arms']),4)
        self.assertEqual(p['gap']['0.025']['median']['R_repetitions'],8)
        self.assertEqual(p['gap']['0.025']['R2']['R_repetitions'],1)
    def test_J_G_A_has_common_denominator_and_reversed_outcomes(self):
        ex=self.data['facts']['example'];self.assertAlmostEqual(ex['mean_J_over_D']+ex['mean_G_over_D'],ex['mean_A_over_D'],places=12)
        self.assertEqual(self.data['facts']['rotate']['g0025_t1_tau20_lower_higher'],[5,3])
        self.assertEqual(self.data['facts']['rotate']['g0025_t5_tau20_lower_higher'],[2,6])
        for lang in ['ko','en']:
            html=render_case019(self.data,lang);self.assertIn('conjugacy',html);self.assertIn('oracle',html)
    def test_no_fake_download_and_no_JS_fallback(self):
        url='https://github.com/Munsik-Kim/inference-lab/releases/download/case019-evidence-v1.0.0/case019-recorded-evidence-v1.tar.xz'
        self.assertEqual(self.data['facts']['distribution']['asset_url'],url)
        for lang in ['ko','en']:
            html=render_case019(self.data,lang)
            self.assertIn('<noscript>',html);self.assertIn('<figure>',html);self.assertIn('run_checks.py',html)
            self.assertNotIn('href="/home/',html);self.assertEqual(html.count('href="'+url+'"'),1)
            self.assertIn('https://arxiv.org/abs/2610.29864v1',html)

if __name__=='__main__':unittest.main()
