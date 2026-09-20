"""Case008 display contracts. Mutations below are test fixtures, never evidence."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/showcase'))
from case008 import load_case008, render_case008
from common import read, json_text, sha
from build import build
from check import check

class Case008(unittest.TestCase):
    def test_summary_matches_original_scopes(self):
        d=load_case008(ROOT);q,r=d['tracks']['Q'],d['tracks']['R']
        self.assertEqual((q['n_scenarios'],r['n_scenarios']),(192,192))
        self.assertNotEqual(q['model'],r['model'])
        self.assertEqual(q['weight_bytes']['Q-BF16'],8044982000)
        self.assertEqual(q['arms']['Q-W4']['transitions']['regression'],1)
        self.assertGreater(q['arms']['Q-W4']['delta_choice_nll']['mean'],0)
        self.assertLess(q['arms']['Q-W4']['delta_full_gold_nll']['mean'],0)
        for row in r['structures'].values():
            self.assertTrue(row['same_structure']);self.assertEqual(row['recovery']['n'],192)
            self.assertLess(row['mean_relative_after'],row['mean_relative_before'])
        self.assertNotIn('token_ids',json_text(d));self.assertNotIn('TINY_RANDOM',json_text(d))
    def test_language_keys_and_html_escape(self):
        self.assertEqual(set(read(ROOT/'presentation/case008/en.json')),set(read(ROOT/'presentation/case008/ko.json')))
        d=load_case008(ROOT);d['tracks']['Q']['model']='<script>alert(1)</script>'
        html=render_case008(ROOT,d,'en')
        self.assertIn('&lt;script&gt;',html);self.assertNotIn('<script>',html)
    def test_changed_or_missing_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'presentation').mkdir()
            m=read(ROOT/'presentation/case008_sources.json')
            (root/'presentation/case008_sources.json').write_text(json_text(m))
            with self.assertRaisesRegex(ValueError,'Missing/unsafe source'):load_case008(root)
            for name in m['files']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
            p=root/'cases/008-build-reconstruct-reload/results/derived/summary.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Changed source'):load_case008(root)
    def test_display_and_order_cannot_be_just_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            site=Path(tmp)/'site';build(ROOT,site)
            for lang in ('en','ko'):
                home=(site/lang/'index.html').read_text()
                self.assertLess(home.index('id="case008-report"'),home.index('class="case-number">007'))
                self.assertIn('href="case008.html"',home)
                page=(site/lang/'case008.html').read_text()
                for anchor in ('track-q','track-r','try-it'):self.assertIn('id="'+anchor+'"',page)
            p=site/'data/case008.json';d=read(p);d['tracks']['Q']['weight_bytes']['Q-W4']=1;p.write_text(json_text(d))
            m=site/'build_manifest.json';v=read(m);v['files']['data/case008.json']=sha(p);m.write_text(json_text(v))
            with self.assertRaisesRegex(ValueError,'Case008 display mismatch'):check(ROOT,site)

if __name__=='__main__':unittest.main()
