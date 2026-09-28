"""Case011 source projection and accessible progressive disclosure contracts."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/showcase'))
from build import build
from check import check, Links
from case011 import C11, SECTIONS, REPORT_SECTIONS, SOURCE_COPIES, FIGURE_SOURCES, load_case011, load_items, render_case011


class ReadoutStudyDisplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.site=Path(cls.temp.name)/'site'
        cls.manifest=build(ROOT,cls.site,'/inference-lab/')
        cls.data=load_case011(ROOT)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def mutate_rehash(self,name,value,message):
        path=self.site/name;manifest=self.site/'build_manifest.json'
        old,old_manifest=path.read_bytes(),manifest.read_bytes()
        try:
            path.write_bytes(value);m=json.loads(old_manifest)
            m['files'][name]=hashlib.sha256(value).hexdigest();manifest.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,message):check(ROOT,self.site)
        finally:path.write_bytes(old);manifest.write_bytes(old_manifest)

    def test_eleven_explicit_cases_and_one_case011_card(self):
        for lang in ('en','ko'):
            html=(self.site/lang/'index.html').read_text()
            self.assertEqual(html.count('id="readout-study"'),1)
            for i in range(1,12):self.assertEqual(html.count(f'<span class="archive-number">{i:03}</span>'),1)
            self.assertIn('case010.html',html)
            for section in ('overview','implementation','results'):self.assertIn('case011.html#'+section,html)

    def test_six_sections_report_anchors_and_language_pair(self):
        for lang in ('en','ko'):
            html=(self.site/lang/'case011.html').read_text();links=Links();links.feed(html)
            self.assertTrue(set(SECTIONS)<=links.ids)
            self.assertEqual(html.count('<h1>'),1)
            for anchor in REPORT_SECTIONS:self.assertIn('.md#'+anchor,html)
            self.assertIn('../'+('ko' if lang=='en' else 'en')+'/case011.html',html)
            self.assertIn('href="#main"',html)
            self.assertNotIn('iframe',html)

    def test_primary_visible_without_javascript_or_details(self):
        from html.parser import HTMLParser
        class Visible(HTMLParser):
            def __init__(self):super().__init__();self.depth=0;self.text=[]
            def handle_starttag(self,tag,attrs):
                if tag=='details':self.depth+=1
            def handle_endtag(self,tag):
                if tag=='details':self.depth-=1
            def handle_data(self,value):
                if not self.depth:self.text.append(value)
        for lang in ('en','ko'):
            p=Visible();p.feed((self.site/lang/'case011.html').read_text());text=' '.join(p.text)
            for row in self.data['primary']:self.assertIn(f"{row['mean_delta_tokens']:.2f}",text)
            self.assertIn('285.84',text);self.assertIn('275.96',text)
            self.assertIn('CE',text);self.assertIn('1,024',text);self.assertIn('2,048',text)

    def test_original_identity_and_intervention_scope(self):
        self.assertEqual(self.data['implementation']['parameter_names'],['mlp.2.weight','mlp.2.bias'])
        self.assertEqual(self.data['implementation']['final_layer_values'],1158)
        self.assertEqual(self.data['implementation']['extra_recurrent_state_bytes'],0)
        self.assertEqual(self.data['units']['recurrent_rollouts'],6)
        self.assertEqual(self.data['units']['logical_readout_conditions'],18)
        self.assertNotIn('bc1da81',json.dumps(self.data['source_identity']))
        self.assertTrue(all(r['interval_tokens'][1]<0 for r in self.data['primary']))

    def test_short_noop_and_solver_caps_retain_log_status(self):
        solver=self.data['posthoc']['solver']['candidates']
        selected=[r for r in solver if r['selected']]
        self.assertEqual(len(solver),18);self.assertEqual(len(selected),6)
        self.assertEqual([r['iterations'] for r in selected if r['head']=='SHORT_REFIT'],[0,0,0])
        self.assertEqual([r['iterations'] for r in selected if r['head']=='MIXED_REFIT'],[200,200,104])
        html=render_case011(self.data,'en')
        self.assertIn('MAX_ITER_NOT_CONVERGED',html)
        self.assertIn('ORIGINAL = SHORT',html)

    def test_fair_item_projection_and_pairing(self):
        items=load_items(ROOT);rows=items['rows']
        self.assertEqual(len(rows),6144)
        self.assertEqual({r['category'] for r in rows},{'longer','shorter','same'})
        self.assertEqual(len({(r['checkpoint_seed'],r['storage'],r['sample_id']) for r in rows}),6144)
        for seed in range(3):
            a=[r for r in rows if r['checkpoint_seed']==seed and r['storage']=='UNIFORM_8']
            self.assertEqual(len(a),1024)
            self.assertEqual(sum(r['rmst_delta'] for r in a)/1024,self.data['primary'][seed]['mean_delta_tokens'])
        js=(self.site/'assets/case011.js').read_text()
        self.assertIn('textContent',js);self.assertNotIn('innerHTML',js);self.assertNotIn('fetch(',js)
        self.assertIn('location.search + location.hash',js);self.assertIn('node.open = true',js)

    def test_sources_and_posthoc_figures_exact_bytes(self):
        for name,path in SOURCE_COPIES.items():self.assertEqual((self.site/'sources'/name).read_bytes(),(ROOT/path).read_bytes())
        for name,path in FIGURE_SOURCES.items():self.assertEqual((self.site/name).read_bytes(),(ROOT/path).read_bytes())
        self.assertEqual(check(ROOT,self.site)['status'],'PASS')

    def test_tampered_primary_rejected_after_rehash(self):
        data=json.loads((self.site/'data/case011.json').read_text());data['primary'][0]['mean_delta_tokens']=10
        self.mutate_rehash('data/case011.json',json.dumps(data).encode(),'Case011 numeric/source mismatch')

    def test_tampered_items_rejected_after_rehash(self):
        data=json.loads((self.site/'data/case011-items.json').read_text());data['rows'][0]['rmst_delta']=99999
        self.mutate_rehash('data/case011-items.json',json.dumps(data).encode(),'Case011 item-source mismatch')

    def test_tampered_figure_rejected_after_rehash(self):
        name=next(iter(FIGURE_SOURCES));value=(self.site/name).read_bytes()+b'<!-- synthetic tamper -->'
        self.mutate_rehash(name,value,'Changed Case011 source-derived figure')

    def test_only_lightweight_case011_deploy_assets(self):
        names=set(self.manifest['files'])
        self.assertFalse(any(name.endswith(('.npz','.npy','.pt','.whl','.zip')) for name in names))
        self.assertEqual({name for name in names if 'case011' in name and name.endswith('.svg')},set(FIGURE_SOURCES))
        self.assertNotIn('data/case011-items.js',(self.site/'en/index.html').read_text())
        for lang in ('en','ko'):
            html=(self.site/lang/'case011.html').read_text()
            self.assertIn('publication/run_checks.py',html)
            self.assertNotIn('pip install diova-compare',html)

if __name__=='__main__':unittest.main()
