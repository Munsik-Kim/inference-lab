"""Case014 artifact, unit, language and untouched-guide display contracts."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/showcase'))
from build import build
from check import check
from case014 import load_case014,render_case014,home_case014,SECTIONS,metric

class Visible(HTMLParser):
    def __init__(self):super().__init__();self.depth=0;self.text=[]
    def handle_starttag(self,t,a):
        if t=='details':self.depth+=1
    def handle_endtag(self,t):
        if t=='details':self.depth-=1
    def handle_data(self,x):
        if not self.depth:self.text.append(x)

class Case014Display(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.site=Path(cls.tmp.name)/'site';build(ROOT,cls.site,'/inference-lab/');cls.data=load_case014(ROOT)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_actual_site_and_source_projection(self):
        self.assertEqual(check(ROOT,self.site)['status'],'PASS');self.assertEqual(json.loads((self.site/'data/case014.json').read_text()),self.data)
    def test_visible_two_corpora_and_original_baseline(self):
        for lang in ['ko','en']:
            p=Visible();p.feed((self.site/lang/'case014.html').read_text());text=' '.join(p.text)
            for term in ['57.600','50.156','45.405','12.452','12.687','ArXiv','64/64','BF16']:self.assertIn(term,text)
    def test_anchors_language_static_and_one_active_entry(self):
        for lang in ['ko','en']:
            html=(self.site/lang/'case014.html').read_text()
            for key in SECTIONS:self.assertIn('id="'+key+'"',html);self.assertIn('href="#'+key+'"',html)
            self.assertNotIn('<script',html);self.assertIn('../'+('en' if lang=='ko' else 'ko')+'/case014.html',html)
            home=(self.site/lang/'index.html').read_text();self.assertEqual(home.count('id="compensation-study"'),1);self.assertEqual(home.count('class="archive-number">014'),1)
            self.assertIn('id="featured-stories"',home);self.assertIn('qwen3-4b.html',home)
    def test_units_core_bytes_and_primary_retained(self):
        self.assertEqual(metric(self.data,'COMP_QUARTER')['document_count'],64);self.assertEqual(metric(self.data,'COMP_QUARTER')['valid_token_count'],64*255)
        self.assertEqual({x['artifact_bytes'] for x in self.data['p2']['storage']},{247733136});self.assertEqual(self.data['p2']['primary']['candidate'],'COMP_QUARTER')
        self.assertEqual(self.data['new_GPU_inference_during_publication'],0)
    def test_renderer_values_follow_data_instead_of_constants(self):
        changed=copy.deepcopy(self.data);r=next(x for x in changed['p2']['methods'] if x['method']=='COMP_QUARTER' and x['split']=='EVAL-ID');r['PPL_relative_change']=-.2;r['lower_documents']=60
        html=home_case014(changed,'en');self.assertIn('20.00%',html);self.assertIn('60/64',html)
    def test_all_four_methods_and_safe_text(self):
        for lang in ['ko','en']:
            html=render_case014(self.data,lang)
            for method in ['BF16','RTN','COMP_QUARTER','COMP_HALF']:self.assertIn(method,html)
            self.assertIn('case014-reproduce.md.txt',html);self.assertIn(self.data['download_url'],html)
        from case014 import table
        self.assertNotIn('<script>',table(['x'],[['<script>bad</script>']],'fixture'))

if __name__=='__main__':unittest.main()
