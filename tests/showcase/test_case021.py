"""Explicit Case021 registry, source identity, units and bilingual routes."""
from pathlib import Path
import sys,unittest,tempfile,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/showcase'))
from case021 import load_case021,render_case021,home_case021,SECTIONS
from build import build,validate_case_ids
from check import check
from common import read,sha
class Case021(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.data=load_case021(ROOT);cls.temp=tempfile.TemporaryDirectory();cls.site=Path(cls.temp.name)/'site';build(ROOT,cls.site,'/inference-lab/')
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_explicit_identity_and_old_registry(self):
  old=[f'cases/{i:03}-fixture' for i in range(1,21)];registered=old+['cases/021-rounding-execution-diagnostics']
  self.assertEqual(validate_case_ids(registered),list(range(1,22)));self.assertEqual(validate_case_ids(old),list(range(1,21)))
  for x in [old+['cases/021-unregistered'],registered+['cases/022-unregistered'],registered[:-2]+registered[-1:]]:
   with self.assertRaises(ValueError):validate_case_ids(x)
 def test_denominators_and_current_verdict(self):
  f=self.data['facts'];self.assertEqual(f['original_primary_verdict'],'INCONCLUSIVE');self.assertEqual(f['primary_documents'],24);self.assertEqual(sum(f['gap_classes'].values()),96)
  self.assertEqual(f['single_term']['A_only'],406626);self.assertEqual(f['multiplicity']['record_rows_first_block'],5120);self.assertEqual(f['multiplicity']['distinct_layer_axis_coefficient_rows'],4102)
  self.assertEqual([(r['layer'],r['document_id'],r['first_observed_K']) for r in f['sparse']],[(0,'dev_code_00',64),(0,'dev_code_01',8),(25,'dev_code_00',64),(25,'dev_code_01',64)])
 def test_routes_and_source_render(self):
  for lang,other in [('ko','en'),('en','ko')]:
   home=(self.site/lang/'index.html').read_text();html=(self.site/lang/'case021.html').read_text()
   self.assertIn(home_case021(self.data,lang),home);self.assertIn('case020.html',home);self.assertIn('href="case021.html"',home)
   self.assertIn('../'+other+'/case021.html',html);self.assertIn(render_case021(self.data,lang),html)
   for s in SECTIONS:self.assertIn('id="'+s+'"',html)
   for v in ['INCONCLUSIVE','406,626','4,102','1,018','ΔG=ΔN+ΔC','A=R_BF16','K1024','nats','CPU']:self.assertIn(v,html)
   self.assertNotIn('/home/',html);self.assertIn('<noscript>',html)
  self.assertEqual(check(ROOT,self.site)['status'],'PASS')
 def test_rehashed_display_tamper_rejected(self):
  p=self.site/'data/case021.json';m=self.site/'build_manifest.json';raw=p.read_bytes();mr=m.read_bytes()
  try:
   data=read(p);data['facts']['single_term']['A_only']+=1;p.write_text(json.dumps(data));manifest=read(m);manifest['files']['data/case021.json']=sha(p);m.write_text(json.dumps(manifest))
   with self.assertRaisesRegex(ValueError,'Case021 data/source'):check(ROOT,self.site)
  finally:p.write_bytes(raw);m.write_bytes(mr)
if __name__=='__main__':unittest.main()
