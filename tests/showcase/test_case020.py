"""Case020 joins the registry without weakening older-case or numeric contracts."""
from pathlib import Path
import sys,unittest,tempfile,shutil,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/showcase'))
from case020 import load_case020,projection,render_case020,home_case020,SECTIONS,C20
from build import build,validate_case_ids
from check import check
from common import read,sha
class Case020(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.data=load_case020(ROOT);cls.temp=tempfile.TemporaryDirectory();cls.site=Path(cls.temp.name)/'site';build(ROOT,cls.site,'/inference-lab/')
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_full_twenty_case_registry_and_unregistered_rejection(self):
  paths=[f'cases/{i:03}-fixture' for i in range(1,21)];self.assertEqual(validate_case_ids(paths),list(range(1,21)))
  for bad in [paths+["cases/021-unregistered"],paths[:-2]+paths[-1:],paths+[paths[-1]]]:
   with self.assertRaisesRegex(ValueError,'Unexpected case'):validate_case_ids(bad)
 def test_bilingual_home_archive_and_detail_exist(self):
  for lang in ['ko','en']:
   home=(self.site/lang/'index.html').read_text();html=(self.site/lang/'case020.html').read_text();self.assertIn(home_case020(self.data,lang),home);self.assertIn('case020.html#overview',home);self.assertIn('href="case020.html"',home)
   for key in SECTIONS:self.assertIn('id="'+key+'"',html)
   self.assertIn('case019.html',home);self.assertIn('case018.html',home);self.assertIn('<noscript>',html);self.assertNotIn('/home/',html);self.assertIn('run_checks.py',html)
 def test_measures_denominators_and_pairing_remain_distinct(self):
  f=self.data['facts'];self.assertEqual((f['P2_final_cell_blocks'],f['P4_method_blocks'],f['P4_C_B_block_comparisons']),(90,96,30));self.assertEqual(f['P4_C_faster_blocks'],0)
  for lang in ['ko','en']:
   h=render_case020(self.data,lang)
   for value in ['304.016','80.790','52.571','1.142','53.0%','23.8%','489.02','24.65']:self.assertIn(value,h)
   self.assertIn('reserved' if lang=='en' else 'reserved',h);self.assertIn('p95',h);self.assertIn('K16',h)
 def test_stale_ratio_rejected_independently_of_manifest(self):
  with tempfile.TemporaryDirectory() as t:
   case=Path(t);(case/'summary').mkdir();source=ROOT/C20/'summary'
   for name in ['facts.json','all_P4_clock_statistics.csv','lifecycle_block.csv','representative_direct_summary.csv','width_storage.csv','lifecycle_summary.csv','representative_direct_block.csv']:shutil.copyfile(source/name,case/'summary'/name)
   f=read(case/'summary/facts.json');f['ratios'][0]['median_of_block_C_B_ratios']*=2;(case/'summary/facts.json').write_text(json.dumps(f))
   with self.assertRaisesRegex(ValueError,'paired median'):projection(case)
 def test_missing_identity_not_accepted_as_a_registered_case(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);case=root/C20;case.mkdir(parents=True);(case/'README.md').write_text('Incomplete Case020')
   with self.assertRaises((FileNotFoundError,ValueError)):load_case020(root)
 def test_artifact_checked_and_no_raw_tensor_payload(self):
  self.assertEqual(check(ROOT,self.site)['status'],'PASS');self.assertFalse(list(self.site.rglob('*.npz')));self.assertFalse(list(self.site.rglob('*.npy')))
  for name in ['assets/case020-01_memory.svg','assets/case020-02_direct_calls.svg','assets/case020-03_width_storage.svg','assets/case020-04_session_worker.svg']:self.assertTrue((self.site/name).is_file())
 def test_rehashed_display_tamper_does_not_pass(self):
  p=self.site/'data/case020.json';m=self.site/'build_manifest.json';before=p.read_bytes();bm=m.read_bytes()
  try:
   x=read(p);x['facts']['P4_C_faster_blocks']=1;p.write_text(json.dumps(x));mdata=read(m);mdata['files']['data/case020.json']=sha(p);m.write_text(json.dumps(mdata))
   with self.assertRaisesRegex(ValueError,'Case020 data/source'):check(ROOT,self.site)
  finally:p.write_bytes(before);m.write_bytes(bm)
if __name__=='__main__':unittest.main()
