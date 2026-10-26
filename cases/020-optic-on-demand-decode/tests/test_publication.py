import pathlib,sys,unittest,tempfile,json,copy
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'publication'))
from verify import safe,check_rows,sha,npz_integrity
from run_checks import equal
from analyze import stat,storage
class Contracts(unittest.TestCase):
 def test_recorded_timing_statistics_and_tail(self):
  s=stat([1,2,3,4,100]);self.assertEqual(s['median_us'],3);self.assertAlmostEqual(s['p95_us'],80.8);self.assertEqual(s['n'],5)
 def test_raw_mutation_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);f=p/'raw.json';f.write_text('{"x":1}');row={'raw.json':{'bytes':f.stat().st_size,'sha256':sha(f)}};check_rows(p,row);f.write_text('{"x":9}')
   with self.assertRaisesRegex(ValueError,'BYTES'):check_rows(p,row)
 def test_paths_reject_escape_and_absolute(self):
  for name in ['../raw','/raw','a/../b','a\\b','a//b']:
   with self.assertRaises(ValueError):safe(name)
 def test_numerical_comparison_does_not_trust_saved_PASS(self):
  a={'status':'PASS','rows':[1.,2.]};b=copy.deepcopy(a);b['rows'][1]=20.
  with self.assertRaisesRegex(ValueError,'NUMERIC'):equal(a,b)
 def test_storage_aliases_and_dtype_sizes(self):
  a={'shape':[4],'element_size':2,'dtype':'torch.float16','payload_bytes':8,'storage_bytes':8,'key':['cuda:0',100,8],'unique':True};b={**a,'unique':False};self.assertEqual(storage([a,b]),8)
  with self.assertRaisesRegex(ValueError,'ALIAS'):storage([a,a])
 def test_binary_npz_integrity(self):
  import numpy as np
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'small.npz';np.savez(p,x=np.arange(4,dtype=np.int64));npz_integrity(p)
 def test_CPU_import_has_no_Torch(self):self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
