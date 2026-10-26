import pathlib,sys,json,tempfile,itertools,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from base import *
import worker
class Contracts(unittest.TestCase):
 def test_plan_balance_and_denominators(self):
  jobs=read(ROOT/'provenance/plans.json');timing=[j for j in jobs if j['kind']=='timing'];self.assertEqual(len(timing),30);self.assertEqual(sum(len(j['methods']) for j in timing),96);self.assertEqual(sum(j['kind']=='lifecycle' for j in jobs),27)
  for j in timing:
   for pas in 'HDE':
    orders=j['orders'][pas];perms=list(itertools.permutations(j['methods']));counts={p:0 for p in perms}
    for o in orders:counts[tuple(o)]+=1
    self.assertEqual(len(set(counts.values())),1)
 def test_preflight_hash_and_source_mismatch_reject(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);(p/'provenance').mkdir();dump(p/'pre.json',{'source_hash':'wrong','config_sha256':'wrong'});dump(p/'PROTOCOL.json',{});dump(p/'provenance/preflight_index.json',{'32':{'path':'pre.json','sha256':info(p/'pre.json')['sha256']}})
   with patch.object(worker,'ROOT',p):
    with self.assertRaisesRegex(ValueError,'SOURCE_CONFIG_MISMATCH'):worker.guard({'kind':'timing','Z':32},'correct')
   doc=read(p/'provenance/preflight_index.json');doc['32']['sha256']='0'*64;dump(p/'provenance/preflight_index.json',doc)
   with patch.object(worker,'ROOT',p):
    with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):worker.guard({'kind':'timing','Z':32},'correct')
 def test_input_boundary_and_CPU_only_import(self):
  for ids in [np.array([-1],np.int64),np.array([32768],np.int64)]:
   with self.assertRaises(IndexError):validate_ids(ids)
  with self.assertRaises(TypeError):validate_ids(np.array([0],np.int32))
  self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
