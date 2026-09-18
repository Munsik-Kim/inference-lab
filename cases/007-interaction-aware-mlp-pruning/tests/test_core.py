import unittest,sys,types,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import *
from src.data import scenario
from src.oracle import interpret,OracleError
from src.surgery import replace_mlp
class TestCore(unittest.TestCase):
 def test_group_partition(self):
  for n in [3072,101]:
   g=groups(n);self.assertEqual(sum(g,[]),list(range(n)));self.assertLessEqual(max(map(len,g))-min(map(len,g)),1)
 def test_removed_mapping(self):
  g=groups(32);self.assertEqual(kept(g,[0,15]),list(range(2,30)))
  with self.assertRaises(ValueError):kept(g,[0,0])
 def test_identity_signed(self):
  c=np.random.default_rng(3).normal(size=(20,16,5));q=np.einsum('tih,tjh->ij',c,c)/20
  for p in random_sets(4):self.assertAlmostEqual(q[np.ix_(p,p)].sum(),np.square(c[:,p].sum(1)).sum()/20,places=10)
 def test_select_cancellation(self):
  q=np.eye(16);q[0,1]=q[1,0]=-.99;v=select(q,4,'PAIRWISE');self.assertTrue(0 in v['removed'] and 1 in v['removed']);self.assertEqual(v,select(q,4,'PAIRWISE'))
 def test_counts_ties(self):
  self.assertEqual(len(candidates(4)),1820);self.assertEqual(len(candidates(8)),12870);self.assertEqual(select(np.eye(16),4,'INDEPENDENT')['removed'],[0,1,2,3])
 def test_random_frozen(self):
  self.assertEqual(random_sets(8),random_sets(8));self.assertEqual(len(set(map(tuple,random_sets(8)))),20)
 def test_no_heldout_selection(self):
  with self.assertRaises(ValueError):calibration_q([{'id':str(i),'split':'HELD_OUT','Q':np.eye(16)} for i in range(96)])
 def test_data_oracle_unique(self):
  rows=[scenario(s,t,i) for s,n in COUNTS.items() for t in TASKS for i in range(n//3)]
  self.assertEqual(len({r['facts_hash'] for r in rows}),336)
  for r in rows:self.assertEqual(len(set(r['options'])),4);self.assertEqual(r['options'][r['gold']],r['gold_value'])
 def test_unsafe_code(self):
  for x in ['import os','answer = open("x")','for i in range(100):\n answer = i','answer = __import__("os")']:
   with self.assertRaises(OracleError):interpret(x)
 def test_positions(self):self.assertEqual(positions(512)[-1],511);self.assertEqual(len(set(positions(512))),32)
 def test_norm_scalar(self):
  s=norm_stats([1,2],[2,4]);self.assertEqual(s['relative_error'],.5);self.assertEqual(s['cosine'],1.)
  self.assertIsNone(norm_stats([0],[0])['relative_error'])
  with self.assertRaises(ValueError):norm_stats([1,float('nan')],[1,2])
 def test_scores(self):
  s=score([0,0,0,0,0],[0,1,2,3],1);self.assertAlmostEqual(s['nll'],math.log(4));self.assertAlmostEqual(s['brier'],.75);self.assertAlmostEqual(s['full_gold_nll'],s['nll']-math.log(s['label_mass']))
 def test_kl_shift(self):self.assertAlmostEqual(kl([1,2,3],[6,7,8]),0)
 def test_outcomes(self):
  b={'gold':0,'correct':True,'prediction':0};c={'gold':0,'correct':False,'prediction':1};self.assertEqual(transition(b,c)['cell'],'regression');self.assertEqual(transition(c,b)['cell'],'gain')
 def test_absent_validity(self):
  with self.assertRaises(ValueError):join([{'id':'x','method':'B','evidence_kind':'gpu_measurement'}],['B'])
  with self.assertRaises(ValueError):join([{'id':'x','method':'B','valid':True,'evidence_kind':'mock'}],['B'])
 def test_missing_duplicate_pair(self):
  r={'id':'x','method':'B','valid':True,'evidence_kind':'gpu_measurement','token_hash':'h'}
  with self.assertRaises(ValueError):join([r,r],['B'])
  with self.assertRaises(ValueError):join([r],['B','P'])
 def test_bootstrap_clusters(self):
  x=bootstrap_indices(['RETRIEVAL']*2+['COMPARISON']*2+['CODE']*2,10)
  self.assertTrue((x[:,:2]<2).all());self.assertTrue((x[:,4:]>=4).all())
 def test_status(self):
  self.assertEqual(decision([[-2,-1],[-2,-1]],True),'COMPLETED_POSITIVE_TRANSFER');self.assertEqual(decision([[1,2],[1,2]],True),'COMPLETED_NEGATIVE_TRANSFER');self.assertEqual(decision([[-1,1],[-2,-1]],True),'COMPLETED_NO_CLEAR_TRANSFER');self.assertEqual(decision([[-2,-1]]*2,False),'BLOCKED_IMPLEMENTATION')
 def test_restore_exception(self):
  layer=types.SimpleNamespace(mlp='old')
  with self.assertRaises(RuntimeError):
   with replace_mlp(layer,'new'):raise RuntimeError('fixture')
  self.assertEqual(layer.mlp,'old')
class TorchCPU(unittest.TestCase):
 def test_slicing_bias_and_equivalence(self):
  try:import torch
  except ImportError:self.skipTest('Torch CPU unavailable; core suite needs only NumPy')
  from src.surgery import sliced,masked
  class Tiny(torch.nn.Module):
   def __init__(self):
    super().__init__();self.config=types.SimpleNamespace(intermediate_size=32);self.intermediate_size=32;self.gate_proj=torch.nn.Linear(5,32,bias=True,dtype=torch.float64);self.up_proj=torch.nn.Linear(5,32,bias=True,dtype=torch.float64);self.down_proj=torch.nn.Linear(32,5,bias=True,dtype=torch.float64);self.act_fn=torch.nn.functional.silu
   def forward(self,x):return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))
  m=Tiny();before=copy.deepcopy(m.state_dict());x=torch.randn(7,5,dtype=torch.float64);g=groups(32);s=sliced(m,g,[0,7,9,15]);torch.testing.assert_close(s(x),masked(m,x,g,[0,7,9,15]),rtol=1e-12,atol=1e-12);self.assertEqual(s.gate_proj.weight.shape,(24,5));self.assertEqual(m.config.intermediate_size,32)
  for k,v in before.items():self.assertTrue(torch.equal(v,m.state_dict()[k]))
if __name__=='__main__':unittest.main()
