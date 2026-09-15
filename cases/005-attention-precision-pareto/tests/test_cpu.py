import importlib.abc,sys,unittest,tempfile,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ['torch','sageattention','transformers','triton']:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from src.decision import numerical,choose_dev,final_decision,dominates
from src.statistics import timing,resample_indices,error_ci
from src.base_metrics import unit_error
from src.reference import attention_fp64
from src.backends import ALIASES,effective
from src.anchor import validate_geometry,assert_unmodified
from src.timing import measured_block
from src.common import fingerprint,sha,ensure_real
class Tests(unittest.TestCase):
 def row(self,i='V1',speed=1.5):return dict(config_id=i,execution_status='EXECUTABLE_VERIFIED',numerical={'status':'PASS','median':.01,'p95':.03},speedup=speed,full_output_nonfinite=0)
 def test_threshold_equality(self):
  u=[dict(relative_output_error=.01,invalid_rows=0)]*20;self.assertEqual(numerical(u,20)['status'],'PASS');self.assertEqual(choose_dev([self.row()]),'V1')
 def test_tail_boundary(self):
  u=[dict(relative_output_error=0,invalid_rows=0)]*19+[dict(relative_output_error=.6,invalid_rows=0)];self.assertAlmostEqual(numerical(u,20)['p95'],.03)
 def test_below_speed(self):self.assertIsNone(choose_dev([self.row(speed=1.4999)]))
 def test_anchor_not_finalist(self):self.assertIsNone(choose_dev([self.row('A_PUBLIC',9)]))
 def test_alias(self):self.assertEqual(ALIASES['A_MATCHED'],'A_PUBLIC');self.assertEqual(effective('A_MATCHED'),effective('A_PUBLIC'))
 def test_ties(self):self.assertEqual(choose_dev([self.row('V2'),self.row('V1')]),'V1')
 def test_missing(self):self.assertEqual(numerical([],128)['status'],'INCONCLUSIVE')
 def test_near_zero(self):
  u=unit_error([[1.]],[[1e-6]]);self.assertIsNone(u['relative_output_error']);self.assertEqual(numerical([u],1)['status'],'INCONCLUSIVE')
 def test_nan(self):self.assertEqual(numerical([unit_error([[np.nan]],[[1.]])],1)['status'],'FAIL')
 def test_full_nonfinite(self):self.assertIsNone(choose_dev([{**self.row(),'full_output_nonfinite':1}]))
 def test_go(self):self.assertEqual(final_decision('V1',True,True,{'status':'PASS'},1.5,[1.001,2]),'GO_LOCAL_CANDIDATE')
 def test_stop(self):self.assertEqual(final_decision(None,False,True,{},0,None),'STOP_DEV_SCREEN')
 def test_no_fresh(self):self.assertEqual(final_decision('V1',False,True,{},2,None),'EXPLORATORY_ONLY_NO_FRESH_CONFIRM')
 def test_ci_boundary(self):self.assertEqual(final_decision('V1',True,True,{'status':'PASS'},1.5,[1,2]),'INCONCLUSIVE_TIMING')
 def test_no_quality_override(self):self.assertEqual(final_decision('V1',True,True,{'status':'FAIL'},5,[4,6]),'NO_QUALIFYING_CANDIDATE')
 def test_undefined(self):self.assertEqual(final_decision('V1',True,True,{'status':'INCONCLUSIVE'},5,[4,6]),'INCONCLUSIVE_FIDELITY')
 def test_pareto_tradeoff(self):
  a=dict(wall_median_ms=1,median_error=.01,p95_error=.04);b=dict(wall_median_ms=2,median_error=.02,p95_error=.03);self.assertFalse(dominates(a,b));self.assertFalse(dominates(b,a))
 def test_pareto(self):
  a=dict(wall_median_ms=1,median_error=.01,p95_error=.03);b=dict(wall_median_ms=2,median_error=.01,p95_error=.03);self.assertTrue(dominates(a,b));self.assertFalse(dominates(a,a))
 def test_ratio(self):
  a=np.ones((3,2,5))*2;b=np.ones_like(a);r=timing(a,b,repetitions=20);self.assertEqual(r['speedup'],2);self.assertEqual(r['ci95'],[2,2])
 def test_pair_not_ratio_of_medians(self):
  a=np.array([1,10,20]*2).reshape(1,2,3);b=np.array([1,2,10]*2).reshape(1,2,3);r=timing(a,b,repetitions=20);self.assertEqual(r['speedup'],2);self.assertEqual(r['ratio_of_medians'],5)
 def test_global_process(self):
  ds,ps,bs=resample_indices(np.random.default_rng(1),16,5,20);self.assertEqual(ps.shape,(5,));self.assertEqual(ds.shape,(16,));self.assertEqual(bs.shape,(16,5,20))
 def test_bad_timing(self):
  with self.assertRaises(ValueError):timing(np.ones((2,2,2)),np.zeros((2,2,2)))
 def test_document_cluster(self):self.assertEqual(error_ci([[0]*16,[1]*16],repetitions=100)['median_ci95'],[0,1])
 def test_mock(self):
  with self.assertRaises(ValueError):ensure_real({'evidence_kind':'cpu_mock'})
 def test_mutation(self):
  with self.assertRaises(RuntimeError):assert_unmodified(['a'],['b'])
 def test_hash(self):self.assertNotEqual(fingerprint([1,2]),fingerprint([2,1]))
 def test_oracle_gqa_causal(self):
  q=np.zeros((1,4,4,2));k=np.zeros((1,2,4,2));v=np.zeros_like(k);v[:,0]=2;v[:,1]=7;o,_=attention_fp64(q,k,v,.5,True);np.testing.assert_allclose(o[:,:2],2);np.testing.assert_allclose(o[:,2:],7)
 def test_causal_flip(self):
  q=np.zeros((1,1,4,2));v=np.arange(8).reshape(q.shape);a,_=attention_fp64(q,q,v,1,True);b,_=attention_fp64(q,q,v,1,False);self.assertFalse(np.allclose(a,b));v[:,:,2:]+=100;a2,_=attention_fp64(q,q,v,1,True);np.testing.assert_equal(a[:,:,:2],a2[:,:,:2])
 def test_fp64_norm(self):self.assertEqual(unit_error([[3,4]],[[0,4]])['relative_output_error'],.75)
 def test_timing_boundary(self):
  actions=[]
  class Event:
   def record(self):actions.append('record')
   def synchronize(self):actions.append('end')
   def elapsed_time(self,last):return 20
  ticks=iter([0,1]);w,e=measured_block(lambda:actions.append('call'),lambda:actions.append('sync'),Event,10,lambda:next(ticks));self.assertEqual(w,100);self.assertEqual(e,2);self.assertEqual(actions[0],'sync');self.assertEqual(actions[11],'sync');self.assertEqual(actions.count('call'),20)
if __name__=='__main__':unittest.main()
