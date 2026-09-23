from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
import torch
from source.evaluation import sequence_call
from source.online_v2 import OnlineAdapter,NativeFloatCodec
from source.checkpoint import save_checkpoint,load_checkpoint
from source.metrics import metrics
class EvaluationContracts(unittest.TestCase):
 def setUp(self):
  self.a=OnlineAdapter(NativeFloatCodec((1,1,1)));self.tokens=np.zeros((2,5),dtype=np.int64)
 def run_fake(self,read):
  with patch('source.evaluation.v1.gather',return_value={}),patch('source.evaluation.v1.torch_transition',side_effect=lambda x,c:x+1):
   return sequence_call(None,{},self.tokens,self.a,readout=read)
 def test_wrong_answers_do_not_terminate(self):
  def read(m,x,c):
   y=torch.zeros((2,6));y[:,2]=1;return y
  p,s,e=self.run_fake(read)
  self.assertTrue(np.all(p==2));self.assertTrue(self.a.terminal_info(s)['active'].all())
  r,_,_=metrics(p,np.zeros((2,5),dtype=np.uint8),family=None)
  self.assertEqual(r['tau'],[1,1]);self.assertEqual(r['RMST0'],0)
 def test_nonfinite_readout_commits_finite_state_and_continues(self):
  def read(m,x,c):
   y=torch.zeros((2,6));y[:,0]=1
   if x[0,0,0,0]==2:y[0,1]=float('nan')
   return y
  p,s,e=self.run_fake(read)
  self.assertEqual(p[0].tolist(),[0,-1,0,0,0,0]);self.assertEqual(e['invalid_readout_counts'],[1,0])
  self.assertTrue(self.a.terminal_info(s)['active'].all());self.assertEqual(float(self.a.represented(s)[0,0,0,0]),6)
 def test_bos_wrong_excluded_bos_terminal_not_excluded(self):
  p=np.zeros((1,4),dtype=np.int8);p[0,0]=2
  self.assertEqual(metrics(p,np.zeros((1,3),dtype=np.uint8),family=None)[0]['tau'],[None])
  p[:]=-1;self.assertEqual(metrics(p,np.zeros((1,3),dtype=np.uint8),family=None)[0]['tau'],[1])
 def test_first_failure_recovery_censor(self):
  p=np.array([[0,0,2,0],[0,0,0,0],[0,1,0,0]],dtype=np.int8)
  m,_,_=metrics(p,np.zeros((3,3),dtype=np.uint8),family=None)
  self.assertEqual(m['tau'],[2,None,1]);self.assertEqual(m['RMST0'],4/3);self.assertAlmostEqual(m['token_accuracy'],7/9)
 def test_job_checkpoint_keeps_earlier_invalid_and_tau(self):
  def read(m,x,c):
   y=torch.zeros((2,6));y[:,0]=1
   if x[0,0,0,0]==2:y[0,1]=float('nan')
   return y
  p,s,e=self.run_fake(read);counts={k:e[k] for k in ['active_update_attempts','terminal_noop_steps','invalid_readout_counts']}
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d)/'complete'
   save_checkpoint(folder,self.a,s,identity={'test':'synthetic_test'},predictions=p,gold=np.zeros((2,5),dtype=np.uint8),sample_ids=['a','b'],counts=counts)
   a2,s2,p2,g2,r=load_checkpoint(folder,expected_identity={'test':'synthetic_test'},expected_sample_ids=['a','b'])
   self.assertEqual(r['tau'],[1,None]);self.assertEqual(r['counts']['invalid_readout_counts'],[1,0]);self.assertEqual(s.payload.tobytes(),s2.payload.tobytes());np.testing.assert_array_equal(p,p2)
   with self.assertRaises(ValueError):load_checkpoint(folder,expected_identity={'test':'synthetic_test'},expected_sample_ids=['b','a'])
   (folder/'runtime.bin').write_bytes(b'corrupt')
   with self.assertRaisesRegex(ValueError,'checksum'):load_checkpoint(folder,expected_identity={'test':'synthetic_test'},expected_sample_ids=['a','b'])
 def test_programming_error_not_terminal(self):
  with patch('source.evaluation.v1.gather',side_effect=RuntimeError('bug')):
   with self.assertRaisesRegex(RuntimeError,'bug'):sequence_call(None,{},self.tokens,self.a)
if __name__=='__main__':unittest.main()
