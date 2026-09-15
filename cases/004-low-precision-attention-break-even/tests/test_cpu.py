"""CPU-only checks: no torch, model, CUDA or SageAttention imports."""
import hashlib,importlib.abc,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
class BlockGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ['torch','sageattention','transformers','triton']:raise AssertionError('CPU test imported GPU/model dependency: '+fullname)
sys.meta_path.insert(0,BlockGPUImports())
from src.reference import attention_fp64
from src.metrics import unit_error,numerical_screen,decision,paired_timing_bootstrap,document_bootstrap
from src.backends import validate_geometry,assert_unmodified
from src.common import fingerprint,verify_spec,allow_confirmation,ensure_real,shapes,CASE
from src.timing import measured_block
from scripts.select_backend import select

class Checks(unittest.TestCase):
    def vectors(self,hq=4,hk=2,n=4):
        return np.zeros((1,hq,n,64)),np.zeros((1,hk,n,64)),np.arange(1*hk*n*64).reshape(1,hk,n,64)/10
    def test_uniform_closed_form(self):
        q,k,v=self.vectors();o,valid=attention_fp64(q,k,v,.125,False)
        np.testing.assert_allclose(o[0,0],np.broadcast_to(v[0,0].mean(0),o[0,0].shape));self.assertTrue(valid.all())
    def test_distinct_gqa_heads(self):
        q,k,v=self.vectors();v[:,0]=2;v[:,1]=7;o,_=attention_fp64(q,k,v,.125,True)
        np.testing.assert_allclose(o[:,:2],2.,rtol=0,atol=2e-15);np.testing.assert_allclose(o[:,2:],7.,rtol=0,atol=2e-15)
    def test_causal_prefix_mean(self):
        q,k,v=self.vectors();o,_=attention_fp64(q,k,v,.125,True)
        np.testing.assert_allclose(o[0,0,1],v[0,0,:2].mean(0))
    def test_causal_future_leak(self):
        q,k,v=self.vectors();a,_=attention_fp64(q,k,v,.125,True);v[:,:,2:]+=1000;b,_=attention_fp64(q,k,v,.125,True)
        np.testing.assert_array_equal(a[:,:,:2],b[:,:,:2])
    def test_causal_flip_changes_answer(self):
        q,k,v=self.vectors();a,_=attention_fp64(q,k,v,.125,True);b,_=attention_fp64(q,k,v,.125,False);self.assertFalse(np.allclose(a,b))
    def test_scale_changes_answer(self):
        q,k,v=self.vectors();q[:]=.1;k[:,:,1]=.2;a,_=attention_fp64(q,k,v,.01,False);b,_=attention_fp64(q,k,v,2.,False);self.assertFalse(np.allclose(a,b))
    def test_all_masked(self):
        q,k,v=self.vectors();o,valid=attention_fp64(q,k,v,.1,False,np.zeros((4,4),bool));self.assertFalse(valid.any());self.assertTrue((o==0).all())
    def test_partial_mask(self):
        q,k,v=self.vectors();mask=np.zeros((4,4),bool);mask[:,1]=True;o,valid=attention_fp64(q,k,v,.1,False,mask);np.testing.assert_allclose(o[0,0,0],v[0,0,1])
    def test_nonfinite_input(self):
        q,k,v=self.vectors();q[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):attention_fp64(q,k,v,.1,True)
    def test_inf_output(self):
        self.assertEqual(unit_error([[np.inf]],[[1.]])['status'],'NUMERICAL_REJECT')
    def test_zero_and_near_zero(self):
        for value in [0.,1e-6]:
            m=unit_error(np.ones((2,3)),np.ones((2,3))*value);self.assertIsNone(m['relative_output_error']);self.assertEqual(m['status'],'NUMERICAL_REVIEW')
    def test_norm_definition(self):
        m=unit_error([[3.,4.]],[[0.,4.]])
        self.assertAlmostEqual(m['relative_output_error'],.75);self.assertAlmostEqual(m['absolute_rms_error'],3/2**.5)
    def test_geometry(self):
        def fake(shape,dtype='bf16'):return SimpleNamespace(shape=shape,dtype=dtype,device='cpu')
        q=fake((1,4,16,64));k=fake((1,2,16,64));validate_geometry(q,k,k)
        with self.assertRaises(ValueError):validate_geometry(fake((1,3,16,64)),k,k)
        with self.assertRaises(ValueError):validate_geometry(fake((1,4,17,64)),k,k)
        with self.assertRaises(TypeError):validate_geometry(q,k,fake(k.shape,'fp32'))
    def test_mutation_detection(self):
        assert_unmodified(['abc'],['abc'])
        with self.assertRaises(RuntimeError):assert_unmodified(['abc'],['xyz'])
    def test_timing_boundary(self):
        calls=[];events=[]
        class Event:
            def record(self):events.append('record')
            def synchronize(self):events.append('end_sync')
            def elapsed_time(self,last):return 6.
        ticks=iter([1.,1.03]);w,e=measured_block(lambda:calls.append(1),lambda:events.append('sync'),Event,3,clock=lambda:next(ticks))
        self.assertEqual(len(calls),6);self.assertAlmostEqual(w,10.);self.assertEqual(e,2.);self.assertEqual(events,['sync','sync','record','record','end_sync'])
    def test_speed_ratio_direction(self):
        d=paired_timing_bootstrap([([2.,2.],[1.,1.]),([4.,4.],[2.,2.])],repetitions=100);self.assertEqual(d['speedup'],2.);self.assertEqual(d['ci95'],[2.,2.])
    def test_timing_document_weights_retained(self):
        d=paired_timing_bootstrap([([[2.,2.],[10.,10.]],[[1.,1.],[1.,1.]]),([[2.,2.],[10.,10.]],[[1.,1.],[1.,1.]])],repetitions=100)
        self.assertEqual(d['speedup'],6.);self.assertEqual(d['ci95'],[6.,6.])
    def test_timing_bad_pairs(self):
        with self.assertRaises(ValueError):paired_timing_bootstrap([([1],[0]),([2],[1])])
    def test_gate_exact_boundary(self):
        self.assertEqual(decision({'speedup':1.10,'ci95':[1.001,1.2]},{'status':'PASS'},True)[0],'LOCAL_CANDIDATE')
        self.assertEqual(decision({'speedup':1.10,'ci95':[1.,1.2]},{'status':'PASS'},True)[0],'INCONCLUSIVE')
        self.assertEqual(decision({'speedup':1.099,'ci95':[1.01,1.2]},{'status':'PASS'},True)[0],'KEEP_BF16')
    def test_synthetic_not_enabled(self):
        self.assertEqual(decision({'speedup':9.,'ci95':[8.,10.]},{'status':'PASS'},False)[0],'SYNTHETIC_ONLY')
    def test_numerical_limits(self):
        a=dict(relative_output_error=.01,invalid_rows=0,near_zero=False)
        self.assertEqual(numerical_screen([a]*20)['status'],'PASS')
        self.assertEqual(numerical_screen([{**a,'relative_output_error':.01001}]*20)['status'],'NUMERICAL_REJECT')
        self.assertEqual(numerical_screen([{**a,'near_zero':True,'relative_output_error':None}])['status'],'NUMERICAL_REVIEW')
    def test_document_cluster(self):
        x=document_bootstrap({'doc0':[0.]*12,'doc1':[1.]*12},repetitions=200)
        self.assertEqual(x['documents'],2);self.assertEqual(x['median_ci95'],[0.,1.])
    def test_manifest_no_leakage(self):
        m=json.loads((CASE/'inputs/manifest.json').read_text());docs=m['documents']
        self.assertEqual(sum(d['split']=='dev' for d in docs),8);self.assertEqual(sum(d['split']=='confirmation' for d in docs),16)
        self.assertEqual(len({d['prefix_sha256']['512'] for d in docs}),24)
        for d in docs:self.assertEqual(fingerprint(d['token_ids_4096'][:512]),d['prefix_sha256']['512'])
    def test_confirmation_gate(self):
        with self.assertRaises(ValueError):allow_confirmation('dev',{'protocol_version':1})
        allow_confirmation('confirmation',{'protocol_version':1})
    def test_spec_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'configs').mkdir();(p/'source').write_text('original');s={'frozen_files':{'source':hashlib.sha256(b'original').hexdigest()}}
            f=p/'configs/experiment_spec.json';f.write_text(json.dumps(s));(p/'configs/experiment_spec.sha256').write_text(hashlib.sha256(f.read_bytes()).hexdigest());verify_spec(p)
            (p/'source').write_text('changed')
            with self.assertRaises(AssertionError):verify_spec(p)
    def test_selector_unknown(self):
        t=dict(environment_fingerprint='a',model_revision='b',input_family='c',spec_sha256='d',entries=[dict(shape={'length':512},status='LOCAL_CANDIDATE',reason_codes=[])])
        q={k:v for k,v in t.items() if k!='entries'};q['shape']={'length':512}
        self.assertEqual(select(t,q)['backend'],'sage')
        self.assertEqual(select(t,{**q,'environment_fingerprint':'unknown'})['status'],'UNKNOWN')
        self.assertEqual(select(t,{**q,'shape':{'length':1024}})['backend'],'bf16')
    def test_mock_export_rejected(self):
        for r in [dict(evidence_kind='cpu_mock',mock=False),dict(evidence_kind='gpu_measurement',mock=True),dict(evidence_kind='gpu_measurement',status='OK',wall_ms=[float('nan')],event_ms=[1.]),dict(evidence_kind='gpu_measurement',status='OK',wall_ms=[0.],event_ms=[1.])]:
            with self.assertRaises(ValueError):ensure_real(r)
    def test_shape_matrix(self):
        ss=shapes();self.assertEqual(len([s for s in ss if s['family']=='synthetic']),24);self.assertEqual(len([s for s in ss if s['family']=='qwen']),3);self.assertEqual(len({s['id'] for s in ss}),27)
if __name__=='__main__':unittest.main(verbosity=2)
