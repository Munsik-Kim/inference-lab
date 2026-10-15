import ast
import copy
import json
import sys
import unittest
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1]/'source';sys.path.insert(0,str(SRC))
from p1b_common import ALPHAS, forbidden
from policies import select, map_policies
from bootstrap import shared_indices, contrast_samples

class SelectionBoundary(unittest.TestCase):
    def rows(self):
        return [dict(target='t',alpha=a,document_id_hash=d,token_hash='input-'+d,split='SELECT-ID',status='COMPLETE',SSE_after=(a-.25)**2+1,NLL_sum=255*((a-.5)**2+2),valid_token_count=255,artifact_sha256='artifact-'+str(a)) for a in ALPHAS for d in ['d1','d2']]
    def test_reject_non_select_and_old_eval(self):
        for split in ['CAL-FIT','EVAL-ID','FRESH-EVAL-ID','FRESH-SHIFT']:
            r=self.rows();r[0]['split']=split
            with self.assertRaises(ValueError):select(r,{'t':1},['t'])
    def test_argmin_and_old_fit(self):
        r=select(self.rows(),{'t':.75},['t'])[0]
        self.assertEqual((r['policy_alpha']['OLD_FIT'],r['policy_alpha']['SELECT_SSE'],r['policy_alpha']['SELECT_NLL']),(.75,.25,.5))
    def test_exact_tie_chooses_small_alpha(self):
        r=self.rows()
        for q in r:q['SSE_after']=1;q['NLL_sum']=255
        p=select(r,{'t':1},['t'])[0]['policy_alpha'];self.assertEqual(p['SELECT_SSE'],0);self.assertEqual(p['SELECT_NLL'],0)
    def test_missing_duplicate_or_bad_input_rejected(self):
        for kind in ['missing','duplicate','input']:
            r=self.rows()
            if kind=='missing':r.pop()
            elif kind=='duplicate':r.append(r[0].copy())
            else:r[0]['token_hash']='different'
            with self.assertRaises(ValueError):select(r,{'t':1},['t'])
    def test_identical_alpha_is_exact_alias(self):
        rr=self.rows();f=select(rr,{'t':1},['t']);mapped=map_policies(rr,f)
        for doc in ['d1','d2']:
            a=next(r for r in mapped if r['policy']=='SELECT_SSE' and r['document_id_hash']==doc).copy()
            b=next(r for r in mapped if r['policy']=='FIXED_QUARTER' and r['document_id_hash']==doc).copy()
            a.pop('policy');b.pop('policy');self.assertEqual(json.dumps(a,sort_keys=True),json.dumps(b,sort_keys=True))
    def test_policy_missing_or_artifact_mismatch_rejected(self):
        rr=self.rows();f=select(rr,{'t':1},['t'])
        for x in [rr[:-1],copy.deepcopy(rr)]:
            if len(x)==len(rr):x[0]['artifact_sha256']='changed'
            with self.assertRaises(ValueError):map_policies(x,f)
    def test_shared_doc_bootstrap_ratios_and_macro(self):
        c=np.array([[3.,9.],[9.,6.]]);b=np.array([[1.,3.],[3.,2.]]);d=np.array([[2.,4.],[5.,1.]])
        ix=np.array([[0,0],[0,1],[1,1]])
        ratios,macro=contrast_samples(c,b,d,ix)
        np.testing.assert_allclose(ratios,[[1,8/6,1.5],[1.2,10/6,4]])
        np.testing.assert_allclose(macro,[1.1,1.5,2.75])
    def test_bootstrap_deterministic_and_no_reselection(self):
        np.testing.assert_array_equal(shared_indices(4,3),shared_indices(4,3))
        tree=ast.parse((SRC/'bootstrap.py').read_text())
        self.assertFalse(any(isinstance(n,ast.Name) and n.id in ['select','argmin','fit'] for n in ast.walk(tree)))
        with self.assertRaises(ValueError):contrast_samples(np.ones((2,3)),np.ones((2,3)),np.zeros((2,3)),shared_indices(3,2))
    def test_runner_has_no_fit_encode_call_or_old_runner_import(self):
        tree=ast.parse((SRC/'run_readonly.py').read_text())
        calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
        # str.encode for source hashing is allowed; quantizer/solver calls are not.
        self.assertFalse(any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id in ['codec','fit'] and n.attr in ['fit_numpy','fit_torch','encode','write_reload','inverse'] for n in calls))
        self.assertFalse(any(isinstance(n,ast.Import) and any(a.name=='run_p1' for a in n.names) for n in ast.walk(tree)))
        with self.assertRaises(RuntimeError):forbidden()
    def test_scalar_aggregation_not_mean_document_ppl(self):
        nll=[255*2.,255*4.];valid=[255,255]
        self.assertEqual(sum(nll)/sum(valid),3.)
        c=np.array([[2.,8.]]);b=np.array([[1.,2.]]);d=np.array([[1.,9.]])
        r,_=contrast_samples(c,b,d,np.array([[0,1]]));self.assertAlmostEqual(r[0,0],.7)

if __name__=='__main__':unittest.main()
