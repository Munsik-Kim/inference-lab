"""Labelled synthetic unit fixtures, never exported as measured results."""
import copy,hashlib,json,math,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from readout_analysis import score,pair,outcomes,describe,join,valid,bootstrap_indices,gap_bin,spearman


def fixture(x=(2.,1.,0.,-1.),gold=0,arm='B',view='H_NATIVE'):
    keys=('full_attention_outputs_finite','full_block_outputs_finite','full_final_hidden_finite','full_logits_finite','inputs_unchanged','same_layer13_qkv','routing_valid','dtype_layout_valid','answer_interface_valid','native_B_validated','effective_backend_verified')
    return dict(option_logits=list(x),gold_index=gold,full_lse=4.,full_argmax=32,label_ids=[32,33,34,35],readout=view,arm=arm,split='standard',base_id='fixture-1',length=4096,task='CODE',token_hash='fixture',evidence_kind='recorded_gpu_measurement',validity=dict.fromkeys(keys,True))
class Tests(unittest.TestCase):
    def test_first_max(self):self.assertEqual(score(fixture((2,2,0,0)))['prediction'],0)
    def test_lower_equality(self):self.assertTrue(score(fixture((2,1,1,0)))['unique'])
    def test_no_jitter(self):self.assertFalse(score(fixture((2,2,1,0)))['unique']);self.assertTrue(score(fixture((2,2-1e-12,1,0)))['unique'])
    def test_strict_flip(self):self.assertEqual(pair(score(fixture()),score(fixture((1,2,0,-1))))['stratum'],'unique/unique')
    def test_partition(self):
        ps=[pair(score(fixture(x)),score(fixture(y))) for x in [(2,1,0,0),(2,2,0,0)] for y in [(1,2,0,0),(2,2,0,0)]]
        d=describe(ps);self.assertEqual(sum(x['n'] for x in d['tie_strata'].values()),4);self.assertEqual(sum(x['flips'] for x in d['tie_strata'].values()),d['flips'])
    def test_correctness(self):
        self.assertEqual(pair(score(fixture()),score(fixture((0,2,1,0))))['cell'],'regression');self.assertEqual(pair(score(fixture((0,2,1,0))),score(fixture()))['cell'],'gain')
    def test_wrong_to_wrong(self):self.assertTrue(pair(score(fixture((0,2,1,0))),score(fixture((0,1,2,0))))['wrong_to_wrong'])
    def test_gold_top_membership(self):self.assertTrue(score(fixture((2,2,0,0),1))['gold_in_top']);self.assertFalse(score(fixture((2,2,0,0),1))['correct'])
    def test_nonfinite(self):
        for x in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):score(fixture((x,0,0,0)))
    def test_unobserved_full_failure(self):
        r=fixture();r['validity']['full_block_outputs_finite']=False
        self.assertTrue(math.isfinite(score(r)['nll']))
        with self.assertRaises(ValueError):valid(r)
    def test_missing_validity(self):
        r=fixture();del r['validity']['full_final_hidden_finite']
        with self.assertRaises(ValueError):valid(r)
    def test_missing_readout_controls(self):
        with self.assertRaises(ValueError):valid(fixture(view='H_FP32'))
    def test_joins(self):
        rows=[fixture(arm=a) for a in ['B','A_PUBLIC','V4']];self.assertEqual(len(join(rows,False)[1]),1)
        with self.assertRaises(ValueError):join(rows+[rows[0]],False)
        with self.assertRaises(ValueError):join(rows[:2],False)
        rows[2]['token_hash']='wrong'
        with self.assertRaises(ValueError):join(rows,False)
    def test_mock_rejected(self):
        r=fixture();r['evidence_kind']='mock'
        with self.assertRaises(ValueError):join([r],False)
    def test_views_not_samples(self):
        rows=[fixture(arm=a) for a in ['B','A_PUBLIC','V4']];self.assertEqual(len(join(rows,False)[1]),1)
    def test_mass_identity(self):
        s=score(fixture());self.assertAlmostEqual(s['full_nll'],s['nll']-math.log(s['label_mass']))
    def test_cast_represented_values(self):
        x=np.array([32.,32.125,31.75,0.],dtype=np.float64);self.assertTrue(np.array_equal(x,x.astype('float32').astype('float64')))
    def test_shift_invariance(self):
        b=score(fixture((1,2,0,-1)));r=fixture((2,3,1,0));r['full_lse']=5;c=score(r);self.assertTrue(np.allclose(b['q'],c['q']));self.assertEqual(b['top'],c['top'])
    def test_empty_conditional(self):self.assertIsNone(outcomes([])['conditional_gain']['rate'])
    def test_exact_tie_bin(self):self.assertEqual(gap_bin(0),'exact_tie');self.assertEqual(gap_bin(.01),'(0,0.1)')
    def test_bootstrap_paired(self):
        a=bootstrap_indices(['CODE']*4+['RETRIEVAL']*4);b=bootstrap_indices(['CODE']*4+['RETRIEVAL']*4);self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)));self.assertTrue(all(np.all(x>=4) or np.all(x<4) for x in a))
    def test_constant_association(self):self.assertIsNone(spearman([1,1,1],[0,1,2])['rho'])
    def test_frozen_inputs(self):
        s=Path(__file__).resolve().parents[1];p=json.loads((s/'protocol.json').read_text())
        for name,key in [('inputs/core.json','inputs_sha256'),('results/raw/native.json','native_sha256')]:self.assertEqual(hashlib.sha256((s/name).read_bytes()).hexdigest(),p[key])
        self.assertEqual(hashlib.sha256((s/'protocol.json').read_bytes()).hexdigest(),(s/'protocol.sha256').read_text().split()[0])
if __name__=='__main__':unittest.main()
