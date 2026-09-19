"""Synthetic CPU failure cases and frozen input contracts."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.modelpack.common import CASE,digest,read
from tools.modelpack.data import scenario,render
from tools.modelpack.r_study import retained,positions
from tools.modelpack.runtime import replace
from tools.modelpack.numerics import bootstrap_indices


class Boundaries(unittest.TestCase):
    def test_frozen_unique_inputs_and_oracles(self):
        for track in ['Q','R']:
            rows=[]
            for split in ['smoke','calibration','development','heldout']:
                for row in read(CASE/'inputs'/track/(split+'.json')):
                    index=int(row['id'].split('-')[-1]);s=scenario(track,split,row['task'],index)
                    self.assertEqual(s['gold_value'],row['options'][row['gold']])
                    self.assertEqual(s['facts'],row['facts']);self.assertEqual(len(set(row['options'])),4)
                    self.assertEqual(digest(row['token_ids']),row['token_hash'])
                    self.assertFalse(row['future_answer_tokens_present']);rows.append(row)
            self.assertEqual(len({r['token_hash'] for r in rows}),len(rows))
            self.assertEqual(len({r['facts_hash'] for r in rows}),len(rows))
    def test_tokenizer_prefix_failure_blocks(self):
        class BadTokenizer:
            def apply_chat_template(self,*a,**k):return 'fixed prefix'
            def encode(self,text,**k):return list(range(40)) if text=='fixed prefix' else list(range(42))
        with self.assertRaises(ValueError):render(scenario('R','smoke','code',0),BadTokenizer(),512)
    def test_groups_positions_and_restore_exception(self):
        self.assertEqual(len(retained('I25')),2304);self.assertEqual(len(retained('S50')),1536)
        for n in [32,103,512]:
            p=positions(n);self.assertEqual(len(set(p)),32);self.assertEqual(p[0],0);self.assertEqual(p[-1],n-1)
        class Layer:mlp=object()
        l=Layer();original=l.mlp
        with self.assertRaises(RuntimeError):
            with replace(l,object()):raise RuntimeError('fixture')
        self.assertIs(l.mlp,original)
    def test_cluster_bootstrap_is_paired_and_stratified(self):
        indices=bootstrap_indices(['a','a','b','b'],20,19)
        self.assertTrue(np.all(indices[:,:2]<2));self.assertTrue(np.all(indices[:,2:]>=2))
        a=np.arange(4);b=a+100
        np.testing.assert_array_equal(b[indices]-a[indices],np.full_like(indices,100))
    def test_quantized_corrupt_scale_and_dense_fallback(self):
        import torch,json
        from safetensors.torch import save_file
        from tools.modelpack.quantized import inspect_quantized
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'config.json').write_text(json.dumps({'quantization_config':{'quant_method':'compressed-tensors'}}))
            values={'x.weight_packed':torch.zeros((2,16),dtype=torch.int32),'x.weight_scale':torch.ones((2,1),dtype=torch.bfloat16)}
            save_file(values,root/'model.safetensors');self.assertEqual(inspect_quantized(root,{'x':[2,128]})['status'],'PASS')
            values['x.weight_scale'][0,0]=0;save_file(values,root/'model.safetensors')
            with self.assertRaises(ValueError):inspect_quantized(root,{'x':[2,128]})
            values['x.weight_scale'].fill_(1);values['x.weight']=torch.ones((2,128),dtype=torch.bfloat16)
            save_file(values,root/'model.safetensors')
            with self.assertRaises(ValueError):inspect_quantized(root,{'x':[2,128]})

class EvidenceChecks(unittest.TestCase):
    def test_semantic_facts_disjoint_without_scenario_ids(self):
        seen=set()
        for track in ['Q','R']:
            for split in ['smoke','calibration','development','heldout']:
                for r in read(CASE/'inputs'/track/(split+'.json')):
                    facts=r['facts']
                    if isinstance(facts,list):facts=sorted([(f['name'],f['value']) for f in facts])
                    fingerprint=digest([r['task'],facts]);self.assertNotIn(fingerprint,seen);seen.add(fingerprint)
    def test_public_analysis_blocks_fixture_and_absent_validity(self):
        from tools.modelpack.analysis import analyze
        import json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'results/raw/Q').mkdir(parents=True)
            f=p/'results/raw/Q/model_records.json'
            f.write_text(json.dumps([{'evidence_kind':'unit_test_fixture','split':'heldout'}]))
            with self.assertRaises(ValueError):analyze(p,p/'out')
            f.write_text(json.dumps([{'evidence_kind':'gpu_measurement','split':'heldout','validity':{'full_final_vocab':True}}]))
            with self.assertRaises(ValueError):analyze(p,p/'out')
    def test_scalar_checker_rejects_altered_choice_loss(self):
        from tools.modelpack.numerics import score
        from tools.modelpack.audit import scalar_score
        r={'score':score(np.array([1.,2.,3.,4.,5.]),[0,1,2,3],0)}
        scalar_score(r);r['score']['choice_nll']+=.01
        with self.assertRaises(ValueError):scalar_score(r)
