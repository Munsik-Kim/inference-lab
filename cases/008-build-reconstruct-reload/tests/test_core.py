"""CPU fixtures only; never mixed with model measurements."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.modelpack.common import checksums, read, safe_name, verify_checksums, write
from tools.modelpack.numerics import fit_ridge, norm_record, paired, recovery, score, full_kl


class Numerics(unittest.TestCase):
    def test_ridge_recovers_known_linear_target(self):
        rng = np.random.default_rng(8); h = rng.normal(size=(5, 100)); w = rng.normal(size=(3, 5))
        w0 = w + .3; y = w@h; g=h@h.T/100; b=y@h.T/100
        fitted, info = fit_ridge(g, b, w0, 1e-4)
        self.assertLess(np.linalg.norm(fitted@h-y), np.linalg.norm(w0@h-y)*.001)
        self.assertLess(info['normal_equation_relative_residual'], 1e-10)
        np.testing.assert_allclose(fitted, np.linalg.solve(g+info['lambda']*np.eye(5),(b+info['lambda']*w0).T).T)

    def test_ridge_rejects_invalid_inputs(self):
        for g in [np.zeros((2,2)),np.array([[1,2],[0,1.]]),np.full((2,2),np.nan)]:
            with self.assertRaises(ValueError): fit_ridge(g,np.ones((3,2)),np.ones((3,2)),.1)
        with self.assertRaises(ValueError): fit_ridge(np.eye(2),np.ones((3,2)),np.ones((3,2)),0)

    def test_nll_mass_and_gold(self):
        z=np.array([1.,2.,2.,0.,3.]);s=score(z,[0,1,2,3],1)
        self.assertEqual(s['top_ties'],[1,2]);self.assertEqual(s['prediction'],1)
        self.assertAlmostEqual(s['full_gold_nll'],s['choice_nll']-np.log(s['allowed_mass']))
        self.assertAlmostEqual(full_kl(z,z+99),0)
        np.testing.assert_allclose(score(z+99,[0,1,2,3],1)['choice_probabilities'],s['choice_probabilities'])

    def test_full_output_nan_blocks_sampled_valid_scores(self):
        with self.assertRaises(ValueError):score(np.array([1.,2.,3.,4.,np.nan]),[0,1,2,3],0)
        with self.assertRaises(ValueError):norm_record(np.array([0.,np.inf]),np.ones(2))

    def test_transitions(self):
        b=score(np.array([3.,2.,1.,0.]),list(range(4)),0)
        c=score(np.array([0.,2.,1.,0.]),list(range(4)),0)
        self.assertTrue(paired(b,c)['regression']);self.assertTrue(paired(c,b)['gain'])
        d=score(np.array([0.,1.,2.,0.]),list(range(4)),0)
        self.assertTrue(paired(c,d)['wrong_to_different_wrong'])

    def test_recovery_is_energy_not_accuracy(self):
        self.assertEqual(recovery(np.array([1.,3.]),np.array([1.,1.])),.5)
        self.assertIsNone(recovery(np.zeros(2),np.zeros(2)))
        self.assertLess(recovery(np.ones(2),np.ones(2)*2),0)


class Storage(unittest.TestCase):
    def test_reject_traversal_and_duplicates(self):
        for name in ['../x','/a','a/../x','a\\x','C:/x']:
            with self.assertRaises(ValueError):safe_name(name)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';p.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):read(p)

    def test_checksums_cover_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);write(p/'x.json',{'fixture':1});checksums(p);verify_checksums(p)
            (p/'extra').write_text('unexpected')
            with self.assertRaises(ValueError):verify_checksums(p)
            with self.assertRaises(ValueError):write(p/'x.json',{'overwrite':True})


class DenseArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        cls.torch=torch;torch.set_num_threads(2);torch.manual_seed(8)
        cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                         num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=True)
        cfg._attn_implementation='sdpa'
        old=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16);cls.model=Qwen3ForCausalLM(cfg).eval()
        finally:torch.set_default_dtype(old)

    def test_noop_separate_process_different_directory(self):
        from tools.modelpack.artifact import save_dense,load_dense,tensor_hash
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);save_dense(self.model,p/'built',{'kind':'unit_test_fixture'},layer=1)
            shutil.copytree(p/'built',p/'copied');shutil.rmtree(p/'built')
            inputs=[{'id':'fixture','token_ids':[1,2,3,4]}];write(p/'input.json',inputs)
            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'}
            reports=[]
            for index in range(2):
                out=p/f'reload-{index}.json'
                subprocess.run([sys.executable,'-B','-m','tools.modelpack','reload-test','--artifact',str(p/'copied'),
                                '--input',str(p/'input.json'),'--output',str(out)],cwd=ROOT,env=env,check=True,capture_output=True)
                reports.append(read(out))
            self.assertEqual(reports[0],reports[1])
            with self.torch.inference_mode():
                z=self.model(input_ids=self.torch.tensor([[1,2,3,4]]),logits_to_keep=1).logits
            self.assertEqual(tensor_hash(z),reports[0]['rows'][0]['full_logits_hashes'][0])
            loaded=load_dense(p/'copied')
            self.assertIs(loaded.lm_head.weight,loaded.model.embed_tokens.weight)

    def test_sliced_shape_strict_reload_and_bad_version(self):
        from tools.modelpack.artifact import save_dense,load_dense,slice_mlp
        model=copy.deepcopy(self.model);ids=list(range(48));model.model.layers[1].mlp=slice_mlp(model.model.layers[1].mlp,ids)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'artifact';save_dense(model,p,{'kind':'unit_test_fixture'},ids,1)
            loaded=load_dense(p)
            self.assertEqual(loaded.model.layers[0].mlp.intermediate_size,64)
            self.assertEqual(loaded.model.layers[1].mlp.intermediate_size,48)
            info=read(p/'artifact.json');info['loader_version']='wrong';(p/'artifact.json').write_text(json.dumps(info))
            (p/'checksums.sha256').unlink();checksums(p)
            with self.assertRaises(ValueError):load_dense(p)

    def test_rehash_cannot_hide_shape_or_missing_tensor(self):
        from safetensors.torch import load_file,save_file
        from tools.modelpack.artifact import save_dense,inspect_artifact
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'artifact';save_dense(self.model,p,{'kind':'unit_test_fixture'})
            state=load_file(p/'weights/model.safetensors');del state['model.layers.0.mlp.gate_proj.weight']
            save_file(state,p/'weights/model.safetensors');(p/'checksums.sha256').unlink();checksums(p)
            with self.assertRaises(ValueError):inspect_artifact(p)


if __name__=='__main__':unittest.main()
