"""Presentation tests; synthetic fixtures never enter measured data files."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/showcase'))
from common import C7, read, new_output, safe_name, script_json, source_manifest, checked_read
from data import index, score, transition, valid6, load, excerpt
from build import build
from check import check
from replay_selection import validate, replay

class Contracts(unittest.TestCase):
    def test_duplicate_join_rejected(self):
        with self.assertRaises(ValueError):index([{'id':'a'},{'id':'a'}],('id',))
    def test_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):score([1,float('nan'),0,2],0,1,1,1,1,0)
    def test_missing_full_validity_rejected(self):
        with self.assertRaises(ValueError):valid6({'validity':{'full_logits_finite':True}})
    def test_tie_first_index_rule(self):
        import math
        s=score([0,0,0,0],0,math.log(4),.75,0,.9,0)
        self.assertEqual(s['top'],[0,1,2,3]);self.assertEqual(s['gap'],0)
        with self.assertRaises(ValueError):score([0,0,0,0],0,math.log(4),.75,0,.9,1)
    def test_gold_pair_mismatch_rejected(self):
        with self.assertRaises(ValueError):transition({'gold':0},{'gold':1})
    def test_script_text_is_escaped(self):
        text=script_json({'prompt':'</script><script>window.pwn=1</script>\u2028'})
        self.assertNotIn('<',text);self.assertNotIn('\u2028',text)
        self.assertEqual(json.loads(text)['prompt'],'</script><script>window.pwn=1</script>\u2028')
    def test_filler_only_is_omitted(self):
        self.assertEqual(excerpt('fact<irrelevant_notes>x</irrelevant_notes>question'),'fact[irrelevant filler omitted]question')
    def test_unsafe_paths(self):
        for path in ('../a','/a','a/../b','a\\b','C:a'):
            with self.assertRaises(ValueError):safe_name(path)
    def test_source_overwrite_rejected(self):
        with self.assertRaises(ValueError):new_output(ROOT,ROOT/'cases/new')
    def test_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.json';p.write_text('{"x":1,"x":2}')
            with self.assertRaises(ValueError):read(p)
    def test_changed_source_hash_rejected(self):
        m=source_manifest(ROOT);m['files'][C7+'/results/raw/selection.json']='0'*64
        with self.assertRaises(ValueError):checked_read(ROOT,C7+'/results/raw/selection.json',m)
    def test_missing_source_rejected(self):
        with self.assertRaises(ValueError):checked_read(ROOT,'cases/missing.json',source_manifest(ROOT))

class Selector(unittest.TestCase):
    def setUp(self):
        self.q=read(ROOT/C7/'results/raw/selection.json')['Q']
        self.groups=read(ROOT/C7/'configs/protocol.json')['groups']
    def test_actual_frozen_replay(self):
        r=replay(ROOT);self.assertEqual(r['model_forwards'],0)
        self.assertTrue(all(x['matches_recorded'] for x in r['results'].values()))
    def test_bad_budget(self):
        for v in (0,3,16,True,4.0):
            with self.assertRaises(ValueError):validate(self.q,self.groups,v)
    def test_shape_finite_symmetry(self):
        for q in ([[0]],[[float('inf')]*16]*16):
            with self.assertRaises(ValueError):validate(q,self.groups,4)
        q=copy.deepcopy(self.q);q[0][1]+=1
        with self.assertRaises(ValueError):validate(q,self.groups,4)
    def test_duplicate_groups(self):
        groups=copy.deepcopy(self.groups);groups[1][0]=groups[0][0]
        with self.assertRaises(ValueError):validate(self.q,groups,4)
    def test_original_tie_policy(self):
        import numpy as np
        spec=importlib.util.spec_from_file_location('showcase_test_original',ROOT/C7/'src/core.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertEqual(m.select(np.zeros((16,16)),4,'PAIRWISE')['removed'],[0,1,2,3])

class Artifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.site=Path(cls.tmp.name)/'site'
        build(ROOT,cls.site,'/inference-lab/')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_actual_artifact(self):self.assertEqual(check(ROOT,self.site)['status'],'PASS')
    def test_unlisted_deploy_file(self):
        p=self.site/'unlisted.txt';p.write_text('fixture')
        try:
            with self.assertRaises(ValueError):check(ROOT,self.site)
        finally:p.unlink()
    def test_tampering_cannot_be_repaired_by_rehash_only(self):
        p=self.site/'data/case007.json';m=self.site/'build_manifest.json';old=p.read_bytes();om=m.read_bytes()
        try:
            d=read(p);d['records'][0]['delta_nll']+=1;p.write_text(json.dumps(d))
            from common import sha
            md=read(m);md['files']['data/case007.json']=sha(p);m.write_text(json.dumps(md))
            with self.assertRaises(ValueError):check(ROOT,self.site)
        finally:p.write_bytes(old);m.write_bytes(om)
    def test_no_raw_tokens_or_full_vectors_in_display(self):
        for case in ('case006','case007'):
            d=read(self.site/f'data/{case}.json')
            self.assertTrue(all('token_ids' not in p for p in d['prompts'].values()))
    def test_language_keys_equal(self):
        self.assertEqual(set(read(ROOT/'presentation/content/en.json')),set(read(ROOT/'presentation/content/ko.json')))
    def test_home_diagram_matches_recorded_groups(self):
        from html.parser import HTMLParser
        class Diagram(HTMLParser):
            def __init__(self):
                super().__init__();self.rows=[];self.current=None
            def handle_starttag(self,tag,attrs):
                a=dict(attrs)
                if a.get('class')=='group-strip':
                    self.current=[];self.rows.append(self.current)
                if tag=='span' and 'group-cell' in a.get('class','').split():
                    self.current.append('removed' in a['class'].split())
        selection=read(ROOT/C7/'results/raw/selection.json')['selections']
        expected=[[],selection['INDEPENDENT_4']['removed'],selection['PAIRWISE_4']['removed']]
        for lang in ('en','ko'):
            d=Diagram();d.feed((self.site/lang/'index.html').read_text())
            self.assertEqual([len(row) for row in d.rows],[16,16,16])
            self.assertEqual([[i for i,v in enumerate(row) if v] for row in d.rows],expected)
    def test_home_rejects_inconsistent_selections(self):
        from layout import home
        d=read(self.site/'data/case007.json')
        row=next(r for r in d['records'] if r['arm']=='PAIRWISE' and r['budget']=='4')
        row['selected_groups']=[0,1,2,3]
        with self.assertRaisesRegex(ValueError,'Inconsistent recorded selection'):
            home({},'en',{'case007':d},[],[],'fixture')
    def test_reused_views_are_not_more_scenarios(self):
        d=read(self.site/'data/case006.json')
        self.assertEqual(len({r['id'] for r in d['records']}),238)
        self.assertEqual(d['n_independent'],{'standard':192,'boundary_pool':46})
    def test_missing_pair_or_wrong_native_view_cannot_hide(self):
        d=read(self.site/'data/case006.json')
        for item in d['prompts']:
            rows=[r for r in d['records'] if r['id']==item]
            self.assertEqual({(r['arm'],r['readout']) for r in rows},{(a,v) for a in ('A_PUBLIC','V4') for v in ('H_NATIVE','H_FP32')})

if __name__=='__main__':unittest.main()
