"""CPU publication fixtures: preserve scientific checks and exercise failure paths."""
from pathlib import Path
import ast
import copy
import json
import math
import re
import shutil
import sys
import tempfile
import unittest

PUB=Path(__file__).resolve().parents[1];ROOT=PUB.parents[2];CASE=PUB.parent
sys.path.insert(0,str(PUB));sys.path.insert(0,str(ROOT))
from verify_publication import (CASE_PATH,check_original,check_manifest,check_documents,check_posthoc,
                                package_files,safe,verify)
from restore_original import restore
from tables import validate_tables
from recalculate import scores,analyze,read

def dump(path,data):path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')

class OriginalProtection(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'repo';self.root.mkdir()
        shutil.copytree(CASE,self.root/CASE_PATH)
        shutil.copytree(ROOT/'tools/modelpack',self.root/'tools/modelpack')
        shutil.copyfile(ROOT/'LICENSE',self.root/'LICENSE');self.case=self.root/CASE_PATH
    def tearDown(self):self.tmp.cleanup()
    def rehash(self):
        (self.case/'PUBLICATION_SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for n,h in package_files(self.root).items()))
    def test_current_publication(self):self.assertEqual(verify(self.root)['original_files_verified'],105)
    def test_rehash_cannot_hide_raw_freeze_code_changes(self):
        for name in ['results/raw/Q/model_records.json','configs/build_freeze.json','results/derived/summary.json']:
            p=self.case/name;old=p.read_bytes();p.write_bytes(old+b'\n');self.rehash()
            with self.assertRaises(ValueError):check_original(self.root)
            p.write_bytes(old)
        p=self.root/'tools/modelpack/numerics.py';p.write_text(p.read_text()+'\n');self.rehash()
        with self.assertRaises(ValueError):check_original(self.root)
    def test_editorial_requires_original_and_published_hash(self):
        p=self.case/'README.md';p.write_text(p.read_text()+'\n');self.rehash()
        with self.assertRaises(ValueError):check_original(self.root)
    def test_recorded_editorial_edit_allowed(self):check_original(self.root)
    def test_missing_file(self):
        (self.case/'inputs/Q/smoke.json').unlink()
        with self.assertRaises(ValueError):check_original(self.root)
    def test_extra_tool_and_case_files_rejected(self):
        for p in [self.root/'tools/modelpack/new.py',self.case/'unknown.json']:
            p.write_text('{}');self.rehash()
            with self.assertRaises(ValueError):check_original(self.root)
            p.unlink()
    def test_duplicate_manifest(self):
        p=self.case/'PUBLICATION_SHA256SUMS';p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
        with self.assertRaises(ValueError):check_manifest(self.root)
    def test_caches_and_weight_payloads_rejected(self):
        for name in ['publication/__pycache__/test.pyc','weights.safetensors','vectors.npy']:
            p=self.case/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
            with self.assertRaises(ValueError):package_files(self.root)
            p.unlink()
            if p.parent.name=='__pycache__':p.parent.rmdir()
    def test_archived_doc_corruption(self):
        p=self.case/'publication/original_docs/README.md';p.write_text('changed')
        with self.assertRaises(ValueError):check_original(self.root)
    def test_restore_exact_original_and_no_overwrite(self):
        dest=Path(self.tmp.name)/'historical';self.assertEqual(restore(self.root,dest)['files'],105)
        old=read(self.case/'publication/original_inventory.json')
        from recalculate import sha
        self.assertEqual({p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()},old)
        with self.assertRaises(ValueError):restore(self.root,dest)
        self.assertFalse((dest/CASE_PATH/'publication').exists())
    def test_source_map_duplicate_and_changed_hash(self):
        p=self.case/'publication/source_map.json';d=read(p);d['claims'].append(d['claims'][0]);dump(p,d)
        with self.assertRaises(ValueError):check_documents(self.root)
    def test_stale_table_denominator_or_sign_rejected(self):
        p=self.case/'REPORT.md';old=p.read_text()
        for before,after in [('137/192','137/193'),('-0.591295','+0.591295'),('Mean relative error','Median relative error')]:
            self.assertIn(before,old);p.write_text(old.replace(before,after,1))
            with self.assertRaises(ValueError):validate_tables(self.case)
        p.write_text(old)
    def test_section_order(self):
        p=self.case/'REPORT.md';p.write_text(p.read_text().replace('id="theory"','id="unknown"'))
        with self.assertRaises(ValueError):check_documents(self.root)

class ScalarDiagnostics(unittest.TestCase):
    def test_stable_nll_identity_and_mean_log_mass(self):
        row=read(CASE/'results/raw/Q/model_records.json')[0];s=scores(row)
        self.assertAlmostEqual(s['full'],s['choice']+s['mass_nll'])
        masses=[.1,.9];self.assertNotAlmostEqual(sum(-math.log(x) for x in masses)/2,-math.log(sum(masses)/2))
    def test_nonfinite_is_not_a_tie(self):
        row=read(CASE/'results/raw/Q/model_records.json')[0];row['score']['option_logits'][2]=float('nan')
        with self.assertRaises(ValueError):scores(row)
    def test_input_join_full_validity_and_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            c=Path(td)/'case';shutil.copytree(CASE,c)
            p=c/'results/raw/Q/model_records.json';rows=read(p)
            for defect in ['duplicate','missing','token','validity']:
                d=copy.deepcopy(rows)
                if defect=='duplicate':d.append(d[0])
                elif defect=='missing':d.pop()
                elif defect=='token':d[0]['token_hash']='wrong'
                else:d[0]['validity'].pop('full_model_outputs')
                dump(p,d)
                with self.assertRaises(ValueError):analyze(c)
    def test_review_and_generator_reconstruction(self):check_posthoc(CASE)
    def test_safe_paths(self):
        for n in ['../x','/x','C:/x','a\\b','a//b','']:
            with self.assertRaises(ValueError):safe(n)
        self.assertEqual(safe('a/b.json'),'a/b.json')

class DocumentationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc=(CASE/'REPRODUCTION.md').read_text()
        cls.blocks=re.findall(r"<<'PY'\n(.*?)\nPY",cls.doc,re.S)
    def test_finalizer_signature_and_documented_arguments(self):
        tree=ast.parse((ROOT/'tools/modelpack/qartifact.py').read_text())
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='finalize')
        self.assertEqual([a.arg for a in fn.args.args],['build','output'])
        self.assertIn('finalize(Path(sys.argv[1]), Path(sys.argv[2]))',self.blocks[0])
        self.assertIn('"$NEW_Q_BUILD" "$NEW_Q_FINALIZED"',self.doc)
    def test_copy_then_inspect_then_q_smoke(self):
        tokens=['build-q --snapshot','from tools.modelpack.qartifact import finalize','Q COPY PASS',
                'inspect-artifact --artifact "$PRIVATE_Q_COPY"','evaluate --track Q --artifact "$PRIVATE_Q_COPY"']
        places=[self.doc.index(t) for t in tokens];self.assertEqual(places,sorted(places))
        self.assertNotIn('reload-test --artifact "$PRIVATE_Q_COPY"',self.doc)
    def test_actual_copy_blocks_with_spaces_and_refuse_existing(self):
        from tools.modelpack.common import checksums,verify_checksums
        for index,arms in [(1,None),(2,['R-B','I25','I25-R','P25','P25-R','S50','S50-R'])]:
            with tempfile.TemporaryDirectory() as td:
                base=Path(td);source=base/'source spaces';target=base/'copy spaces';source.mkdir()
                for folder in [source] if arms is None else [source/x for x in arms]:
                    folder.mkdir(exist_ok=True);(folder/'fixture.json').write_text('{"evidence_kind":"unit_test_fixture"}');checksums(folder)
                prior=sys.argv;sys.argv=['copy',str(source),str(target)]
                try:
                    exec(compile(self.blocks[index],'<trusted-doc-copy-fixture>','exec'),{})
                    with self.assertRaises(ValueError):exec(compile(self.blocks[index],'<trusted-doc-copy-fixture>','exec'),{})
                finally:sys.argv=prior
                for arm in arms or ['']:
                    self.assertEqual(verify_checksums(source/arm),verify_checksums(target/arm))
    def test_r_export_root_vs_individual_artifact(self):
        code=(ROOT/'tools/modelpack/r_study.py').read_text()
        self.assertIn("out/'R-B'",code);self.assertIn('out/arm',code);self.assertIn("load_dense(artifacts/arm,'cuda')",code)
        self.assertIn('inspect-artifact --artifact "$PRIVATE_R_COPY_ROOT/I25-R"',self.doc)
        self.assertIn('evaluate --track R --artifact "$PRIVATE_R_COPY_ROOT"',self.doc)

if __name__=='__main__':unittest.main()
