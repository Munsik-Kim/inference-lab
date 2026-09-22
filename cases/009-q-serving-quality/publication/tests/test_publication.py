"""Current identity, original snapshot and diagnostic contracts; CPU only."""
import copy,hashlib,importlib.util,json,shutil,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
PUB=ROOT/'cases/009-q-serving-quality/publication'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v=module('review_publication',PUB/'verify_publication.py')
a=module('review_posthoc',PUB/'posthoc/analyze.py')

class PublicationTests(unittest.TestCase):
    def copy_root(self,tmp):
        r=Path(tmp)/'repo';r.mkdir()
        for name in ['cases/009-q-serving-quality','packages/diova-compare']:shutil.copytree(ROOT/name,r/name)
        (r/'downloads').mkdir()
        for name in ['diova_compare-0.1.1-py3-none-any.whl','diova_compare-0.1.1.metadata.json']:shutil.copyfile(ROOT/'downloads'/name,r/'downloads'/name)
        return r
    def rehash(self,r):
        p=r/v.CASE/'publication/PUBLICATION_SHA256SUMS';p.write_text(''.join(h+'  '+n+'\n' for n,h in sorted(v.inventory(r).items())))
    def test_current_reviewed_edits_accepted(self):self.assertEqual(v.verify(ROOT)['status'],'PASS')
    def test_rehash_cannot_hide_protected_change(self):
        for rel in ['results/raw/quality_scalars.jsonl','configs/serving_protocol.json','results/derived/quality_summary.json']:
            with self.subTest(path=rel),tempfile.TemporaryDirectory() as t:
                r=self.copy_root(t);p=r/v.CASE/rel;p.write_bytes(p.read_bytes()+b'\n');self.rehash(r)
                with self.assertRaisesRegex(ValueError,'protected original changed'):v.verify(r)
    def test_overlay_cannot_change(self):
        with tempfile.TemporaryDirectory() as t:
            r=self.copy_root(t);p=r/v.CASE/'publication/original_overlay'/v.CASE/'README.md';p.write_text('modified');self.rehash(r)
            with self.assertRaisesRegex(ValueError,'editorial original copy'):v.verify(r)
    def test_unapproved_added_file_even_rehashed(self):
        with tempfile.TemporaryDirectory() as t:
            r=self.copy_root(t);(r/v.CASE/'unexpected.txt').write_text('x');self.rehash(r)
            with self.assertRaisesRegex(ValueError,'unapproved case additions'):v.verify(r)
    def test_duplicate_manifest_and_unsafe_path(self):
        with tempfile.TemporaryDirectory() as t:
            r=self.copy_root(t);p=r/v.CASE/'publication/PUBLICATION_SHA256SUMS';s=p.read_text();p.write_text(s+s.splitlines()[0]+'\n')
            with self.assertRaisesRegex(ValueError,'duplicate manifest'):v.verify(r)
        for path in ['../escape','/abs','C:/data','x\\y']:
            with self.assertRaisesRegex(ValueError,'unsafe'):v.safe(path)
    def test_missing_current_file(self):
        with tempfile.TemporaryDirectory() as t:
            r=self.copy_root(t);(r/v.CASE/'results/raw/quality_scalars.jsonl').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):v.verify(r)
    def test_posthoc_reference_denominators(self):
        d=a.analyze(ROOT/v.CASE);m=d['mmlu'];self.assertEqual([m[k] for k in ['n_items','n_subjects','subjects_declining','baseline_correct','candidate_correct','lost','gained']],[14042,57,57,8720,7616,1452,348])
        self.assertEqual(m['D_choice_counts'],{'BF16':5842,'W4':7163});self.assertEqual(d['wikitext']['lower_candidate_loglikelihood'],62)
        self.assertAlmostEqual(d['wikitext']['word_PPL_relative_increase'],.10766682980643205)
        self.assertEqual(d['gsm8k']['filter_views']['W4']['strict_correct_flexible_wrong'],2)
        self.assertEqual(d['serving']['total_preemptions'],324)
    def test_lifecycle_status_not_historical_repair(self):
        m=module('life_contract',ROOT/v.CASE/'supplemental/lifecycle-v1/validate.py');d=json.loads((ROOT/v.CASE/'supplemental/lifecycle-v1/summary.json').read_text());m.validate(d)
        bad=copy.deepcopy(d);bad['ledger'][0]['returncode']=-6
        with self.assertRaises(ValueError):m.validate(bad)
    def test_parent_rejects_bad_numeric_and_shape(self):
        m=module('life_parent',ROOT/v.CASE/'supplemental/lifecycle-v1/run.py')
        self.assertFalse(m.valid_result([float('inf')],'loglikelihood_rolling',1))
        self.assertFalse(m.valid_result([[1.,'yes']],'loglikelihood',1))
        self.assertTrue(m.valid_result([[-2.,False]],'loglikelihood',1))
    def test_original_manifest_stays_historical(self):
        restore=module('restore',PUB/'restore_review.py')
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)/'old';restore.restore(ROOT,dest)
            original=module('old_public',dest/v.CASE/'scripts/verify_public.py')
            self.assertEqual(original.manifest(dest/v.CASE),56)
            self.assertNotEqual((ROOT/v.CASE/'README.md').read_bytes(),(dest/v.CASE/'README.md').read_bytes())
if __name__=='__main__':unittest.main()
