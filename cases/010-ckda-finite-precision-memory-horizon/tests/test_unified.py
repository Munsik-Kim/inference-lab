"""Integration contracts; tiny identity fixtures are synthetic, never measured evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

CASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CASE/'scripts'))
import verify_unified as verifier
from restore_workspace import restore
from derive_summary import derive

class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.case=Path(self.tmp.name)/'case';self.case.mkdir()
        files=[]
        for version in ('v1','v2'):
            p=self.case/f'versions/{version}/sample.txt';p.parent.mkdir(parents=True);p.write_bytes(version.encode())
            files.append({'version':version,'path':p.relative_to(self.case).as_posix(),'original_path':verifier.ROOTS[version]+'sample.txt','bytes':2,'sha256':verifier.sha(p)})
        (self.case/'provenance').mkdir()
        self.manifest=self.case/'provenance/snapshot_manifest.json'
        self.manifest.write_text(json.dumps({'files':files}))
        for mock in (patch.object(verifier,'SNAPSHOT_SHA256',verifier.sha(self.manifest)),patch.object(verifier,'COUNTS',{'v1':1,'v2':1})):
            mock.start();self.addCleanup(mock.stop)
    def current(self):
        lines=[f'{verifier.sha(self.case/p)}  {p}' for p in verifier.inventory(self.case) if p!='UNIFIED_SHA256SUMS']
        (self.case/'UNIFIED_SHA256SUMS').write_text('\n'.join(lines)+'\n')
    def test_original_and_current_identity_pass(self):
        self.current();self.assertEqual(verifier.verify(self.case,derived=False)['status'],'PASS')
    def test_protected_edit_rejected_after_current_rehash(self):
        (self.case/'versions/v1/sample.txt').write_bytes(b'xx');self.current()
        with self.assertRaisesRegex(ValueError,'protected snapshot'):verifier.verify(self.case,derived=False)
    def test_original_manifest_rehash_rejected(self):
        (self.case/'versions/v1/sample.txt').write_bytes(b'xx')
        data=json.loads(self.manifest.read_text());data['files'][0]['sha256']=verifier.sha(self.case/'versions/v1/sample.txt')
        self.manifest.write_text(json.dumps(data));self.current()
        with self.assertRaisesRegex(ValueError,'original snapshot identity'):verifier.verify(self.case,derived=False)
    def test_extra_original_file_rejected(self):
        (self.case/'versions/v1/extra').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError,'inventory mismatch'):verifier.verify_snapshots(self.case)
    def test_missing_original_file_rejected(self):
        (self.case/'versions/v1/sample.txt').unlink()
        with self.assertRaisesRegex(ValueError,'inventory mismatch'):verifier.verify_snapshots(self.case)
    def test_symlink_rejected(self):
        (self.case/'versions/v1/link').symlink_to('sample.txt')
        with self.assertRaisesRegex(ValueError,'symlink'):verifier.verify_snapshots(self.case)
    def test_traversal_and_noncanonical_paths_rejected(self):
        for name in ('../x','/x','C:/x','a\\b','a//b','a/./b','a/../b',''):
            with self.subTest(name=name),self.assertRaises(ValueError):verifier.safe_path(name)
    def test_duplicate_current_entry_rejected(self):
        self.current();p=self.case/'UNIFIED_SHA256SUMS';p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError,'duplicate'):verifier.verify(self.case,derived=False)
    def test_current_missing_or_extra_entry_rejected(self):
        self.current();(self.case/'extra.txt').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError,'current integration manifest'):verifier.verify(self.case,derived=False)
    def test_restore_historical_paths_exactly(self):
        out=Path(self.tmp.name)/'restored';restore(self.case,out)
        for version in ('v1','v2'):
            self.assertEqual((out/verifier.ROOTS[version]/'sample.txt').read_bytes(),version.encode())
    def test_restore_refuses_existing_and_internal_destinations(self):
        for out in (self.case,self.case/'new'):
            with self.subTest(out=out.name),self.assertRaises(ValueError):restore(self.case,out)
    def test_integrated_document_not_part_of_original_inventory(self):
        (self.case/'README.md').write_text('integration layer')
        self.current();self.assertEqual(verifier.verify(self.case,derived=False)['status'],'PASS')

class ScalarMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=derive(CASE)
    def test_summary_matches_independent_regeneration(self):
        self.assertEqual(self.data,json.loads((CASE/'summary/project.json').read_text()))
    def test_stages_and_confidence_families_remain_separate(self):
        self.assertEqual(self.data['stages']['v1']['N'],512)
        self.assertEqual(self.data['stages']['v2']['N'],1024)
        self.assertEqual(self.data['stages']['v1']['families'],[546,630])
        self.assertEqual(self.data['stages']['v2']['family'],195)
        self.assertFalse(self.data['fresh']['pooled_checkpoint_estimate'])
    def test_bytes_and_mean_difference_have_correct_units(self):
        head=self.data['headline'];self.assertEqual((head['native_stream_bytes'],head['int8_stream_bytes']),(12305,3137))
        self.assertAlmostEqual(head['reduction_pct'],100*(1-3137/12305))
        self.assertTrue(all(abs(r['delta_tokens'])<1 for r in head['int8_minus_native_RMST0_by_seed']))
    def test_rank2_mixed_negative_checkpoint_preserved(self):
        pairs=[r for r in self.data['fresh']['paired_comparisons'] if r['baseline']=='MIXED_5_6_BUDGET']
        self.assertEqual([r['RMST0_delta']>0 for r in pairs],[True,False,True])
        self.assertEqual(self.data['storage']['comparison_stream_count'],128)
        self.assertEqual(self.data['storage']['evaluation_sequence_count'],1024)
    def test_zero_terminal_does_not_mean_all_tokens_correct(self):
        self.assertTrue(all(r['terminal_count']==0 for r in self.data['fresh']['rows']))
        self.assertTrue(all(r['token_accuracy']<1 for r in self.data['fresh']['rows']))
    def test_sources_identify_actual_original_bytes(self):
        self.assertTrue(self.data['sources'])
        for source in self.data['sources']:
            self.assertTrue(source['path'].startswith('versions/'))
            self.assertEqual(source['sha256'],verifier.sha(CASE/source['path']))

if __name__=='__main__':unittest.main()
