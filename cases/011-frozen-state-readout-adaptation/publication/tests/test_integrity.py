"""Historical identity cannot be replaced by merely refreshing public hashes."""
from pathlib import Path
import hashlib,json,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrity import safe,read,verify_original,original_file,CASE,anchored,verify_manifest

class IdentityContracts(unittest.TestCase):
    def test_original_anchor_is_the_reviewed_archive(self):
        self.assertEqual(anchored()['archive_sha256'],'dcc8c18abf7333a38024b59e498f20fcace8a95f5a6d810e1e8f5890e1c27d27')
    def test_original_bytes_pass(self):
        self.assertEqual(verify_original()['research_files'],140)
    def test_doc_overlay_is_separate_from_frozen_code(self):
        self.assertEqual(original_file(CASE,'REPORT.md'),CASE/'publication/original_docs/REPORT.md')
        self.assertEqual(original_file(CASE,'source/adapter.py'),CASE/'source/adapter.py')
    def test_path_traversal_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['../secret','/absolute','a\\b']:
                with self.assertRaises(ValueError):safe(root,name)
            (root/'link').symlink_to('/tmp')
            with self.assertRaises(ValueError):safe(root,'link')
    def test_json_duplicate_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json'
            for raw in ['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}']:
                p.write_text(raw)
                with self.assertRaises(ValueError):read(p)
    def test_rehashing_publication_cannot_validate_changed_original(self):
        with tempfile.TemporaryDirectory() as d:
            c=Path(d);(c/'publication').mkdir();(c/'raw.json').write_bytes(b'changed')
            h=hashlib.sha256(b'original').hexdigest()
            original={'research_files':{'raw.json':{'bytes':8,'sha256':h}}}
            (c/'publication/PUBLICATION_SHA256SUMS').write_text(hashlib.sha256(b'changed').hexdigest()+'  raw.json\n')
            self.assertEqual(verify_manifest(c)['status'],'PASS')
            with patch('integrity.anchored',return_value=original):
                with self.assertRaisesRegex(ValueError,'Original review file changed'):verify_original(c)
    def test_extra_files_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            c=Path(d);(c/'publication').mkdir();(c/'publication/PUBLICATION_SHA256SUMS').write_text('')
            (c/'surprise').write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'inventory'):verify_manifest(c)

if __name__=='__main__':unittest.main()
