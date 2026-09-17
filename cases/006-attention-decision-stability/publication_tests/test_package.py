"""Publication-only integrity tests; historical experiment tests stay unchanged."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_publication import verify, sha, safe


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.case = Path(self.tmp.name)
        (self.case / 'provenance').mkdir()
        (self.case / 'results').mkdir()
        (self.case / 'results/raw.json').write_text('{"measured":true}\n')
        self.record = {'original_editorial_changes': {},
                       'original_review_files': {'results/raw.json': sha(self.case / 'results/raw.json')},
                       'supplement_review_files': {},
                       'publication_added_files': ['provenance/publication_snapshot.json'],
                       'frozen_identities': {'results/raw.json': sha(self.case / 'results/raw.json')}}
        self.freeze()

    def freeze(self):
        (self.case / 'provenance/publication_snapshot.json').write_text(json.dumps(self.record))
        files = sorted(p for p in self.case.rglob('*') if p.is_file() and p.name != 'PUBLICATION_SHA256SUMS')
        (self.case / 'PUBLICATION_SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(self.case).as_posix()+'\n' for p in files))

    def test_valid_snapshot(self):
        self.assertEqual(verify(self.case)['status'], 'PASS')

    def test_rehashing_does_not_authorize_raw_change(self):
        (self.case / 'results/raw.json').write_text('{"measured":false}\n')
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'reviewed evidence changed'):
            verify(self.case)

    def test_unapproved_extra_file(self):
        (self.case / 'extra.json').write_text('{}')
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'outside approved'):
            verify(self.case)

    def test_duplicate_manifest(self):
        p = self.case / 'PUBLICATION_SHA256SUMS'
        p.write_text(p.read_text() + p.read_text().splitlines()[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            verify(self.case)

    def test_symlink_rejected(self):
        (self.case / 'link').symlink_to(self.case / 'results/raw.json')
        with self.assertRaisesRegex(ValueError, 'Symlinks'):
            verify(self.case)

    def test_paths(self):
        for name in ('../x', '/absolute', 'dir/../../x', 'drive:thing', 'dir\\x', ''):
            self.assertFalse(safe(name))
        self.assertTrue(safe('results/raw.json'))

if __name__ == '__main__':
    unittest.main()
