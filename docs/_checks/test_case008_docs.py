"""A declared archive source must not make arbitrary cases writable."""
from pathlib import Path
import tempfile
import unittest
from check_docs import check_additional_publications, check_claims

class AdditionalPublication(unittest.TestCase):
    def test_unknown_publication_is_rejected(self):
        errors=[]
        self.assertEqual(check_additional_publications(Path('.'),{'additional_publications':[{'case_path':'cases/other','archive_sha256':'0'*64}]},errors),set())
        self.assertTrue(errors)
    def test_missing_publication_not_authorized(self):
        with tempfile.TemporaryDirectory() as d:
            errors=[]
            known=check_additional_publications(Path(d),{'additional_publications':[{'case_path':'cases/008-build-reconstruct-reload','archive_sha256':'49cc2640b63d7662cf3d40c168eaf036505bbf5f7f5f73de8de48099d452daca'}]},errors)
            self.assertEqual(known,set());self.assertTrue(errors)
    def test_case_eight_navigation_and_report_exists(self):
        root=Path(__file__).resolve().parents[2]
        for lang,report in [('en','REPORT.md'),('ko','REPORT.ko.md')]:
            text=(root/f'docs/{lang}/CASEBOOK.md').read_text()
            self.assertIn('id="case-008"',text)
            self.assertIn('cases/008-build-reconstruct-reload/'+report,text)
            self.assertTrue((root/'cases/008-build-reconstruct-reload'/report).is_file())
