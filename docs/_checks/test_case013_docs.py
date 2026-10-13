"""Do not allow an incomplete Case013 folder to weaken preservation checks."""
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_docs import check_case013

class Case013Docs(unittest.TestCase):
    def test_missing_case_is_not_added(self):
        with tempfile.TemporaryDirectory() as t:
            errors=[];self.assertEqual(check_case013(Path(t),errors),set());self.assertEqual(errors,[])
    def test_incomplete_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);case=root/'cases/013-dualquant-reference-quality';case.mkdir(parents=True)
            (case/'README.md').write_text('not a verified publication')
            errors=[];self.assertEqual(check_case013(root,errors),set());self.assertTrue(errors)

if __name__=='__main__':unittest.main()
