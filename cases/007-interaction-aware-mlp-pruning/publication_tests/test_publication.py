"""Publication-boundary fixtures only; no mock measurements are exported."""
import copy
import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
CASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CASE/'scripts'))
import verify_publication as v
from package_publication import verify_zip

class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.original={name:'a'*64 for name in ['README.md','results/raw/sample.json',
                     'configs/protocol.json','results/raw/selection.json','results/derived/summary.json','SHA256SUMS']}
        self.actual=dict(self.original,**{n:'b'*64 for n in v.ADDED})
        self.edits={}
    def check(self):
        current={n:h for n,h in self.actual.items() if n!='PUBLICATION_SHA256SUMS'}
        v.verify_maps(self.original,self.actual,self.edits,sorted(v.ADDED),current)
    def test_valid_unchanged_original(self):self.check()
    def test_rehash_does_not_hide_protected_change(self):
        for name in ['results/raw/sample.json','configs/protocol.json','results/raw/selection.json','results/derived/summary.json']:
            with self.subTest(name=name):
                old=self.actual[name];self.actual[name]='c'*64
                with self.assertRaisesRegex(ValueError,'Protected original'):self.check()
                self.actual[name]=old
    def test_document_with_history_and_reason(self):
        self.actual['README.md']='d'*64
        self.edits={'README.md':{'reviewed_sha256':'a'*64,'published_sha256':'d'*64,'reason':'Explain recorded scope'}}
        self.check()
    def test_unrecorded_document_or_missing_reason(self):
        self.actual['README.md']='d'*64
        with self.assertRaises(ValueError):self.check()
        self.edits={'README.md':{'reviewed_sha256':'a'*64,'published_sha256':'d'*64,'reason':''}}
        with self.assertRaises(ValueError):self.check()
    def test_unknown_and_missing_files(self):
        self.actual['unapproved.txt']='c'*64
        with self.assertRaises(ValueError):self.check()
        del self.actual['unapproved.txt'];del self.actual['configs/protocol.json']
        with self.assertRaises(ValueError):self.check()
    def test_duplicate_or_unsafe_manifest(self):
        for text in ['a'*64+'  x\n'+'a'*64+'  x\n','a'*64+'  ../x\n','a'*64+'  /x\n','a'*64+'  a//x\n']:
            with self.subTest(text=text),self.assertRaises(ValueError):v.manifest(text)
    def test_duplicate_added_mapping(self):
        with self.assertRaises(ValueError):v.verify_maps(self.original,self.actual,{},[*v.ADDED,'PUBLICATION_SHA256SUMS'])
    def test_original_manifest_is_not_current_manifest(self):
        with self.assertRaisesRegex(ValueError,'coverage'):
            v.verify_maps(self.original,self.actual,{},list(v.ADDED),self.original)
    def test_historical_manifest_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'SHA256SUMS').write_text('a'*64+'  changed.txt\n')
            with self.assertRaisesRegex(ValueError,'reviewed v2'):v.original_inventory(root)

class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.values=v.editorial_values(CASE)
    def text(self,lang='en'):
        return '\n'.join(f'<!-- publication-table: {k} -->\n{v.table(self.values,k,lang)}\n<!-- /publication-table: {k} -->' for k in ['random','quality'])
    def test_both_language_tables(self):
        for lang in ['en','ko']:v.validate_tables(self.text(lang),self.values,lang)
    def test_mean_median_confusion(self):
        with self.assertRaises(ValueError):v.validate_tables(self.text().replace('Mean local relative','Median local relative'),self.values,'en')
    def test_nll_sign(self):
        value=f"{self.values['quality']['INDEPENDENT_4']['nll_delta']:+.6f}"
        with self.assertRaises(ValueError):v.validate_tables(self.text().replace(value,value.replace('-','+')),self.values,'en')
    def test_conditional_denominator(self):
        with self.assertRaises(ValueError):v.validate_tables(self.text().replace('15/94','15/192'),self.values,'en')
    def test_duplicate_table(self):
        with self.assertRaises(ValueError):v.validate_tables(self.text()+self.text(),self.values,'en')

class PackageTests(unittest.TestCase):
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').write_text('x');(root/'b').symlink_to(root/'a')
            with self.assertRaises(ValueError):v.file_map(root)
    def test_unsafe_zip_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'case';root.mkdir();(root/'a').write_text('x');z=Path(tmp)/'bad.zip'
            with zipfile.ZipFile(z,'w') as archive:archive.writestr('../a','x')
            with self.assertRaises(ValueError):verify_zip(z,root)
    def test_current_publication(self):
        result=v.verify(CASE);self.assertEqual(result['reviewed_files'],1282)
        self.assertEqual(result['protected_original_files'],1277)

if __name__=='__main__':unittest.main()
