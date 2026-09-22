"""Presentation fixtures only; these are not measured outcomes."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/showcase'))
from case009 import render_case009


class NewStudyDisplay(unittest.TestCase):
    def test_offline_payload_cannot_be_changed_by_rehashing_build(self):
        from build import build
        from check import check
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            site=Path(tmp)/'site';build(root,site)
            payload=json.loads((site/'data/case009.json').read_text())
            payload['serving']['complete_cells']=59
            target=site/'data/case009.js'
            target.write_text('window.CASE009_EVIDENCE = '+json.dumps(payload)+';\n')
            manifest=site/'build_manifest.json';data=json.loads(manifest.read_text())
            data['files']['data/case009.js']=hashlib.sha256(target.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'offline numeric/source mismatch'):
                check(root,site)
    def fixture(self):
        return {'serving':{'contrasts':[{'execution':'graph','input_tokens':128,'client_concurrency':1,'status':'INCOMPLETE_PAIRS'}]},
                'quality':{'tasks':[{'task':'<script>bad()</script>','status':'INCOMPLETE','arms':{'BF16':'NOT_RUN','W4':'FAILED'}}]}}
    def test_missing_cells_are_not_filled(self):
        for lang in ['en','ko']:
            s=render_case009(self.fixture(),lang)
            self.assertIn('INCOMPLETE',s);self.assertIn('NOT_RUN',s);self.assertIn('FAILED',s)
    def test_text_escaped(self):
        s=render_case009(self.fixture(),'en')
        self.assertNotIn('<script>bad()',s);self.assertIn('&lt;script&gt;',s)
    def test_language_pair_sections_and_download(self):
        for lang in ['en','ko']:
            s=render_case009(self.fixture(),lang)
            for key in ['serving','quality','scope']:self.assertIn('id="'+key+'"',s)
            self.assertIn('../data/case009.json',s);self.assertIn('NOT_ASSESSED',s)
            self.assertIn('serving-L128.svg',s);self.assertIn('serving-L1024.svg',s)
    def test_retained_calculation_keeps_failed_process_visible(self):
        d=self.fixture();d['quality']['tasks']=[{'task':'wikitext','status':'COMPUTED_WITH_PROCESS_FAILURE',
            'process_status':{'BF16':'COMPLETE','W4':'FAILED'},'groups':{'wikitext/none':{'arms':{
                'BF16':{'word_perplexity':12.0},'W4':{'word_perplexity':13.0}}}}}]
        for lang in ['en','ko']:
            text=render_case009(d,lang)
            self.assertIn('PROCESS EXIT',text);self.assertIn('FAILED',text)
            self.assertIn('12.0000',text);self.assertIn('13.0000',text)


if __name__=='__main__':unittest.main()
