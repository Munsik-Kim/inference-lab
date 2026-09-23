"""Case010 display contracts; synthetic mutations are not measured evidence."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/showcase'))
from build import build
from case010 import C10, FIGURE_ASSET, FIGURE_SOURCE, SOURCE_COPIES, load_case010, render_case010
from check import check, Links


class UnifiedCaseDisplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.site = Path(cls.temp.name)/'site'
        cls.manifest = build(ROOT, cls.site, '/inference-lab/')
        cls.data = load_case010(ROOT)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def mutate_and_rehash(self, relative, content, expected):
        target = self.site/relative
        manifest = self.site/'build_manifest.json'
        original, previous = target.read_bytes(), manifest.read_bytes()
        try:
            target.write_bytes(content)
            data = json.loads(previous)
            data['files'][relative] = hashlib.sha256(content).hexdigest()
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, expected):
                check(ROOT, self.site)
        finally:
            target.write_bytes(original)
            manifest.write_bytes(previous)

    def test_current_project_preserves_two_distinct_stages(self):
        data = self.data['project']
        self.assertEqual(data['stages']['v1']['N'], 512)
        self.assertEqual(data['stages']['v2']['N'], 1024)
        self.assertEqual(len(data['model']['checkpoint_sha256']), 3)
        self.assertEqual(data['stages']['v1']['families'], [546,630])
        self.assertEqual(data['stages']['v2']['family'], 195)
        for lang in ('en','ko'):
            html = (self.site/lang/'case010.html').read_text()
            parser=Links();parser.feed(html)
            self.assertTrue({'objective','theory','implementation','stage-a','stage-b','results','diagnostics','reproduce'} <= parser.ids)
            self.assertIn('../sources/case010-report-'+lang+'.md.txt', html)
            self.assertIn('../sources/case010-reproduction-'+lang+'.md.txt', html)
            self.assertIn('LOWRANK_4_8_R2', html)
            self.assertIn('MIXED_5_6_BUDGET', html)
            self.assertIn('29/192, 29/192, 29/192', html)
            self.assertIn('292,633', html)
            self.assertIn('292,469', html)
            self.assertIn('-20.76', html)
            self.assertIn('../'+('ko' if lang=='en' else 'en')+'/case010.html', html)

    def test_single_project_home_and_archive(self):
        for lang in ('en','ko'):
            html = (self.site/lang/'index.html').read_text()
            self.assertEqual(html.count('id="memory-study"'), 1)
            self.assertEqual(html.count('<span class="archive-number">010</span>'), 1)
            self.assertNotIn('<span class="archive-number">011</span>',html)
            for i in range(1,10):
                self.assertIn(f'<span class="archive-number">{i:03}</span>', html)

    def test_original_source_copies_and_exact_binary_asset(self):
        for name, path in SOURCE_COPIES.items():
            self.assertEqual((self.site/'sources'/name).read_bytes(), (ROOT/path).read_bytes())
        self.assertEqual((self.site/FIGURE_ASSET).read_bytes(), (ROOT/FIGURE_SOURCE).read_bytes())
        self.assertEqual(check(ROOT,self.site)['status'],'PASS')

    def test_scalar_tamper_is_not_fixed_by_manifest_rehash(self):
        value=json.loads((self.site/'data/case010.json').read_text())
        value['project']['headline']['reduction_pct']=100
        self.mutate_and_rehash('data/case010.json',json.dumps(value).encode(),'Case010 numeric/source mismatch')

    def test_figure_tamper_is_not_fixed_by_manifest_rehash(self):
        content=(self.site/FIGURE_ASSET).read_bytes()+b'synthetic corruption'
        self.mutate_and_rehash(FIGURE_ASSET,content,'Changed Case010 figure bytes')

    def test_arbitrary_binary_is_not_exempted(self):
        target=self.site/'assets/other.png';target.write_bytes(b'\x89PNG\r\n\x1a\n')
        try:
            with self.assertRaisesRegex(ValueError,'Unexpected/missing deploy artifact'):
                check(ROOT,self.site)
        finally:target.unlink()

    def test_rendering_uses_source_values_and_escapes_text(self):
        data=copy.deepcopy(self.data)
        data['project']['headline']['int8_stream_bytes']=3000
        for lang in ('en','ko'):
            text=render_case010(data,lang)
            self.assertIn('3,000',text)
        data['project']['fresh']['rows'][0]['arm']='<script>bad()</script>'
        # A schema mismatch is an error, not a fabricated row silently inserted.
        with self.assertRaises(KeyError):render_case010(data,'en')
        from case010 import table
        self.assertIn('&lt;script&gt;',table(['header'],[['<script>bad()</script>']],'synthetic'))

    def test_deploy_excludes_snapshot_tensors_and_archives(self):
        names=set(self.manifest['files'])
        self.assertFalse(any('/versions/' in name or name.endswith(('.zip','.pt','.npy','.npz')) for name in names))
        self.assertEqual({name for name in names if name.endswith('.png')},{FIGURE_ASSET})
        self.assertEqual(self.manifest['case010_units']['source_files'],self.data['sources'])


if __name__=='__main__':unittest.main()
