"""Display contracts for the four entry pages; no model or GPU execution."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import sys
import copy
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/showcase'))
from build import build
from common import read, source_manifest
from data import load
from layout import home, report8_url, C8_REVISION, C8_PATH, output_metrics
from case008 import load_case008
from check import check_home_outputs


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []; self.cards = {}; self.card = None
        self.scripts = []; self.text = []; self.h1_count = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if tag == 'article' and a.get('class') == 'cap-card':
            self.card = a['id']; self.cards[self.card] = []
        if tag == 'a' and self.card: self.cards[self.card].append(a['href'])
        if tag == 'script': self.scripts.append(a.get('src'))
        if tag == 'h1': self.h1_count += 1

    def handle_endtag(self, tag):
        if tag == 'article': self.card = None

    def handle_data(self, data): self.text.append(data)


def check_order(text):
    positions = [text.index(f'id="{name}"') for name in ('capabilities','tech-stack','projects')]
    if positions != sorted(positions): raise ValueError('Wrong reading order')


def check_capabilities(html, root):
    check_order(html)
    page = Page(); page.feed(html)
    if set(page.cards) != {'model-structure','gpu-comparison','result-delivery'}:
        raise ValueError('Missing capability')
    for links in page.cards.values():
        if len(links) < 2: raise ValueError('Missing implementation links')
        for link in links:
            url = urlsplit(link)
            if url.hostname != 'github.com' or not url.path.startswith('/Munsik-Kim/inference-lab/blob/'):
                raise ValueError('Wrong implementation repository')
            _, relative = url.path.split('/blob/',1)[1].split('/',1)
            target = (root/unquote(relative)).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                raise ValueError('Missing implementation source')
    return page


class Homepages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.site = Path(cls.tmp.name)/'site'
        build(ROOT, cls.site, '/inference-lab/')

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_four_entries_have_the_same_reading_order(self):
        for name in ('README.md','README.ko.md'):
            check_order((ROOT/name).read_text())
        for lang in ('en','ko'):
            check_order((self.site/lang/'index.html').read_text())

    def test_paired_capabilities_link_actual_sources(self):
        pages = [check_capabilities((self.site/lang/'index.html').read_text(), ROOT) for lang in ('en','ko')]
        self.assertEqual(pages[0].cards, pages[1].cards)

    def test_projects_first_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Wrong reading order'):
            check_order('<section id="projects"></section><section id="capabilities"></section><section id="tech-stack"></section>')

    def test_missing_card_and_source_are_rejected(self):
        html = (self.site/'en/index.html').read_text()
        with self.assertRaisesRegex(ValueError,'Missing capability'):
            check_capabilities(html.replace('id="model-structure"','id="missing"'),ROOT)
        with self.assertRaisesRegex(ValueError,'Missing implementation source'):
            check_capabilities(html.replace('/tools/modelpack/artifact.py','/tools/modelpack/absent.py'),ROOT)

    def test_essential_content_is_static_and_anchors_survive(self):
        for lang in ('en','ko'):
            html = (self.site/lang/'index.html').read_text();page=check_capabilities(html,ROOT)
            self.assertEqual(page.h1_count,1);self.assertEqual(page.scripts,[])
            text=' '.join(page.text)
            for term in ('Python','PyTorch','Transformers','vLLM','NumPy'):
                self.assertIn(term,text)
            for anchor in ('studies','projects','capabilities','tech-stack'):
                self.assertEqual(page.ids.count(anchor),1)
            self.assertLess(html.index('id="projects"'),html.index('class="diagram project-diagram"'))
            self.assertIn(f'href="../{"ko" if lang=="en" else "en"}/index.html"',html)

    def test_evidence_projection_and_revision_stay_source_backed(self):
        manifest=read(self.site/'build_manifest.json')
        self.assertEqual(manifest['evidence_revision'],source_manifest(ROOT)['evidence_revision'])
        for case,data in load(ROOT).items():self.assertEqual(read(self.site/f'data/{case}.json'),data)
        self.assertEqual(manifest['independent_scenarios'],{'case007':192,'case006_standard':192,'case006_selected_stress':46,'case008_Q':192,'case008_R':192})

    def test_case008_report_is_linked_without_loading_data_on_home(self):
        for lang in ('en','ko'):
            for page in ('index','guide'):
                html=(self.site/lang/(page+'.html')).read_text()
                self.assertIn('href="'+report8_url(lang)+'"',html)
                self.assertNotIn('data/case008',html)
            html=(self.site/lang/'index.html').read_text()
            self.assertIn('id="case008-report"',html)
            for i in range(1,9):self.assertIn(f'#case-{i:03}',html)
        reports=read(self.site/'build_manifest.json')['linked_reports']['case008']
        self.assertEqual(reports['revision'],C8_REVISION)
        self.assertEqual(set(reports['files']),{C8_PATH+'/REPORT.md',C8_PATH+'/REPORT.ko.md'})
        with self.assertRaises(ValueError):report8_url('unknown')

    def test_archive_cannot_silently_drop_an_untranslated_case(self):
        with self.assertRaisesRegex(ValueError,'Every case needs'):
            home({},'ko',{},['case001','case002'],['Only one title'],'fixture')

    def test_outcomes_are_separate_scoped_and_linked(self):
        data=load_case008(ROOT)
        for lang in ('en','ko'):
            html=(self.site/lang/'index.html').read_text()
            check_home_outputs(html,data,lang)
            locale=read(ROOT/f'presentation/content/{lang}.json')
            self.assertIn('<h1>'+locale['homeHeroTitle']+'</h1>',html)
            self.assertIn('Deep-learning Inference Optimization, Validation &amp; Analysis',html)
            self.assertNotIn('class="card-limit"',html)
            self.assertNotIn(locale['projectResult8'],html)
            self.assertNotIn(locale['homeLimit6'],html)
            self.assertNotIn('-0.2003',html)
            self.assertIn('#tiny-model-demo',html)
            self.assertIn('#cpu-selector',html)

    def test_wrong_home_units_scope_and_missing_evaluation_rejected(self):
        data=load_case008(ROOT);html=(self.site/'en/index.html').read_text()
        mutations=[('8.045 GB','8.045 GiB','decimal GB'),
                   ('strict','permissive','loader implementation'),
                   ('192 short synthetic held-out prompts','384 prompts','error/model/input scope'),
                   ('case008.html#q-evaluation','case008.html','quality/runtime'),
                   ('16 channel groups','all layers','tool scope')]
        for old,new,error in mutations:
            with self.subTest(old=old),self.assertRaisesRegex(ValueError,error):
                check_home_outputs(html.replace(old,new),data,'en')

    def test_rounding_uses_retained_values_instead_of_fixed_headlines(self):
        d=load_case008(ROOT);m=output_metrics(d)
        self.assertEqual(m['weight_files'],'8.045 GB → 2.652 GB')
        changed=copy.deepcopy(d);changed['tracks']['Q']['weight_bytes']['Q-W4']=3_000_000_000
        changed['tracks']['Q']['file_reduction_fraction']=1-3_000_000_000/d['tracks']['Q']['weight_bytes']['Q-BF16']
        self.assertIn('3.000 GB',output_metrics(changed)['weight_files'])
        self.assertNotEqual(m['reduction'],output_metrics(changed)['reduction'])
        for lang in ('en','ko'):
            doc=(ROOT/('README.md' if lang=='en' else 'README.ko.md')).read_text()
            for value in (m['reduction']+'%',m['weight_files']):self.assertIn(value,doc)
            self.assertNotIn(m['recovery'],doc)
            for structure in d['tracks']['R']['structures'].values():
                self.assertIn(f"{100*structure['recovery']['pooled_recovery']:.2f}%",(self.site/lang/'case008.html').read_text())

    def test_evaluation_links_reach_preserved_detail(self):
        for lang in ('en','ko'):
            detail=(self.site/lang/'case008.html').read_text()
            locale=read(ROOT/f'presentation/case008/{lang}.json')
            for key in ('qscore','nllScope','timeScope','rConclusion'):
                from html import escape
                self.assertIn(escape(locale[key]),detail)
            for name in ('q-evaluation','q-timing','r-evaluation'):self.assertIn('id="'+name+'"',detail)
            self.assertIn('NOT_ASSESSED',detail)
            self.assertIn('COMPLETED_NO_CLEAR_TRANSFER',(self.site/lang/'case007.html').read_text())
            self.assertIn('0/24',(self.site/lang/'case006.html').read_text())


if __name__=='__main__':unittest.main()
