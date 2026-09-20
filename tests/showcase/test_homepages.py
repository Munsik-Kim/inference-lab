"""Display contracts for the four entry pages; no model or GPU execution."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/showcase'))
from build import build
from common import read, source_manifest
from data import load
from layout import home, report8_url, C8_REVISION, C8_PATH


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


if __name__=='__main__':unittest.main()
