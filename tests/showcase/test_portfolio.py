"""Editorial coverage and source-driven claims; no model or GPU imports."""
from pathlib import Path
import sys, unittest, copy
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/showcase'))
from portfolio import catalog,latest,hero,TOPICS
from common import read

class Markup(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.cases=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:self.ids.append(d['id'])
        if 'data-case-id' in d:self.cases.append((d['data-case-id'],d['data-topic']))
        if tag=='a':self.links.append(d.get('href'))

class Portfolio(unittest.TestCase):
    def test_all_cases_once_and_correct_direct_routes(self):
        paths=[f'cases/{n:03}-fixture' for n in range(1,22)]
        for lang in ('ko','en'):
            html=catalog(paths,[f'Case {n:03}' for n in range(1,22)],lang);p=Markup();p.feed(html)
            self.assertEqual(sorted(n for n,_ in p.cases),[f'{n:03}' for n in range(1,22)])
            self.assertEqual(len(p.ids),len(set(p.ids)))
            for n in range(6,22):self.assertIn(f'case{n:03}.html',p.links)
            for n in range(1,6):self.assertTrue(any(u.endswith(f'CASEBOOK.md#case-{n:03}') for u in p.links))
            self.assertIn('id="case-controls" class="catalog-controls" hidden',html)
            self.assertNotIn('data-topic="undefined"',html)
        with self.assertRaises(ValueError):catalog(paths[:-1],['x']*20,'en')
        with self.assertRaises(ValueError):catalog(paths,['x']*20,'en')

    def test_recent_metrics_follow_verified_loader_facts(self):
        d20={'facts':read(ROOT/'cases/020-optic-on-demand-decode/summary/facts.json')}
        d21={'facts':read(ROOT/'cases/021-rounding-execution-diagnostics/summary/facts.json')}
        for lang in ('ko','en'):
            h=latest(d20,d21,lang);self.assertIn('0/30',h);self.assertIn('INCONCLUSIVE',h)
            changed=copy.deepcopy(d20);changed['facts']['P4_C_faster_blocks']=2
            self.assertIn('2/30',latest(changed,d21,lang))
            changed=copy.deepcopy(d21);changed['facts']['primary_documents']=7
            self.assertIn('7문서' if lang=='ko' else '7 documents',latest(d20,changed,lang))
        with self.assertRaises(ValueError):latest(d20,None,'ko')

    def test_safe_text_and_offline_progressive_controls(self):
        paths=[f'cases/{n:03}-fixture' for n in range(1,22)]
        h=catalog(paths,['<img src=x onerror=alert(1)>']*21,'en')
        self.assertNotIn('<img',h);self.assertIn('&lt;img',h)
        js=(ROOT/'presentation/assets/portfolio.js').read_text()
        for unwanted in ('fetch(', 'XMLHttpRequest', 'localStorage', 'innerHTML', 'eval('):self.assertNotIn(unwanted,js)
        self.assertIn('textContent',js);self.assertIn('hashchange',js)
        self.assertIn('[hidden] { display: none !important; }',(ROOT/'presentation/assets/portfolio.css').read_text())

if __name__=='__main__':unittest.main()
