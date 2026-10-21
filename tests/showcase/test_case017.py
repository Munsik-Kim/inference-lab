"""Case017 registry/rendering keeps raw provenance and prominent deviation."""
import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/showcase'))
from case017 import load_case017,render_case017,home_case017,check_evidence_download,wording,EVIDENCE_SOURCE,SECTIONS

class Case017(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=load_case017(ROOT)

    def test_bilingual_core_results_and_gate_are_static(self):
        for lang in ('ko','en'):
            html=render_case017(self.data,lang)
            self.assertTrue(all('id="'+s+'"' in html for s in SECTIONS))
            self.assertLess(html.index('case017-deviation'),html.index('id="workflow"'))
            self.assertEqual(html.count('class="case017-episode"'),len(self.data['selected_episodes']))
            self.assertNotIn('<script',html)
            self.assertIn('5/8',html);self.assertIn('6/8',html)

    def test_home_numbers_are_projected_from_data(self):
        data=copy.deepcopy(self.data)
        g=next(x for x in data['groups'] if (x['stage'],x['kv'],x['policy'])==('P2','K1','G'))
        g['verified']=3
        for lang in ('ko','en'):
            self.assertIn('3/24',home_case017(data,lang))
            self.assertNotEqual(home_case017(data,lang),home_case017(self.data,lang))

    def test_raw_output_and_request_are_html_escaped(self):
        data=copy.deepcopy(self.data)
        data['selected_episodes'][0]['request']='<img src=x onerror=alert(1)>'
        data['selected_episodes'][0]['turns'][0]['raw_text']='</pre><script>alert(1)</script>'
        html=render_case017(data,'en')
        self.assertIn('&lt;img',html);self.assertIn('&lt;script&gt;',html)
        self.assertNotIn('<script>alert',html)

    def test_only_the_source_matched_evidence_download_is_valid(self):
        self.assertEqual(check_evidence_download((ROOT/EVIDENCE_SOURCE).read_bytes()),418)
        with self.assertRaises(ValueError):check_evidence_download(b'not an archive')
        for lang in ('ko','en'):
            html=render_case017(self.data,lang)
            self.assertLess(html.index('C0'),html.index('<table>'))

    def test_intro_numbers_and_deviation_match_the_document_source(self):
        from html import escape
        for lang in ('ko','en'):
            copy=wording(self.data,lang)
            report=(ROOT/'cases/017-agent-contract-cost-bench'/('REPORT.ko.md' if lang=='ko' else 'REPORT.md')).read_text()
            html=render_case017(self.data,lang)
            for key in ('headline','cost','deviation'):
                self.assertIn(copy[key],report)
                self.assertIn(escape(copy[key]),html)
            self.assertIn('7/24',copy['headline']);self.assertIn('22/24',copy['headline'])

if __name__=='__main__':unittest.main()
