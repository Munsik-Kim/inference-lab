"""Shared reader contracts; browser rendering is checked separately with Edge."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
import tempfile
import shutil
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/showcase'))
from reader import reading_layout
from portfolio import research_map, tools_section, recovery_finding, comparison_finding, OBSERVATIONS


class Tags(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.links=[]
    def handle_starttag(self, tag, attrs):
        d=dict(attrs)
        if 'id' in d:self.ids.append(d['id'])
        if tag=='a':self.links.append(d.get('href'))


class Reader(unittest.TestCase):
    def test_contents_preserve_existing_destinations_and_body(self):
        body='<article><h1>Question</h1><nav><a href="#result">Result</a></nav><section id="result"><h2>Result</h2><p>INCONCLUSIVE</p></section><nav class="related-pages"><a href="case020.html">Related</a></nav></article>'
        for lang in ('en','ko'):
            result=reading_layout(body,'case021',lang); p=Tags();p.feed(result)
            self.assertEqual(p.ids,['result'])
            self.assertEqual(p.links.count('#result'),1)
            self.assertIn('case020.html',p.links)
            self.assertIn('<p>INCONCLUSIVE</p>',result)
            self.assertIn('<details class="reader-toc" open>',result)
            self.assertEqual(result.count('<h1>'),1)
        self.assertEqual(reading_layout(body,'index','en'),body)
        self.assertEqual(reading_layout(body,'case012-annotate','en'),body)

    def test_existing_study_rail_is_not_nested_in_a_second_grid(self):
        body='<div class="study-layout"><aside class="study-toc"><nav><a href="#design">Design</a></nav></aside><section id="design">Evidence</section></div>'
        result=reading_layout(body,'case006','en')
        self.assertNotIn('class="reader-layout"',result)
        self.assertEqual(result.count('class="reader-toc"'),1)
        self.assertIn('id="design"',result)

    def test_current_claim_map_keeps_source_bytes_and_named_boundaries(self):
        data=json.loads((ROOT/'presentation/portfolio_claims.json').read_text())
        for claim in data['claims']:
            p=ROOT/claim['source_path']; self.assertTrue(p.is_file())
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),claim['source_sha256'])
            self.assertTrue(claim['scope'] and claim['unit'] and claim['rendered_by'])
        self.assertEqual(len(OBSERVATIONS),21)
        self.assertEqual(len({c['claim_id'] for c in data['claims']}),len(data['claims']))

    def test_public_research_and_tools_have_distinct_destinations(self):
        for lang in ('en','ko'):
            p=Tags();p.feed(research_map(lang)+tools_section(lang))
            for repo in ('teacher-error-geometry-lab','rtpa-research','coupling-matters-dllm','fdca-visual-token','direction-scale-archive','rcfs-dq'):
                self.assertIn('https://github.com/Munsik-Kim/'+repo,p.links)
            self.assertIn('compare.html#run',p.links)
            self.assertIn('https://github.com/Munsik-Kim/inference-engineering-kit/releases/tag/v0.2.1',p.links)

    def test_synthetic_card_uses_paired_records_and_rejects_mismatched_ids(self):
        for lang in ('en','ko'):
            self.assertIn('2/4 → 2/4',comparison_finding(ROOT,lang))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rel=Path('packages/diova-compare/examples/compare')
            shutil.copytree(ROOT/rel,root/rel)
            path=root/rel/'candidate.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['sample_id']='wrong-pair'
            path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            with self.assertRaisesRegex(ValueError,'IDs differ'):
                comparison_finding(root,'en')

    def test_recovery_endpoint_values_come_from_existing_projection(self):
        data=json.loads((ROOT/'cases/018-agent-state-recovery/publication/summary/project.json').read_text())
        for lang in ('en','ko'):
            self.assertIn('Q 24/24 · W 0/24',recovery_finding(data,lang))
        for g in data['phases'][1]['groups']:
            if (g['kv'],g['policy'])==('K0','R0'):g['W']=3
        self.assertIn('Q 24/24 · W 3/24',recovery_finding(data,'en'))

if __name__=='__main__': unittest.main()
