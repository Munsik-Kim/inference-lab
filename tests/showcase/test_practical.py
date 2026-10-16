"""Guide integration, provenance and reading contracts; no model execution."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/showcase'))
from build import build
from check import check
from common import read, sha
from practical import PAGES, REVISION, SOURCE_PATHS, graph_memory_pairs, values


class Structure(HTMLParser):
    def __init__(self):
        super().__init__(); self.headings=[]; self.links=[]; self.visible=[]; self.details=0
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if re.fullmatch('h[1-6]',tag): self.headings.append(int(tag[1]))
        if tag=='a': self.links.append(a.get('href',''))
        if tag=='details': self.details+=1
    def handle_endtag(self, tag):
        if tag=='details': self.details-=1
    def handle_data(self, data):
        if not self.details:self.visible.append(data)


class PracticalGuide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(); cls.site=Path(cls.tmp.name)/'site'
        build(ROOT,cls.site,'/inference-lab/')
        cls.data=read(cls.site/'data/practical-guide.json')
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_new_sources_are_the_exact_public_commit_blobs(self):
        manifest=read(ROOT/'presentation/practical_sources.json')
        self.assertEqual(set(manifest['files']),set(SOURCE_PATHS))
        for name,digest in manifest['files'].items():
            blob=subprocess.check_output(['git','show',REVISION+':'+name],cwd=ROOT)
            self.assertEqual(hashlib.sha256(blob).hexdigest(),digest,name)

    def test_original_values_conditions_and_denominators(self):
        raw=read(ROOT/'cases/009-q-serving-quality/results/derived/serving_summary.json')
        self.assertEqual(self.data['serving']['contrasts'],raw['contrasts'])
        source=read(ROOT/'research/agent_tool_contract_repair_20261014/records/summary.json')
        self.assertEqual(self.data['tool_repair'],source)
        self.assertEqual(source['totals']['unique_previously_observed_tasks'],8)
        self.assertEqual(source['totals']['episodes'],80)
        v=values(self.data,'ko')
        self.assertEqual(v['weight_files'],'8.045 GB → 2.652 GB')
        self.assertEqual((v['speed_small'],v['speed_large']),('2.090','1.237'))
        self.assertEqual((v['d16_before'],v['d16_after'],v['d4_before'],v['d4_after']),('6/8','8/8','0/8','2/8'))
        changed=copy.deepcopy(self.data); changed['storage']['weight_bytes']['Q-W4']=3_000_000_000
        self.assertIn('3.000 GB',values(changed,'ko')['weight_files'])

    def test_entries_static_outline_language_and_two_primary_links(self):
        for lang in ('en','ko'):
            home=(self.site/lang/'index.html').read_text()
            hero=home[home.index('<section class="cap-hero">'):home.index('id="featured-stories"')]
            self.assertEqual(hero.count('class="reading-link"'),2)
            self.assertNotIn('core-tech',hero)
            self.assertIn('href="llm-guide.html"',hero)
            self.assertIn('href="#featured-stories"',hero)
            stories=home[home.index('id="featured-stories"'):home.index('id="contributions"')]
            self.assertEqual(stories.count('<article class="story">'),3)
            for page in PAGES:
                html=(self.site/lang/(page+'.html')).read_text(); p=Structure();p.feed(html)
                self.assertEqual(p.headings.count(1),1)
                self.assertTrue(all(b<=a+1 for a,b in zip(p.headings,p.headings[1:])),page)
                self.assertIn('../'+('ko' if lang=='en' else 'en')+'/'+page+'.html',p.links)
                self.assertNotIn('<script',html)
                self.assertNotIn('{{',html)
                if page == 'compare':
                    self.assertIn('Synthetic CPU example' if lang=='en' else '합성 CPU 예제',html)
                    self.assertNotIn(('Recorded public evidence ' if lang=='en' else '공개된 실험 기록의 버전 ')+REVISION,html)
                else:
                    self.assertIn(('Recorded public evidence ' if lang=='en' else '공개된 실험 기록의 버전 ')+REVISION,html)

    def test_memory_reductions_pair_rounds_and_exclude_eager(self):
        processes=self.data['serving']['processes']
        expected=graph_memory_pairs(processes)
        self.assertEqual(expected,self.data['device_memory'])
        self.assertEqual(expected,graph_memory_pairs(list(reversed(processes))))
        pairs=expected['pairs']
        self.assertEqual([p['saved_mib'] for p in pairs],[5116,5180,5104])
        for p in pairs:
            self.assertAlmostEqual(p['reduction_fraction'],1-p['w4_peak_mib']/p['bf16_peak_mib'])
        changed=copy.deepcopy(processes)
        for p in changed:
            if p['process_id'].endswith('-eager'):p['sampled_device_peak_used_MiB']=1
        self.assertEqual(expected,graph_memory_pairs(changed))
        with self.assertRaisesRegex(ValueError,'Incomplete graph memory pair'):
            graph_memory_pairs([p for p in processes if p['process_id']!='round2-W4-graph'])
        with self.assertRaisesRegex(ValueError,'Duplicate graph memory arm'):
            graph_memory_pairs(processes+[processes[0]])
        v=values(self.data,'ko')
        self.assertEqual((v['memory_bf16_gib'],v['memory_w4_gib']),('14.08–14.51 GiB','9.08–9.52 GiB'))
        self.assertEqual((v['memory_saved_gib'],v['memory_saved_percent']),('4.98–5.06 GiB','34.4–35.8%'))
        for lang in ('ko','en'):
            html=(self.site/lang/'qwen3-4b.html').read_text()
            self.assertIn(v['memory_saved_percent'],html)
            self.assertIn('id="device-memory"',html)

    def test_limitations_remain_visible_outside_disclosures(self):
        p=Structure();p.feed((self.site/'ko/qwen3-4b.html').read_text());text=' '.join(p.visible)
        for phrase in ('가중치 파일','전체 그래픽카드 전용 메모리','BF16으로 복원','전체 모델 계산값 검사는 실패했습니다',
                       '비정상 종료','문제(DEV) 8개를 재사용했습니다','서로 다른 문제가 80개라는 뜻은 아닙니다','6개 실패','추가 효과는 입증하지 못했습니다',
                       '메모리 정책상 미실행','Qwen3-0.6B','실제 독자 평가'):
            self.assertIn(phrase,text)
        en=(self.site/'en/qwen3-4b.html').read_text()
        self.assertIn('not a complete translation',en)

    def test_reader_order_common_routes_and_page_metadata(self):
        for lang in ('ko','en'):
            html=(self.site/lang/'index.html').read_text()
            positions=[html.index('id="'+name+'"') for name in ('featured-stories','model-guides','collaboration','contributions','capabilities','tech-stack','compensation-study','projects','cases')]
            self.assertEqual(positions,sorted(positions))
            for page in self.site.joinpath(lang).glob('*.html'):
                text=page.read_text()
                if 'class="site-header"' not in text:continue  # retained image-review tool
                nav=text.split('<header class="site-header">',1)[1].split('</nav>',1)[0].split('<nav ',1)[1]
                p=Structure();p.feed('<nav '+nav+'</nav>')
                self.assertEqual(p.links,['index.html','llm-guide.html','index.html#cases','guide.html'],page.name)
                title=re.search(r'<title>(.*?)</title>',text)[1]
                self.assertIn('property="og:title" content="'+title+'"',text)
                self.assertRegex(text,r'<meta name="description" content="[^\"]+">')
            guide=(self.site/lang/'guide.html').read_text()
            self.assertIn('NumPy',guide)
            self.assertIn('--base-path /inference-lab/',guide)
            self.assertIn('measurements.html',guide)
            self.assertIn('개발자 재현' if lang=='ko' else 'Developer reproduction',guide)

    def test_beginner_routes_start_online_and_cover_current_cases(self):
        for lang in ('ko','en'):
            text=(ROOT/f'docs/{lang}/START_HERE.md').read_text()
            self.assertNotIn('일곱 사례' if lang=='ko' else 'seven cases',text)
            for number in range(1,14):self.assertIn(f'CASEBOOK.md#case-{number:03}',text)
            self.assertLess(text.index('case008.html#track-q'),text.index('case009.html#serving'))
            self.assertLess(text.index('case009.html#serving'),text.index('agent_tool_contract_repair_20261014'))
            self.assertLess(text.index('https://munsik-kim.github.io'),text.index('GETTING_STARTED.md#offline-explorers'))
            reproduction=(ROOT/f'docs/{lang}/GETTING_STARTED.md').read_text()
            self.assertLess(reproduction.index('https://munsik-kim.github.io'),reproduction.index('tools/case008_checks/run.py'))

    def test_new_github_code_and_doc_targets_exist(self):
        # The general checker verifies local site links. This covers new external
        # GitHub source paths too, avoiding plausible-looking nonexistent files.
        for lang in ('ko','en'):
            for page in PAGES:
                p=Structure();p.feed((self.site/lang/(page+'.html')).read_text())
                for link in p.links:
                    u=urlsplit(link)
                    prefix='/Munsik-Kim/inference-lab/blob/'
                    if u.hostname=='github.com' and u.path.startswith(prefix):
                        revision,name=u.path[len(prefix):].split('/',1)
                        target=ROOT/unquote(name)
                        self.assertTrue(target.is_file(),link)
                        if u.fragment and target.suffix=='.md':
                            sys.path.insert(0,str(ROOT/'docs/_checks'))
                            from check_docs import anchors
                            self.assertIn(unquote(u.fragment),anchors(target.read_text()),link)

    def test_manifest_and_scalar_tampering_rejected_even_after_rehash(self):
        p=self.site/'data/practical-guide.json';m=self.site/'build_manifest.json'
        before=p.read_bytes();bm=m.read_bytes()
        try:
            data=read(p);data['storage']['weight_bytes']['Q-W4']-=1;p.write_text(json.dumps(data))
            manifest=read(m);manifest['files']['data/practical-guide.json']=sha(p);m.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Practical guide numeric/source mismatch'): check(ROOT,self.site)
        finally:p.write_bytes(before);m.write_bytes(bm)

    def test_allowlist_addition_and_full_build_check(self):
        manifest=read(self.site/'build_manifest.json')
        self.assertEqual(manifest['practical_guide_units']['new_gpu_runs'],0)
        self.assertEqual(manifest['practical_guide_units']['evidence_revision'],REVISION)
        self.assertEqual(check(ROOT,self.site)['status'],'PASS')

    def test_relative_site_path_keeps_fragment_links_valid(self):
        self.assertEqual(check(ROOT,Path(os.path.relpath(self.site)))['status'],'PASS')

    def test_readme_portfolio_and_new_document_entry_links(self):
        sys.path.insert(0,str(ROOT/'docs/_checks'))
        from check_docs import check_links
        errors=[]
        check_links(ROOT,['docs/ko/LLM_GUIDE.md','docs/en/LLM_GUIDE.md'],errors)
        self.assertEqual(errors,[])
        for lang in ('ko','en'):
            readme=ROOT/('README.ko.md' if lang=='ko' else 'README.md')
            self.assertIn('docs/'+lang+'/LLM_GUIDE.md',readme.read_text())
            self.assertIn('](LLM_GUIDE.md)',(ROOT/f'docs/{lang}/PORTFOLIO.md').read_text())


if __name__=='__main__': unittest.main()
