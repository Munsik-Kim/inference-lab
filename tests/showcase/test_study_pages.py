"""Study explanations must retain scope, evidence values and reading order."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/showcase'))
from build import build
from common import read, sha
from check import check
from data import load
from study import SECTION_IDS, content, paired_counts, tables, table_html

class StudyPages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.site=Path(cls.tmp.name)/'site'
        cls.data=load(ROOT);build(ROOT,cls.site)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_all_cases_in_both_casebooks_have_eight_named_sections(self):
        for lang in ('en','ko'):
            doc=(ROOT/f'docs/{lang}/CASEBOOK.md').read_text()
            for case in range(1,8):
                positions=[doc.index(f'id="case-{case:03}-{name}"') for name in SECTION_IDS]
                self.assertEqual(positions,sorted(positions))
                self.assertTrue(all(doc.count(f'id="case-{case:03}-{name}"')==1 for name in SECTION_IDS))

    def test_static_explanations_precede_filters_and_are_bilingual(self):
        for case in ('case006','case007'):
            for lang in ('en','ko'):
                html=(self.site/lang/f'{case}.html').read_text()
                positions=[html.index('id="'+name+'"') for name in (*SECTION_IDS,'explorer','set')]
                self.assertEqual(positions,sorted(positions))
                c,_=content(ROOT,lang,case)
                self.assertEqual(len(c['sections']),8)
                for section in c['sections']:self.assertIn(section['title'],html)
                self.assertIn('NOT_ASSESSED',html)
                self.assertEqual(html.count('<h1>'),1)

    def test_correctness_denominators_reconstruct_original_counts(self):
        d=self.data['case007']['records']
        counts=[]
        for arm in ('INDEPENDENT','PAIRWISE'):
            c=paired_counts([r for r in d if r['arm']==arm and r['budget']=='4'])
            self.assertEqual((c['n'],c['B_correct']),(192,94));counts.append(c)
        self.assertEqual([c['flips'] for c in counts],[48,12])
        self.assertEqual([c['regression'] for c in counts],[15,2])
        self.assertEqual([c['gain'] for c in counts],[14,3])

    def test_sets_and_readouts_are_not_added_as_independent_samples(self):
        rows=self.data['case006']['records']
        for readout,flips in [('H_NATIVE',[5,8]),('H_FP32',[3,3])]:
            for arm,n in zip(('A_PUBLIC','V4'),flips):
                c=paired_counts([r for r in rows if r['set']=='standard' and r['readout']==readout and r['arm']==arm])
                self.assertEqual((c['n'],c['flips']),(192,n))
        with self.assertRaisesRegex(ValueError,'Do not pool'):
            paired_counts([r for r in rows if r['arm']=='V4'])

    def test_native_standard_flip_tie_statement_uses_exact_stored_top_sets(self):
        for r in self.data['case006']['records']:
            if r['set']=='standard' and r['readout']=='H_NATIVE' and r['flip']:
                self.assertTrue(len(r['B']['top'])>1 or len(r['candidate']['top'])>1)

    def test_methods_define_groups_and_attention_operands_before_results(self):
        for lang in ('ko','en'):
            seven,_=content(ROOT,lang,'case007');six,_=content(ROOT,lang,'case006')
            definition=' '.join(seven['sections'][0]['paragraphs'])
            for term in ('192','0–191','192–383','3,072'):self.assertIn(term,definition)
            methods=seven['sections'][0]['table']
            self.assertIn('INDEPENDENT',methods['rows'][0][0]);self.assertIn('PAIRWISE',methods['rows'][1][0])
            attn=six['sections'][0]['table']
            self.assertIn('A_PUBLIC',attn['rows'][1][0]);self.assertIn('V4',attn['rows'][2][0])
            self.assertIn('fp32+fp16',attn['rows'][1][2]);self.assertIn('fp32',attn['rows'][2][2])
            html=(self.site/lang/'case007.html').read_text()
            self.assertLess(html.index('data-table="method-definitions"'),html.index('id="results"'))
            self.assertIn('href="#objective"',html[html.index('id="results"'):])

    def test_teaching_example_is_labelled_and_not_added_to_evidence(self):
        for lang,label in [('ko','실험 측정값 아님'),('en','not measured evidence')]:
            c,_=content(ROOT,lang,'case007')
            self.assertIn(label,c['sections'][2]['table']['title'])
            self.assertEqual(read(self.site/'data/case007.json'),self.data['case007'])
            self.assertNotIn('Teaching example',(self.site/'data/case007.json').read_text())

    def test_duplicate_and_missing_comparisons_are_rejected(self):
        r=copy.deepcopy(self.data['case007']['records'][0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):paired_counts([r,r])
        with self.assertRaisesRegex(ValueError,'Missing'):paired_counts([])
        r['cell']='missing'
        with self.assertRaisesRegex(ValueError,'Missing correctness'):paired_counts([r])

    def test_random_mean_is_not_replaced_with_prompt_median(self):
        d=copy.deepcopy(self.data['case007'])
        old=tables(d,'en')
        for k,v in d['summary']['local']['HELD_OUT'].items():
            if k.startswith('RANDOM_'):v['relative_error']['median']=999
        self.assertEqual(old,tables(d,'en'))
        del d['summary']['local']['HELD_OUT']['RANDOM_4_00']
        with self.assertRaisesRegex(ValueError,'Incomplete frozen random'):tables(d,'en')

    def test_changed_rendered_number_cannot_pass_by_rehashing(self):
        p=self.site/'ko/case007.html';m=self.site/'build_manifest.json';before=p.read_bytes();bm=m.read_bytes()
        try:
            self.assertIn('29.8291%',p.read_text())
            p.write_text(p.read_text().replace('29.8291%','0.0000%'))
            manifest=read(m);manifest['files']['ko/case007.html']=sha(p);m.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Study table values'):check(ROOT,self.site)
        finally:p.write_bytes(before);m.write_bytes(bm)

    def test_changed_source_or_missing_section_blocks_build(self):
        from study import read as original_read
        def changed(path):
            d=original_read(path)
            if path.name=='sources.json':d['files'][next(iter(d['files']))]='0'*64
            return d
        with patch('study.read',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'Changed study source'):content(ROOT,'ko','case007')
        def missing(path):
            d=original_read(path)
            if path.name=='en.json':d['case007']['sections'].pop()
            return d
        with patch('study.read',side_effect=missing):
            with self.assertRaisesRegex(ValueError,'section order/coverage'):content(ROOT,'ko','case007')

    def test_plain_explanations_escape_html(self):
        h=table_html({'headers':['<script>','x'],'rows':[['a','</td><img onerror=alert(1)>']]},'a'*40,'en')
        self.assertNotIn('<script>',h);self.assertNotIn('<img',h);self.assertIn('&lt;script&gt;',h)

    def test_home_pages_do_not_load_new_study_styles_or_data(self):
        for lang in ('en','ko'):
            home=(self.site/lang/'index.html').read_text()
            self.assertNotIn('study.css',home);self.assertNotIn('data/case006.js',home)
            self.assertNotIn('study-layout',home)

if __name__=='__main__':unittest.main()
