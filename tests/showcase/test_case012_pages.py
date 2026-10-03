"""Source-backed Pages and synthetic human-file contracts; no model forward."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/showcase'))
from build import build
from check import check, Links
from case012_pages import load_case012, annotation_data, C12, SECTIONS
from score_case012 import score


class Case012Pages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.site = Path(cls.tmp.name)/'site'
        cls.manifest = build(ROOT,cls.site,'/inference-lab/')
        cls.data = load_case012(ROOT)
        cls.inputs = annotation_data(ROOT,cls.data)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_original_MAIN_PNGs_have_exact_bytes(self):
        self.assertEqual(len(self.data['images']),192)
        for name,path in self.data['images'].items():
            self.assertEqual((self.site/name).read_bytes(),(ROOT/path).read_bytes())

    def test_human_input_masks_config_and_has_no_AI_prefill(self):
        inputs = json.loads((self.site/'data/case012-annotation.json').read_text())
        self.assertEqual(inputs,self.inputs)
        self.assertEqual(len(inputs['items']),192)
        for item in inputs['items']:
            self.assertFalse({'setting','loops','steps','seed_label','complete_seconds','values'} & set(item))

    def test_visible_AI_results_are_separate_from_human_scope(self):
        self.assertEqual(self.data['recorded_quality_status'],'ANNOTATION_PENDING')
        self.assertEqual(self.data['evaluator']['human_raters'],0)
        self.assertEqual([self.data['quality']['settings'][s]['passed_images'] for s in ('A_time','B_time','C_time')],[51,53,53])
        for lang in ('en','ko'):
            main=(self.site/lang/'case012.html').read_text(); parser=Links(); parser.feed(main)
            self.assertTrue(set(SECTIONS)<=parser.ids)
            self.assertIn('51/64',main); self.assertIn('53/64',main)
            self.assertIn('case012-annotate.html',main)
            form=(self.site/lang/'case012-annotate.html').read_text()
            self.assertNotIn('data/case012.json',form)
            self.assertNotIn('51/64',form)
            self.assertIn('id="export"',form); self.assertIn('id="import"',form)
            self.assertIn('../'+('ko' if lang=='en' else 'en')+'/case012-annotate.html',form)

    def test_unknown_artifact_and_tampered_PNG_cannot_be_rehashed_to_pass(self):
        name=next(iter(self.data['images'])); target=self.site/name
        original=target.read_bytes(); manifest=self.site/'build_manifest.json'; old=manifest.read_bytes()
        try:
            target.write_bytes(original+b'corruption'); m=json.loads(old)
            m['files'][name]=hashlib.sha256(target.read_bytes()).hexdigest(); manifest.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'Changed Case012 original PNG'): check(ROOT,self.site)
        finally: target.write_bytes(original); manifest.write_bytes(old)
        extra=self.site/'assets/unlisted.png'; extra.write_bytes(original)
        try:
            with self.assertRaisesRegex(ValueError,'Unexpected/missing deploy'): check(ROOT,self.site)
        finally: extra.unlink()

    def test_tampered_human_identity_is_rejected_even_after_manifest_update(self):
        target=self.site/'data/case012-annotation.json'; old=target.read_bytes()
        manifest=self.site/'build_manifest.json'; old_m=manifest.read_bytes()
        try:
            data=json.loads(old); data['items'][0]['image_sha256']='0'*64; target.write_text(json.dumps(data))
            m=json.loads(old_m); m['files']['data/case012-annotation.json']=hashlib.sha256(target.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'Case012 human input mismatch'): check(ROOT,self.site)
        finally: target.write_bytes(old); manifest.write_bytes(old_m)

    def annotation_fixture(self, complete=False):
        items=self.inputs['items']; rows=[]
        if complete:
            rows=[{'image_id':i['image_id'],'image_sha256':i['image_sha256'],
                   'values':{c['id']:'satisfied' for c in i['constraints']}} for i in items]
        return {'schema':'case012-annotations-v1','scope':'MAIN_ONLY',
                'image_set_sha256':self.inputs['image_set_sha256'],'rubric_sha256':self.inputs['rubric_sha256'],
                'evaluator':{'type':'human','id':'SYNTHETIC_TEST_ONLY'},
                'status':'COMPLETE' if complete else 'PARTIAL','rows':rows,'draft_rows':[],
                'incomplete_image_ids':[] if complete else [i['image_id'] for i in items],
                'cursor':0,'current_image_id':items[0]['image_id']}

    def test_partial_human_file_has_no_quality_score(self):
        q=score(ROOT,self.annotation_fixture())['quality']
        self.assertEqual(q['status'],'ANNOTATION_PENDING'); self.assertIsNone(q['primary'])
        self.assertEqual(q['coverage']['expected'],192)

    def test_synthetic_complete_labels_preserve_full_denominator(self):
        # Contract fixture only; these fabricated labels are never published as observations.
        q=score(ROOT,self.annotation_fixture(True))['quality']
        self.assertEqual(q['coverage']['annotated'],192)
        self.assertEqual(q['primary']['estimate'],0)
        self.assertEqual(q['paired']['both_pass'],64)

    def test_bad_human_files_and_AI_import_are_rejected(self):
        f=self.annotation_fixture(True)
        for change in ('AI','duplicate','hash','status','cursor'):
            x=copy.deepcopy(f)
            if change=='AI': x['evaluator']['type']='model-assisted'
            if change=='duplicate': x['rows'].append(x['rows'][0])
            if change=='hash': x['rows'][0]['image_sha256']='other'
            if change=='status': x['status']='PARTIAL'
            if change=='cursor': x['cursor']=192
            with self.subTest(change=change),self.assertRaises(ValueError): score(ROOT,x)

    def grouped_annotation_fixture(self):
        from case012_review import review_copy
        f=self.annotation_fixture(True);f['schema']='case012-annotations-v2'
        display=review_copy(ROOT,self.data)
        f['review']={'mode':'similarity','grouping_sha256':display['visual_group_manifest_sha256'],'separated_image_ids':[]}
        for row in f['rows']:row['origins']={c:{'kind':'direct'} for c in row['values']}
        group=next(g for g in display['visual_groups'] if len(g['members'])>1)
        source=group['representative'];target=next(i for i in group['members'] if i!=source)
        row=next(r for r in f['rows'] if r['image_id']==target);cid=next(iter(row['values']))
        row['origins'][cid]={'kind':'similarity_reuse','source_image_id':source,'group_id':group['group_id']}
        return f,target,cid

    def test_grouped_recalculation_is_descriptive_and_discloses_reuse(self):
        f,_,_=self.grouped_annotation_fixture();result=score(ROOT,f)
        self.assertEqual(result['evidence_kind'],'USER_GROUPED_HUMAN_REVIEW')
        self.assertEqual(result['label_provenance']['reused_constraints'],1)
        self.assertEqual(result['quality']['status'],'GROUPED_REVIEW_DESCRIPTIVE')
        self.assertEqual(result['quality']['coverage']['annotated'],192)
        self.assertIsNone(result['quality']['primary']['interval'])
        self.assertFalse(result['published_AI_scores_modified'])

    def test_independent_CPU_checker_rejects_stale_or_invalid_reuse(self):
        f,target,cid=self.grouped_annotation_fixture()
        for error in ('value','source','group','separated','config'):
            x=copy.deepcopy(f);row=next(r for r in x['rows'] if r['image_id']==target)
            if error=='value':row['values'][cid]='not_satisfied'
            if error=='source':row['origins'][cid]['source_image_id']=target
            if error=='group':row['origins'][cid]['group_id']='unknown'
            if error=='separated':x['review']['separated_image_ids']=[target]
            if error=='config':x['review']['grouping_sha256']='changed'
            with self.subTest(error=error),self.assertRaises(ValueError):score(ROOT,x)

    def test_v2_all_direct_rows_retain_original_arithmetic(self):
        f,_,_=self.grouped_annotation_fixture()
        for row in f['rows']:row['origins']={c:{'kind':'direct'} for c in row['values']}
        result=score(ROOT,f)
        self.assertEqual(result['evidence_kind'],'USER_HUMAN_ANNOTATIONS')
        self.assertEqual(result['label_provenance']['reused_constraints'],0)
        self.assertEqual(result['quality']['primary']['estimate'],0)
        self.assertIsNotNone(result['quality']['primary']['interval'])

    def test_site_has_no_model_weights_or_new_quality_run(self):
        self.assertFalse(any(name.endswith(('.pt','.bin','.zip','.npz')) for name in self.manifest['files']))
        self.assertEqual(self.manifest['case012_units']['images'],192)
        check(ROOT,self.site)

class Case012ReviewUX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(); cls.site=Path(cls.tmp.name)/'site'
        cls.manifest=build(ROOT,cls.site,'/inference-lab/')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_display_translation_keeps_saved_input_identity(self):
        # Pin the published v1 cohort: correcting UI language must not discard users' JSON.
        previous=ROOT/'cases/012-looped-dit-inference-budget/analysis/annotation_items.json'
        from case012_review import review_copy
        data=load_case012(ROOT); inputs=annotation_data(ROOT,data); display=review_copy(ROOT,data)
        self.assertEqual(len(display['groups']),16); self.assertEqual(len(display['items']),192)
        for g in display['groups']:
            self.assertEqual(sum(pid==g['prompt_id'] for pid in display['items'].values()),12)
        self.assertEqual(inputs,json.loads((self.site/'data/case012-annotation.json').read_text()))
        self.assertEqual(inputs['image_set_sha256'],'d204dd88a27b317e1e94b5f77e840cede78bb4945e59b817d8e4f96ebb2a3e30')
        self.assertEqual(set(display['items']),{i['image_id'] for i in inputs['items']})
        # The original awkward strings are retained in the identity-bearing input, not shown as UI copy.
        self.assertIn('풍선가',previous.read_text())
        self.assertEqual(display['groups'][2]['ko']['constraints']['c1'],'풍선이 정확히 네 개 보이나요?')

    def test_comparison_is_collapsed_without_omitting_images(self):
        for lang in ('en','ko'):
            html=(self.site/lang/'case012-images.html').read_text()
            self.assertEqual(html.count('class="comparison"'),64)
            self.assertEqual(html.count('class="pair"'),16)
            self.assertEqual(html.count('class="card"'),192)
            self.assertEqual(html.count('<details open class="comparison"'),1)
            self.assertEqual(html.count('<details open class="pair"'),1)
            for name in self.manifest['case012_units']['image_assets']:
                self.assertIn(name,html)

    def test_each_compound_translation_stays_with_its_own_objects(self):
        html=(self.site/'ko/case012-images.html').read_text()
        for pid,correct,wrong in [
            ('main-compound-1','머그잔은 모두 빨간색이고 병은 파란색인가요?','정육면체는 모두 노란색'),
            ('main-compound-2','정육면체는 모두 노란색이고 구는 초록색인가요?','머그잔은 모두 빨간색'),
            ('main-compound-4','자동차는 모두 파란색이고 버스는 빨간색인가요?','머그잔은 모두 빨간색')]:
            import re
            section=re.search(r'<details class="pair" id="'+pid+r'">(.*?)(?=<details class="pair"|<section id="method")',html,re.S).group(1)
            self.assertIn(correct,section); self.assertNotIn(wrong,section)
        self.assertNotIn('풍선가',html); self.assertNotIn('머그잔가',html)

    def test_copy_is_blind_and_has_no_answers_or_setting_identity(self):
        copy=json.loads((self.site/'data/case012-review-copy.json').read_text())
        self.assertEqual(copy['schema'],'case012-review-copy-v1')
        for g in copy['groups']:
            self.assertFalse({'loops','steps','seed_label','values','time'}&set(g))
        self.assertTrue(all(isinstance(v,str) for v in copy['items'].values()))

    def test_similarity_groups_have_exact_coverage_and_never_mix_requests(self):
        from case012_similarity import load_groups
        data=load_case012(ROOT);similar=load_groups(ROOT,data)
        self.assertEqual(len(similar['groups']),97)
        self.assertEqual(len([i for g in similar['groups'] for i in g['members']]),192)
        self.assertFalse(similar['uses_AI_labels']);self.assertEqual(similar['new_model_inference'],0)
        copy=json.loads((self.site/'data/case012-review-copy.json').read_text())
        self.assertEqual(copy['visual_groups'],similar['groups'])
        for g in similar['groups']:
            self.assertTrue(all(copy['items'][i]==g['prompt_id'] for i in g['members']))
        for lang in ('en','ko'):
            form=(self.site/lang/'case012-annotate.html').read_text()
            for name in ('review-mode','apply-group','similar-members','separate-image'):
                self.assertIn('id="'+name+'"',form)
            viewer=(self.site/lang/'case012-images.html').read_text()
            self.assertIn('id="representative-view"',viewer)
            self.assertIn('../data/case012-review-copy.js',viewer)

    def test_complete_link_does_not_merge_via_a_middle_image(self):
        from case012_similarity import complete_link
        d={frozenset(('a','b')):.2,frozenset(('b','c')):.3,frozenset(('a','c')):1.2}
        self.assertEqual(complete_link(['a','b','c'],d),[['a','b'],['c']])
