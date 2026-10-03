"""Actual saved human answers, independent arithmetic and publication contracts."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/showcase'))
from common import read
from case012_human import PUBLIC, ANNOTATION_SHA, calculate, load_human, report_markdown
from case012_human_display import human_results
from case012_pages import load_case012, image_viewer
from score_case012 import score


class Case012HumanReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.annotations=read(ROOT/PUBLIC/'annotations.json')
        cls.result=load_human(ROOT)

    def test_completed_export_has_exact_source_identity(self):
        self.assertEqual(hashlib.sha256((ROOT/PUBLIC/'annotations.json').read_bytes()).hexdigest(),ANNOTATION_SHA)
        self.assertEqual(len(self.annotations['rows']),192)
        self.assertEqual(self.result['label_provenance'],dict(direct_images=100,reused_images=92,
            direct_constraints=254,reused_constraints=226,review_groups=97))

    def test_independent_arithmetic_matches_existing_model_free_scorer(self):
        q=score(ROOT,self.annotations)['quality'];d=self.result
        self.assertEqual(d['paired'],q['paired']);self.assertEqual(d['primary'],q['primary'])
        for setting in d['settings']:
            for key,value in q['settings'][setting].items():
                self.assertAlmostEqual(d['settings'][setting][key],value,places=14)
        self.assertEqual(d['constraint_transition']['C_gains'],sum(q['constraint_gains'].values()))
        self.assertEqual(d['constraint_transition']['C_losses'],sum(q['constraint_losses'].values()))

    def test_image_category_and_pair_denominators(self):
        d=self.result
        self.assertEqual([d['settings'][s]['passed_images'] for s in ('A_time','B_time','C_time')],[52,51,53])
        self.assertEqual(sum(d['paired'].values()),64)
        self.assertEqual(d['paired'],dict(both_pass=49,A_only=3,C_only=4,neither=8))
        self.assertEqual(d['primary']['estimate'],1/64)
        for values in d['categories'].values():
            self.assertTrue(all(v['images']==16 for v in values.values()))

    def test_uncertain_is_unmet_and_sensitivity_remains_separate(self):
        d=self.result;self.assertEqual(d['uncertain_constraint_count'],3)
        self.assertEqual([round(d['settings'][s]['all_uncertain_as_pass_rate']*64) for s in ('A_time','B_time','C_time')],[52,53,54])
        self.assertEqual(d['settings']['B_time']['passed_images'],51)

    def test_no_interval_or_new_generations_for_reused_judgments(self):
        d=self.result
        self.assertIsNone(d['primary']['interval']);self.assertIsNone(d['primary']['level'])
        self.assertEqual(d['new_generations'],0);self.assertFalse(d['published_AI_scores_modified'])
        json.dumps(d,allow_nan=False)

    def test_bad_ID_hash_or_constraint_cannot_enter_aggregation(self):
        for error in ('duplicate','missing','PNG','value','scope'):
            a=copy.deepcopy(self.annotations)
            if error=='duplicate':a['rows'].append(a['rows'][0])
            if error=='missing':a['rows'].pop()
            if error=='PNG':a['rows'][0]['image_sha256']='0'*64
            if error=='value':a['rows'][0]['values'][next(iter(a['rows'][0]['values']))]='yes'
            if error=='scope':a['status']='PARTIAL'
            with self.subTest(error=error),self.assertRaises(ValueError):calculate(ROOT,a)

    def test_reuse_cannot_hide_wrong_source_or_answer(self):
        for error in ('source','value','group'):
            a=copy.deepcopy(self.annotations)
            row=next(r for r in a['rows'] if any(o['kind']=='similarity_reuse' for o in r['origins'].values()))
            cid=next(c for c,o in row['origins'].items() if o['kind']=='similarity_reuse')
            if error=='source':row['origins'][cid]['source_image_id']=row['image_id']
            if error=='value':row['values'][cid]='not_satisfied' if row['values'][cid]=='satisfied' else 'satisfied'
            if error=='group':row['origins'][cid]['group_id']='unknown'
            with self.subTest(error=error),self.assertRaises(ValueError):calculate(ROOT,a)

    def test_both_languages_share_generated_counts_and_reproduction(self):
        for lang,name in [('en','README.md'),('ko','README.ko.md')]:
            doc=(ROOT/PUBLIC/name).read_text()
            self.assertEqual(doc,report_markdown(self.result,lang))
            self.assertIn('52/64',doc);self.assertIn('51/64',doc);self.assertIn('53/64',doc)
            self.assertIn('tools/showcase/case012_human.py --output',doc)

    def test_preserved_AI_and_new_human_results_have_different_fields(self):
        data=load_case012(ROOT)
        self.assertEqual([data['quality']['settings'][s]['passed_images'] for s in ('A_time','B_time','C_time')],[51,53,53])
        self.assertEqual(data['evaluator']['human_raters'],0) # Historical AI phase only.
        self.assertEqual(data['human_review']['coverage']['annotated'],192)
        self.assertIn('DISCLOSED_VISUAL_GROUP_REUSE',data['human_evaluation'])

    def test_every_image_has_both_labels_and_primary_is_visible_without_JS(self):
        data=load_case012(ROOT)
        for lang in ('en','ko'):
            html=image_viewer(ROOT,data,lang)
            self.assertEqual(html.count('data-human-image='),192)
            self.assertEqual(html.count('id="human-results"'),1)
            self.assertIn('<details id="ai-assessment">',html)
            self.assertIn('52/64',human_results(data,lang))
            self.assertIn('100',human_results(data,lang));self.assertIn('92',human_results(data,lang))


if __name__=='__main__':unittest.main()
