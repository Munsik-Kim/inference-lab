import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from diova_compare.core import compare, validate, load
from diova_compare.adapters import harness_sample


def row(i='0', answer='a', gold='a', kind='synthetic_test'):
    return dict(schema_version=1,sample_id=i,task='fixture',task_version='1',prompt_hash='a'*64,
        gold=[gold],gold_definition='exact-label',model_id='B',artifact_id='b',
        output_type='choice',answer=answer,correct=answer==gold,evidence_kind=kind)


def dist(r, p=(.6,.3,.1), labels=('a','b','c')):
    r['distribution']=dict(kind='choice',labels=list(labels),probabilities=list(p),
        tokenizer_id='fixture',prefix_hash=r['prompt_hash'],dtype='float64',normalization='probabilities_sum_to_one')
    return r


class Contract(unittest.TestCase):
    def paired(self,b,c,**kw):return compare(b,c,replicates=50,**kw)
    def test_source_process_failure_retained(self):
        b=row()|{'process_exit_code':0};c=row()|{'process_exit_code':-6}
        result=self.paired([b],[c])
        self.assertEqual(result['pairs'][0]['candidate_process_exit_code'],-6)
        self.assertEqual(result['tasks']['fixture@1']['source_process_exits']['candidate'],[-6])
    def test_invalid_or_missing_process_status(self):
        for value in [True,0.0,'0',None]:
            with self.assertRaisesRegex(ValueError,'exit code'):validate([row()|{'process_exit_code':value}])
        with self.assertRaisesRegex(ValueError,'one-sided'):self.paired([row()|{'process_exit_code':0}],[row()])
        with self.assertRaisesRegex(ValueError,'availability'):validate([row()|{'process_exit_code':0},row('1')])
    def test_transitions(self):
        b=[row('0'),row('1','b'),row('2','b'),row('3')]
        c=[row('0','b'),row('1'),row('2','c'),row('3')]
        q=self.paired(b,c)['tasks']['fixture@1']['counts']
        self.assertEqual([q[x] for x in ['correct_to_wrong','wrong_to_correct','wrong_to_wrong','both_correct']], [1]*4)
        self.assertEqual(q['all_answer_disagreement'],3);self.assertEqual(q['correctness_flip'],2)
    def test_duplicate(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):validate([row(),row()])
    def test_duplicate_json_field(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.jsonl';p.write_text('{"sample_id":"one","sample_id":"two"}\n')
            with self.assertRaisesRegex(ValueError,'duplicate JSON field'):load(p)
    def test_schema_version_is_integer(self):
        for value in (True,1.0,'1',2):
            r=row();r['schema_version']=value
            with self.assertRaisesRegex(ValueError,'schema version'):validate([r])
    def test_missing(self):
        with self.assertRaisesRegex(ValueError,'missing pair'):self.paired([row(),row('1')],[row()])
    def test_intersection_explicit(self):
        x=self.paired([row(),row('1')],[row()],intersection=True)
        self.assertEqual(x['excluded_baseline'],[['fixture','1','1']]);self.assertEqual(x['n_pairs'],1)
    def test_mismatches(self):
        for field,value in [('gold',['c']),('prompt_hash','b'*64),('gold_definition','other'),('evidence_kind','historical')]:
            c=row();c[field]=value
            if field=='gold':c['correct']=False
            with self.subTest(field=field),self.assertRaises(ValueError):self.paired([row()],[c])
    def test_reorder_not_silently_repaired(self):
        with self.assertRaisesRegex(ValueError,'labels'):self.paired([dist(row())],[dist(row(),labels=('b','a','c'))])
    def test_nonfinite(self):
        for value in [float('nan'),float('inf'),-.1,True]:
            with self.subTest(value=value),self.assertRaises(ValueError):validate([dist(row(),(value,.3,.1))])
    def test_invalid_sum(self):
        with self.assertRaisesRegex(ValueError,'sum'):validate([dist(row(),(.6,.3,.2))])
    def test_wrong_length(self):
        with self.assertRaisesRegex(ValueError,'length'):validate([dist(row(),(.6,.4))])
    def test_not_four_labels(self):
        r=self.paired([dist(row())],[dist(row(),(.5,.4,.1))])['pairs'][0]
        self.assertAlmostEqual(r['delta_gold_nll'],math.log(.6/.5))
        self.assertAlmostEqual(r['choice_kl_B_to_C'],.6*math.log(.6/.5)+.3*math.log(.3/.4))
    def test_free_generation(self):
        a=row();a.update(output_type='extracted_answer',answer='42',gold=['42'],correct=True)
        b=copy.deepcopy(a);b.update(answer='41',correct=False)
        p=self.paired([a],[b])['pairs'][0]
        self.assertTrue(p['correct_to_wrong']);self.assertNotIn('delta_brier',p)
    def test_full_requires_complete(self):
        x=dist(row());x['distribution']['kind']='full_vocabulary'
        with self.assertRaisesRegex(ValueError,'cover'):validate([x])
    def test_full_kl_named_separately(self):
        b=dist(row());c=dist(row(),(.5,.4,.1))
        for r in [b,c]:r['distribution'].update(kind='full_vocabulary',complete=True,vocabulary_size=3)
        p=self.paired([b],[c])['pairs'][0]
        self.assertIn('full_vocabulary_kl_B_to_C',p);self.assertNotIn('choice_kl_B_to_C',p)
    def test_infinite_kl_visible(self):
        c=dist(row(),(.9,.1,0));p=self.paired([dist(row())],[c])['pairs'][0]
        self.assertIsNone(p['choice_kl_B_to_C']);self.assertTrue(p['choice_kl_B_to_C_infinite'])
    def test_missing_distribution(self):
        with self.assertRaisesRegex(ValueError,'one-sided'):self.paired([row()],[dist(row())])
    def test_seed_reproducibility(self):
        self.assertEqual(self.paired([row(),row('1')],[row(),row('1','c')]),self.paired([row(),row('1')],[row(),row('1','c')]))
    def test_zero_conditional_denominator(self):
        r=self.paired([row('0','b')],[row()])['tasks']['fixture@1']
        self.assertEqual(r['regression_denominator'],0)
    def test_official_adapter(self):
        s=dict(doc_id=3,arguments=[['prompt',' a'],['prompt',' b']],filtered_resps=[[-2,False],[-1,True]],target='1',acc=1)
        r=harness_sample(s,task='arc_challenge',version='1',model_id='B',artifact_id='b',metric='acc')
        self.assertTrue(validate([r])[0]['correct'])
        s['acc']=0
        with self.assertRaises(ValueError):harness_sample(s,task='arc',version='1',model_id='B',artifact_id='b',metric='acc')
    def test_history_retained(self):
        r=self.paired([row(kind='historical')],[row(kind='historical')])
        self.assertEqual(r['pairs'][0]['evidence_kind'],'historical')
    def test_json_duplicate_load(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.jsonl';p.write_text((json.dumps(row())+'\n')*2)
            with self.assertRaises(ValueError):load(p)

if __name__=='__main__':unittest.main()
