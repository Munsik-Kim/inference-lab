"""Labelled CPU fixtures only. None are experimental observations."""
from __future__ import annotations
import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src import metrics as m
from src.storage import Ledger, digest, external_directory, load, write_new
from src.tasks import COUNTS, LABELS, TASKS, OracleError, build_prompt, interpret, permute, scenario, user_content
from src.statistics import balanced_interval, paired_join, scenario_draws, select_boundary, zero_event_upper
from src.reference import positions, tiny_reference
from src.intervention import ScopedAttention, route
from src.validity import REQUIRED, assess, study_status
from src.trajectory import alignment, parse_structured, reference_prefixes
from src.timing import paired_ratios, timing_interval, wall_block
from scripts.verify_results import scalar_bootstrap, scalar_norm, scalar_score, verify
from scripts.package import safe_name
from scripts.build_demo import embedded_json, build
from src.analysis import summarize

CASE=Path(__file__).resolve().parents[1]


def score(x,gold=0):
    return m.score_full(np.array([*x,-3.,-4.]),[0,1,2,3],gold)


class TaskTests(unittest.TestCase):
    def test_all_budgeted_latents_have_unique_gold_and_base(self):
        # Latents only; not evaluation prompts or model outputs.
        rows=[scenario(s,t,i) for s,n in COUNTS.items() for t in TASKS for i in range(n)]
        self.assertEqual(len(rows),486)
        self.assertEqual(len({r['base_id'] for r in rows}),len(rows))
        self.assertEqual(len({r['base_facts_hash'] for r in rows}),len(rows))
        for r in rows:
            self.assertEqual(len(set(r['options'])),4)
            self.assertEqual(r['options'][LABELS.index(r['gold'])],r['gold_value'])

    def test_permutation_updates_gold_and_preserves_scenario(self):
        r=scenario('dev','RETRIEVAL',0)
        for shift in range(4):
            p=permute(r,shift)
            self.assertEqual(p['options'][LABELS.index(p['gold'])],r['gold_value'])
            self.assertEqual(p['base_id'],r['base_id'])

    def test_no_answer_metadata_in_model_render(self):
        r=scenario('dev','CODE',1); expected=user_content(r)
        r['gold']='SECRET_SENTINEL';r['support_ids']=['PRIVATE_SENTINEL'];r['seed']='META_SENTINEL'
        self.assertEqual(user_content(r),expected)
        self.assertNotIn('SENTINEL',expected)

    def test_exact_oracle(self):
        self.assertEqual(interpret('x = 7\nfor i in range(4):\n    x = x + 3 * i\nanswer = x % 13'),12)

    def test_oracle_rejects_unsafe_and_unbounded(self):
        for program in ['import os','answer = open("x")','answer = ().__class__','while True: pass',
                        'for i in range(9):\n    answer = i','answer = 999999 * 999999','answer = 1 % 0',
                        'answer = missing','answer = True','for i in range(0):\n    import os']:
            with self.subTest(program=program),self.assertRaises((OracleError,SyntaxError)):
                interpret(program)

    def test_retained_lengths_and_label_convention(self):
        rows=load(CASE/'inputs/prompts.json')
        self.assertEqual(len(rows),102)
        for r in rows:
            self.assertEqual(len(r['token_ids']),512 if r['split']=='smoke' else 4096)
            self.assertEqual(r['label_token_ids'],[32,33,34,35])
            self.assertEqual(digest(r['token_ids']),r['token_hash'])
            self.assertFalse(r['future_answer_tokens_present'])
            self.assertNotIn('gold',r)

    def test_independent_gold_check(self):
        report=verify(CASE)
        self.assertEqual(report['independent_gold_records_checked'],102)
        self.assertEqual(report['measured_scores_checked'],0)

    def test_saved_gold_matches_current_generator(self):
        for row in load(CASE/'inputs/gold.json'):
            index=int(row['base_id'].split('-')[-1])
            self.assertEqual(row,scenario(row['split'],row['task'],index))

    def test_unstable_continuation_context_is_rejected(self):
        class TokenizerFixture:
            def apply_chat_template(self,messages,**kwargs):return messages[0]['content']+'|'
            def encode(self,text,**kwargs):
                ids=list(range(len(text)))
                if text.endswith(tuple('ABCD')):ids[0]=-1
                return ids
        # Pick the no-filler exact length so this tests label boundary, not filler.
        item=scenario('smoke','CODE',0);target=len(user_content(item)+'|')
        with self.assertRaisesRegex(ValueError,'ANSWER_INTERFACE'):
            build_prompt(item,TokenizerFixture(),target)


class MetricTests(unittest.TestCase):
    def test_uniform_brier_convention(self):
        s=score([0,0,0,0]);self.assertAlmostEqual(s['choice_nll'],math.log(4));self.assertAlmostEqual(s['choice_brier'],.75)
        self.assertTrue(s['winner_tied']);self.assertEqual(s['winner_gap'],0)

    def test_full_mass_identity_and_wrong_vocabulary_argmax(self):
        s=m.score_full([0,0,0,0,10],[0,1,2,3],0)
        self.assertAlmostEqual(s['full_gold_nll'],s['choice_nll']-s['log_label_mass'])
        self.assertFalse(s['full_argmax_allowed']);self.assertLess(s['label_mass'],.001)

    def test_independent_scalar_scores(self):
        for logits in [[1000,999,998,997],[-1000,-999,-998,-997],[1,3,2,4]]:
            actual=score(logits,2);independent=scalar_score(actual)
            for k in ('choice_nll','choice_brier','gold_margin','label_mass','full_gold_nll'):
                self.assertAlmostEqual(actual[k],independent[k],places=10)

    def test_reject_nan_inf(self):
        for x in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(ValueError): score([1,2,3,x])

    def test_regression_gain_wrong_wrong(self):
        cases=[([4,3,2,1],[3,4,2,1],'regression'),([3,4,2,1],[4,3,2,1],'gain'),([1,4,3,2],[1,3,4,2],'both_wrong')]
        pairs=[m.paired(score(b),score(c)) for b,c,_ in cases]
        self.assertEqual([p['cell'] for p in pairs],[x[2] for x in cases]);self.assertTrue(pairs[-1]['wrong_to_wrong_flip'])
        out=m.outcomes(pairs);self.assertEqual(out['accuracy_delta'],0);self.assertEqual(out['flips']['numerator'],3)
        self.assertEqual(out['regression_given_B_correct']['denominator'],1)

    def test_empty_denominator_is_undefined(self):
        out=m.outcomes([m.paired(score([4,3,2,1]),score([4,3,2,1]))])
        self.assertIsNone(out['gain_given_B_wrong']['rate'])

    def test_common_shift(self):
        b=score([4,2,1,0]);c=score([5,2,1,0]);shifted=score([105,102,101,100])
        self.assertTrue(np.allclose(c['q'],shifted['q']))
        self.assertAlmostEqual(m.paired(b,c)['oscillation'],m.paired(b,shifted)['oscillation'])

    def test_gap_not_gold_margin(self):
        s=score([1,5,3,0]);self.assertEqual(s['winner_gap'],2);self.assertEqual(s['gold_margin'],-4)

    def test_sufficient_bound_and_R_counterexample(self):
        p=m.paired(score([5,4,0,0]),score([5.1,4,0,0]))
        self.assertTrue(p['sufficient_unchanged']);self.assertFalse(p['flip'])
        p=m.paired(score([5,4,0,0]),score([7,4,0,0]))
        self.assertGreaterEqual(p['R'],1);self.assertFalse(p['flip'])

    def test_tie_R_undefined(self):
        self.assertIsNone(m.paired(score([1,1,0,0]),score([1,1,0,0]))['R'])

    def test_KL_full_vectors_and_shift(self):
        self.assertAlmostEqual(m.kl_logits([1,2,3],[101,102,103]),0)
        self.assertGreater(m.kl_logits([1,2,3],[3,2,1]),0)
        with self.assertRaises(ValueError):m.kl_logits([1,2],[1,2,3])

    def test_norms_independent_and_near_zero(self):
        u=m.norm_evidence([2,4],[1,2]);v=scalar_norm(u)
        self.assertEqual(u['relative_error'],1)
        for k in v:self.assertAlmostEqual(v[k],u[k])
        self.assertIsNone(m.norm_evidence([1],[1e-7])['relative_error'])
        self.assertFalse(m.norm_evidence([np.nan],[1])['valid'])

    def test_pooled_RMS_not_mean_of_ratios(self):
        units=[m.norm_evidence([2],[1]),m.norm_evidence([11],[10])]
        self.assertAlmostEqual(m.pool_norms(units)['relative_error'],math.sqrt(2/101))


class ValidityTests(unittest.TestCase):
    def test_missing_evidence_blocks(self):
        self.assertFalse(assess({})['valid']);self.assertFalse(assess({k:None for k in REQUIRED})['valid'])

    def test_unsampled_nonfinite_blocks_whole_verdict(self):
        # The four sampled logits are finite, but an unobserved full output is not.
        self.assertTrue(np.isfinite([1,2,3,4]).all())
        checks={k:True for k in REQUIRED};checks['full_attention_outputs_finite']=False
        result=study_status([{'validity':checks}],1,{'status':'PASS'})
        self.assertEqual(result['execution_status'],'BLOCKED_NUMERICAL_VALIDITY')
        self.assertNotEqual(result['semantic_validity_status'],'VALID')

    def test_incomplete_semantics_never_complete(self):
        r={'validity':{k:True for k in REQUIRED}}
        self.assertNotEqual(study_status([r],1,None)['execution_status'],'COMPLETED_CONTROLLED_STUDY')

    def test_empty_results_not_no_effect(self):
        s=study_status([],576,None)
        self.assertEqual(s['observable_effect_summary'],'NOT_ESTIMABLE')
        self.assertEqual(s['deployment_verdict'],'NOT_ASSESSED')

    def test_layer_prefill_only_and_masks(self):
        self.assertEqual(route('V4',12,512,512,mask_present=False,dropout=0,geometry=(1,16,8,128)),'native_BF16')
        self.assertEqual(route('V4',13,1,513,mask_present=False,dropout=0,geometry=(1,16,8,128)),'native_BF16')
        self.assertEqual(route('V4',13,512,512,mask_present=False,dropout=0,geometry=(1,16,8,128)),'sageattn_qk_int8_pv_fp16_cuda')
        for params in [(True,0,(1,16,8,128)),(False,.1,(1,16,8,128)),(False,0,(1,16,4,128))]:
            with self.assertRaises(ValueError):route('V4',13,512,512,mask_present=params[0],dropout=params[1],geometry=params[2])

    def test_scoped_restoration_on_exception_and_passthrough(self):
        class Registry(dict):
            def register(self,name,value):self[name]=value
        sentinel=object();original=lambda *a,**k:(sentinel,None)
        reg=Registry(sdpa=original);q=SimpleNamespace(shape=(1,16,2,128));k=SimpleNamespace(shape=(1,8,2,128))
        with self.assertRaises(RuntimeError):
            with ScopedAttention(reg,'B'):
                out=reg['sdpa'](SimpleNamespace(layer_idx=13),q,k,k,None,scaling=.1)
                self.assertIs(out[0],sentinel)
                raise RuntimeError('fixture exception')
        self.assertIs(reg['sdpa'],original)


class ReferenceTests(unittest.TestCase):
    def test_distinct_GQA_and_causal_leakage(self):
        q=np.zeros((1,4,3,2));k=np.zeros((1,2,3,2));v=np.zeros_like(k)
        v[:,0]=2;v[:,1]=9
        out=tiny_reference(q,k,v,1)
        self.assertTrue(np.all(out[:,0:2]==2));self.assertTrue(np.all(out[:,2:]==9))
        v[0,0,-1]=100
        causal=tiny_reference(q,k,v,1,True);noncausal=tiny_reference(q,k,v,1,False)
        self.assertTrue(np.all(causal[0,0,0]==2));self.assertTrue(np.all(noncausal[0,0,0]>2))

    def test_reference_immutable_and_scale(self):
        q=np.ones((1,1,2,1));k=np.array([[[[0.],[2.]]]]);v=k.copy();saved=[x.copy() for x in (q,k,v)]
        low=tiny_reference(q,k,v,0,False);high=tiny_reference(q,k,v,2,False)
        self.assertTrue(np.all(high>low))
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip((q,k,v),saved)))

    def test_invalid_reference_inputs_and_positions(self):
        for n in (512,2048,4096):
            p=positions(n);self.assertEqual(len(p),32);self.assertEqual(p[-1],n-1)
        with self.assertRaises(ValueError):tiny_reference(np.zeros((1,3,2,1)),np.zeros((1,2,2,1)),np.zeros((1,2,2,1)),1)


class StatisticsTests(unittest.TestCase):
    def fixture_rows(self):
        return [{'base_id':f'{t}-{i}','task':t,'length':n,'delta_choice_nll':i/10} for t in TASKS for i in range(3) for n in (512,4096)]

    def test_cluster_draws_keep_repeats_and_independent_check(self):
        rows=self.fixture_rows();draws=scenario_draws(rows,repetitions=100)
        self.assertEqual(sum(len(v) for v in draws['ids'].values()),9)
        a=balanced_interval(rows,'delta_choice_nll',draws);b=scalar_bootstrap(rows,'delta_choice_nll',draws)
        self.assertAlmostEqual(a['mean'],b['mean']);self.assertTrue(np.allclose(a['ci95'],b['ci95']))
        self.assertEqual(draws,scenario_draws(rows,repetitions=100))

    def test_boundary_selection_is_B_only_and_ties_not_refilled(self):
        pool=[{'arm':'B','split':'boundary_pool','base_id':str(i),'task':'CODE','score':{'winner_gap':0 if i==0 else .2}} for i in range(9)]
        selected=select_boundary(pool)
        self.assertEqual(len(selected['selected_ids']),6);self.assertNotIn('0',selected['selected_ids'])
        pool[0]['arm']='V4'
        with self.assertRaises(ValueError):select_boundary(pool)

    def test_zero_events_nonzero_population_bound(self):
        self.assertGreater(zero_event_upper(64)['upper'],0);self.assertIsNone(zero_event_upper(0)['upper'])

    def test_join_rejects_mock_missing_duplicate_and_hash_mismatch(self):
        # In-memory schema fixture; it is never exported to results.
        template={'item_id':'fixture','base_id':'fixture','task':'CODE','split':'standard','length':4096,
                  'token_hash':'t','design_hash':'d','manifest_hash':'m','evidence_kind':'gpu_measurement','mock':False}
        rows=[dict(template,arm=a) for a in ('B','A_PUBLIC','V4')]
        self.assertEqual(len(paired_join(rows)),1)
        for changed in [rows[:2],rows+[rows[0]],[dict(r,mock=True) for r in rows]]:
            with self.assertRaises(ValueError):paired_join(changed)
        rows[1]['token_hash']='different'
        with self.assertRaises(ValueError):paired_join(rows)

    def test_timing_block_boundary_and_ratio(self):
        events=[];ticks=iter([1.,2.])
        self.assertEqual(wall_block(lambda:events.append('call'),lambda:events.append('sync'),5,lambda:next(ticks)),.2)
        self.assertEqual(events,['sync']+['call']*5+['sync'])
        rows=[{'base_id':'fixture','process':0,'block':i,'arm':a,'seconds_per_call':t,
               'token_hash':'t','evidence_kind':'gpu_timing','mock':False} for i in range(2) for a,t in [('B',2.),('V4',1.)]]
        pairs=paired_ratios(rows,'V4');self.assertEqual(pairs[0]['ratio'],2.)
        self.assertEqual(timing_interval(pairs,repetitions=20)['ci95'],[2.,2.])
        rows[0]['mock']=True
        with self.assertRaises(ValueError):paired_ratios(rows,'V4')


class TrajectoryStorageTests(unittest.TestCase):
    def test_demo_escapes_embedded_script_terminators(self):
        payload={'prompt':'</script><script>alert(1)</script>&\u2028'}
        encoded=embedded_json(payload)
        self.assertNotIn('<',encoded);self.assertEqual(json.loads(encoded),payload)

    def test_demo_and_summary_do_not_invent_measurements(self):
        s,p=summarize(CASE);self.assertEqual(s['measured_record_count'],0);self.assertEqual(p,[])
        self.assertEqual(set(s['arms']),{'B','A_PUBLIC','V4'})
        self.assertTrue(all(v['status']=='NOT_RUN' and v['metrics'] is None for v in s['arms'].values()))
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'index.html';build(CASE,output);html=output.read_text()
            self.assertIn('MEASUREMENT_NOT_RUN',html);self.assertNotIn('fetch(',html)
            self.assertNotIn('innerHTML',html);self.assertNotIn('eval(',html)

    def test_common_prefix_never_contains_future_target(self):
        states=list(reference_prefixes([1,2],[3,4,5]))
        self.assertEqual(states,[([1,2],3),([1,2,3],4),([1,2,3,4],5)])

    def test_token_match_not_state_recovery(self):
        r=alignment([1,2,3,4],[9,2,8,4],{99})
        self.assertEqual(r['first_aligned_match_after_difference'],1)
        self.assertEqual(r['realignment_status'],'NOT_APPLICABLE')
        self.assertFalse(r['state_recovery_claim'])

    def test_EOS_not_padding_equality(self):
        with self.assertRaises(ValueError):alignment([1,99,0],[1,99],{99})
        self.assertTrue(alignment([1,2],[3,4],{99})['right_censored'])

    def test_secondary_schema_and_duplicate_keys(self):
        good=parse_structured('{"choice":"A","evidence":["R01"]}','A',['R01'])
        self.assertTrue(good['schema_valid']);self.assertTrue(good['evidence_correct'])
        bad=parse_structured('{"choice":"A","choice":"B","evidence":[]}','A',[])
        self.assertFalse(bad['schema_valid'])

    def test_atomic_no_overwrite_resume_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d));identity={'tokens':'abc'}
            ledger.put('cell',identity,{'data':1})
            self.assertEqual(ledger.get('cell',identity)['payload']['data'],1)
            with self.assertRaises(FileExistsError):ledger.put('cell',identity,{'data':2})
            with self.assertRaises(ValueError):ledger.get('cell',{'tokens':'changed'})
            with self.assertRaises(ValueError):ledger.path('../escape')

    def test_serialization_no_nan_and_output_safety(self):
        with self.assertRaises(ValueError):digest({'x':float('nan')})
        with self.assertRaises(ValueError):external_directory(CASE/'results/new',CASE)
        for name in ('../secret','/absolute','a/../../b','C:\\secret','a\\b'):
            self.assertFalse(safe_name(name))
        self.assertTrue(safe_name('inputs/prompts.json'))


if __name__=='__main__':unittest.main()
