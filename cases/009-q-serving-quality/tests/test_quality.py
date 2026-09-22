import copy
from pathlib import Path
import sys
import unittest
import math
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_quality import wiki_aggregate,pair_rows,scalar_sample,transition_summary,process_status,validate_coverage,generation_summary


class QualityContracts(unittest.TestCase):
    def test_missing_filter_or_subject_rejected(self):
        freeze={'tasks':{'gsm8k':{'n_documents':1}}}
        rows=[{'task':'gsm8k','id':'0','filter':f} for f in ['strict-match','flexible-extract']]
        validate_coverage('gsm8k',rows,freeze)
        with self.assertRaisesRegex(ValueError,'filter'):validate_coverage('gsm8k',rows[:1],freeze)
        with self.assertRaisesRegex(ValueError,'task'):validate_coverage('mmlu',[],{'tasks':{'mmlu_a':{'n_documents':1}}})
    def test_generation_coverage_and_cap(self):
        audit=[{'outputs':[{'tokens':1024,'finish_reason':'length'}]}]
        self.assertEqual(generation_summary(audit,1)['at_1024_token_cap'],1)
        with self.assertRaisesRegex(ValueError,'coverage'):generation_summary(audit,2)
        audit[0]['outputs'][0]['tokens']=1025
        with self.assertRaisesRegex(ValueError,'length'):generation_summary(audit,1)
    def test_result_marker_does_not_hide_failed_teardown(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'complete.json').write_text('{"status":"PASS"}')
            self.assertEqual(process_status(p),'INCOMPLETE')
            (p/'process_exit.json').write_text('{"returncode":-6}')
            self.assertEqual(process_status(p),'FAILED')
            (p/'process_exit.json').write_text('{"returncode":0}')
            self.assertEqual(process_status(p),'COMPLETE')
    def test_ppl_uses_summed_denominator(self):
        r=wiki_aggregate([{'loglikelihood':-2.,'words':1,'bytes':2},{'loglikelihood':-6.,'words':3,'bytes':6}])
        self.assertAlmostEqual(r['word_perplexity'],math.exp(2))
        self.assertAlmostEqual(r['byte_perplexity'],math.e)
        self.assertAlmostEqual(r['bits_per_byte'],1/math.log(2))
    def test_ppl_not_document_average(self):
        r=wiki_aggregate([{'loglikelihood':-2.,'words':1,'bytes':2},{'loglikelihood':-3.,'words':3,'bytes':6}])
        self.assertNotAlmostEqual(r['word_perplexity'],(math.exp(2)+math.e)/2)
    def test_zero_invalid_denominator(self):
        with self.assertRaises(ValueError):wiki_aggregate([{'loglikelihood':-1.,'words':0,'bytes':0}])
    def test_duplicate_and_missing_pairs(self):
        r={'task':'t','filter':'none','id':'0','prompt_hash':'a','doc_hash':'b'}
        with self.assertRaisesRegex(ValueError,'duplicate'):pair_rows([r,r],[r])
        with self.assertRaisesRegex(ValueError,'missing'):pair_rows([r],[])
    def test_prompt_gold_and_denominator_mismatch(self):
        r={'task':'t','filter':'none','id':'0','prompt_hash':'a','doc_hash':'b','gold':0,'words':2}
        for key in ('prompt_hash','gold','words'):
            c=copy.deepcopy(r);c[key]='different'
            with self.assertRaisesRegex(ValueError,'mismatch'):pair_rows([r],[c])
    def test_correctness_flip_not_wrong_to_wrong(self):
        p=[({'correct':False,'prediction':1},{'correct':False,'prediction':2}),
           ({'correct':True,'prediction':0},{'correct':False,'prediction':1})]
        r=transition_summary(p,np.random.default_rng(1))
        self.assertEqual(r['all_answer_disagreement'],2)
        self.assertEqual(r['correctness_flips_Dutta_definition'],1)
        self.assertEqual(r['wrong_to_different_wrong'],1)
    def test_official_metric_inconsistency_rejected(self):
        r={'doc_id':0,'filter':'none','arguments':[['q','a'],['q','b']], 'doc_hash':'h','filtered_resps':[[-1,False],[-2,False]],'target':0,'acc':0}
        with self.assertRaisesRegex(ValueError,'argmax'):scalar_sample('arc_challenge',r)
    def test_normalized_choice_is_not_raw_argmax(self):
        r={'doc_id':0,'filter':'none','arguments':[], 'doc_hash':'h','filtered_resps':[[-1,False],[-2,False]],'target':1,'acc':0,'acc_norm':1,'doc':{'choices':{'text':['a','bbbb']}}}
        s=scalar_sample('arc_challenge',r)
        self.assertEqual(s['prediction'],0);self.assertEqual(s['prediction_norm'],1)
    def test_filtered_outputs_not_mixed(self):
        a={'task':'gsm8k','id':'1','filter':'strict-match'};b=a|{'filter':'flexible-extract'}
        with self.assertRaisesRegex(ValueError,'missing'):pair_rows([a],[b])


if __name__=='__main__':unittest.main()
