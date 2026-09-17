"""Study calculation fixtures; never exported as GPU measurements."""
import unittest
import numpy as np
from src.study_analysis import distribution,wilson
from src.statistics import scenario_draws,balanced_interval,select_boundary
from src.metrics import score_full,paired
from src.trajectory import reference_prefixes,alignment
from scripts.verify_results import scalar_bootstrap
from src.validity import REQUIRED,assess

class StudyTests(unittest.TestCase):
    def test_empty_bins_not_zero_risk(self):
        self.assertIsNone(wilson(0,0));self.assertGreater(wilson(0,64)[1],0)
    def test_measured_distribution_preserves_null_count(self):
        self.assertEqual(distribution([None])['n'],0)
        self.assertEqual(distribution([1,2,3])['median'],2)
    def test_gain_and_regression_can_cancel_accuracy_not_scores(self):
        a=paired(score_full([3,1,0,0],[0,1,2,3],0),score_full([1,3,0,0],[0,1,2,3],0))
        b=paired(score_full([1,2,0,0],[0,1,2,3],0),score_full([2,1,0,0],[0,1,2,3],0))
        self.assertEqual(a['cell'],'regression');self.assertEqual(b['cell'],'gain')
        self.assertNotAlmostEqual(a['delta_choice_nll']+b['delta_choice_nll'],0)
    def test_paired_bootstrap_does_not_count_heads_as_documents(self):
        rows=[{'task':t,'base_id':f'{t}-{i}','metric':float(i)} for t in ('RETRIEVAL','COMPARISON','CODE') for i in range(3) for head in range(16)]
        draws=scenario_draws(rows,repetitions=40);a=balanced_interval(rows,'metric',draws);b=scalar_bootstrap(rows,'metric',draws)
        self.assertEqual(a['independent_scenarios'],9);self.assertTrue(np.allclose(a['ci95'],b['ci95']))
    def test_validity_cannot_be_replaced_by_only_finite_options(self):
        checks={k:True for k in REQUIRED};del checks['full_final_hidden_finite']
        self.assertFalse(assess(checks)['valid'])
        checks['full_final_hidden_finite']=False
        self.assertEqual(assess(checks)['status'],'BLOCKED_NUMERICAL_VALIDITY')
    def test_boundary_never_uses_candidate_predictions(self):
        rows=[{'arm':'B','split':'boundary_pool','task':'CODE','base_id':str(i),'score':{'winner_gap':.25,'correct':i%2==0}} for i in range(10)]
        a=select_boundary(rows)['selected_ids']
        for r in rows:r['score']['correct']=not r['score']['correct']
        self.assertEqual(a,select_boundary(rows)['selected_ids'])
    def test_reference_path_scores_before_feeding_target(self):
        history=list(reference_prefixes([7,8],[3,4,5]))
        self.assertEqual([len(p) for p,_ in history],[2,3,4]);self.assertEqual(history[1],([7,8,3],4))
    def test_eight_token_match_is_not_state_recovery(self):
        a=alignment([1]+list(range(10,19)),[2]+list(range(10,19)),{99})
        self.assertEqual(a['eight_token_realignment_start'],1);self.assertFalse(a['state_recovery_claim'])

if __name__=='__main__':unittest.main()
