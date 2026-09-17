"""No fixture from these tests is exported as measured evidence."""
import unittest
from src.completion import completion_status
from src.validity import REQUIRED

class CompletionTests(unittest.TestCase):
    def test_good_sample_logits_do_not_override_bad_full_output(self):
        r={'split':'standard','length':4096,'arm':'B','base_id':'fixture',
           'option_logits':[1.,2.,3.,4.],'validity':{k:True for k in REQUIRED}}
        r['validity']['full_final_hidden_finite']=False
        s=completion_status([r],{'status':'PASS'},{},{},[])
        self.assertEqual(s['execution_status'],'BLOCKED_NUMERICAL_VALIDITY')
    def test_absent_evidence_blocks_completion(self):
        s=completion_status([],{}, {}, {}, [])
        self.assertNotEqual(s['execution_status'],'COMPLETED_CONTROLLED_STUDY')
        self.assertEqual(s['deployment_verdict'],'NOT_ASSESSED')
    def test_semantic_report_is_not_all_required_evidence(self):
        s=completion_status([],{'status':'PASS'},{'status':'PASS_CPU_MEASUREMENT_CONSISTENCY'},{'status':'PASS'},[])
        self.assertTrue(s['problems']);self.assertEqual(s['evidence_completeness'],'INCOMPLETE')

if __name__=='__main__':unittest.main()
