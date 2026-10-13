"""Publication/scalar tests; synthetic examples are never measured model evidence."""
import importlib.util, unittest
from pathlib import Path
CASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('c013audit_tests',CASE/'analysis/audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class ScalarContracts(unittest.TestCase):
    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):audit.indexed([{'id':'a'},{'id':'a'}],('id',))
    def test_pairing_different_ids_rejected(self):
        with self.assertRaises(ValueError):audit.paired_counts({'a':{'ok':True}},{'b':{'ok':True}},'ok')
    def test_all_paired_outcomes_and_denominator(self):
        a={str(i):{'ok':x} for i,x in enumerate([1,1,0,0])};b={str(i):{'ok':x} for i,x in enumerate([1,0,1,0])}
        self.assertEqual(audit.paired_counts(a,b,'ok'),dict(both_success=1,baseline_only=1,candidate_only=1,both_failed=1))
    def test_first_object_policy_not_strict_document(self):
        self.assertEqual(audit.extract_json('prefix {"answer":"21"} suffix'),{'answer':'21'})
        self.assertIsNone(audit.extract_json('no object'))
    def test_correct_numeric_expression_is_not_exact(self):
        task={'answer':'21','evidence':['D1']}
        self.assertTrue(audit.exact_long('{"answer":"21","evidence":["D1"]}',task))
        self.assertFalse(audit.exact_long('{"answer":"7*3=21","evidence":["D1"]}',task))
    def test_wrong_evidence_and_key_case_remain_failures(self):
        task={'answer':'21','evidence':['D1']}
        for text in ('{"answer":"21","evidence":["D2"]}','{"Answer":"21","Evidence":["D1"]}'):
            self.assertFalse(audit.exact_long(text,task))
    def test_complete_recorded_scalar_readback(self):
        result=audit.audit(CASE)
        self.assertEqual(result['units']['valid_tokens_per_condition'],65504)
        self.assertEqual(len(result['quality']),8)
        self.assertEqual(result['units']['agent_unique_tasks'],8)
        self.assertEqual(result['packed']['status'],'BLOCKED')
        agent={(x['cohort'],x['method'],x['mode']):x['successes'] for x in result['agents']}
        self.assertEqual(agent[('base','D16','FREE')],6)
        self.assertEqual(agent[('base','D4','FREE')],0)
        self.assertEqual(agent[('extension16k','D16','FREE')],3)

if __name__=='__main__':unittest.main()
