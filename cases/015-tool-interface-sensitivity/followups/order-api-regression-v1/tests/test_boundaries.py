import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from order_regression.common import ROOT,METHODS,APIS
from order_regression.analysis import summarize


class Boundaries(unittest.TestCase):
    def test_cpu_help_and_guard_imports_without_torch(self):
        script="import sys; from order_regression import registry,fixture,scoring,analysis,audit,cli; from interface_sensitivity.model_adapter import forbid; assert 'torch' not in sys.modules;\ntry: forbid()\nexcept RuntimeError as e: assert 'FORBIDDEN' in str(e)\nelse: raise AssertionError('quantizer guard absent')\n"
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
        subprocess.run([sys.executable,'-B','-c',script],env=env,check=True,capture_output=True)
        result=subprocess.run([sys.executable,'-B','-m','order_regression.cli','--help'],env=env,check=True,capture_output=True,text=True)
        self.assertIn('oracle-check',result.stdout)

    def test_missing_duplicate_synthetic_and_mismatched_rows(self):
        plan=[];records=[]
        for i in range(16):
            for m in METHODS:
                for a in APIS:
                    p={'episode_id':f'{m}-{a}-{i}','method':m,'api_version':a,'scenario_id':str(i),'template_id':str(i//4),
                        'canonical_task_hash':f'task{i}','surface_prompt_hash':f'{i}{a}','input_token_sha256':f'token{i}{a}'}
                    plan.append(p)
                    r={**p,'input_token_hash':p['input_token_sha256'],'evidence_kind':'measurement','VERIFIED_WORKFLOW_SUCCESS':False,
                        **{k:False for k in ['enum_enabled','enhanced_feedback','aliases_enabled','STATE_ONLY_SUCCESS','FINAL_ANSWER_CORRECT','REQUIRED_ACTIONS_DONE','OBSERVATION_ORDER_VALID','UNKNOWN_TOOL','ARGUMENT_VALUE_WRONG','UNSUPPORTED_SUCCESS_CLAIM']},
                        'observation_violations':[],'model_calls':1,'attempted_tool_calls':0,'generated_tokens':10,'wall_seconds':.1}
                    records.append(r)
        self.assertEqual(summarize(records,plan,replicates=10)['status'],'COMPLETED_A1')
        self.assertEqual(len(summarize(records[:-1],plan,replicates=10)['missing']),1)
        with self.assertRaises(ValueError):summarize(records+[records[0]],plan)
        for key,value in [('evidence_kind','synthetic_contract_test'),('surface_prompt_hash','changed'),('aliases_enabled',True),('input_token_hash','other')]:
            changed=copy.deepcopy(records);changed[0][key]=value
            with self.assertRaises(ValueError):summarize(changed,plan)


if __name__=='__main__':unittest.main()
