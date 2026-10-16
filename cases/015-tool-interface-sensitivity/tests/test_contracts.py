"""Meaningful CPU execution boundaries, independent of model availability."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from interface_sensitivity.common import CASE, METHODS, INTERFACES, digest, inputs, parse_calls, atomic
from interface_sensitivity.registry import Registry, ORIGINAL_NAMES
from interface_sensitivity.transform import make_interface, reverse, lower_camel
from interface_sensitivity.fixture import Fixture
from interface_sensitivity.scoring import score
from interface_sensitivity.equivalence import oracle_check, oracle_actions
from interface_sensitivity.runner import run_episode, ObservationError
from interface_sensitivity.analysis import summarize, validate_records
from interface_sensitivity.model_adapter import forbid, verify_saved_artifacts
from interface_sensitivity.audit import independent_verdict


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tasks,self.messages,self.tools=inputs()
        self.tmp=tempfile.TemporaryDirectory(prefix='case015-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def spec(self,t=6,i='I0_ORIGINAL'):
        task=self.tasks[t]
        return make_interface(task,self.messages[task['id']],self.tools,i)

    def fixture(self,t=6,i='I0_ORIGINAL',name='episode'):
        return Fixture(self.tasks[t],Registry(self.spec(t,i)['mapping']),self.root/name)

    def test_identity_and_typed_roundtrip_all_tasks(self):
        for t,task in enumerate(self.tasks):
            for i in INTERFACES:
                spec=self.spec(t,i)
                self.assertEqual(reverse(spec),{'tools':self.tools,'messages':self.messages[task['id']]})
                if i=='I0_ORIGINAL':
                    self.assertEqual(spec['tools'],self.tools)
                    self.assertEqual(spec['messages'],self.messages[task['id']])

    def test_bijection_collision_missing_rejected(self):
        for mapping in [{},dict.fromkeys(ORIGINAL_NAMES,'same')]:
            with self.assertRaises(ValueError):Registry(mapping)

    def test_order_only_nonidentity_content_unchanged(self):
        for t in range(8):
            spec=self.spec(t,'I1_ORDER_ONLY')
            self.assertNotEqual(spec['permutation'],list(range(6)))
            self.assertCountEqual(spec['tools'],self.tools)
            self.assertEqual(spec['messages'],self.messages[self.tasks[t]['id']])

    def test_style_is_lower_not_upper_camel(self):
        self.assertEqual(lower_camel('apply_fixture_patch'),'applyFixturePatch')
        self.assertNotEqual(lower_camel('apply_fixture_patch'),'ApplyFixturePatch')

    def test_opaque_allocation_does_not_read_gold_or_position(self):
        task=copy.deepcopy(self.tasks[6]);task['gold']={'valid':'wrong'}
        a=self.spec(6,'I3_OPAQUE_RENAME')['mapping']
        b=make_interface(task,self.messages[task['id']],list(reversed(self.tools)),'I3_OPAQUE_RENAME')['mapping']
        self.assertEqual(a,b)

    def test_no_global_free_text_replacement(self):
        task=copy.deepcopy(self.tasks[6]);messages=copy.deepcopy(self.messages[task['id']])
        messages[0]['content']+=' Customer note: validate_fixture_test'
        spec=make_interface(task,messages,self.tools,'I3_OPAQUE_RENAME')
        self.assertEqual(spec['messages'][0],messages[0])
        self.assertNotIn('validate_fixture_test',spec['messages'][-1]['content'])

    def test_oracle_each_step_real_fixture_all32(self):
        result=oracle_check(CASE)
        self.assertEqual(result['oracle_episodes'],32)
        self.assertEqual(result['oracle_steps'],48)

    def test_old_and_typo_names_are_unknown_no_side_effect(self):
        for i in INTERFACES[2:]:
            fixture=self.fixture(i=i,name=i)
            before=fixture.state
            for name in ['apply_fixture_patch','ApplyFixturePatch','applyFixturePatcH','../../config.json','__import__']:
                event=fixture.execute(name,{'enabled':True,'limit':14},0,0)
                self.assertIn('UNKNOWN_TOOL',event['error_labels'])
                self.assertEqual(event['tool_result'],{'error':'unknown tool/incorrect argument keys'})
                self.assertEqual(fixture.state,before)

    def test_schema_key_type_and_value_are_separate(self):
        fixture=self.fixture()
        for args,kind in [({'enabled':True},'KEYS'),({'enabled':1,'limit':14},'TYPE'),({'enabled':True,'limit':2},'VALID')]:
            e=fixture.execute('apply_fixture_patch',args,0,0)
            self.assertEqual(e['schema_result'],kind)
            self.assertIn('BAD_ARGUMENTS',e['error_labels'])
        self.assertEqual(fixture.state,{'enabled':True,'limit':2})

    def test_unsupported_valid_claim_same_json_different_verdict(self):
        fixture=self.fixture();task=self.tasks[6]
        self.assertFalse(score(task,{'valid':True},fixture.state,fixture.events)['functional_success'])
        fixture.execute('apply_fixture_patch',task['fixture']['expected_config'],0,0)
        self.assertFalse(score(task,{'valid':True},fixture.state,fixture.events)['functional_success'])
        fixture.execute('validate_fixture_test',{},1,0)
        self.assertTrue(score(task,{'valid':True},fixture.state,fixture.events)['functional_success'])

    def test_legacy_vs_functional_required_document_read(self):
        fixture=self.fixture(t=0);task=self.tasks[0]
        r=score(task,task['gold'],fixture.state,[])
        self.assertTrue(r['legacy_success']);self.assertFalse(r['functional_success'])

    def test_not_first_tool_scorer_and_commuted_operands(self):
        fixture=self.fixture(t=2,i='I1_ORDER_ONLY');task=self.tasks[2]
        for j,(semantic,args) in enumerate([('record.read',{}),('math.multiply',{'a':3,'b':10})]):
            fixture.execute(fixture.registry.expose(semantic),args,j,0)
        self.assertTrue(score(task,task['gold'],fixture.state,fixture.events)['functional_success'])

    def test_episode_state_and_message_deepcopy_isolation(self):
        a=self.fixture();a.execute('apply_fixture_patch',{'enabled':True,'limit':14},0,0)
        b=self.fixture(name='other')
        self.assertEqual(b.state,self.tasks[6]['fixture']['config'])
        spec=self.spec();spec['messages'][0]['content']='changed'
        self.assertNotEqual(spec['messages'],self.messages['AGENT-6'])

    def test_error_feedback_has_no_hints(self):
        e=self.fixture().execute('missing',{},0,0)
        self.assertEqual(set(e['tool_result']),{'error'})
        self.assertNotIn('allowed',e['tool_result']['error'])

    def test_repeated_bad_call_is_observation_only(self):
        fixture=self.fixture()
        fixture.execute('missing',{},0,0);e=fixture.execute('missing',{},1,0)
        self.assertIn('REPEATED_FAILED_CALL',e['error_labels'])
        self.assertEqual(e['tool_result'],{'error':'unknown tool/incorrect argument keys'})

    def test_parse_failures_kept_alongside_valid_objects(self):
        calls,errors=parse_calls('<tool_call>broken</tool_call><tool_call>{"name":"read_record","arguments":{}}</tool_call>')
        self.assertEqual(len(calls),1);self.assertEqual(len(errors),1)

    def test_multi_call_budget_counts_attempts_and_connects(self):
        task=self.tasks[6];spec=self.spec();seen=[]
        replies=[''.join('<tool_call>'+json.dumps({'name':'missing','arguments':{}})+'</tool_call>' for _ in range(4))]*2
        def generate(messages,tools,deadline):
            seen.append(copy.deepcopy(messages));raw=replies.pop(0)
            return {'raw_text':raw,'text':raw,'output_tokens':1,'input_tokens':1}
        gen={'max_model_calls':2,'max_tool_calls':6,'deadline_seconds':900}
        row=run_episode(task,spec,generate,self.root/'multi',gen,{'method':'BF16'},'test')
        self.assertEqual(row['tool_attempts'],6)
        self.assertEqual(len(seen[1][-5:]),5)
        self.assertEqual([m['name'] for m in seen[1] if m['role']=='tool'],['missing']*4)
        self.assertEqual(len({e['call_id'] for e in row['tool_events']}),6)

    def test_generated_code_is_data_never_executed(self):
        payload='__import__("pathlib").Path("'+str(self.root/'unsafe')+'").write_text("bad")'
        def generate(*args):return {'raw_text':payload,'text':payload,'output_tokens':1,'input_tokens':1}
        row=run_episode(self.tasks[0],self.spec(0),generate,self.root/'unsafe-episode',{'max_model_calls':1,'max_tool_calls':6,'deadline_seconds':900},{},'unsafe')
        self.assertFalse((self.root/'unsafe').exists())
        self.assertEqual(row['terminal_reason'],'PARSE_ERROR')

    def test_infrastructure_missing_is_not_zero_score(self):
        def generate(*args):raise ObservationError('OOM','synthetic OOM')
        row=run_episode(self.tasks[0],self.spec(0),generate,self.root/'oom',{'max_model_calls':1,'max_tool_calls':6,'deadline_seconds':900},{},'oom')
        self.assertIsNone(row['functional_success']);self.assertEqual(row['observation_status'],'UNOBSERVED_INFRA')

    def test_decoder_solver_mitigation_guards_fail_closed(self):
        for name in ['fit','solve','encode','ENUM','feedback']:
            with self.assertRaisesRegex(RuntimeError,'FORBIDDEN'):forbid(name)

    def test_artifact_preflight_missing_rejected_without_gpu(self):
        with self.assertRaises(FileNotFoundError):
            verify_saved_artifacts({'artifact_root':str(self.root/'absent')},CASE)

    def test_cpu_import_and_help_do_not_import_torch(self):
        code='import sys; from interface_sensitivity import registry,analysis,cli; assert "torch" not in sys.modules; sys.argv=["test","--help"]; cli.main()'
        r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('oracle-check',r.stdout)

    def fake_matrix(self):
        plan=[];records=[]
        for ti,task in enumerate(self.tasks):
            for m in METHODS:
                for i in INTERFACES:
                    p={'episode_id':m+'__'+i+'__'+task['id'],'method':m,'interface_id':i,'task_id':task['id'],
                       'family_id':task['family'],'canonical_task_hash':digest(task),'rendered_prompt_hash':digest([task['id'],i])}
                    plan.append(p)
                    records.append({**p,'functional_success':not (m=='D16' and i==INTERFACES[2] and ti<2),'legacy_success':True,
                        'input_token_hash':digest([task['id'],i]),'enum_enabled':False,'aliases_enabled':False,'enhanced_feedback_enabled':False,
                        'tool_events':[],'recovery':[],'first_error_type':None,'tool_attempts':0,'model_calls':1,'generated_tokens':1})
        return plan,records

    def test_interaction_loss_gain_dependency_and_bootstrap(self):
        plan,records=self.fake_matrix();s=summarize(records,plan,1000)
        self.assertAlmostEqual(s['contrasts']['D16-minus-BF16']['estimate'],-1/12)
        self.assertEqual(s['sensitivity'][-1]['changed_pairs'],2)
        self.assertEqual(s['paired_interface_counts'][-2]['baseline_only'],2)
        for value in s['primary_worst_best_all_planned_tasks']:
            self.assertAlmostEqual(value,-1/12)

    def test_duplicate_and_mapping_input_mismatch_rejected(self):
        plan,records=self.fake_matrix()
        with self.assertRaises(ValueError):validate_records(records+[records[0]],plan)
        records[0]['rendered_prompt_hash']='mismatch'
        with self.assertRaises(ValueError):validate_records(records,plan)

    def test_within_interface_prompt_pairing_rejects_mismatch(self):
        plan,records=self.fake_matrix();records[4]['input_token_hash']='wrong'
        with self.assertRaises(ValueError):validate_records(records,plan)

    def test_missing_cells_bounds_shared_I0(self):
        plan,records=self.fake_matrix();records=[r for r in records if r['episode_id']!=plan[0]['episode_id']]
        s=summarize(records,plan,100)
        self.assertEqual(s['status'],'PARTIAL');self.assertEqual(len(s['missing']),1)
        lo,hi=s['primary_worst_best_all_planned_tasks'];self.assertAlmostEqual(hi-lo,1/8)

    def test_independent_oracle_does_not_trust_claim(self):
        f=self.fixture();self.assertEqual(independent_verdict(self.tasks[6],{'valid':True},f.state,[]),(False,False))


if __name__=='__main__':unittest.main()
