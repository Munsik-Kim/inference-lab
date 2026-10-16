"""A1 execution/observation invariants, independent of model availability."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from order_regression.common import digest
from order_regression.prepare import generate_tasks
from order_regression.fixture import Fixture
from order_regression.registry import Registry,MAPS
from order_regression.scoring import assess,expected_total
from order_regression.equivalence import gold_trace
from order_regression.runner import run_episode
from order_regression.audit import independent_endpoint


class OrderContracts(unittest.TestCase):
    def setUp(self):
        self.scenario=generate_tasks()[0];self.target=self.scenario['target_order_id'];self.total=expected_total(self.scenario)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.final={'order_id':self.target,'total_minor':self.total,'validated':True}

    def fixture(self,name='fixture',api='API_V1'):
        return Fixture(self.scenario,Registry(api),self.root/name)

    def test_both_gold_traces_match_every_state_result_and_receipt(self):
        traces=[]
        for api in MAPS:
            f,r,v=gold_trace(self.scenario,api,self.root/api)
            self.assertTrue(v['VERIFIED_WORKFLOW_SUCCESS'])
            traces.append([{k:e[k] for k in ['semantic_tool_id','raw_arguments','tool_result','generated_after_event_ids','state_before_execution','state_after_execution']} for e in f.events])
        self.assertEqual(traces[0],traces[1])

    def test_only_names_differ_in_two_schemas(self):
        tools=[Registry(api).tools() for api in MAPS]
        for a,b in zip(*tools):
            aa=copy.deepcopy(a);bb=copy.deepcopy(b);aa['function'].pop('name');bb['function'].pop('name');self.assertEqual(aa,bb)

    def test_old_names_typos_and_collision_rejected_without_effect(self):
        f=self.fixture(api='API_V2');before=f.snapshot()
        for i,name in enumerate(['read_order','write_order_total','WriteOrderTotal','writeOrderTotaL']):
            e=f.execute(name,{'order_id':self.target,'total_minor':self.total},0,i,[])
            self.assertEqual(e['execution_status'],'ERROR');self.assertEqual(f.snapshot(),before)
        with self.assertRaises(ValueError):Registry('API_V2',dict.fromkeys(MAPS['API_V2'],'duplicate'))

    def invalid_observation_trace(self):
        f=self.fixture();r=self.scenario['orders'][self.target]
        for i,(name,args) in enumerate([('read_order',{'order_id':self.target}),('multiply_amount',{'a':r['quantity'],'b':r['unit_price_minor']}),('write_order_total',{'order_id':self.target,'total_minor':self.total}),('validate_order',{'order_id':self.target})]):f.execute(name,args,0,i,[])
        responses=[{'available_result_event_ids':[]},{'available_result_event_ids':[e['event_id'] for e in f.events]}]
        v=assess(self.scenario,f.events,responses,self.final,f.snapshot(),f.version,f.integrity_violation)
        return f,v

    def test_same_response_read_multiply_write_is_observation_failure(self):
        f,v=self.invalid_observation_trace()
        self.assertTrue(v['STATE_ONLY_SUCCESS']);self.assertTrue(v['REQUIRED_ACTIONS_DONE']);self.assertFalse(v['OBSERVATION_ORDER_VALID']);self.assertFalse(v['VERIFIED_WORKFLOW_SUCCESS'])
        self.assertEqual(len(v['observation_violations']),2)
        self.assertTrue(all(e['generated_after_event_ids']==[] for e in f.events))

    def test_write_and_validate_same_response_are_allowed(self):
        f,r,v=gold_trace(self.scenario,'API_V1',self.root/'combined')
        self.assertEqual(f.events[-2]['generated_in_turn'],f.events[-1]['generated_in_turn'])
        self.assertTrue(v['VERIFIED_WORKFLOW_SUCCESS'])

    def test_wrong_schema_valid_total_is_really_stored(self):
        f=self.fixture();f.execute('write_order_total',{'order_id':self.target,'total_minor':self.total+1},0,0,[])
        self.assertEqual(f.snapshot()[self.target]['total_minor'],self.total+1)
        e=f.execute('validate_order',{'order_id':self.target},1,0,['g0-c0']);self.assertIs(e['tool_result']['valid'],False)
        v=assess(self.scenario,f.events,[{'available_result_event_ids':['g0-c0','g1-c0']}],self.final,f.snapshot(),f.version)
        self.assertFalse(v['VERIFIED_WORKFLOW_SUCCESS']);self.assertTrue(v['UNSUPPORTED_SUCCESS_CLAIM'])

    def test_other_order_change_then_restore_still_violates(self):
        f,r,v=gold_trace(self.scenario,'API_V1',self.root/'other');control=self.scenario['control_order_id'];old=f.snapshot()[control]['total_minor']
        # Ensure an integer restore case without changing the frozen scenario.
        if old is None:
            f.execute('write_order_total',{'order_id':control,'total_minor':100},5,0,[])
            from order_regression.common import atomic
            atomic(f.paths[control],self.scenario['orders'][control])
        else:
            f.execute('write_order_total',{'order_id':control,'total_minor':100},5,0,[]);f.execute('write_order_total',{'order_id':control,'total_minor':old},6,0,[])
        state=f.snapshot();v=assess(self.scenario,f.events,r,self.final,state,f.version,f.integrity_violation)
        self.assertTrue(v['STATE_ONLY_SUCCESS']);self.assertFalse(v['integrity_valid_throughout']);self.assertFalse(v['VERIFIED_WORKFLOW_SUCCESS'])

    def test_immutable_field_change_is_detected(self):
        f,r,v=gold_trace(self.scenario,'API_V1',self.root/'immutable')
        from order_regression.common import atomic
        state=f.snapshot();state[self.target]['quantity']+=1;atomic(f.paths[self.target],state[self.target]);state=f.snapshot()
        self.assertFalse(assess(self.scenario,f.events,r,self.final,state,f.version,f.integrity_violation)['VERIFIED_WORKFLOW_SUCCESS'])

    def test_final_only_correct_json_is_not_workflow_success(self):
        f=self.fixture();v=assess(self.scenario,[],[{'available_result_event_ids':[]}],self.final,f.snapshot(),f.version)
        self.assertTrue(v['FINAL_ANSWER_CORRECT']);self.assertFalse(v['REQUIRED_ACTIONS_DONE']);self.assertFalse(v['VERIFIED_WORKFLOW_SUCCESS'])

    def test_stale_validation_cannot_certify_restored_same_digest(self):
        f,r,v=gold_trace(self.scenario,'API_V1',self.root/'stale');old_version=f.version
        f.execute('write_order_total',{'order_id':self.target,'total_minor':self.total+1},5,0,[]);f.execute('write_order_total',{'order_id':self.target,'total_minor':self.total},6,0,[])
        self.assertGreater(f.version,old_version)
        v=assess(self.scenario,f.events,r,self.final,f.snapshot(),f.version)
        self.assertTrue(v['STATE_ONLY_SUCCESS']);self.assertTrue(v['REQUIRED_ACTIONS_DONE']);self.assertFalse(v['VERIFIED_WORKFLOW_SUCCESS'])

    def test_validation_must_be_seen_before_final_generation(self):
        f,r,v=gold_trace(self.scenario,'API_V1',self.root/'unseen');r[-1]['available_result_event_ids'].remove(f.events[-1]['event_id'])
        self.assertFalse(assess(self.scenario,f.events,r,self.final,f.snapshot(),f.version)['VERIFIED_WORKFLOW_SUCCESS'])

    def test_extra_duplicate_reads_and_commuted_operands_allowed(self):
        f=self.fixture();row=self.scenario['orders'][self.target]
        f.execute('read_order',{'order_id':self.target},0,0,[]);f.execute('read_order',{'order_id':self.target},0,1,[])
        f.execute('multiply_amount',{'a':row['unit_price_minor'],'b':row['quantity']},1,0,['g0-c0','g0-c1'])
        f.execute('write_order_total',{'order_id':self.target,'total_minor':self.total},2,0,['g0-c0','g0-c1','g1-c0']);f.execute('validate_order',{'order_id':self.target},2,1,['g0-c0','g0-c1','g1-c0'])
        v=assess(self.scenario,f.events,[{'available_result_event_ids':[e['event_id'] for e in f.events]}],self.final,f.snapshot(),f.version)
        self.assertTrue(v['VERIFIED_WORKFLOW_SUCCESS'])

    def test_strict_integer_and_boolean_final_types(self):
        f,r,v=gold_trace(self.scenario,'API_V1',self.root/'types')
        for change in [{'validated':1},{'total_minor':True},{'total_minor':float(self.total)},{'extra':1}]:
            final={**self.final,**change};self.assertFalse(assess(self.scenario,f.events,r,final,f.snapshot(),f.version)['FINAL_ANSWER_CORRECT'])
        e=f.execute('multiply_amount',{'a':True,'b':5},8,0,[]);self.assertEqual(e['schema_result'],'TYPE')

    def test_same_execution_state_different_receipt_is_not_same_input(self):
        f=self.fixture();s=digest(f.snapshot());a={'state':s,'available':[]};b={'state':s,'available':['previous-read']}
        self.assertEqual(a['state'],b['state']);self.assertNotEqual(digest(a),digest(b))

    def test_isolated_reset_and_path_symlink_guards(self):
        f=self.fixture();f.execute('write_order_total',{'order_id':self.target,'total_minor':self.total},0,0,[])
        g=self.fixture(name='reset');self.assertNotEqual(g.snapshot()[self.target]['total_minor'],self.total)
        before=g.snapshot();e=g.execute('read_order',{'order_id':'../../secret'},0,0,[]);self.assertEqual(e['execution_status'],'ERROR');self.assertEqual(g.snapshot(),before)
        target=self.root/'outside.json';target.write_text('{}');g.paths[self.target].unlink();g.paths[self.target].symlink_to(target)
        with self.assertRaises(ValueError):g.snapshot()

    def test_raw_response_reexecution_and_frozen_visibility(self):
        scenario=self.scenario;row=scenario['orders'][self.target]
        stages=[[('read_order',{'order_id':self.target})],[('multiply_amount',{'a':row['quantity'],'b':row['unit_price_minor']})],[('write_order_total',{'order_id':self.target,'total_minor':self.total}),('validate_order',{'order_id':self.target})]]
        raws=[''.join('<tool_call>'+json.dumps({'name':n,'arguments':a})+'</tool_call>' for n,a in ss) for ss in stages]+[json.dumps(self.final)]
        def factory():
            queue=iter(raws)
            def generate(messages,tools,deadline):
                raw=next(queue);return {'raw_text':raw,'text':raw,'input_token_sha256':'0'*64,'input_tokens':10,'output_tokens':1,'output_token_ids':[1],
                    'raw_utf8_sha256':hashlib.sha256(raw.encode()).hexdigest(),'messages_tools_hash':digest({'messages':messages,'tools':tools})}
            return generate
        gen={'max_model_calls':8,'max_tool_calls':6,'deadline_seconds':900};identity={'method':'BF16','evidence_kind':'synthetic_contract_test'}
        a=run_episode(scenario,'API_V1',factory(),self.root/'a',gen,identity,'a');b=run_episode(scenario,'API_V1',factory(),self.root/'b',gen,identity,'b')
        self.assertTrue(a['VERIFIED_WORKFLOW_SUCCESS']);self.assertTrue(independent_endpoint(scenario,a));self.assertEqual(a['tool_events'],b['tool_events'])
        self.assertEqual(a['tool_events'][2]['generated_after_event_ids'],a['tool_events'][3]['generated_after_event_ids'])


if __name__=='__main__':unittest.main()
