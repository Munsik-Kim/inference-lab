import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from kv_agent_bench.common import digest
from kv_agent_bench.fixture_generator import make_tasks,oracle,tools,short_messages
from kv_agent_bench.tools import Fixture
from kv_agent_bench.scorer import assess
from kv_agent_bench.audit import independent_endpoint
from kv_agent_bench.vllm_client import StreamAssembly,request_cap

class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.task=make_tasks('TEST',1,6101907)[0];self.fx=Fixture(self.task,pathlib.Path(self.tmp.name)/'fx');self.gold=oracle(self.task)
    def gold_trace(self):
        g=self.gold;f=self.fx
        r=f.execute('read_source',{'source_id':g['source_id']},0,0,[])
        c=f.execute('calculate_amount',{k:g[k] for k in ('quantity','unit_price_minor','discount_percent')},1,0,[r['event_id']])
        w=f.execute('update_state',{k:g[k] for k in ('record_id','total_minor','source_version','status')},2,0,[r['event_id'],c['event_id']])
        v=f.execute('validate_state',{'record_id':g['record_id']},2,1,[r['event_id'],c['event_id']])
        final={k:g[k] for k in ('record_id','total_minor','source_version','status')};final['validated']=True
        return final,[{'available_result_event_ids':[r['event_id'],c['event_id'],w['event_id'],v['event_id']]}]
    def score(self,final,responses):return assess(self.task,self.fx.events,responses,final,self.fx.state,self.fx.version,'FINAL')
    def test_gold_trace_all_families(self):
        for i,t in enumerate(make_tasks('ORACLE',1,6101909)):
            f=Fixture(t,pathlib.Path(self.tmp.name)/f'g{i}');g=oracle(t)
            r=f.execute('read_source',{'source_id':g['source_id']},0,0,[])
            c=f.execute('calculate_amount',{k:g[k] for k in ('quantity','unit_price_minor','discount_percent')},1,0,['e0'])
            f.execute('update_state',{k:g[k] for k in ('record_id','total_minor','source_version','status')},2,0,['e0','e1'])
            f.execute('validate_state',{'record_id':g['record_id']},2,1,['e0','e1'])
            answer={k:g[k] for k in ('record_id','total_minor','source_version','status')};answer['validated']=True
            responses=[{'available_result_event_ids':['e0','e1','e2','e3']}]
            self.assertTrue(assess(t,f.events,responses,answer,f.state,f.version,'FINAL')['VERIFIED_WORKFLOW_SUCCESS'])
            self.assertTrue(independent_endpoint(t,f.events,answer,f.state,f.version,'FINAL',responses))
    def test_write_validate_same_response_allowed(self):
        final,rs=self.gold_trace();self.assertTrue(self.score(final,rs)['VERIFIED_WORKFLOW_SUCCESS'])
    def test_final_without_execution_fails(self):
        final={k:self.gold[k] for k in ('record_id','total_minor','source_version','status')};final['validated']=True
        self.assertFalse(self.score(final,[])['VERIFIED_WORKFLOW_SUCCESS'])
    def test_same_response_cannot_observe_new_read(self):
        final,rs=self.gold_trace();self.fx.events[1]['generated_after_event_ids']=[]
        self.assertFalse(self.score(final,rs)['VERIFIED_WORKFLOW_SUCCESS'])
    def test_stale_validation_fails(self):
        final,rs=self.gold_trace();self.fx.execute('update_state',{k:self.gold[k] for k in ('record_id','total_minor','source_version','status')},4,0,[])
        self.assertFalse(self.score(final,rs)['VERIFIED_WORKFLOW_SUCCESS'])
    def test_public_validator_does_not_reveal_gold(self):
        self.fx.execute('update_state',{'record_id':self.gold['record_id'],'total_minor':self.gold['total_minor']+1,'source_version':99,'status':'active'},0,0,[])
        result=self.fx.execute('validate_state',{'record_id':self.gold['record_id']},1,0,[])['result']
        self.assertTrue(result['valid']);self.assertNotIn('expected',json.dumps(result));self.assertNotIn('total_minor',result)
    def test_wrong_value_really_stored(self):
        self.fx.execute('update_state',{'record_id':self.gold['record_id'],'total_minor':17,'source_version':1,'status':'active'},0,0,[])
        self.assertEqual(self.fx.state[self.gold['record_id']]['total_minor'],17)
    def test_bool_is_not_integer(self):
        e=self.fx.execute('calculate_amount',{'quantity':True,'unit_price_minor':200,'discount_percent':0},0,0,[])
        self.assertEqual(e['error'],'BAD_ARGUMENT_TYPE')
    def test_numeric_string_rejected(self):
        e=self.fx.execute('calculate_amount',{'quantity':'2','unit_price_minor':200,'discount_percent':0},0,0,[])
        self.assertEqual(e['error'],'BAD_ARGUMENT_TYPE')
    def test_unknown_tool_no_state_effect(self):
        before=digest(self.fx.state);self.fx.execute('exec',{'cmd':'touch anything'},0,0,[]);self.assertEqual(before,digest(self.fx.state))
    def test_arbitrary_path_not_dispatchable(self):
        e=self.fx.execute('read_source',{'source_id':'../../etc/passwd'},0,0,[]);self.assertEqual(e['error'],'UNKNOWN_SOURCE')
    def test_symlink_root_rejected(self):
        p=pathlib.Path(self.tmp.name)/'link';p.symlink_to(self.fx.root,target_is_directory=True)
        with self.assertRaises(ValueError):Fixture(self.task,p)
    def test_control_record_change_sticky_failure(self):
        final,rs=self.gold_trace();target=next(k for k in self.fx.state if k!=self.gold['record_id'])
        self.fx.execute('update_state',{'record_id':target,'total_minor':12,'source_version':1,'status':'active'},3,0,[])
        self.assertFalse(self.score(final,rs)['integrity_valid_throughout'])
    def test_final_bool_integer_strict(self):
        final,rs=self.gold_trace();final['validated']=1;self.assertFalse(self.score(final,rs)['FINAL_ANSWER_CORRECT'])
    def test_menu_keeps_required_order_and_schema(self):self.assertEqual(tools(8),tools(32)[:8])
    def test_oracle_does_not_depend_on_filler(self):
        t=copy.deepcopy(self.task);t['filler']='irrelevant'*1000;self.assertEqual(oracle(t),self.gold)
    def test_essential_quantity_changes_oracle(self):
        t=copy.deepcopy(self.task);t['visible']['quantity']+=1;self.assertNotEqual(oracle(t)['total_minor'],self.gold['total_minor'])
    def test_rule_version_dependency(self):
        t=make_tasks('DEPENDENCY',1,6101909)[1];a=oracle(t);t['visible']['sources'][0]['version']=100
        if t['visible']['sources'][0]['source_id']==a['source_id']:t['visible']['sources'][1]['version']=101
        self.assertNotEqual(oracle(t)['source_id'],a['source_id'])
    def test_initial_prompt_has_no_oracle_object(self):
        task=copy.deepcopy(self.task);task['oracle']={'PRIVATE_GOLD_SENTINEL':'must not enter model request'}
        msg=short_messages(task);self.assertNotIn('PRIVATE_GOLD_SENTINEL',json.dumps(msg));self.assertNotIn('initial_state',json.dumps(msg))
    def test_split_ids_and_hashes_disjoint(self):
        a=make_tasks('DEV',3,6101901);b=make_tasks('EVAL',12,6101911)
        self.assertFalse({t['task_id'] for t in a}&{t['task_id'] for t in b});self.assertEqual(len(a),12);self.assertEqual(len(b),48)
    def test_stream_interleaved_calls(self):
        a=StreamAssembly()
        a.add({'choices':[{'delta':{'role':'assistant','tool_calls':[{'index':1,'id':'b','function':{'name':'validate_state','arguments':'{"record_id":'}},{'index':0,'id':'a','function':{'name':'update_state','arguments':'{"record_id":"r",'}}]}}]},0.1)
        a.add({'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'"total_minor":1}'}},{'index':1,'function':{'arguments':'"r"}'}}]},'token_ids':[11,12],'finish_reason':'tool_calls'}],'usage':{'completion_tokens':2}},0.2)
        r=a.result();self.assertEqual([c['id'] for c in r['tool_calls']],['a','b']);self.assertEqual(r['output_token_ids'],[11,12]);self.assertEqual(r['first_role_chunk_seconds'],0.1)
    def test_cap_uses_actual_context_remaining(self):
        value,bounds,factors=request_cap(1024,20,9000,8990);self.assertEqual(value,10);self.assertIn('remaining_context_tokens',factors)
    def test_zero_correct_cost_undefined(self):
        from kv_agent_bench.metrics import workload_cost
        cost=workload_cost(20,120,0,1000);self.assertIsNone(cost['seconds_per_correct_job']);self.assertIsNone(cost['output_tokens_per_correct_job']);self.assertEqual(cost['correct_jobs_per_minute'],0)
    def test_cost_uses_workload_makespan_with_failures(self):
        from kv_agent_bench.metrics import workload_cost
        cost=workload_cost(10,130,2,1200);self.assertEqual(cost['correct_jobs_per_minute'],1);self.assertEqual(cost['output_tokens_per_correct_job'],600)
    def test_mixed_response_preserves_unaccepted_slot_order(self):
        from kv_agent_bench.observations import ordered_slots
        raw='<tool_call>{bad}</tool_call><tool_call>{"name":"validate_state","arguments":{"record_id":"x"}}</tool_call>'
        calls=[{'id':'a','function':{'name':'validate_state','arguments':'{"record_id":"x"}'}}]
        s=ordered_slots(raw,calls);self.assertEqual(len(s),2);self.assertIsNone(s[0]['parsed']);self.assertEqual(s[1]['parsed']['name'],'validate_state')
    def test_server_rejection_is_not_repaired(self):
        from kv_agent_bench.observations import ordered_slots
        s=ordered_slots('<tool_call>{"name":"read_source","arguments":{"source_id":"x"}}</tool_call>',[])
        self.assertIsNone(s[0]['parsed'])
    def test_engine_history_normalization_is_typed_only(self):
        from kv_agent_bench.tokenization import render
        class Capture:
            def apply_chat_template(self,messages,**kw):return messages
        m=[{'role':'assistant','content':'free text remains {bad}','tool_calls':[{'function':{'name':'read_source','arguments':'{ "source_id" : "x" }'}}]}]
        n=render(Capture(),m,[]);self.assertEqual(n[0]['tool_calls'][0]['function']['arguments'],{'source_id':'x'});self.assertIsInstance(m[0]['tool_calls'][0]['function']['arguments'],str);self.assertEqual(n[0]['content'],m[0]['content'])
    def test_core_imports_no_torch_or_transformers(self):
        code='import kv_agent_bench.fixture_generator,kv_agent_bench.tools,kv_agent_bench.scorer,kv_agent_bench.audit,kv_agent_bench.metrics,kv_agent_bench.episode_runner;import sys;assert "torch" not in sys.modules and "transformers" not in sys.modules'
        subprocess.run([sys.executable,'-B','-c',code],check=True)

if __name__=='__main__':unittest.main()
