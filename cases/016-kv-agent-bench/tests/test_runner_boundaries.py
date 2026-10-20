"""Scripted traces below are synthetic contract tests, never LLM measurements."""
import asyncio
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from kv_agent_bench.fixture_generator import make_tasks,short_messages,tools
from kv_agent_bench.common import digest
from kv_agent_bench.tokenization import render,token_hash
from kv_agent_bench.episode_runner import run_episode

class ByteTokenizer:
    def apply_chat_template(self,messages,**kw):return json.dumps({'messages':messages,'tools':kw.get('tools')},sort_keys=True)
    def encode(self,text,**kw):return list(text.encode())
    def decode(self,ids,**kw):return bytes(ids).decode()

class RunnerBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.tok=ByteTokenizer();self.task=make_tasks('UNIT',1,6101908)[0]
        m=short_messages(self.task);ts=tools(8);ids=self.tok.encode(render(self.tok,m,ts));self.prepared={'messages':m,'tools':ts,'menu':8,'input_token_sha256':token_hash(ids),'surface_hash':digest(m)}
        self.config={'max_model_calls':8,'max_tool_attempts':6,'episode_token_budget':8192,'request_seconds':180,'episode_seconds':900,'max_model_len':24576,'eos_token_ids':[]}
    def run_script(self,script,cap=256):
        counter=iter(script)
        async def request(base,payload,timeout):
            raw,content,calls=next(counter);ids=self.tok.encode(render(self.tok,payload['messages'],payload['tools']));out=self.tok.encode(raw)
            return {'content':content,'tool_calls':calls,'output_token_ids':out,'prompt_token_ids':ids,'usage':{'completion_tokens':len(out),'prompt_tokens':len(ids)},'server_finish_reason':'stop','server_stop_reason':None,'raw_sse_lines':[]}
        e=asyncio.run(run_episode(self.task,self.prepared,self.tok,'http://unused','B16',cap,pathlib.Path(self.temp.name)/'fixture',self.config,'SYNTHETIC',request=request));e['evidence_kind']='synthetic_contract_test';return e
    def test_malformed_with_remaining_budget_is_parse_stop(self):
        e=self.run_script([('<tool_call>{oops}</tool_call>','',[])])
        self.assertEqual(e['terminal_reason'],'PARSE_STOP');self.assertEqual(e['model_calls'],1);self.assertEqual(e['tool_attempts'],1);self.assertEqual(e['tool_events'],0);self.assertEqual(e['final_state_version'],0)
    def test_mixed_response_executes_only_valid_call(self):
        call={'id':'a','type':'function','function':{'name':'service_status','arguments':'{}'}}
        raw='<tool_call>{bad}</tool_call><tool_call>{"name":"service_status","arguments":{}}</tool_call>'
        e=self.run_script([(raw,'',[call]),('{}','{}',[])])
        self.assertEqual(e['tool_attempts'],2);self.assertEqual(e['tool_events'],1);self.assertEqual(e['events'][0]['call_index'],1)
        self.assertEqual(e['responses'][0]['available_result_event_ids'],[]);self.assertEqual(e['responses'][1]['available_result_event_ids'],['e0']);self.assertFalse(e['VERIFIED_WORKFLOW_SUCCESS'])
    def test_model_call_limit_is_actual_branch(self):
        self.config['max_model_calls']=1
        call={'id':'a','type':'function','function':{'name':'service_status','arguments':'{}'}}
        e=self.run_script([('<tool_call>{"name":"service_status","arguments":{}}</tool_call>','',[call])]);self.assertEqual(e['terminal_reason'],'MODEL_CALL_LIMIT')
    def test_server_over_cap_is_unobserved(self):
        e=self.run_script([('{}','{}',[])],cap=1);self.assertEqual(e['availability'],'SERVER_BUDGET_VIOLATION');self.assertIsNone(e['VERIFIED_WORKFLOW_SUCCESS'])
    def test_partial_server_call_stays_unaccepted(self):
        call={'id':'a','type':'function','function':{'name':'read_source','arguments':'{"source_id":'}}
        e=self.run_script([('<tool_call>{"name":"read_source","arguments":{"source_id":','',[call])]);self.assertEqual(e['tool_events'],0);self.assertEqual(e['tool_attempts'],1)
    def test_cpu_help_does_not_import_ml(self):
        code='import builtins,sys;original=builtins.__import__\ndef guarded(name,*a,**k):\n if name.split(".")[0] in ("torch","transformers"):raise AssertionError(name)\n return original(name,*a,**k)\nbuiltins.__import__=guarded\nfrom kv_agent_bench.cli import main\nsys.argv=["cli","--help"];main()'
        subprocess.run([sys.executable,'-B','-c',code],check=True,stdout=subprocess.DEVNULL)

if __name__=='__main__':unittest.main()
