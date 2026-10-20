"""Small raw-recording/cancellation probes; excluded from task success tables."""
import asyncio
import json
import pathlib
import re
import time
from .common import load,save,utc
from .server_manager import Server
from .workload_runner import inspect,support_check,metrics,freeze
from .vllm_client import chat

def active_counts(text):
    rows={}
    for key in ('num_requests_running','num_requests_waiting'):
        values=[]
        for line in text.splitlines():
            if line.startswith('vllm:'+key+'{') or line.startswith('vllm:'+key+' '):values.append(float(line.rsplit(' ',1)[-1]))
        rows[key]=sum(values) if values else None
    return rows

async def cancelled_probe(base,payload):
    import httpx
    before=metrics(base);started=time.monotonic();lines=[];cancelled=False
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        async with client.stream('POST',base+'/v1/chat/completions',json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                lines.append(line)
                if line.startswith('data: ') and line[6:]!='[DONE]':
                    obj=json.loads(line[6:])
                    if any(c.get('token_ids') for c in obj.get('choices',[])):
                        cancelled=True;break
    samples=[];after=''
    for _ in range(40):
        after=metrics(base);active=active_counts(after);samples.append({'elapsed_seconds':time.monotonic()-started,**active})
        if active=={'num_requests_running':0.0,'num_requests_waiting':0.0}:break
        await asyncio.sleep(.25)
    return {'client_closed_after_generated_token':cancelled,'raw_sse_lines_before_close':lines,'samples':samples,
      'metrics_before':before,'metrics_after':after,'drained':bool(cancelled and samples[-1]['num_requests_running']==0 and samples[-1]['num_requests_waiting']==0),
      'abort_confirmation_scope':'stream closed after a token; dedicated-server running/waiting counters returned to zero; correlate owned server abort log',
      'wall_seconds':time.monotonic()-started}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--case',type=pathlib.Path,required=True);p.add_argument('--runtime',required=True);args=p.parse_args()
    case=args.case.resolve();runtime=load(args.runtime);private=pathlib.Path(runtime['private_root']);out=case/'results'/runtime.get('diagnostics_run_id','p0_diagnostics_r1')
    if out.exists():raise ValueError('fresh diagnostics result root required')
    out.mkdir(parents=True);freeze(case,out/'code_freeze.json',runtime);started=time.monotonic();registered=utc();rows=[]
    reuse=case/'results'/runtime['diagnostics_reuse_root'] if runtime.get('diagnostics_reuse_root') else None
    save(out/'registration.json',{'registered_utc':registered,'per_supported_kv':['reuse completed raw ON/OFF/incomplete probes when present','close-stream abort/drain'],'extra_f4_only_probe':'one32-token generation under Torch profiler','counts_as_workflow_performance':False,'reused_from':str(reuse.relative_to(case)) if reuse else None})
    budget=load(private/'p0_budget.json')
    for kv in ('B16','F8','F4'):
        if time.monotonic()-budget['start_monotonic']>=budget['limit_seconds']:rows.append({'kv':kv,'status':'NOT_RUN_BUDGET'});continue
        if kv not in load(case/'results'/runtime['p0_run_id']/'decision.json')['supported']:rows.append({'kv':kv,'status':'NOT_RUN_SUPPORT'});continue
        root=private/'runs'/f'{out.name}_{kv}_00';s=Server(runtime['python'],runtime['snapshot'],{'B16':'auto','F8':'fp8_e4m3','F4':'nvfp4'}[kv],runtime['p1_max_model_len'],runtime['pool_bytes'],root,case/'src',extension='kv_agent_bench.profiler_extension.ProfileInspector' if kv=='F4' else 'kv_agent_bench.worker_inspection.CacheInspector')
        try:
            s.start(ready_seconds=900);before=inspect(s.base);support_check(before,kv);save(out/f'{kv}_final_config_cache_before.json',before)
            prepared=load(pathlib.Path(runtime['prepared_root'])/'SMOKE-A00_m8_l0.json')
            base={'model':'case016-qwen3-4b','messages':prepared['messages'],'tools':prepared['tools'],'tool_choice':'auto','temperature':0,'max_tokens':256,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False},'ignore_eos':False}
            # One pair per KV; order is frozen to ON then OFF, not a latency benchmark.
            pair=[]
            for recording in (True,False):
                payload={**base,'return_token_ids':recording}
                if reuse:r=load(reuse/f'{kv}_raw_{"on" if recording else "off"}.json')
                else:r=asyncio.run(chat(s.base,payload));save(out/f'{kv}_raw_{"on" if recording else "off"}.json',r)
                pair.append({'raw_recording':recording,'client_http_seconds':r['http_end_seconds'],'usage':r['usage'],'observed_ids':len(r['output_token_ids'])})
            if reuse:incomplete=load(reuse/f'{kv}_incomplete_raw.json')
            else:incomplete=asyncio.run(chat(s.base,{**base,'max_tokens':2,'return_token_ids':True}));save(out/f'{kv}_incomplete_raw.json',incomplete)
            incomplete_ok=len(incomplete['output_token_ids'])==incomplete['usage']['completion_tokens']==2 and incomplete['server_finish_reason']=='length'
            cancel_payload={**base,'messages':[{'role':'user','content':'Write every integer from 1 to 100000, one per line, with no commentary. Do not stop early.'}],'tool_choice':'none','max_tokens':1024,'return_token_ids':True}
            cancel_payload.pop('tools')
            cancellation=asyncio.run(cancelled_probe(s.base,cancel_payload))
            # Abort is logged by the official AsyncLLM cancellation path.
            time.sleep(.5)
            cancellation['server_abort_log_confirmed']='Aborted request(s)' in (root/'server.log').read_text(errors='replace')
            save(out/f'{kv}_cancellation.json',cancellation)
            if kv=='F4':
                def rpc(method):
                    import urllib.request
                    request=urllib.request.Request(s.base+'/collective_rpc',data=json.dumps({'method':method,'timeout':60}).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=70) as r:return json.load(r)['results'][0]
                rpc('case016_profile_start')
                profile_response=asyncio.run(chat(s.base,{**base,'max_tokens':32,'return_token_ids':True}));save(out/'F4_profile_request.json',profile_response)
                save(out/'F4_kernel_profile.json',rpc('case016_profile_stop'))
            after=inspect(s.base);save(out/f'{kv}_final_config_cache_after.json',after)
            prompt_match=load((reuse or out)/f'{kv}_raw_on.json')['prompt_token_ids']==prepared['input_token_ids']
            status='PASS' if cancellation['drained'] and cancellation['server_abort_log_confirmed'] and prompt_match and incomplete_ok else 'BLOCKED_DRAIN_OR_INPUT'
            rows.append({'kv':kv,'status':status,'raw_cost_pair':pair,'raw_policy_p1':True,'prompt_identity':prompt_match,'incomplete_raw_observable':incomplete_ok,'final_model_len':runtime['p1_max_model_len'],'pool_requested_bytes':runtime['pool_bytes'],'final_config_support':True,'cancellation_drained':cancellation['drained'],'server_abort_log_confirmed':cancellation['server_abort_log_confirmed'],'additional_generation_requests':(0 if reuse else 3)+1+int(kv=='F4'),'reused_raw_probe_root':str(reuse.relative_to(case)) if reuse else None,'evidence_kind':'p0_instrumentation_probe'})
        except Exception as e:
            save(out/f'{kv}_failure.json',{'type':type(e).__name__,'message':str(e)});rows.append({'kv':kv,'status':'BLOCKED','reason':str(e)})
        finally:s.stop()
    save(out/'summary.json',{'rows':rows,'status':'PASS' if all(x['status']=='PASS' for x in rows if x['kv']in runtime['p1_kvs']) else 'BLOCKED','wall_seconds':time.monotonic()-started,'cumulative_p0_seconds':time.monotonic()-budget['start_monotonic'],'protocol':'P0 raw ON/OFF1pair, incomplete1 and disconnect/drain1 per supported KV. Never counted as workflow episodes.','frozen_before_requests_utc':registered})

if __name__=='__main__':main()
