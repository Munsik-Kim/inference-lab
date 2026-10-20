"""One isolated episode; all same-response calls share immutable observations."""
import asyncio
import copy
import hashlib
import json
import time
from .common import digest
from .tools import Fixture
from .scorer import assess
from .tokenization import render,token_hash
from .vllm_client import chat,parse_response,request_cap

async def run_episode(task,prepared,tok,base,kv,cap,root,config,episode_id,request=chat):
    start=time.monotonic();fixture=Fixture(task,root,prepared['menu']);messages=copy.deepcopy(prepared['messages']);tool_list=prepared['tools']
    responses=[];available=[];attempts=0;used=0;terminal='UNKNOWN';final=None;availability='OBSERVED';errors=[]
    for turn in range(config['max_model_calls']):
        if time.monotonic()-start>=config['episode_seconds']:terminal='EPISODE_DEADLINE';break
        prompt=render(tok,messages,tool_list);ids=tok.encode(prompt,add_special_tokens=False)
        if turn==0 and token_hash(ids)!=prepared['input_token_sha256']:raise ValueError('initial input identity mismatch')
        limit,bounds,factors=request_cap(cap,config['episode_token_budget']-used,config['max_model_len'],len(ids))
        if limit<=0:
            terminal='CONTEXT_LIMIT' if bounds['remaining_context_tokens']<=0 else 'TOKEN_LIMIT';break
        frozen=tuple(available);state_before=fixture.snapshot();payload={'model':'case016-qwen3-4b','messages':copy.deepcopy(messages),'tools':tool_list,
          'tool_choice':'auto','temperature':0,'max_tokens':limit,'stream':True,'stream_options':{'include_usage':True},
          'return_token_ids':True,'chat_template_kwargs':{'enable_thinking':False},'ignore_eos':False}
        remaining=config['episode_seconds']-(time.monotonic()-start)
        try:r=await asyncio.wait_for(request(base,payload,min(config['request_seconds'],remaining)),timeout=min(config['request_seconds'],remaining))
        except (asyncio.TimeoutError,TimeoutError) as e:terminal='EPISODE_DEADLINE' if time.monotonic()-start>=config['episode_seconds'] else 'REQUEST_TIMEOUT';errors.append({'type':terminal,'message':str(e)});break
        except Exception as e:terminal='SERVER_ERROR';availability='UNOBSERVED_INFRA';errors.append({'type':type(e).__name__,'message':str(e)});break
        raw,slots,answer=parse_response(r,tok);out=r['output_token_ids'];used+=len(out)
        if r.get('prompt_token_ids')!=ids:
            terminal='SERVER_ERROR';availability='INPUT_IDENTITY_MISMATCH';errors.append({'type':'INPUT_IDENTITY_MISMATCH'})
        if raw is None or not r.get('usage') or r['usage']['completion_tokens']!=len(out):
            terminal='SERVER_ERROR';availability='RAW_UNOBSERVABLE';errors.append({'type':'RAW_UNOBSERVABLE','usage':r.get('usage'),'observed_ids':len(out)})
        r.update({'turn':turn,'available_result_event_ids':list(frozen),'messages':copy.deepcopy(messages),'tools_hash':digest(tool_list),
          'input_token_sha256':token_hash(ids),'local_prompt_tokens':len(ids),'input_token_ids':ids,
          'actual_prompt_tokens':len(r.get('prompt_token_ids')or[]),'server_input_token_sha256':token_hash(r['prompt_token_ids']) if r.get('prompt_token_ids')is not None else None,
          'state_digest':digest(state_before),'state_version':fixture.version,'canonical_observation_hash':digest([e for e in fixture.events if e['event_id']in frozen]),
          'request_cap':limit,'cap_bounds':bounds,'limiting_factors':factors,'raw_text':raw,
          'raw_utf8_sha256':hashlib.sha256(raw.encode()).hexdigest() if raw is not None else None,
          'parsed_slots':slots,'output_tokens':len(out),'eos_observed':bool(out and out[-1]in config['eos_token_ids']),
          'output_cap_reached':len(out)>=limit})
        responses.append(r)
        if availability!='OBSERVED':break
        if slots:
            accepted=[];results=[]
            for i,s in enumerate(slots):
                if attempts>=config['max_tool_attempts']:terminal='TOOL_CALL_LIMIT';break
                attempts+=1
                if s['parsed']is None:errors.append({'type':'PARSE_ERROR','turn':turn,'slot':i});continue
                obj=s['parsed'];e=fixture.execute(obj['name'],obj['arguments'],turn,i,frozen)
                if len(tok.encode(json.dumps(e['result']),add_special_tokens=False))>256:raise ValueError('tool return contract exceeds 256 tokens')
                call=s['server_call'] or {'id':f'local_{turn}_{i}','type':'function','function':{'name':obj['name'],'arguments':json.dumps(obj['arguments'])}}
                accepted.append(call);results.append(e)
            if accepted:
                messages.append({'role':'assistant','content':r['content'] or None,'tool_calls':accepted})
                for call,e in zip(accepted,results):
                    messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(e['result'],ensure_ascii=False)});available.append(e['event_id'])
                if terminal=='TOOL_CALL_LIMIT':break
                continue
            if terminal!='TOOL_CALL_LIMIT':terminal='PARSE_STOP'
            break
        final=answer
        if answer is not None:terminal='FINAL'
        elif r['server_finish_reason']=='length':terminal='TOKEN_LIMIT'
        else:terminal='PARSE_STOP'
        break
    else:terminal='MODEL_CALL_LIMIT'
    if terminal=='UNKNOWN':raise AssertionError('unassigned terminal branch')
    verdict=assess(task,fixture.events,responses,final,fixture.snapshot(),fixture.version,terminal)
    if availability!='OBSERVED':verdict['VERIFIED_WORKFLOW_SUCCESS']=None
    return {'episode_id':episode_id,'task_id':task['task_id'],'family':task['family'],'split':task['split'],'kv':kv,'menu':prepared['menu'],'cap':cap,
      'canonical_task_hash':task['canonical_task_hash'],'initial_input_token_sha256':prepared['input_token_sha256'],'initial_surface_hash':prepared['surface_hash'],
      'evidence_kind':'model_measurement','slot_order_policy':'raw_hermes_slots_v2','availability':availability,'terminal_reason':terminal,'errors':errors,
      'responses':responses,'events':fixture.events,'final':final,'final_state':fixture.snapshot(),'final_state_version':fixture.version,
      'model_calls':len(responses),'tool_attempts':attempts,'tool_events':len(fixture.events),'output_tokens':used,
      'input_tokens_total':sum(r['local_prompt_tokens'] for r in responses),'wall_seconds':time.monotonic()-start,**verdict}
