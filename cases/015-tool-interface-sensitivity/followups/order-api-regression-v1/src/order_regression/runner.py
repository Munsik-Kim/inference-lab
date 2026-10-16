"""Immutable per-generation result visibility, distinct from execution state."""
import copy
import time
from .common import digest, history, parse_slots, extract, same
from .registry import Registry
from .fixture import Fixture
from .scoring import assess
from interface_sensitivity.runner import ObservationError


def run_episode(scenario,api,generate,root,generation,identity,episode_id):
    started=time.monotonic();registry=Registry(api);fixture=Fixture(scenario,registry,root)
    messages=copy.deepcopy(scenario['messages']);responses=[];available=[];attempted=0;errors=[]
    final=None;terminal='BUDGET_EXHAUSTED';observed='OBSERVED'
    for turn in range(generation['max_model_calls']):
        if time.monotonic()-started>=generation['deadline_seconds']:
            terminal='TIMEOUT';break
        frozen_available=tuple(available)
        before_generation=fixture.snapshot()
        try:
            response=generate(messages,registry.tools(),started+generation['deadline_seconds'])
        except ObservationError as e:
            terminal=e.label;errors.append({'type':e.label,'turn':turn,'message':str(e)})
            if e.label in ('OOM','INFRA_ERROR','WORKER_BUDGET'):observed='UNOBSERVED_INFRA'
            break
        response.update({'generated_in_turn':turn,'available_result_event_ids':list(frozen_available),
            'pre_generation_state_digest':digest(before_generation),'pre_generation_history_hash':digest(messages),
            'pre_generation_state_version':fixture.version,
            'canonical_observation_hash':digest([{'semantic_tool_id':e['semantic_tool_id'],'arguments':e['raw_arguments'],'result':e['tool_result']}
                                                for e in fixture.events if e['event_id'] in frozen_available])})
        responses.append(response);slots=parse_slots(response['raw_text']);accepted=[];results=[];new_available=[]
        if slots:
            for slot in slots:
                if attempted>=generation['max_tool_calls']:
                    terminal='BUDGET_EXHAUSTED';errors.append({'type':terminal,'turn':turn,'reason':'attempted_tool_call_limit'});break
                attempted+=1;obj=slot['parsed']
                if obj is None:
                    errors.append({'type':'PARSE_ERROR','turn':turn,'call_index':slot['call_index']});continue
                event=fixture.execute(obj.get('name'),obj.get('arguments'),turn,slot['call_index'],frozen_available)
                accepted.append(obj);results.append(event['tool_result']);new_available.append(event['event_id'])
            # Only after all calls from this generated response have been executed
            # and inserted into the next request can any result become available.
            if accepted:
                messages=history(messages,accepted,results);available.extend(new_available)
                continue
            if not accepted and terminal!='BUDGET_EXHAUSTED':terminal='PARSE_ERROR'
            break
        final=extract(response['text']);terminal='FINAL_ANSWER' if final is not None else 'PARSE_ERROR';break
    state=fixture.snapshot();verdict=assess(scenario,fixture.events,responses,final,state,fixture.version,fixture.integrity_violation)
    if terminal!='FINAL_ANSWER':verdict['VERIFIED_WORKFLOW_SUCCESS']=False
    if observed!='OBSERVED':verdict['VERIFIED_WORKFLOW_SUCCESS']=None
    return {'episode_id':episode_id,'scenario_id':scenario['scenario_id'],'template_id':scenario['template_id'],
        'method':identity.get('method'),'api_version':api,'canonical_task_hash':scenario['canonical_task_hash'],
        'surface_prompt_hash':digest({'messages':scenario['messages'],'tools':registry.tools()}),
        'input_token_hash':responses[0]['input_token_sha256'] if responses else None,
        'evidence_kind':identity.get('evidence_kind','measurement'),'phase':'A1_POST_P1_APPLICATION',**identity,**verdict,
        'observation_status':observed,'terminal_reason':terminal,'errors':errors,
        'responses':responses,'tool_events':fixture.events,'final_answer':final,'final_fixture':state,'final_state_version':fixture.version,
        'model_calls':len(responses),'attempted_tool_calls':attempted,'executed_tool_calls':len(fixture.events),
        'generated_tokens':sum(r['output_tokens'] for r in responses),'input_tokens_total':sum(r['input_tokens'] for r in responses),
        'wall_seconds':time.monotonic()-started,
        'peak_allocated_bytes':max((r.get('peak_allocated_bytes',0) for r in responses),default=0),
        'peak_reserved_bytes':max((r.get('peak_reserved_bytes',0) for r in responses),default=0),
        'enum_enabled':False,'enhanced_feedback':False,'aliases_enabled':False}
