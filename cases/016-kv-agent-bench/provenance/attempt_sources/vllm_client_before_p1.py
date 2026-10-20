"""Raw HTTP/SSE client. Chunk clocks are client observations, not token ITL."""
import asyncio
import copy
import json
import time
from .common import digest

class StreamAssembly:
    def __init__(self):
        self.calls={};self.content='';self.events=[];self.token_ids=[];self.prompt_token_ids=None
        self.finish_reason=None;self.stop_reason=None;self.usage=None;self.first_role=None;self.first_token=None;self.first_complete_call=None
    def add(self,obj,elapsed):
        self.events.append({'elapsed_seconds':elapsed,'data':obj})
        if obj.get('usage'):self.usage=obj['usage']
        if obj.get('prompt_token_ids')is not None:self.prompt_token_ids=obj['prompt_token_ids']
        for c in obj.get('choices',[]):
            delta=c.get('delta')or{}
            if delta.get('role') and self.first_role is None:self.first_role=elapsed
            ids=c.get('token_ids')
            if ids is not None:self.token_ids.extend(ids)
            if (ids or delta.get('content') or delta.get('tool_calls')) and self.first_token is None:self.first_token=elapsed
            if delta.get('content'):self.content+=delta['content']
            for t in delta.get('tool_calls',[]):
                i=t['index'];a=self.calls.setdefault(i,{'index':i,'id':None,'type':'function','function':{'name':'','arguments':''}})
                if t.get('id'):a['id']=t['id']
                fn=t.get('function')or{}
                if fn.get('name'):a['function']['name']+=fn['name']
                if fn.get('arguments'):a['function']['arguments']+=fn['arguments']
                try:
                    parsed=json.loads(a['function']['arguments'])
                    if type(parsed)is dict and a['function']['name'] and self.first_complete_call is None:self.first_complete_call=elapsed
                except (ValueError,TypeError):pass
            if c.get('finish_reason')is not None:self.finish_reason=c['finish_reason']
            if c.get('stop_reason')is not None:self.stop_reason=c['stop_reason']
    def result(self):
        return {'content':self.content,'tool_calls':[self.calls[k] for k in sorted(self.calls)],
          'output_token_ids':self.token_ids,'prompt_token_ids':self.prompt_token_ids,'usage':self.usage,
          'server_finish_reason':self.finish_reason,'server_stop_reason':self.stop_reason,
          'sse_events':self.events,'first_role_chunk_seconds':self.first_role,
          'first_generated_bearing_event_seconds':self.first_token,'first_complete_call_seconds':self.first_complete_call}

async def chat(base,payload,timeout_seconds=180):
    import httpx
    a=StreamAssembly();started=time.monotonic();raw=[]
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds),trust_env=False) as client:
        async with client.stream('POST',base+'/v1/chat/completions',json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                raw.append(line)
                if line.startswith('data: ') and line[6:]!='[DONE]':a.add(json.loads(line[6:]),time.monotonic()-started)
    x=a.result();x.update({'raw_sse_lines':raw,'http_end_seconds':time.monotonic()-started,'request_hash':digest(payload)})
    return x

def parse_response(r,tokenizer):
    raw=tokenizer.decode(r['output_token_ids'],skip_special_tokens=False) if r['output_token_ids'] else None
    slots=[]
    for c in r['tool_calls']:
        try:
            arguments=json.loads(c['function']['arguments'])
            parsed={'name':c['function']['name'],'arguments':arguments} if type(arguments)is dict else None
        except (ValueError,TypeError):parsed=None
        slots.append({'call_id':c.get('id'),'parsed':parsed,'server_call':copy.deepcopy(c)})
    # Unaccepted Hermes marker slots consume attempt budget; they never execute.
    marker_count=raw.count('<tool_call>') if raw is not None else 0
    for _ in range(max(0,marker_count-len(slots))):slots.append({'call_id':None,'parsed':None,'server_call':None})
    final=None
    if not slots:
        try:
            final=json.loads(r['content'].strip())
            if type(final)is not dict:final=None
        except (ValueError,TypeError):pass
    return raw,slots,final

def request_cap(config_cap,remaining_tokens,max_model_len,prompt_tokens):
    values={'configured_cap':config_cap,'remaining_episode_tokens':remaining_tokens,'remaining_context_tokens':max_model_len-prompt_tokens}
    value=min(values.values())
    return value,values,[k for k,v in values.items() if v==value]
