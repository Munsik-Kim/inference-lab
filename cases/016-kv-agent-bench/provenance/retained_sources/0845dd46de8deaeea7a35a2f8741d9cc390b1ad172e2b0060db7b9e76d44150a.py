"""Keep raw slot order without accepting anything the engine parser rejected."""
import json
import re

def ordered_slots(raw,calls):
    server=[]
    for call in calls:
        try:
            args=json.loads(call['function']['arguments']);parsed={'name':call['function']['name'],'arguments':args} if type(args)is dict else None
        except (ValueError,TypeError):parsed=None
        server.append({'call_id':call.get('id'),'parsed':parsed,'server_call':call})
    if not raw or '<tool_call>'not in raw:return server
    slots=[];unused=list(range(len(server)))
    for match in re.finditer(r'<tool_call>(.*?)(?:</tool_call>|(?=<tool_call>)|$)',raw,re.S):
        body=match.group(1);chosen=None
        try:obj=json.JSONDecoder().raw_decode(body.lstrip())[0]
        except ValueError:obj=None
        for i in unused:
            candidate=server[i]['parsed']
            if candidate is not None and type(obj)is dict and obj.get('name')==candidate['name'] and obj.get('arguments')==candidate['arguments']:chosen=i;break
        if chosen is None:
            # This is alignment with already-accepted engine output, not JSON repair.
            for i in unused:
                candidate=server[i]['parsed']
                if candidate is not None and re.search(r'"name"\s*:\s*'+re.escape(json.dumps(candidate['name'])),body):chosen=i;break
        if chosen is None:slots.append({'call_id':None,'parsed':None,'server_call':None})
        else:slots.append(server[chosen]);unused.remove(chosen)
    if unused:raise ValueError('engine calls could not be aligned to raw Hermes slots')
    return slots
