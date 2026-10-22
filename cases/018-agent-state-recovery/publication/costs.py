"""Independent recorded SSE token counts, technical missingness and worker walls."""
import json
from pathlib import Path

def tech(path):
    raw=json.loads(path.read_text());ids=[];prompt=None;usage=None;done=False
    for frame in raw['raw_sse_lines']:
        line=frame['line'] if isinstance(frame,dict) else frame
        if not line.startswith('data:'):continue
        value=line[5:].strip()
        if value=='[DONE]':done=True;continue
        if not value:continue
        d=json.loads(value)
        if d.get('prompt_token_ids') is not None:prompt=d['prompt_token_ids']
        if d.get('usage') is not None:usage=d['usage']
        for c in d.get('choices',[]):ids.extend(c.get('token_ids') or [])
    if done:
        if usage is None or prompt is None or usage['completion_tokens']!=len(ids) or usage['prompt_tokens']!=len(prompt):raise ValueError('Technical token cost mismatch')
    return {'source':path.name,'complete':done,'received_IDs':len(ids),
            'engine_output_tokens':len(ids) if done else None,'input_tokens':len(prompt) if done else None,
            'reason':None if done else 'Incomplete cancelled stream; engine-generated tokens NOT_RECORDED'}

def recompute(view,project):
    results=view/'case/results';phases=[]
    for phase in ['P1','P2']:
        groups=next(p['groups'] for p in project['phases'] if p['phase']==phase)
        runs=['p1_r'+str(i) for i in range(1,6)] if phase=='P1' else ['p2_r1']
        traces=[];walls=[]
        for run in runs:
            for f in sorted((results/run).glob('*_transport/*.json')):
                if f.parent.name.startswith(('TECH_','RELOAD_')):traces.append(tech(f))
            walls.append(json.loads((results/run/'worker_cost.json').read_text())['worker_wall_seconds'])
        model_requests=sum(g['requests'] for g in groups)
        phases.append({'phase':phase,'model_requests':model_requests,'model_input':sum(g['input_tokens'] for g in groups),
                       'model_output':sum(g['output_tokens'] for g in groups),'technical_attempts':len(traces),
                       'technical_complete':sum(t['complete'] for t in traces),'technical_input':sum(t['input_tokens'] or 0 for t in traces),
                       'technical_observed_output':sum(t['received_IDs'] for t in traces),
                       'unknown_engine_output_attempts':sum(t['engine_output_tokens'] is None for t in traces),
                       'technical_rows':traces,'worker_wall_seconds':sum(walls),'worker_attempt_walls':walls})
    attempts=sum(p['model_requests']+p['technical_attempts'] for p in phases)
    complete=sum(p['model_requests']+p['technical_complete'] for p in phases)
    lower=sum(p['model_output']+p['technical_observed_output'] for p in phases)
    if (attempts,complete,lower)!=(917,915,57987):raise ValueError('Cumulative raw costs differ')
    return {'phases':phases,'attempts':attempts,'complete':complete,'observed_output_lower_bound':lower,
            'exact_engine_output_total':None,'unknown_engine_output_attempts':attempts-complete,
            'worker_wall_seconds':sum(p['worker_wall_seconds'] for p in phases),
            'scope':'Worker includes load/compile/failures/cleanup and overlaps episode wall. Not GPU busy time or serving throughput.'}
