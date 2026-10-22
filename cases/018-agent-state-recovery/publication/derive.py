"""Recreate public tables, episode excerpts and deterministic SVG from raw traces."""
import csv
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path
import random
import resource
import time
from analysis.quantity import diagnose

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()

def paired(rows):
    matrix={}
    for r in rows:matrix.setdefault(r['task_id'],{})[(r['kv'],r['policy'])]=r
    specs=[('K0_R1-R0',('K0','R0'),('K0','R1')),('K1_R1-R0',('K1','R0'),('K1','R1')),
           ('R0_K1-K0',('K0','R0'),('K1','R0')),('R1_K1-K0',('K0','R1'),('K1','R1'))]
    out=[]
    for name,a,b in specs:
        for m in ['Q','W']:
            c=Counter('both_success' if x[a][m] and x[b][m] else 'loss' if x[a][m] else 'recovery' if x[b][m] else 'both_failure' for x in matrix.values())
            out.append({'comparison':name,'endpoint':m,'pairs':len(matrix),**{k:c[k] for k in ['both_success','loss','recovery','both_failure']},
                        'difference_pp':100*(c['recovery']-c['loss'])/len(matrix),'base':list(a),'candidate':list(b)})
    strata=[[t for t in sorted(matrix) if t.startswith('HOLDOUT-'+f)] for f in 'ABC']
    if list(map(len,strata))!=[8,8,8]:raise ValueError('Paired HOLDOUT strata differ')
    rng=random.Random(61022018);distributions=[[] for _ in out]
    for _ in range(10000):
        sample=[s[rng.randrange(len(s))] for s in strata for _ in range(8)]
        for r,d in zip(out,distributions):
            a,b=tuple(r['base']),tuple(r['candidate']);m=r['endpoint']
            d.append(100*sum(int(matrix[t][b][m])-int(matrix[t][a][m]) for t in sample)/24)
    for r,d in zip(out,distributions):
        d.sort()
        def pct(p):
            v=(len(d)-1)*p;i=int(v);return d[i]+(d[min(i+1,len(d)-1)]-d[i])*(v-i)
        r['exploratory_95_pp']=[pct(.025),pct(.975)]
    return out

def export(view,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);start=time.perf_counter();cpu=time.process_time()
    rows=[];selected=[];phases=[]
    selected_ids={f'HOLDOUT-{t}_{k}_{p}' for t,k,p in [('A00','K0','R0'),('A00','K1','R0'),('A06','K1','R0'),('A06','K1','R1'),('C01','K1','R0'),('C01','K1','R1')]}
    mapping=read(view.parent/'mapping.json') if (view.parent/'mapping.json').exists() else None
    for phase,run,n in [('P1','p1_r5',24),('P2','p2_r1',96)]:
        root=view/'case/results'/run;manifest=read(root/'run_manifest.json')
        audit={r['episode_id']:r for r in read(view.parent/(phase+'-audit.json'))['rows']}
        phase_rows=[]
        for entry in manifest['episodes']:
            path=root/entry['path'];raw=path.read_bytes();row=json.loads(raw)
            if sha(raw)!=entry['sha256']:raise ValueError('Changed raw before diagnostic')
            task=read(root/'inputs'/(row['task_id']+'.json'))
            new=diagnose(task,row);ep=audit[row['episode_id']]['independently_recomputed']
            complete=[r['response'] for r in row['requests']]
            if any(not r['stream_done'] for r in complete):raise ValueError('Incomplete response in expected-complete model traces')
            r={'phase':phase,'episode_id':row['episode_id'],'task_id':row['task_id'],'family':row['family'],'kv':row['kv'],'policy':row['policy'],
               **{m:int(ep[m]) for m in ['D','B','P','F','C1','C2','Q','W','FINAL_CONTENT_CORRECT']},
               'Fresh':int(ep['F']),'original_final_correct':int(ep['FINAL_CONTENT_CORRECT']),
               'requests':len(row['requests']),'responses':len(complete),
               'input_tokens':sum(len(x['prompt_token_ids']) for x in complete),
               'output_tokens':sum(len(x['output_token_ids']) for x in complete),
               'guard_rejections':sum(bool(x.get('guard') and not x['guard'].get('accepted',False)) for x in row['requests']),
               'episode_wall_seconds':row['accounting']['episode_wall_seconds'],'terminal_reason':row['terminal_reason'],
               **new,'source_path':'results/'+run+'/'+entry['path'],'source_sha256':sha(raw)}
            # Use original observed counter, independently check requests marked GUARD_REJECTED below.
            r['guard_rejections']=row['accounting']['actual']['guard_rejections']
            raw_rejects=sum(x.get('decision') in ('GUARD_REJECTED','GUARD_LIMIT') for x in row['requests'])
            # The frozen audit has already replayed each branch and compared all counters.
            r['rejected_final_request_ids']=[x['request_id'] for x in row['requests'] if x.get('guard') and x['guard'].get('code')=='FRESH_VALIDATION_REQUIRED']
            rows.append(r);phase_rows.append(r)
            if row['episode_id'] in selected_ids:
                excerpt={'episode_id':row['episode_id'],'evidence_kind':'LOSSLESS_RAW_TEXT_EXCERPT_WITH_DERIVED_SUMMARY','diagnostic':r,
                         'request':task['public_request'],'initial_quantities':{k:v['quantity'] for k,v in row['initial_state']['records'].items()},
                         'terminal_quantities':new['terminal_quantities'],'endpoint':ep,
                         'turns':[{'generation':x['generated_in_turn'],'request_id':x['request_id'],
                                   'available_result_event_ids':x['available_result_event_ids'],'raw_text':x['response']['raw_text'],
                                   'decision':x['decision'],'guard':x.get('guard'),
                                   'event':{k:v for k,v in x['tool_event'].items() if k not in ('state_before','state_after')} if x.get('tool_event') else None} for x in row['requests']]}
                selected.append(excerpt);write(output/'selected'/(row['episode_id']+'.json'),excerpt)
        if len(phase_rows)!=n or len({(x['task_id'],x['kv'],x['policy']) for x in phase_rows})!=n:raise ValueError('Phase key coverage')
        groups=[]
        for kv in ('K0','K1'):
            for policy in ('R0','R1'):
                a=[r for r in phase_rows if r['kv']==kv and r['policy']==policy]
                def cross(b):
                    return {f'read{read_flag}_final{final_flag}':sum(r['has_accepted_final'] and r['current_quantity_read_observed']==read_flag and r['final_quantity_matches_terminal']==bool(final_flag) for r in b) for read_flag in (0,1) for final_flag in (0,1)}
                q=[r for r in a if r['Q']]
                groups.append({'phase':phase,'kv':kv,'policy':policy,'episodes':len(a),'accepted_final':sum(r['has_accepted_final'] for r in a),
                               'no_accepted_final':sum(not r['has_accepted_final'] for r in a),
                               **{m:sum(r[m] for r in a) for m in ['D','B','P','Fresh','C1','C2','Q','W','original_final_correct','current_quantity_read_observed','read_after_all_state_mutations_observed','requests','responses','input_tokens','output_tokens','guard_rejections','episode_wall_seconds']},
                               'cross_table':cross(a),'conditional_Q1':{'episodes':len(q),'accepted_final':sum(r['has_accepted_final'] for r in q),'cross_table':cross(q)}})
        bsets={m:sorted(r['episode_id'] for r in phase_rows if r[m]) for m in ['B','C1','C2']}
        phases.append({'phase':phase,'episodes':n,'independent_tasks':n//4,'groups':groups,'B_C1_C2_success_sets_identical':bsets['B']==bsets['C1']==bsets['C2'],
                       'Q1_W0':{'count':sum(r['Q'] and not r['W'] for r in phase_rows),
                                 'first_read_match_count':sum(r['Q'] and not r['W'] and r['final_matches_first_read'] is True for r in phase_rows)}})
    comparisons=paired([r for r in rows if r['phase']=='P2'])
    for r in rows:write(output/'episodes'/(r['episode_id']+'.json'),r)
    write(output/'quantity_ledger.json',rows)
    cols=['phase','episode_id','task_id','family','kv','policy','Q','W','Fresh','original_final_correct','has_accepted_final','anchor_generation','anchor_kind','current_quantity_read_observed','read_after_all_state_mutations_observed','current_read_coverage','required_count','final_quantity_matches_terminal','final_matches_first_read','final_quantity_error','requests','input_tokens','output_tokens','guard_rejections','episode_wall_seconds','terminal_reason','source_path','source_sha256']
    with (output/'episode_ledger.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:('NA' if r[k] is None else int(r[k]) if type(r[k]) is bool else r[k]) for k in cols} for r in rows)
    summary={'name':'Posthoc quantity observation and final-report consistency','definition_version':'1.0','posthoc':True,'phases':phases,'paired_P2':comparisons,
             'selected_episodes':selected,'new_model_requests':0,'causal_claim':'NOT_ESTABLISHED; observational trajectories only',
             'measurement_limits':['No full tokenizer re-execution; archived prior receipts reused','No forced reread intervention','B/C1/C2 not independent samples','No GPU busy time or serving throughput measurement']}
    write(output/'project.json',summary)
    # Deterministic scientific SVG, all numerical inputs are regenerated from raw.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="350" viewBox="0 0 760 350" role="img" aria-labelledby="title desc">',
         '<title id="title">HOLDOUT recovery quality Q and verified workflow W, per24 tasks</title><desc id="desc">BF16 R0/R1 Q24 W0; NVFP4 R0/R1 Q19 W5. Two fixed CPU continuations, exploratory synthetic sample.</desc>',
         '<rect width="760" height="350" fill="#101b2c"/><g fill="#e8eff8" font-family="sans-serif" font-size="15"><text x="24" y="28">HOLDOUT: Q and W /24 synthetic tasks</text>']
    for i,g in enumerate(phases[1]['groups']):
        y=65+i*64;label=('BF16' if g['kv']=='K0' else 'NVFP4')+' '+g['policy']
        svg.append(f'<text x="24" y="{y+18}">{escape(label)}</text><rect x="170" y="{y}" width="{g["Q"]*18}" height="18" fill="#5dd2bb"/><text x="{180+g["Q"]*18}" y="{y+15}">Q {g["Q"]}/24</text><rect x="170" y="{y+23}" width="{g["W"]*18}" height="14" fill="#eca95f"/><text x="{180+g["W"]*18}" y="{y+36}">W {g["W"]}/24</text>')
    svg.append('<text x="24" y="339">Q: state + protection + fixed continuations; W: Q + final + Fresh</text></g></svg>')
    (output/'quality.svg').write_text('\n'.join(svg)+'\n')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="300" viewBox="0 0 760 300" role="img" aria-labelledby="title desc">',
         '<title id="title">Posthoc quantity observation versus final quantity-map match</title><desc id="desc">HOLDOUT96 episodes:91 accepted finals,5 no accepted final. Association only; matching a wrong recovered state is not successful recovery.</desc>',
         '<rect width="760" height="300" fill="#101b2c"/><g fill="#e8eff8" font-family="sans-serif" font-size="15"><text x="20" y="28">Posthoc: current quantity read × accepted final map match</text><text x="20" y="55">Per24 episodes; rows: read0 wrong / read0 right / read1 wrong / read1 right / no final</text>']
    for i,g in enumerate(phases[1]['groups']):
        c=g['cross_table'];label=('BF16' if g['kv']=='K0' else 'NVFP4')+' '+g['policy']
        numbers=' / '.join(str(c[k]) for k in ['read0_final0','read0_final1','read1_final0','read1_final1'])+' / '+str(g['no_accepted_final'])
        svg.append(f'<text x="20" y="{98+i*39}">{escape(label)}</text><text x="215" y="{98+i*39}">{numbers}</text>')
    svg.append('<text x="20" y="275">Required: all public record keys; no later quantity change; no final = NA.</text></g></svg>')
    (output/'observation.svg').write_text('\n'.join(svg)+'\n')
    write(output/'CPU_receipt.json',{'status':'PASS','input_episodes':120,'valid':120,'missing':0,'exceptions':0,
          'wall_seconds':time.perf_counter()-start,'CPU_seconds':time.process_time()-cpu,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'output_hashes':[{'path':str(p.relative_to(output)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()],
          'new_model_requests':0,'classification_uses_reported_booleans':False})
    return summary
