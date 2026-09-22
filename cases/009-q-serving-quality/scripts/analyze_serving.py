"""Round-paired serving analysis. Concurrent requests are not independent rounds."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics


def quantile(values,p):
    if not values:return None
    s=sorted(values);i=(len(s)-1)*p;lo=math.floor(i);hi=math.ceil(i)
    return s[lo]+(s[hi]-s[lo])*(i-lo)


def aggregate_cell(cell,condition,expected_ids):
    rows=cell['records'];length,c=condition
    if len(rows)!=max(64,8*c):raise ValueError('missing/extra requests')
    if [r['request_id'] for r in rows]!=expected_ids:raise ValueError('request order/identity mismatch')
    if len(set(r['request_id'] for r in rows))!=len(rows):raise ValueError('duplicate request')
    elapsed=cell['elapsed_seconds']
    if not math.isfinite(elapsed) or elapsed<=0:raise ValueError('invalid block elapsed')
    failures=sum(r['status']!='OK' for r in rows)
    result={'n_requests':len(rows),'failures':failures,'status':'INVALID' if failures else 'COMPLETE','elapsed_seconds':elapsed}
    for r in rows:
        if r['status']!='OK':continue
        if r['input_tokens']!=length or r['output_tokens']!=256 or r['finish_reason']!='length':raise ValueError('token/finish mismatch')
        for k in ('latency_seconds','ttft_seconds','tpot_seconds'):
            if not isinstance(r[k],(int,float)) or not math.isfinite(r[k]) or r[k]<0:raise ValueError('nonfinite/negative timing')
        if not math.isclose(r['tpot_seconds'],(r['latency_seconds']-r['ttft_seconds'])/255,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('TPOT denominator')
    if failures:return result
    for metric in ['ttft','tpot','latency']:
        v=[1000*r[metric+'_seconds'] for r in rows]
        result[metric+'_p50_ms']=quantile(v,.5);result[metric+'_p95_ms']=quantile(v,.95)
    result['output_tokens_per_second']=sum(r['output_tokens'] for r in rows)/elapsed
    result['generated_tokens']=sum(r['output_tokens'] for r in rows)
    return result


def round_ratio(baseline,candidate,metric,inverse=False):
    if baseline.keys()!=candidate.keys():raise ValueError('missing paired round')
    ratios=[(candidate[k][metric]/baseline[k][metric] if inverse else baseline[k][metric]/candidate[k][metric]) for k in sorted(baseline)]
    rng=random.Random(909221)
    boot=[statistics.fmean(rng.choices(ratios,k=len(ratios))) for _ in range(5000)]
    return {'n_independent_rounds':len(ratios),'round_ratios':ratios,'mean_ratio':statistics.fmean(ratios),'pointwise_95_interval':[quantile(boot,.025),quantile(boot,.975)],'direction':'W4/BF16' if inverse else 'BF16/W4'}


def counter(path,name):
    if not path.exists():return None
    vals=[]
    for line in path.read_text().splitlines():
        if line.startswith(name+'{') or line.startswith(name+' '):
            vals.append(float(line.rsplit(' ',1)[1]))
    return sum(vals) if vals else None


def analyze(work,protocol,requests):
    cells=[];raw=[];process=[]
    for run in protocol['process_order']:
        name=f"round{run['round']}-{run['arm']}-{run['execution']}";d=work/name
        tele=[]
        if (d/'telemetry.jsonl').exists():
            for line in (d/'telemetry.jsonl').read_text().splitlines():
                t=json.loads(line)
                if t['returncode']==0:
                    tele.append([float(v.strip()) for v in t['values'].split(',')])
        log=(d/'server.log').read_text() if (d/'server.log').exists() else ''
        running=[int(x) for x in re.findall(r'Running: (\d+) reqs',log)]
        waiting=[int(x) for x in re.findall(r'Waiting: (\d+) reqs',log)]
        process.append({'process_id':name,'sampled_device_peak_used_MiB':max(t[0] for t in tele) if tele else None,
          'telemetry_samples':len(tele),'sampled_running_max':max(running) if running else None,
          'sampled_waiting_max':max(waiting) if waiting else None,'capture_logged':'Capturing CUDA graphs' in log,
          'memory_scope':'whole-device 1Hz, includes desktop/background, process startup and warmup; not Torch allocated/reserved peak'})
        for length,c in run['conditions']:
            key=f'L{length}-C{c}';path=d/(key+'.json')
            ident={'round':run['round'],'arm':run['arm'],'execution':run['execution'],'input_tokens':length,'output_tokens':256,'client_concurrency':c,'process_id':name,'condition':key}
            if not path.exists():
                ident['status']='UNMEASURED'
                if (d/'failure.json').exists():
                    ident['status']='FAILED_PROCESS';ident['failure']=json.loads((d/'failure.json').read_text())
                cells.append(ident);continue
            cell=json.loads(path.read_text());ids=[r['id'] for r in requests if r['input_length']==length][:max(64,8*c)]
            result=aggregate_cell(cell,(length,c),ids)
            for row,reference in zip(cell['records'],[r for r in requests if r['input_length']==length]):
                if row['prompt_hash']!=reference['prompt_hash']:raise ValueError('prompt changed')
            b=counter(d/(key+'-metrics-before.txt'),'vllm:num_preemptions_total');a=counter(d/(key+'-metrics-after.txt'),'vllm:num_preemptions_total')
            result['preemptions']=a-b if b is not None and a is not None else None
            cells.append(ident|result);raw.extend(cell['records'])
    contrasts=[]
    for mode in ['graph','eager']:
        for l,c in sorted({(r['input_tokens'],r['client_concurrency']) for r in cells if r['execution']==mode}):
            members=[r for r in cells if (r['execution'],r['input_tokens'],r['client_concurrency'])==(mode,l,c)]
            b={r['round']:r for r in members if r['arm']=='BF16' and r['status']=='COMPLETE'}
            q={r['round']:r for r in members if r['arm']=='W4' and r['status']=='COMPLETE'}
            item={'execution':mode,'input_tokens':l,'client_concurrency':c}
            if len(b)==len(q)==3 and b.keys()==q.keys():
                item['tpot_ratio']=round_ratio(b,q,'tpot_p50_ms');item['throughput_ratio']=round_ratio(b,q,'output_tokens_per_second',True)
            else:item['status']='INCOMPLETE_PAIRS';item['complete_rounds']={'BF16':sorted(b),'W4':sorted(q)}
            contrasts.append(item)
    return {'cells':cells,'processes':process,'contrasts':contrasts,'complete_cells':sum(x['status']=='COMPLETE' for x in cells),'planned_cells':len(cells),'independent_rounds_planned':3,'uncertainty':'pointwise descriptive paired server-round bootstrap; n=3 is small; concurrent requests share scheduler','raw_request_count':len(raw)},raw


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--requests',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result,rows=analyze(a.work,json.loads(a.protocol.read_text()),json.loads(a.requests.read_text()))
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'serving_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (a.output/'requests.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n' for r in rows))
    print(json.dumps({'complete_cells':result['complete_cells'],'planned_cells':result['planned_cells']}))

if __name__=='__main__':main()
