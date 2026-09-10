"""Paired descriptive analysis; round 1 is the only primary quality sample."""
import argparse,collections,hashlib,json,pathlib,re
import numpy as np
from data import FIELDS,read_jsonl,dump,jsonl
from score import score

def distribution(values):
    a=np.array([x for x in values if x is not None],dtype=float)
    return {'n':int(len(a)),'median':float(np.median(a)) if len(a) else None,'p95':float(np.percentile(a,95)) if len(a) else None,'min':float(a.min()) if len(a) else None,'max':float(a.max()) if len(a) else None}

def quality(rows,gold):
    n=len(rows);scored=[]
    for r in rows:
        s=score(r['raw_output'],gold[r['case_id']]['values'])
        if r['error']:s['document_correct']=False;s['fields']={f:False for f in FIELDS}
        scored.append((r,s))
    result={'n':n,'document_correct':sum(s['document_correct'] for _,s in scored),'json_valid':sum(s['json_valid'] for _,s in scored),'schema_valid':sum(s['schema_valid'] for _,s in scored),'execution_errors':sum(bool(r['error']) for r,_ in scored),'fields':{},'schema_error_counts':dict(collections.Counter(s['error'] for _,s in scored if s['error']))}
    for f in FIELDS:
        result['fields'][f]={'correct':sum(s['fields'][f] for _,s in scored),'n':n}
        for kind,isnull in [('null',True),('non_null',False)]:
            pairs=[(r,s) for r,s in scored if (gold[r['case_id']]['values'][f] is None)==isnull]
            result['fields'][f][kind]={'correct':sum(s['fields'][f] for _,s in pairs),'n':len(pairs)}
    for key in ['document_correct','json_valid','schema_valid']:result[key+'_pct']=100*result[key]/n if n else None
    return result

def paired_quality(primary,docs,gold):
    maps={m:{r['case_id']:r for r in primary if r['model']==m} for m in ['bf16','fp8']}
    assert set(maps['bf16'])==set(maps['fp8'])==set(docs)
    cells={'both_correct':0,'bf16_only_correct':0,'fp8_only_correct':0,'both_wrong':0};pairs=[];failures=[]
    for cid,d in docs.items():
        b=maps['bf16'][cid];f=maps['fp8'][cid]
        sb=score(b['raw_output'],gold[cid]['values']);sf=score(f['raw_output'],gold[cid]['values'])
        bc=sb['document_correct'] and not b['error'];fc=sf['document_correct'] and not f['error']
        category='both_correct' if bc and fc else 'bf16_only_correct' if bc else 'fp8_only_correct' if fc else 'both_wrong'
        cells[category]+=1
        item={'case_id':cid,'bucket':d['bucket'],'category':category,'bf16_correct':bool(bc),'fp8_correct':bool(fc),'bf16_score':sb,'fp8_score':sf}
        pairs.append(item)
        if category!='both_correct':failures.append({**item,'document':d['document'],'target_id':d['target_id'],'gold':gold[cid], 'bf16_output':b['raw_output'],'fp8_output':f['raw_output']})
    rng=np.random.default_rng(20260910);boot=np.zeros(10000)
    for bucket in ['short','long']:
        dif=np.array([int(p['fp8_correct'])-int(p['bf16_correct']) for p in pairs if p['bucket']==bucket])
        assert len(dif)==50
        boot+=rng.choice(dif,size=(10000,len(dif)),replace=True).mean(axis=1)*.5*100
    delta=100*(cells['fp8_only_correct']-cells['bf16_only_correct'])/len(pairs)
    return {'n_paired_documents':len(pairs),'cells':cells,'fp8_minus_bf16_percentage_points':delta,'paired_stratified_bootstrap_95pct_interval_pp':[float(x) for x in np.percentile(boot,[2.5,97.5])],'bootstrap_resamples':10000,'bootstrap_seed':20260910,'bootstrap_note':'Resample paired document IDs within short/long strata, preserving 50/50 weights. This interval reflects only the observed synthetic sample; a degenerate interval is not proof of equivalence or absence of rare errors.'},pairs,failures

def memory_summary(run_dir):
    meta=json.loads((run_dir/'server.json').read_text());samples=read_jsonl(run_dir/'memory.jsonl');log=(run_dir/'server.log').read_text(errors='replace')
    result={k:meta.get(k) for k in ['run_id','model','round','startup_to_health_s','server_exit_code','error','measurement_complete']}
    result['phase_device_memory_bytes']={}
    for phase in ['idle_before_start','startup','loaded','warmup','measured','shutdown']:
        ss=[r for r in samples if r['phase']==phase and 'device_used_bytes' in r]
        result['phase_device_memory_bytes'][phase]=distribution([r['device_used_bytes'] for r in ss])
    matches=re.findall(r'Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds',log)
    result['engine_model_loading_memory_gib_and_seconds']=[list(map(float,m)) for m in matches]
    result['gpu_kv_cache_tokens']=[int(x.replace(',','')) for x in re.findall(r'GPU KV cache size: ([\d,]+) tokens',log)]
    result['memory_log_lines']=[s for s in log.splitlines() if any(k in s for k in ['KV cache','Model loading took','Capturing CUDA graphs','Graph capturing','Memory profiling','compile took','compilation takes','engine init','weight memory','non_torch_memory','activation_peak_memory'])]
    result['metric_scope']='Whole GPU NVML samples; 100 ms nominal interval, sampled peak only. No per-process isolation or allocator telemetry.'
    return result

def analyze(root):
    eval_dirs=sorted(p for p in (root/'results').iterdir() if p.is_dir() and '-eval-r' in p.name)
    assert len(eval_dirs)==6,len(eval_dirs)
    allrows=[]
    for d in eval_dirs:
        rows=read_jsonl(d/'requests.jsonl');measured=[r for r in rows if r['phase']=='measured'];assert len(measured)==100,d
        assert len({r['case_id'] for r in measured})==100
        assert all(r['split']=='eval' for r in measured)
        assert len([r for r in rows if r['phase']=='warmup'])==2
        allrows+=measured
    docs={r['id']:r for r in read_jsonl(root/'data/eval_inputs.jsonl')};gold={r['id']:r for r in read_jsonl(root/'data/eval_gold.jsonl')}
    primary=[r for r in allrows if r['round']==1]
    result={'primary_quality_round':1,'unique_eval_documents':100,'total_measured_requests':600,'quality':{},'latency':{},'paired_latency':{},'output_reproducibility':{}}
    for m in ['bf16','fp8']:
        result['quality'][m]={b:quality([r for r in primary if r['model']==m and (b=='all' or r['bucket']==b)],gold) for b in ['all','short','long']}
        result['latency'][m]={}
        for rnd in [1,2,3,'pooled_descriptive']:
            rr=[r for r in allrows if r['model']==m and (rnd=='pooled_descriptive' or r['round']==rnd)]
            result['latency'][m][str(rnd)]={}
            for b in ['short','long']:
                bucket=[r for r in rr if r['bucket']==b]
                result['latency'][m][str(rnd)][b]={metric:distribution([r.get(metric) for r in bucket]) for metric in ['ttft_s','total_s','input_tokens','output_tokens','approx_post_first_s_per_token']}
                result['latency'][m][str(rnd)][b]['request_errors']=sum(bool(r['error']) for r in bucket)
        by_round={i:{r['case_id']:r for r in allrows if r['model']==m and r['round']==i} for i in [1,2,3]}
        result['output_reproducibility'][m]={}
        for rnd in [2,3]:
            same_raw=sum(by_round[1][cid]['raw_output']==by_round[rnd][cid]['raw_output'] for cid in docs)
            normalized_pairs=[(score(by_round[1][cid]['raw_output'],gold[cid]['values']),score(by_round[rnd][cid]['raw_output'],gold[cid]['values'])) for cid in docs]
            same_normalized=sum(x['schema_valid'] and y['schema_valid'] and x['parsed']==y['parsed'] for x,y in normalized_pairs)
            result['output_reproducibility'][m][str(rnd)]={'same_raw_as_round1':same_raw,'same_parsed_as_round1':same_normalized,'n':100,'quality':quality(list(by_round[rnd].values()),gold)}
    paired,pairrows,failures=paired_quality(primary,docs,gold);result['paired_quality']=paired
    for b in ['short','long']:
        pairs=[]
        for rnd in [1,2,3]:
            bm={r['case_id']:r for r in allrows if r['model']=='bf16' and r['round']==rnd and r['bucket']==b}
            fm={r['case_id']:r for r in allrows if r['model']=='fp8' and r['round']==rnd and r['bucket']==b}
            assert [x['case_id'] for x in bm.values()]==[x['case_id'] for x in fm.values()]
            for cid in bm:
                br,fr=bm[cid],fm[cid]
                item={'case_id':cid,'round':rnd,'bucket':b,'output_tokens_bf16':br['output_tokens'],'output_tokens_fp8':fr['output_tokens'],'same_output_tokens':br['output_tokens']==fr['output_tokens']}
                for metric in ['ttft_s','total_s','approx_post_first_s_per_token']:
                    if br.get(metric) is not None and fr.get(metric) is not None:
                        item[metric+'_fp8_minus_bf16']=fr[metric]-br[metric];item[metric+'_fp8_over_bf16']=fr[metric]/br[metric]
                pairs.append(item)
        result['paired_latency'][b]={metric:distribution([p.get(metric) for p in pairs]) for metric in ['ttft_s_fp8_minus_bf16','ttft_s_fp8_over_bf16','total_s_fp8_minus_bf16','total_s_fp8_over_bf16','approx_post_first_s_per_token_fp8_over_bf16']}
        result['paired_latency'][b]['same_output_token_count_pairs']=sum(p['same_output_tokens'] for p in pairs)
        jsonl(root/'results'/f'paired_latency_{b}.jsonl',pairs)
    result['memory']=[memory_summary(d) for d in eval_dirs]
    result['finish_reasons']=dict(collections.Counter(str(r['finish_reason']) for r in allrows))
    result['input_token_check_passed']=all(r['input_tokens']==r['expected_input_tokens'] for r in allrows)
    result['measurement_note']='Client streaming TTFT starts before HTTP send and stops at first nonempty content chunk. Total includes completion/[DONE]. Post-first time per token is approximate and includes final transport overhead. Chunks are not tokens. All failures remain in accuracy denominators; latency missing values are counted by metric n.'
    result['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in eval_dirs for p in sorted(d.iterdir()) if p.is_file()}
    dump(root/'results/aggregate.json',result);jsonl(root/'results/quality_pairs.jsonl',pairrows);jsonl(root/'results/failures.jsonl',failures)
    print(json.dumps({'quality':result['quality'],'paired_quality':paired,'finish_reasons':result['finish_reasons']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);a=p.parse_args();analyze(a.root)
