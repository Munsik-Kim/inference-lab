"""Independent CPU readback of compact records; never loads a model or weights."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np

CASE = Path(__file__).resolve().parents[1]
METHODS = ('BF16', 'RTN', 'D16', 'D4')

def require(ok, message):
    if not ok: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def indexed(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(key not in result, 'duplicate record: '+str(key))
        result[key] = row
    return result

def extract_json(text):
    # An independent scalar parser; mirrors the documented first-object policy.
    for i in range(len(text)):
        if text[i] != '{': continue
        try:
            value, _ = json.JSONDecoder().raw_decode(text[i:])
            if isinstance(value, dict): return value
        except ValueError: pass
    return None

def exact_long(text, task):
    value = extract_json(text)
    return bool(value and str(value.get('answer')) == task['answer']
                and sorted(value.get('evidence', [])) == sorted(task['evidence']))

def paired_counts(base, candidate, field):
    require(set(base) == set(candidate), 'pair IDs differ')
    result = dict(both_success=0, baseline_only=0, candidate_only=0, both_failed=0)
    for key in base:
        a, b = bool(base[key][field]), bool(candidate[key][field])
        name = 'both_success' if a and b else 'baseline_only' if a else 'candidate_only' if b else 'both_failed'
        result[name] += 1
    return result

def verify_export(case):
    mapping = read(case/'provenance/export_map.json')
    seen=set()
    for row in mapping['files']:
        relative = Path(row['public'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe export path')
        require(str(relative) not in seen, 'duplicate export path'); seen.add(str(relative))
        p=case/relative
        require(p.is_file() and not p.is_symlink(), 'missing/linked export: '+str(relative))
        require(p.stat().st_size == row['public_bytes'] and sha(p) == row['public_sha256'], 'changed export: '+str(relative))
        if row['operation'] == 'BYTE_IDENTICAL_COPY':
            require(row['original_sha256'] == row['public_sha256'], 'invalid identity mapping')
        else: require(row['operation'] == 'LOCAL_PATH_REDACTION', 'unknown export operation')
    return len(seen)

def audit(case=CASE):
    case=Path(case); r=case/'records'; export_count=verify_export(case)
    contract=read(r/'reference_contract.json'); manifest=read(r/'experiment_manifest.json')
    require(contract['label'] == 'PAPER_INT4_REFERENCE_VARIANT' and contract['group'] == 64
            and contract['iterations'] == 15 and contract['scale_dtype'] == 'float32', 'execution contract changed')
    quality=[]
    for model in ('06b','4b'):
        with (r/f'quality_{model}.csv').open() as csv_file:
            reported={row['method']:row for row in csv.DictReader(csv_file)}
        require(set(reported) == set(METHODS), 'missing quality methods')
        bases=[]
        for i in range(32):
            with np.load(r/f'raw/{model}/BF16/tokens{i:02}.npz', allow_pickle=False) as raw:
                bases.append(raw['argmax'])
        base_windows=lines(r/f'raw/{model}/BF16/quality_windows.jsonl')
        for method in METHODS:
            folder=r/f'raw/{model}/{method}'; windows=lines(folder/'quality_windows.jsonl')
            require(len(windows) == 32 and len(indexed(windows,('window',))) == 32, 'quality window inventory')
            nll_sum=0.; prefix_sum=0.; flip_sum=0; kl=[]
            for i,row in enumerate(windows):
                require(row['window'] == i and row['valid_nll_tokens'] == 2047, 'window/token identity')
                require(row['input_sha256'] == base_windows[i]['input_sha256'], 'quality input pairing')
                with np.load(folder/f'tokens{i:02}.npz',allow_pickle=False) as raw:
                    require(set(raw.files) == {'nll','argmax'}, 'unexpected token fields')
                    loss=raw['nll'].astype(np.float64); pred=raw['argmax']
                    require(loss.shape == pred.shape == (2047,) and np.isfinite(loss).all() and (loss>=0).all(), 'invalid losses/shape')
                    total=float(loss.sum()); flips=int(np.count_nonzero(pred != bases[i]))
                    require(abs(total-row['nll_sum']) < 1e-8 and flips == row['argmax_flips'], 'window scalar mismatch')
                    nll_sum += total; prefix_sum += float(loss[:511].sum()); flip_sum += flips
                # KL is recorded full-vocabulary reduction at 32 selected positions.
                # Full logits are not public, so this is an aggregation, not new KL computation.
                require(row['kl_positions'] == 32, 'KL position denominator')
                kl.append(row['kl_sum']/row['kl_positions'])
            nll=nll_sum/65504; ppl=math.exp(nll); summary=read(folder/'summary.json')
            values={'valid_tokens':65504,'nll':nll,'ppl':ppl,
                    'prefix512_nll_from_saved2048_records':prefix_sum/(32*511),
                    'prefix512_ppl_from_saved2048_records':math.exp(prefix_sum/(32*511)),
                    'full_token_argmax_flip_fraction':flip_sum/65504,
                    'forward_KL_BF16_to_candidate':float(np.mean(kl))}
            for field,value in values.items():
                require(math.isfinite(value) and abs(value-float(reported[method][field])) < 1e-8,
                        f'quality CSV mismatch: {model}/{method}/{field}')
            require(abs(nll-summary['quality']['nll'])<1e-10 and summary['status']=='COMPLETE', 'worker summary mismatch')
            row={'model':model,'method':method,**values}
            for field in ('weight_normalized_mse','artifact_payload_bytes','conversion_or_reload_seconds',
                          'quality_wall_seconds','peak_allocated_bytes','peak_reserved_bytes'):
                row[field]=float(reported[method][field])
            row['artifact_payload_bytes']=int(row['artifact_payload_bytes'])
            quality.append(row)

    # Metadata-only reconstruction audit; this does not read private weight bytes.
    for model in ('06b','4b'):
        for method in ('RTN','D16'):
            meta=read(r/f'artifact_metadata/{model}-{method}.json'); entries=meta['files']
            require(len(entries)==(196 if model=='06b' else 252),'projection inventory')
            indexed(entries,('layer',))
            payload=sum(x['bytes'] for x in entries)
            nmse=sum(x['weight_error_sse'] for x in entries)/sum(x['weight_energy'] for x in entries)
            for entry in entries:
                ledger=entry['ledger']
                require(ledger['serialized_bytes']==entry['bytes'] and ledger['header_bytes']==24,'artifact byte ledger')
                require(sum(ledger[k] for k in ('code_bytes','gamma_bytes','alpha_bytes','beta_bytes'))+24==entry['bytes'],'artifact scale bytes')
            selected=next(x for x in quality if x['model']==model and x['method']==method)
            require(payload==selected['artifact_payload_bytes']==meta['total_payload_bytes']
                    and abs(nmse-selected['weight_normalized_mse'])<1e-10,'weight/byte metadata aggregation')
        a=next(x for x in quality if x['model']==model and x['method']=='D16')
        b=next(x for x in quality if x['model']==model and x['method']=='D4')
        require(a['weight_normalized_mse']==b['weight_normalized_mse'] and a['artifact_payload_bytes']==b['artifact_payload_bytes'], 'D4 weight reuse')

    tasks=indexed(read(r/'inputs/long_tasks.json'),('id',))
    require(len(tasks) == 12, 'long DEV task denominator')
    long_rows=[]; long_summary=[]; long_pairing={}; not_run=[]
    for method in METHODS:
        rows=lines(r/f'raw/4b/{method}/long_dev.jsonl')
        completed=[row for row in rows if row['status']=='COMPLETE']
        indexed(completed,('id','length'))
        for row in rows:
            if row['status'] != 'COMPLETE':
                require(row['status']=='NOT_RUN_MEMORY_POLICY' and row['length']==32768, 'unknown long status')
                not_run.append({'method':method,'length':row['length'],'status':row['status']}); continue
            task=tasks[(row['id'],)]; outcome=exact_long(row['text'],task)
            require(outcome == row['exact_success'] and row['gold']==task['answer'] and row['evidence']==task['evidence'], 'long exact score/gold mismatch')
            ids=np.load(r/f'inputs/long_{row["length"]}.npy',allow_pickle=False)
            pos=list(tasks).index((row['id'],))
            require(row['input_token_sha256']==hashlib.sha256(ids[pos].tobytes()).hexdigest()
                    and row['input_tokens']==row['length'], 'long input hash/length')
            key=(row['id'],row['length']); digest=row['input_token_sha256']
            require(key not in long_pairing or long_pairing[key]==digest, 'long methods input mismatch');long_pairing[key]=digest
            long_rows.append({'method':method,'id':row['id'],'family':row['family'],'length':row['length'],
                              'success':outcome,'text':row['text'],'seconds':row['seconds'],
                              'input_hash':digest,'peak_allocated_bytes':row['peak_allocated_bytes']})
        for length in (8192,16384,32768):
            selected=[x for x in long_rows if x['method']==method and x['length']==length]
            if selected:
                require(len(selected)==12,'long cell incomplete')
                long_summary.append({'method':method,'length':length,'successes':sum(x['success'] for x in selected),'n':12})
    require(len(long_rows)==108 and len(not_run)==3,'long execution inventory')
    pair=[]
    for length in (8192,16384):
        for a,b in [('BF16','RTN'),('RTN','D16'),('RTN','D4')]:
            ix=lambda m:indexed([x for x in long_rows if x['method']==m and x['length']==length],('id',))
            pair.append({'baseline':a,'candidate':b,'length':length,**paired_counts(ix(a),ix(b),'success')})

    agent_summary=[]; agent_rows=[]
    for cohort,prefix,expected_n in [('base',r,8),('extension16k',r/'agent16k_extension',4)]:
        taskmap=indexed(read(prefix/'inputs/agent_tasks.json'),('id',))
        require(len(taskmap)==expected_n,'agent task denominator')
        tokens=read(r/'inputs/agent_token_manifest.json') if cohort=='base' else read(prefix/'preregistration.json')['input_manifest']
        tokenmap={row['id']:row for row in tokens}; pairing={}
        for method in METHODS:
            folder=prefix/f'raw/4b/{method}' if cohort=='base' else prefix/f'raw/{method}'
            rows=lines(folder/'agent_dev.jsonl'); require(len(rows)==expected_n*2,'agent cell count')
            indexed(rows,('id','mode'))
            for row in rows:
                task=taskmap[(row['id'],)]; calls=[e for e in row['events'] if 'text' in e]
                require(calls and len(calls)==row['model_calls'],'model call count')
                require(sum(e['output_tokens'] for e in calls)==row['total_output_tokens']
                        and sum(e['input_tokens'] for e in calls)==row['total_input_tokens'],'agent token count')
                token=tokenmap[row['id']]; expected=token.get('initial_token_sha256',token.get('sha256'))
                if row['mode']=='FREE':require(calls[0]['input_token_sha256']==expected,'agent initial input hash')
                key=(row['id'],row['mode']); digest=calls[0]['input_token_sha256']
                require(key not in pairing or pairing[key]==digest,'agent pairing'); pairing[key]=digest
                final=extract_json(calls[-1]['text']); success=bool(final and all(final.get(k)==v for k,v in task['gold'].items()))
                if task['family']=='fixture_patch' and row['mode']=='FREE':
                    actual=read(folder/f'{row["id"]}-FREE-config.json')
                    success=success and actual==task['fixture']['expected_config'] and any(
                        e.get('tool')=='validate_fixture_test' and e.get('result',{}).get('valid') for e in row['events'])
                require(success==row['success'],'agent final/action score mismatch')
                require(math.isfinite(row['wall_seconds']) and row['wall_seconds']>=0,'agent elapsed')
                agent_rows.append({'cohort':cohort,'method':method,'id':row['id'],'mode':row['mode'],
                                   'success':success,'tool_errors':row['tool_errors'],'parse_errors':row['parse_errors'],
                                   'status':row['status'],'wall_seconds':row['wall_seconds']})
            for mode in ('REPLAY','FREE'):
                chosen=[x for x in agent_rows if x['cohort']==cohort and x['method']==method and x['mode']==mode]
                agent_summary.append({'cohort':cohort,'method':method,'mode':mode,'n':expected_n,
                                      'successes':sum(x['success'] for x in chosen),
                                      'tool_errors':sum(x['tool_errors'] for x in chosen),
                                      'parse_errors':sum(x['parse_errors'] for x in chosen),
                                      'all_attempt_wall_seconds':sum(x['wall_seconds'] for x in chosen)})
    agent_pairs=[]
    for cohort in ('base','extension16k'):
        for mode in ('REPLAY','FREE'):
            for a,b in [('BF16','RTN'),('RTN','D16'),('RTN','D4')]:
                ix=lambda method:indexed([x for x in agent_rows if x['cohort']==cohort and x['mode']==mode and x['method']==method],('id',))
                agent_pairs.append({'cohort':cohort,'mode':mode,'baseline':a,'candidate':b,
                                    **paired_counts(ix(a),ix(b),'success')})
    packed=read(r/'diagnostics/packed_precision_aligned_03.json')
    require(packed['status']=='BLOCKED' and packed['short_full_decode']['status']=='FAIL_TOLERANCE'
            and packed['short_full_decode']['forward_relative_rmse']>packed['thresholds_before_probe']['full_forward_relative_rmse'], 'packed gate falsely accepted')
    return {'schema':'case013-audit-v1','status':'PASS_RECORDED_RESULTS','model_execution':False,
            'exported_files_verified':export_count,'quality':quality,'long':long_summary,'long_pairs':pair,
            'long_not_run':not_run,'agents':agent_summary,'agent_pairs':agent_pairs,
            'units':{'models':2,'quality_windows_per_model':32,'valid_tokens_per_condition':65504,
                     'long_unique_tasks':12,'agent_unique_tasks':8,'long_generations':108,'agent_episode_conditions':96},
            'scope':{'recomputed':['per-token NLL/PPL','argmax flips','exact long scores','agent final/action scores','pair IDs','projection metadata SSE/energy and byte totals'],
                     'aggregated_recorded':['selected-position KL','weight MSE','Torch memory','wall times'],
                     'not_publicly_recomputed':['KL from full logits','original weight MSE','large weight artifact payload hash']},
            'packed':{'status':packed['status'],'relative_rmse':packed['short_full_decode']['forward_relative_rmse'],
                      'threshold':packed['thresholds_before_probe']['full_forward_relative_rmse']}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=CASE);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists() and not a.output.resolve().is_relative_to(a.case.resolve()),'output must be new and outside case')
    result=audit(a.case);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','model_execution','units')}))
if __name__=='__main__':main()
