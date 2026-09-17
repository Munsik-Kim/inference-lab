"""Independent scalar checker. Empty GPU data means zero GPU scores audited."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import digest, load, write_new


def scalar_score(row):
    """Separate math.fsum/exp route; does not import src.metrics."""
    x=row['option_logits']; y=row['gold_index']
    if len(x)!=4 or y not in range(4) or not all(math.isfinite(v) for v in x):
        raise ValueError('Invalid saved logits')
    peak=max(x); normalizer=peak+math.log(math.fsum(math.exp(v-peak) for v in x))
    p=[math.exp(v-normalizer) for v in x]
    winner=max(range(4),key=lambda j:x[j])
    return {'q':p,'prediction':winner,'correct':winner==y,
            'choice_nll':normalizer-x[y],
            'choice_brier':math.fsum((v-(j==y))**2 for j,v in enumerate(p)),
            'gold_margin':x[y]-max(v for j,v in enumerate(x) if j!=y),
            'label_mass':math.exp(normalizer-row['full_lse']),
            'full_gold_nll':row['full_lse']-x[y]}


def scalar_norm(row):
    n=row['count']; r2=row['reference_squared_sum']; e2=row['error_squared_sum']
    if n<=0 or not math.isfinite(r2+e2) or min(r2,e2)<0:
        raise ValueError('Invalid norm sufficient statistics')
    return {'absolute_rms':math.sqrt(e2/n),'reference_rms':math.sqrt(r2/n),
            'relative_error':math.sqrt(e2/r2) if r2/n>1e-12 else None}


def linear_quantile(values,p):
    v=sorted(values); rank=(len(v)-1)*p; left=int(rank); right=math.ceil(rank)
    return v[left]+(v[right]-v[left])*(rank-left)


def scalar_bootstrap(rows,metric,draws):
    """Reuses recorded indices, independently calculates task/scenario means."""
    by_task={}
    for task,ids in draws['ids'].items():
        by_task[task]=[]
        for sid in ids:
            values=[r[metric] for r in rows if r['task']==task and r['base_id']==sid]
            if not values:
                raise ValueError('Missing scenario')
            by_task[task].append(math.fsum(values)/len(values))
    samples=[]
    for rep in range(draws['repetitions']):
        task_means=[]
        for task,values in by_task.items():
            indices=draws['indices'][task][rep]
            task_means.append(math.fsum(values[i] for i in indices)/len(indices))
        samples.append(math.fsum(task_means)/len(task_means))
    point=math.fsum(math.fsum(v)/len(v) for v in by_task.values())/len(by_task)
    return {'mean':point,'ci95':[linear_quantile(samples,.025),linear_quantile(samples,.975)]}


def close(a,b):
    if a is None or b is None:
        if a is not b: raise ValueError('Undefined statistic mismatch')
    elif not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError(f'Scalar mismatch: {a} / {b}')


def verify(case):
    inputs=load(case/'inputs/prompts.json'); gold=load(case/'inputs/gold.json'); manifest=load(case/'inputs/manifest.json')
    if digest(inputs)!=manifest['inputs_digest'] or digest(gold)!=manifest['gold_digest']:
        raise ValueError('Input manifest mismatch')
    ids=[r['base_id'] for r in gold]
    if len(ids)!=len(set(ids)) or len({r['base_facts_hash'] for r in gold})!=len(ids):
        raise ValueError('Duplicate scenarios')
    for prompt in inputs:
        if prompt['base_id'] not in ids or len(prompt['token_ids'])!=prompt['length'] or digest(prompt['token_ids'])!=prompt['token_hash']:
            raise ValueError('Token manifest mismatch')
        if prompt['future_answer_tokens_present'] or any(x in prompt for x in ('gold','gold_value','support_ids')):
            raise ValueError('Answer metadata in request record')
    for row in gold:
        values=row['facts']
        if row['task']=='CODE':
            answer=str((values['initial']+values['step']*sum(range(values['count'])))%values['modulus'])
        else:
            supporting=[r for r in values if r['id'] in row['support_ids']]
            answer=str(supporting[0]['value']) if row['task']=='RETRIEVAL' else max(supporting,key=lambda r:r['value'])['entity']
        if answer!=row['gold_value'] or row['options']['ABCD'.index(row['gold'])]!=answer or row['options'].count(answer)!=1 or len(set(row['options']))!=4:
            raise ValueError('Independent gold/foil disagreement')
    records=load(case/'results/records.json'); n=0
    if records:
        raise ValueError('End-to-end measured-study audit is unfinished; scalar helpers alone cannot certify a GPU study')
    for r in records:
        if r.get('evidence_kind')!='gpu_measurement' or r.get('mock'):
            raise ValueError('Non-measurement record')
        independent=scalar_score(r)
        if 'score' in r:
            for key in ('choice_nll','choice_brier','gold_margin','label_mass','full_gold_nll'):
                close(independent[key],r['score'][key])
        for unit in r.get('local_units',[]):
            for key,value in scalar_norm(unit).items(): close(value,unit[key])
        n+=1
    hashes=0
    if (case/'SHA256SUMS').exists():
        names=[]
        for line in (case/'SHA256SUMS').read_text().splitlines():
            expected,name=line.split('  ',1); path=Path(name)
            names.append(name)
            if path.is_absolute() or '..' in path.parts or (case/path).is_symlink():
                raise ValueError('Unsafe checksum path')
            actual=hashlib.file_digest((case/path).open('rb'),'sha256').hexdigest()
            if actual!=expected: raise ValueError(f'Hash mismatch: {name}')
            hashes+=1
        actual_names={p.relative_to(case).as_posix() for p in case.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
        if len(names)!=len(set(names)) or set(names)!=actual_names:
            raise ValueError('Manifest does not cover exactly the public files')
    snapshot=case/'provenance/code_snapshot.json'
    if snapshot.exists():
        for name,expected in load(snapshot)['files'].items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or (case/path).is_symlink():
                raise ValueError('Unsafe source snapshot path')
            if hashlib.sha256((case/path).read_bytes()).hexdigest()!=expected:
                raise ValueError('Code/draft integrity snapshot mismatch')
    if not records:
        semantic=load(case/'results/semantic_status.json')
        if semantic['status']!='NOT_RUN': raise ValueError('Empty results cannot imply GPU semantic PASS')
        summary=load(case/'results/summary.json')
        if summary['status']['execution_status']!='CPU_ONLY_HARNESS_READY' or summary['measured_record_count']!=0:
            raise ValueError('Stored summary overstates empty evidence')
        if set(summary['arms'])!={'B','A_PUBLIC','V4'} or any(v['status']!='NOT_RUN' or v['metrics'] is not None for v in summary['arms'].values()):
            raise ValueError('Missing arm or fabricated measured summary')
        if summary['prepared_inputs']!=manifest['counts']:
            raise ValueError('Summary/input counts differ')
    return {'status':'PASS_CPU_CONSISTENCY','independent_gold_records_checked':len(gold),
            'exact_token_records_checked':len(inputs),'measured_scores_checked':n,'file_hashes_checked':hashes,
            'gpu_reproduction':'NOT_RUN','measured_intervals_checked':0 if not records else None,
            'limitation':'Validates retained scalar/input consistency; no independent GPU replication. No measured metrics exist in the CPU-only package.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.resolve().is_relative_to(a.case.resolve()): p.error('Use a new external report')
    result=verify(a.case);write_new(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__': main()
