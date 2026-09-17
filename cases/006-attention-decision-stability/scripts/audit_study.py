"""Independent scalar recalculation, distinct from the study analysis implementation."""
import argparse,json,math,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import load,write_new,digest
from scripts.verify_results import scalar_score,scalar_norm,scalar_bootstrap,close


def check(records,summary):
    index={};norm_count=0;allpairs=[]
    required=('full_attention_outputs_finite','full_block_outputs_finite','full_final_hidden_finite','full_logits_finite','inputs_unchanged','same_layer13_qkv','routing_valid','dtype_layout_valid','answer_interface_valid','native_B_validated','effective_backend_verified')
    for row in records:
        if row.get('evidence_kind')!='gpu_measurement' or row.get('mock'):raise ValueError('Not measured evidence')
        if any(row.get('validity',{}).get(k) is not True for k in required):raise ValueError('Incomplete full-output validity chain')
        key=row['item_id'],row['arm']
        if key in index:raise ValueError('Duplicate cell')
        index[key]=row;score=scalar_score(row)
        for k in ('choice_nll','choice_brier','gold_margin','label_mass','full_gold_nll'):close(score[k],row['score'][k])
        if score['prediction']!=row['score']['prediction'] or score['correct']!=row['score']['correct']:raise ValueError('Decision mismatch')
        for unit in row['local_units']:
            for target in (unit,unit['last_query']):
                values=scalar_norm(target)
                for k,v in values.items():close(v,target[k])
                norm_count+=1
        count=sum(u['count'] for u in row['local_units']);r2=math.fsum(u['reference_squared_sum'] for u in row['local_units']);e2=math.fsum(u['error_squared_sum'] for u in row['local_units'])
        pooled=math.sqrt(e2/r2) if r2/count>1e-12 else None;close(pooled,row['local_pooled']['relative_error'])
    for item in sorted({r['item_id'] for r in records}):
        if any((item,a) not in index for a in ('B','A_PUBLIC','V4')):raise ValueError('Missing paired arm')
        b=index[item,'B'];bs=scalar_score(b)
        for arm in ('A_PUBLIC','V4'):
            c=index[item,arm];cs=scalar_score(c)
            for k in ('token_hash','design_hash','manifest_hash','runtime_hash','base_id','task','split','length'):
                if b[k]!=c[k]:raise ValueError('Pair identity mismatch')
            if b['qkv_hashes']!=c['qkv_hashes']:raise ValueError('QKV mismatch')
            cell='both_correct' if bs['correct'] and cs['correct'] else 'regression' if bs['correct'] else 'gain' if cs['correct'] else 'both_wrong'
            allpairs.append({k:b[k] for k in ('item_id','base_id','task','split','length')}|{'arm':arm,'cell':cell,'flip':bs['prediction']!=cs['prediction'],
                 'delta_choice_nll':cs['choice_nll']-bs['choice_nll'],'delta_brier':cs['choice_brier']-bs['choice_brier'],
                 'delta_gold_margin':cs['gold_margin']-bs['gold_margin'],'delta_accuracy':float(cs['correct'])-float(bs['correct'])})
    ci_count=0
    for name,reported in summary['groups'].items():
        split,l=name.split('/L');length=int(l);rows=[r for r in allpairs if r['split']==split and r['length']==length]
        ids={task:sorted({r['base_id'] for r in rows if r['task']==task}) for task in ('RETRIEVAL','COMPARISON','CODE')}
        rng=np.random.default_rng(606901)
        draws={'ids':ids,'indices':{t:rng.integers(len(v),size=(5000,len(v))).tolist() for t,v in ids.items()},'repetitions':5000}
        if digest(draws['indices'])!=reported['bootstrap_indices_sha256']:raise ValueError('Bootstrap index mismatch')
        for arm in ('A_PUBLIC','V4'):
            selected=[r for r in rows if r['arm']==arm];stats=reported['arms'][arm]
            for cell in ('both_correct','regression','gain','both_wrong'):
                if sum(r['cell']==cell for r in selected)!=stats['outcomes'][cell]:raise ValueError('Outcome cell mismatch')
            if sum(r['flip'] for r in selected)!=stats['outcomes']['flips']['numerator']:raise ValueError('Flip mismatch')
            for metric in ('delta_choice_nll','delta_brier','delta_gold_margin','delta_accuracy'):
                independent=scalar_bootstrap(selected,metric,draws);published=stats['task_balanced'][metric]
                close(independent['mean'],published['mean'])
                for a,b in zip(independent['ci95'],published['ci95']):close(a,b)
                ci_count+=1
    primary=summary['groups'].get('standard/L4096')
    complete=primary is not None and primary['independent_scenarios']==192
    return {'status':'PASS_CPU_MEASUREMENT_CONSISTENCY','records_checked':len(records),'norm_units_checked':norm_count,'paired_comparisons':len(allpairs),'bootstrap_intervals_checked':ci_count,
            'primary_192_scenarios_complete':complete,'independence_scope':'Distinct scalar/norm/paired/bootstrap calculation path. Not third-party GPU replication. Full-vocabulary KL and hidden differences cannot be fully reconstructed from public four-logit/norm evidence alone.',
            'derived_execution_status':'COMPLETED_CONTROLLED_STUDY' if complete else 'PARTIAL_TECHNICAL_BLOCK','deployment_verdict':'NOT_ASSESSED'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',nargs='+',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=[]
    for path in a.records:
        if path.is_dir():records.extend(load(f)['payload'] for f in sorted(path.glob('*.json')))
        else:records.extend(load(path))
    result=check(records,load(a.summary));write_new(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
