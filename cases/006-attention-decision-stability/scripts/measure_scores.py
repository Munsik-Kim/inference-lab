"""Serial, immutable paired prompt scoring. All model work uses cached assets."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import load,write_new,digest,file_hash,Ledger
from src.model_runtime import Runtime,ARMS
from src.metrics import kl_logits
from scripts.run_case006 import gpu_gate


def read_inputs(paths):
    rows=[]
    for path in paths:rows.extend(load(path))
    if len({r['item_id'] for r in rows})!=len(rows):raise ValueError('Duplicate input item')
    return rows


def save_base(path,private):
    if path.exists():raise FileExistsError(path)
    fields={'reference':private['reference'],'native_samples':private['native_samples'],'full_logits':private['full_logits'],
            'qkv_hashes':np.array(private['qkv_hashes'])}
    fields.update({'hidden__'+k:v for k,v in private['hidden'].items()})
    with path.open('xb') as out:np.savez_compressed(out,**fields)


def restore_base(path):
    with np.load(path,allow_pickle=False) as z:
        return {'reference':z['reference'],'native_samples':z['native_samples'],'full_logits':z['full_logits'],
                'qkv_hashes':z['qkv_hashes'].tolist(),'hidden':{k[8:]:z[k] for k in z.files if k.startswith('hidden__')}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['dev-b','dev-paired','pool','eval','length'],required=True)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--inputs',nargs='+',type=Path,required=True)
    p.add_argument('--gold',nargs='+',type=Path,required=True);p.add_argument('--semantic',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--freeze',type=Path)
    p.add_argument('--selection',type=Path,help='For stress: frozen selected IDs; never select on candidate scores')
    a=p.parse_args();case=Path(__file__).resolve().parents[1]
    if a.work_dir.resolve().is_relative_to(case):p.error('Write raw measurements outside source case')
    semantic=load(a.semantic)
    if semantic.get('status')!='PASS':raise ValueError('GPU semantics have not passed')
    inputs=read_inputs(a.inputs);gold_rows=read_inputs_gold(a.gold)
    if a.stage.startswith('dev'):
        if any(r['split']!='dev' for r in inputs):raise ValueError('DEV stage cannot access evaluation')
    else:
        if a.freeze is None:raise ValueError('Real design/evaluation freeze required')
        frozen=load(a.freeze)
        expected='DESIGN_FREEZE' if a.stage=='pool' else 'EVAL_MANIFEST_FREEZE'
        if frozen.get('kind')!=expected:raise ValueError('Wrong freeze phase')
        for name,sha in frozen['code_hashes'].items():
            if file_hash(case/name)!=sha:raise ValueError('Frozen code changed: '+name)
        for row in inputs:
            if frozen.get('input_hashes') and frozen['input_hashes'].get(row['item_id'])!=row['token_hash']:
                raise ValueError('Input not in frozen manifest')
    if a.selection:
        selected=set(load(a.selection)['selected_ids']);inputs=[r for r in inputs if r['base_id'] in selected]
        if {r['base_id'] for r in inputs}!=selected:raise ValueError('Incomplete selected input set')
    if a.stage in ('eval','length'):
        allowed=set(frozen['candidate_item_ids'])
        if any(r['item_id'] not in allowed for r in inputs):raise ValueError('Candidate access outside frozen evaluation set')
    arms=('B',) if a.stage in ('dev-b','pool') else ARMS
    manifest_hash=digest({r['item_id']:r['token_hash'] for r in inputs})
    design_hash=file_hash(a.freeze) if a.freeze else 'DEV_NOT_FROZEN'
    runtime_hash=digest({n:file_hash(case/n) for n in ('src/model_runtime.py','src/intervention.py','src/reference.py','src/metrics.py')})
    a.work_dir.mkdir(parents=True,exist_ok=True);ledger=Ledger(a.work_dir/'cells');private_dir=a.work_dir/'private_vectors';private_dir.mkdir(exist_ok=True)
    gate=gpu_gate();write_new(a.work_dir/f'gate-{time.time_ns()}.json',gate)
    if gate['status']!='RESOURCE_GATE_PASSED':return 20
    runtime=Runtime(a.snapshot);counts={arm:0 for arm in arms};start=time.perf_counter()
    for item in inputs:
        identity={'token_hash':item['token_hash'],'runtime_hash':runtime_hash,'semantic_hash':file_hash(a.semantic),'stage':a.stage}
        baseline=None
        for arm in arms:
            cell=item['item_id']+'--'+arm;basefile=private_dir/(item['item_id']+'--B.npz')
            existing=ledger.get(cell,{**identity,'arm':arm})
            if existing is not None:
                if arm=='B':baseline=restore_base(basefile)
                counts[arm]+=1;continue
            try:
                result,private=runtime.diagnose(item,'ABCD'.index(gold_rows[item['base_id']]['gold']),arm,semantic,baseline)
                if not result['validity_status']['valid']:raise ValueError('Invalid complete-output evidence')
                result.update({k:item[k] for k in ('item_id','base_id','task','split','length','token_hash')})
                result.update({'evidence_kind':'gpu_measurement','evidence_set':{'dev-b':'DEVELOPMENT','dev-paired':'DEVELOPMENT','pool':'BASELINE_CONDITIONED_POOL','eval':'STANDARD_OR_SELECTED_STRESS','length':'SECONDARY_LENGTH'}[a.stage],
                               'design_hash':design_hash,'manifest_hash':manifest_hash,'runtime_hash':runtime_hash,
                               'model_id':'Qwen/Qwen3-0.6B','model_revision':'c1899de289a04d12100db370d81485cdf75e47ca',
                               'full_vocab_kl_B_to_candidate':kl_logits(baseline['full_logits'],private['full_logits']) if baseline else 0.0})
                if arm=='B':save_base(basefile,private);baseline=private
                else:
                    with (private_dir/(cell+'.npy')).open('xb') as out:np.save(out,private['full_logits'],allow_pickle=False)
                ledger.put(cell,{**identity,'arm':arm},result);counts[arm]+=1
            except Exception as exc:
                write_new(a.work_dir/f'failure-{cell}-{time.time_ns()}.json',{'cell':cell,'stage':a.stage,'type':type(exc).__name__,'message':str(exc),'successful_verdict':False})
                raise
        print(json.dumps({'stage':a.stage,'item':item['item_id'],'counts':counts,'elapsed_seconds':round(time.perf_counter()-start,2)}),flush=True)
    write_new(a.work_dir/f'completion-{time.time_ns()}.json',{'stage':a.stage,'counts':counts,'scenario_count':len({r['base_id'] for r in inputs}),
               'load_seconds':runtime.load_seconds,'elapsed_seconds':time.perf_counter()-start,'manifest_hash':manifest_hash,'design_hash':design_hash})
    return 0


def read_inputs_gold(paths):
    gold={}
    for path in paths:
        for row in load(path):
            if row['base_id'] in gold:raise ValueError('Duplicate gold base')
            gold[row['base_id']]=row
    return gold


if __name__=='__main__':raise SystemExit(main())
