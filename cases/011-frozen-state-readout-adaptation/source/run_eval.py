"""Six frozen recurrent rollouts, three independent heads sharing each feature."""
from pathlib import Path
import argparse, datetime, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from .data import ROOT, load, file_sha, sha, write_json
from .references import load_verified_model
from .adapter import FrozenRollout
from .head_patch import load_patch
from .controls import InputOnlyPredictor, shuffle_permutation
from .metrics import summarize_predictions, score_logits

HEADS=['ORIGINAL','SHORT_REFIT','MIXED_REFIT']

def verify_freeze():
    freeze=json.loads((ROOT/'configs/pretest_freeze.json').read_text())
    for rel,digest in freeze['files'].items():
        if file_sha(ROOT/rel)!=digest:raise ValueError('Pre-TEST file changed: '+rel)
    return freeze

def save_boundary(private,rollout,pred,shuffled,scores,offset,identity):
    dest=private/f'offset-{offset:04d}'
    if dest.exists():raise ValueError('Cannot replace execution checkpoint')
    partial=private/(dest.name+'.partial');partial.mkdir()
    (partial/'state.bin').write_bytes(rollout.export_bytes())
    np.savez_compressed(partial/'outputs.npz',predictions=pred[:,:,:offset],shuffled_predictions=shuffled[:,:,:offset],
                        **{k:v[:,:offset] for k,v in scores.items()})
    record={'offset':offset,'identity':identity,'files':{p.name:file_sha(p) for p in partial.iterdir()}}
    write_json(partial/'receipt.json',record)
    for name,digest in record['files'].items():
        if file_sha(partial/name)!=digest:raise ValueError('Execution checkpoint checksum failure')
    raw=(partial/'state.bin').read_bytes();rollout.import_bytes(raw,pred.shape[1])
    if rollout.export_bytes()!=raw:raise ValueError('Execution checkpoint byte mismatch')
    os.rename(partial,dest)
    return {'offset':offset,'state_sha256':sha(raw),'state_bytes':len(raw),'receipt_sha256':file_sha(dest/'receipt.json')}

def cell(args,seed,storage,tokens,gold,input_identity,selected,control):
    out=ROOT/'results/fresh'/f'seed{seed}'/storage
    if (out/'receipt.json').exists():raise ValueError('Completed TEST cell cannot be rerun')
    out.mkdir(parents=True,exist_ok=True)
    private=Path(args.private)/'eval'/f'seed{seed}'/storage;private.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(args.checkpoint_root)/f'training-seed{seed}'/'final.pt'
    model,table=load_verified_model(args.upstream,checkpoint,seed)
    weights=[(model.mlp[2].weight.detach(),model.mlp[2].bias.detach())]
    for condition in HEADS[1:]:
        chosen=next(x for x in selected['heads'] if x['model_seed']==seed and x['condition']==condition)
        folder=Path(args.private)/'patches'/chosen['patch_directory']
        if file_sha(folder/'manifest.json')!=chosen['manifest_sha256']:raise ValueError('Selected patch changed')
        patch=load_patch(folder,expected_base_sha256=file_sha(checkpoint))
        weights.append((patch['tensors']['mlp.2.weight'],patch['tensors']['mlp.2.bias']))
    n,t=tokens.shape;rollout=FrozenRollout(model,table,storage,model_seed=seed,batch_size=n)
    pred=np.full((3,n,t),-1,dtype=np.int8);shuffled=pred.copy()
    scores={k:np.zeros((3,t),dtype=np.int64 if k=='score_valid_count' else np.float64)
            for k in ['ce_sum','gold_margin_sum','top_margin_sum','score_valid_count']}
    identity={'seed':seed,'storage':storage,'checkpoint_sha256':file_sha(checkpoint),
       'input_hash':input_identity['tokens_sha256'],'selected_heads_sha256':file_sha(ROOT/'configs/selected_heads.json'),
       'pretest_freeze_sha256':file_sha(ROOT/'configs/pretest_freeze.json')}
    offset=0;boundaries=[];cache_checks=[];active_updates=0;terminal_noops=0;singleton_total=0
    previous=sorted(private.glob('offset-*'))
    previous=[p for p in previous if p.is_dir() and not p.name.endswith('.partial')]
    if previous:
        last=previous[-1];receipt=json.loads((last/'receipt.json').read_text())
        if receipt['identity']!=identity:raise ValueError('Resume identity mismatch')
        for name,digest in receipt['files'].items():
            if file_sha(last/name)!=digest:raise ValueError('Resume checksum mismatch')
        offset=receipt['offset'];rollout.import_bytes((last/'state.bin').read_bytes(),n)
        with np.load(last/'outputs.npz',allow_pickle=False) as z:
            pred[:,:,:offset]=z['predictions'];shuffled[:,:,:offset]=z['shuffled_predictions']
            for k in scores:scores[k][:,:offset]=z[k]
        # Derive consumed updates from frozen terminal metadata rather than losing earlier events.
        info=rollout.adapter.terminal_info(rollout.state)
        writes=offset+1
        active_updates=int(sum(writes if a else min(int(f)+1,writes) for a,f in zip(info['active'],info['first_terminal_write'])))
        terminal_noops=n*writes-active_updates
        for p in previous:
            r=json.loads((p/'receipt.json').read_text());boundaries.append({'offset':r['offset'],'state_sha256':r['files']['state.bin'],'receipt_sha256':file_sha(p/'receipt.json')})
    else:
        step=rollout.step(np.full(n,6,dtype=np.int64));active_updates+=n
    started=time.perf_counter()
    for i in range(offset,t):
        step=rollout.step(tokens[:,i]);active_updates+=int(step.status['active_before'].sum());terminal_noops+=int((~step.status['active_before']).sum())
        check=(i+1 in (32,128,256,512,1024,1536,2048))
        before=sha(rollout.export_bytes()) if check else None
        logits=[]
        for h,(w,b) in enumerate(weights):
            with torch.inference_mode():logit=F.linear(step.phi,w,b).numpy()
            if h==0 and np.any(step.status['active']) and not np.array_equal(logit,step.logits_original.numpy(),equal_nan=True):raise ValueError('Original head boundary changed')
            valid=step.status['active'] & np.isfinite(logit).all(-1)
            pred[h,:,i]=np.where(valid,logit.argmax(-1),-1)
            s=score_logits(logit,gold[:,i]);mask=valid&s['finite']
            scores['score_valid_count'][h,i]=mask.sum()
            for target,key in [('ce_sum','gold_ce'),('gold_margin_sum','gold_margin'),('top_margin_sum','top1_top2_margin')]:
                scores[target][h,i]=s[key][mask].sum()
            logits.append(logit)
        # Linear acts row-wise: permuting its output is exactly the declared phi permutation.
        permutation,ledger=shuffle_permutation(tokens[:,i],i+1)
        shuffled[:,:,i]=pred[:,permutation,i]
        shuffled[:,~step.status['active'],i]=-1
        singleton_total+=ledger['singleton_unchanged_rows']
        if check:
            after=sha(rollout.export_bytes())
            if before!=after:raise ValueError('Head/control evaluation changed persistent cache')
            cache_checks.append({'position':i+1,'before_heads_sha256':before,'after_heads_sha256':after})
        if i+1 in (256,512,1024,1536,2048):
            boundaries.append(save_boundary(private,rollout,pred,shuffled,scores,i+1,identity))
            print(json.dumps({'phase':'TEST','seed':seed,'storage':storage,'offset':i+1,'seconds':time.perf_counter()-started}),flush=True)
    rollout.assert_frozen();info=rollout.adapter.terminal_info(rollout.state)
    ids=json.loads((ROOT/input_identity['sample_ids_path']).read_text())
    np.savez_compressed(out/'predictions.npz',predictions=pred,gold=gold,head_names=np.array(HEADS),
       sample_ids=np.array(ids),input_hash=np.array(input_identity['tokens_sha256']),
       checkpoint_sha256=np.array(file_sha(checkpoint)),checkpoint_seed=np.array(seed),storage=np.array(storage),
       shuffled_predictions=shuffled,input_only_predictions=control.predict(tokens),**scores,
       terminal_code=info['terminal_code'],first_terminal_write=info['first_terminal_write'])
    write_json(out/'summary.json',summarize_predictions(pred,gold))
    write_json(out/'receipt.json',{'status':'COMPLETE','identity':identity,'logical_arms':3,'actual_recurrent_rollouts':1,
      'N':n,'T':t,'resumed_offset':offset,'elapsed_seconds_this_attempt_including_control_and_io':time.perf_counter()-started,
      'active_update_attempts':active_updates,'terminal_noop_steps':terminal_noops,'terminal_count':int((~info['active']).sum()),
      'cache_unchanged_by_heads':cache_checks,'checkpoint_boundaries':boundaries,
      'shuffle_singleton_unchanged_rows_this_attempt':singleton_total,
      'shuffle_implementation':'same-token/position phi permutation evaluated by identical rowwise head-output permutation',
      'source_sha256':file_sha(Path(__file__)),'raw_sha256':file_sha(out/'predictions.npz'),
      'summary_sha256':file_sha(out/'summary.json'),'ledger':rollout.reference.common.ledger_with_table(rollout.adapter,rollout.state,seed),
      'model_parameter_value_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
      'feature_temporary_bytes':n*192*4,'final_cache_sha256':sha(rollout.export_bytes()),
      'head_patch_bytes_separate_from_cache':True,'performance_timing':False})

def main(args):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);verify_freeze()
    selected=json.loads((ROOT/'configs/selected_heads.json').read_text())
    if len(selected['heads'])!=6 or selected['test_outputs_observed'] is not False:raise ValueError('Missing pre-TEST head selection')
    tokens,gold,identity=load('fresh')
    control=InputOnlyPredictor(json.loads((ROOT/'results/input_only_fit.json').read_text())['counts'])
    for seed in range(3):
        for storage in ['NATIVE_FP32','UNIFORM_8']:
            if (ROOT/'results/fresh'/f'seed{seed}'/storage/'receipt.json').exists():continue
            cell(args,seed,storage,tokens,gold,identity,selected,control)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--checkpoint-root',required=True);p.add_argument('--private',required=True);main(p.parse_args())
