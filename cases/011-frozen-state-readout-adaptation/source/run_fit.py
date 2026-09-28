"""Extract frozen Native features, fit 18 bounded heads, freeze six DEV selections."""
from pathlib import Path
import argparse, datetime, json, time
import numpy as np
import torch
from .data import ROOT, load, sha, file_sha, write_json
from .adapter import FrozenRollout, FEATURE_BOUNDARY
from .references import load_verified_model
from .fitting import fit_head, select_candidate, equal_band_ce, LAMBDAS
from .head_patch import save_patch
from .controls import InputOnlyPredictor

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def extract(model, table, seed, role, private):
    tokens,gold,identity=load(role)
    destination=private/f'seed{seed}-{role}.npz'
    receipt=ROOT/'results'/f'seed{seed}-{role}-features.json'
    if destination.exists():
        r=json.loads(receipt.read_text())
        if file_sha(destination)!=r['private_features_sha256']:raise ValueError('Feature cache changed')
        with np.load(destination,allow_pickle=False) as x:return {k:x[k] for k in x.files}
    n,t=tokens.shape; rollout=FrozenRollout(model,table,'NATIVE_FP32',model_seed=seed,batch_size=n)
    rollout.step(np.full(n,6,dtype=np.int64))
    if role=='fit':
        positions=np.load(ROOT/'inputs/mixed_positions.npy',allow_pickle=False)
        xshort=np.empty((n,32,192),np.float32); xmixed=np.empty_like(xshort)
    else: features=np.empty((n,t,192),np.float32)
    started=time.perf_counter()
    for i in range(t):
        step=rollout.step(tokens[:,i])
        if not step.status['active'].all() or not step.status['finite_features'].all():
            raise ValueError(f'Nonfinite or terminal {role} feature at seed{seed},position{i+1}; no row dropped')
        phi=step.phi.numpy()
        if role=='fit':
            if i<32:xshort[:,i]=phi
            rows,columns=np.where(positions==i+1);xmixed[rows,columns]=phi[rows]
        else:features[:,i]=phi
    elapsed=time.perf_counter()-started;rollout.assert_frozen()
    if role=='fit':
        values={'short_phi':xshort.reshape(-1,192),'short_gold':gold[:,:32].reshape(-1),
                'mixed_phi':xmixed.reshape(-1,192),
                'mixed_gold':np.take_along_axis(gold,positions-1,axis=1).reshape(-1)}
    else:values={'phi':features.reshape(-1,192),'gold':gold.reshape(-1),'positions':np.tile(np.arange(1,t+1),n)}
    np.savez(destination,**values)
    write_json(receipt,{'role':role,'model_seed':seed,'input_sha256':identity['tokens_sha256'],
      'sequences':n,'group_tokens_per_sequence':t,'recurrence_group_tokens':n*t,'bos_writes':n,
      'elapsed_seconds':elapsed,'selected_rows_per_condition':16384 if role=='fit' else n*t,
      'private_features_sha256':file_sha(destination),'private_feature_file_bytes':destination.stat().st_size,
      'feature_boundary':FEATURE_BOUNDARY,'frozen_parameter_check':'PASS','terminal_count':0,
      'short_is_prefix_of_same_fit_rollout':role=='fit','source_code_sha256':file_sha(Path(__file__))})
    print(json.dumps({'phase':'features','seed':seed,'role':role,'seconds':elapsed}),flush=True)
    return values

def main(args):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    private=Path(args.private);private.mkdir(exist_ok=True)
    patches=private/'patches';patches.mkdir(exist_ok=True)
    final=ROOT/'configs/selected_heads.json'
    if final.exists():raise ValueError('Selections already frozen; no new fitting allowed')
    protocol=ROOT/'configs/protocol.json'
    freeze=json.loads((ROOT/'configs/prefit_freeze.json').read_text())
    for rel,digest in freeze['files'].items():
        if file_sha(ROOT/rel)!=digest:raise ValueError('Pre-FIT file changed: '+rel)
    fit_tokens,fit_gold,_=load('fit')
    control=InputOnlyPredictor.fit(fit_tokens,fit_gold)
    write_json(ROOT/'results/input_only_fit.json',control.to_dict())
    selected=[]
    for seed in range(3):
        checkpoint=Path(args.checkpoint_root)/f'training-seed{seed}'/'final.pt'
        model,table=load_verified_model(args.upstream,checkpoint,seed)
        f=extract(model,table,seed,'fit',private);d=extract(model,table,seed,'dev',private)
        w0=model.mlp[2].weight.detach().clone();b0=model.mlp[2].bias.detach().clone()
        original=equal_band_ce(d['phi'],d['gold'],d['positions'],w0,b0)
        for condition,prefix in [('SHORT_REFIT','short'),('MIXED_REFIT','mixed')]:
            out=ROOT/'results'/f'fit-seed{seed}-{condition}.json'
            if out.exists():
                record=json.loads(out.read_text());selected.append(record['selected']);continue
            candidates=[]
            for lam in LAMBDAS:
                c=fit_head(f[prefix+'_phi'],f[prefix+'_gold'],w0,b0,lam)
                candidates.append(c)
                print(json.dumps({'phase':'fit','seed':seed,'condition':condition,'lambda':lam,
                  'status':c['solver']['status'],'iterations':c['solver']['iterations'],
                  'seconds':c['solver']['elapsed_seconds']}),flush=True)
            pick=select_candidate(candidates,d['phi'],d['gold'],d['positions'])
            chosen=candidates[pick['candidate_index']]
            name=f'seed{seed}-{condition}'
            metadata={'base_checkpoint_sha256':file_sha(checkpoint),
               'fit_input_sha256':json.loads((ROOT/'inputs/manifest.json').read_text())['cohorts']['fit']['tokens_sha256'],
               'dev_input_sha256':json.loads((ROOT/'inputs/manifest.json').read_text())['cohorts']['dev']['tokens_sha256'],
               'code_sha256':file_sha(ROOT/'source/fitting.py'),'feature_boundary':FEATURE_BOUNDARY,
               'solver':chosen['solver'],'regularization':pick['regularization'],
               'condition':condition,'model_seed':seed,'protocol_sha256':file_sha(protocol)}
            saved=save_patch(patches/name,chosen['weight'],chosen['bias'],metadata)
            item={'model_seed':seed,'condition':condition,'patch_directory':name,
                  'regularization':pick['regularization'],'base_checkpoint_sha256':file_sha(checkpoint),**saved}
            write_json(out,{'original_dev':original,'selection':pick,'selected':item,
                    'solvers':[x['solver'] for x in candidates]})
            selected.append(item)
    write_json(final,{'schema':'case011-selected-heads-v1','frozen_utc':utc(),
      'test_outputs_observed':False,'max_fitting_candidates':18,'actual_fitting_candidates':18,
      'input_manifest_sha256':file_sha(ROOT/'inputs/manifest.json'),
      'protocol_sha256':file_sha(protocol),'heads':selected})
    print(json.dumps({'phase':'SELECTED_HEADS_FROZEN','sha256':file_sha(final)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--checkpoint-root',required=True);p.add_argument('--private',required=True)
    main(p.parse_args())
