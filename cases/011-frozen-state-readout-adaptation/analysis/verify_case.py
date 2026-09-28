"""Verify immutable design, inputs, selected patch metadata and six recorded cells."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np

CASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CASE))
from source.data import load

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    def bad(x):raise ValueError('Nonfinite JSON '+x)
    return json.loads(Path(path).read_text(),parse_constant=bad)

def verify(root=CASE):
    root=Path(root);checked=0
    for label in ['prefit','pretest']:
        freeze=read(root/f'configs/{label}_freeze.json')
        for rel,h in freeze['files'].items():
            p=root/rel
            if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or digest(p)!=h:
                raise ValueError('Frozen design changed: '+rel)
            checked+=1
    inputs=read(root/'inputs/manifest.json')
    for name in ['smoke','fit','dev','fresh']:load(name)
    selected=read(root/'configs/selected_heads.json')
    if len(selected['heads'])!=6 or selected['actual_fitting_candidates']!=18 or selected['test_outputs_observed']:
        raise ValueError('Invalid selected-head contract')
    for x in selected['heads']:
        p=root/'provenance/head_patches'/(x['patch_directory']+'.json')
        if digest(p)!=x['manifest_sha256']:raise ValueError('Head manifest changed')
        m=read(p)
        if m['payload_bytes']!=4632 or m['additional_recurrent_state_bytes']!=0:
            raise ValueError('Unexpected head footprint')
        if {z['name'] for z in m['tensors']}!={'mlp.2.weight','mlp.2.bias'}:raise ValueError('Unexpected parameter change')
    # Independently recover the FIT-only control without importing its predictor.
    ft,fg,_=load('fit');counts=np.zeros((4,6,6),np.int64)
    for band,(lo,hi) in enumerate([(0,32),(32,64),(64,128),(128,256)]):
        for token in range(6):
            counts[band,token]=np.bincount(fg[:,lo:hi][ft[:,lo:hi]==token],minlength=6)
    if counts.tolist()!=read(root/'results/input_only_fit.json')['counts']:raise ValueError('Control did not use frozen FIT counts')
    cells=[];tokens,gold,identity=load('fresh')
    bands=np.minimum(np.searchsorted([32,64,128],np.arange(1,2049)),3)
    expected_control=counts.argmax(-1)[bands,tokens]
    for seed in range(3):
        for storage in ['NATIVE_FP32','UNIFORM_8']:
            folder=root/'results/fresh'/f'seed{seed}'/storage;receipt=read(folder/'receipt.json')
            if receipt['status']!='COMPLETE' or receipt['resumed_offset']!=0:raise ValueError('Unexpected execution state')
            if digest(folder/'predictions.npz')!=receipt['raw_sha256'] or digest(folder/'summary.json')!=receipt['summary_sha256']:
                raise ValueError('Recorded output changed')
            if receipt['identity']['input_hash']!=identity['tokens_sha256']:raise ValueError('Input pairing mismatch')
            if not all(x['before_heads_sha256']==x['after_heads_sha256'] for x in receipt['cache_unchanged_by_heads']):
                raise ValueError('Head mutation affected cache')
            with np.load(folder/'predictions.npz',allow_pickle=False) as a:
                if not np.array_equal(a['gold'],gold):raise ValueError('Raw labels differ from independent input oracle')
                if a['predictions'].shape!=(3,1024,2048):raise ValueError('Incomplete raw result')
                if not np.array_equal(a['input_only_predictions'],expected_control):raise ValueError('Input control prediction mismatch')
            cells.append({'seed':seed,'storage':storage,'terminal_count':receipt['terminal_count'],'raw_sha256':receipt['raw_sha256']})
    return {'status':'PASS','frozen_file_checks':checked,'input_cohorts':4,'selected_heads':6,
            'complete_recurrent_rollouts':6,'logical_arms':18,'cells':cells,
            'scope':'identity, independent S3 gold, source/input freeze, recorded execution; no model forward'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Choose a new verification output file')
    value=verify();out.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(json.dumps(value))
