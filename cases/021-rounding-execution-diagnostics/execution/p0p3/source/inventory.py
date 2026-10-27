"""Read only selected existing artifacts and checkpoint; never downloads."""
import argparse
import json
import platform
import sys
from pathlib import Path
import numpy as np
from common import CASE, file_row, write, utc, byte_sha
from layout import load_rpi4

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); p.add_argument('--private',required=True); a=p.parse_args()
    run=Path(a.run); old=json.loads((run/'artifacts/06b/D16/manifest.json').read_text()); model=Path(old['model']['path'])
    items=[]
    for row in old['files']:
        path=run/'artifacts/06b/D16'/(row['layer']+'.rpi4')
        actual=file_row(path)
        if actual!={k:row[k] for k in ('bytes','sha256')}: raise ValueError('ARTIFACT_IDENTITY_MISMATCH:'+str(path))
        q=load_rpi4(path)
        if list(q.shape)!=row['shape']: raise ValueError('ARTIFACT_SHAPE_MISMATCH')
        arrays={}
        for name in ('codes','gamma','alpha_inv','beta_inv'):
            v=getattr(q,name); arrays[name]={'dtype':str(v.dtype),'shape':list(v.shape),'bytes':v.nbytes,'sha256':byte_sha(v.tobytes()),'min':float(v.min()),'max':float(v.max()),'finite':bool(np.isfinite(v).all())}
        items.append({'projection':row['layer'],'path':str(path),**actual,'shape':list(q.shape),'method':q.method,'iterations':q.iterations,'arrays':arrays})
    checkpoints={f.name:{'path':str(f),'resolved_path':str(f.resolve()),**file_row(f)} for f in model.iterdir() if f.is_file() and (f.suffix in ('.json','.safetensors') or f.name=='LICENSE')}
    # Safetensors header only: validates projection bias inventory without loading weights.
    with (model/'model.safetensors').open('rb') as f:
        length=int.from_bytes(f.read(8),'little'); header=json.loads(f.read(length))
    bias=[]
    for row in items:
        key=row['projection']+'.bias'; desc=header.get(key)
        bias.append({'projection':row['projection'],'bias':'NONE' if desc is None else desc,'weight_header':header[row['projection']+'.weight']})
    historical=run/'inputs/wiki_06b.npy'; ids=np.load(historical,allow_pickle=False)[0,:32].astype(np.int64)
    dest=Path(a.private)/'historical_ids.npy'; np.save(dest,ids,allow_pickle=False)
    result={'created_utc':utc(),'artifact_status':'VERIFIED','historical_manifest':{'path':str(run/'artifacts/06b/D16/manifest.json'),**file_row(run/'artifacts/06b/D16/manifest.json')},'model':old['model']['id'],'revision':old['model']['revision'],'model_files':checkpoints,'weight_hash_scope':'All bytes of the one actually used safetensors shard; no other model hashed','projection_count':len(items),'artifact_total_bytes':sum(x['bytes'] for x in items),'artifacts':items,'bias_inventory':bias,'historical_input':{'source':str(historical),**file_row(historical),'selection':'row0[:32]','ids_sha256':byte_sha(ids.tobytes()),'private_ids_path':str(dest)},'cpu_inventory_runtime':{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'numpy_path':np.__file__,'platform':platform.platform()}}
    write(CASE/'artifact_manifest.json',result)
    write(Path(a.private)/'locations.json',{'run':str(run),'artifact_dir':str(run/'artifacts/06b/D16'),'model':str(model),'historical_ids':str(dest)})
    print(json.dumps({'status':'VERIFIED','count':len(items),'bytes':result['artifact_total_bytes'],'bias_count':sum(x['bias']!='NONE' for x in bias),'weight_bytes':checkpoints['model.safetensors']['bytes']}))

if __name__=='__main__': main()
