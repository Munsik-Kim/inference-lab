"""Atomic evaluation-job checkpoints, separate from charged runtime cache bytes.

Same-filesystem rename provides atomic visibility, not a claim of fsync-based
crash durability. Failed partial directories are retained for inspection.
"""
from pathlib import Path,PurePosixPath
import json
import os
import uuid
import numpy as np
from .common import canonical,sha,file_sha,write_json

def _history(predictions,gold):
    correct=predictions[:,1:]==gold
    bad=~correct
    first=bad.argmax(1)+1 if correct.shape[1] else np.zeros(len(correct),dtype=int)
    return correct,[int(t) if fail else None for t,fail in zip(first,bad.any(1))]

def save_checkpoint(target,adapter,state,*,identity,predictions,gold,sample_ids,counts):
    target=Path(target)
    if target.exists():raise ValueError('Never overwrite a completed checkpoint')
    pred=np.asarray(predictions)
    if pred.ndim!=2 or pred.dtype.kind not in 'iu' or np.any((pred<-1)|(pred>5)):
        raise ValueError('Malformed prediction history')
    if pred.shape[1]<1 or np.asarray(gold).shape!=(len(pred),pred.shape[1]-1):
        raise ValueError('Checkpoint must include exactly one consumed BOS and matching gold prefix')
    if len(sample_ids)!=len(pred) or len(set(sample_ids))!=len(pred):raise ValueError('Duplicate/missing sample ID')
    info=adapter.terminal_info(state)
    if not np.all(info['cursor']==pred.shape[1]):raise ValueError('Cursor/output boundary mismatch')
    correct,taus=_history(pred,gold)
    partial=target.with_name(target.name+'.partial-'+uuid.uuid4().hex)
    partial.mkdir(parents=True,exist_ok=False)
    (partial/'runtime.bin').write_bytes(state.payload.tobytes())
    (partial/'codec.json').write_bytes(adapter.config_bytes)
    (partial/'basis.bin').write_bytes(adapter.basis_bytes)
    np.savez_compressed(partial/'history.npz',predictions=pred.astype('i1'),
        correctness=np.packbits(correct,axis=1,bitorder='little'),gold=np.asarray(gold,dtype='u1'))
    record={'schema':'ckda-v2-evaluation-checkpoint-v1','identity':identity,'sample_ids':list(sample_ids),
        'sequences':len(pred),'scored_offset':pred.shape[1]-1,'bos_consumed':True,
        'tau':taus,'counts':counts,'runtime_config_sha256':sha(adapter.config_bytes),
        'basis_sha256':sha(adapter.basis_bytes),'runtime_bytes':state.payload.nbytes,
        'evaluator_disk_bytes_not_model_cache':True}
    write_json(partial/'record.json',record)
    inventory={p.name:{'bytes':p.stat().st_size,'sha256':file_sha(p)} for p in sorted(partial.iterdir())}
    write_json(partial/'inventory.json',inventory)
    # The same validation used for external resume checks every written byte.
    load_checkpoint(partial,expected_identity=identity,expected_sample_ids=sample_ids)
    if target.exists():raise ValueError('Concurrent checkpoint destination appeared')
    os.rename(partial,target)
    return {'path_name':target.name,'runtime_bytes':state.payload.nbytes,
        'evaluation_checkpoint_disk_bytes':sum(p.stat().st_size for p in target.iterdir()),
        'inventory_sha256':file_sha(target/'inventory.json'),'atomic_visibility':'same-filesystem rename; no fsync durability claim'}

def load_checkpoint(folder,*,expected_identity,expected_sample_ids):
    from .online_v2 import OnlineAdapter
    folder=Path(folder)
    inventory=json.loads((folder/'inventory.json').read_text())
    expected={'runtime.bin','codec.json','basis.bin','history.npz','record.json'}
    if set(inventory)!=expected or {p.name for p in folder.iterdir()}!=expected|{'inventory.json'}:
        raise ValueError('Checkpoint exact inventory mismatch')
    for name,entry in inventory.items():
        if PurePosixPath(name).name!=name or (folder/name).is_symlink():raise ValueError('Unsafe checkpoint entry')
        raw=(folder/name).read_bytes()
        if len(raw)!=entry['bytes'] or sha(raw)!=entry['sha256']:raise ValueError('Checkpoint checksum mismatch')
    record=json.loads((folder/'record.json').read_text())
    if record['schema']!='ckda-v2-evaluation-checkpoint-v1' or record['identity']!=expected_identity:
        raise ValueError('Checkpoint study/input/checkpoint/runtime identity mismatch')
    if record['sample_ids']!=list(expected_sample_ids) or not record['bos_consumed']:
        raise ValueError('Checkpoint sample order or BOS policy mismatch')
    config=(folder/'codec.json').read_bytes();basis=(folder/'basis.bin').read_bytes()
    if sha(config)!=record['runtime_config_sha256'] or sha(basis)!=record['basis_sha256']:
        raise ValueError('Checkpoint codec identity mismatch')
    adapter=OnlineAdapter.from_shared(config,basis)
    state=adapter.from_bytes((folder/'runtime.bin').read_bytes(),batch_size=record['sequences'])
    with np.load(folder/'history.npz',allow_pickle=False) as z:
        if set(z.files)!={'predictions','correctness','gold'}:raise ValueError('Unexpected evaluator arrays')
        pred=z['predictions'].copy();gold=z['gold'].copy();bits=z['correctness'].copy()
    n,t=record['sequences'],record['scored_offset']
    if pred.shape!=(n,t+1) or gold.shape!=(n,t) or bits.shape!=(n,(t+7)//8):raise ValueError('Checkpoint history shape mismatch')
    if pred.dtype!=np.int8 or gold.dtype!=np.uint8 or bits.dtype!=np.uint8 or np.any((pred<-1)|(pred>5)) or np.any(gold>5):raise ValueError('Invalid history values')
    correct,taus=_history(pred,gold)
    if not np.array_equal(np.packbits(correct,axis=1,bitorder='little'),bits) or taus!=record['tau']:
        raise ValueError('Earlier first-failure history lost or corrupted')
    info=adapter.terminal_info(state)
    if not np.all(info['cursor']==t+1):raise ValueError('Cursor/history boundary mismatch')
    for row in np.flatnonzero(~info['active']):
        first=int(info['first_terminal_write'][row])
        if not np.all(pred[row,first:]==-1):raise ValueError('Terminal output history contains revival')
    counts=record['counts']
    invalid=counts['invalid_readout_counts']
    if len(invalid)!=n or any(type(x)!=int or not 0<=x<=t+1 for x in invalid):raise ValueError('Invalid cumulative readout counts')
    if any(type(counts.get(k))!=int or counts[k]<0 for k in ['active_update_attempts','terminal_noop_steps']):raise ValueError('Invalid runtime counts')
    if counts['active_update_attempts']+counts['terminal_noop_steps']!=n*(t+1):raise ValueError('Lost consumed-step history')
    return adapter,state,pred,gold,record
