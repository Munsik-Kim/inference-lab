"""Identities and read-only loaders for the fixed-checkpoint follow-up."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    0: '2c3a4fb99aa9bf94c14f28ee38c1022e5a6f62dbf896f3dad18bd19e35249535',
    1: '1ebb25155612a66f53a2e59be0dab819360307d02256f55104906579bf40644e',
    2: '381bd6b7f8db4f2416021433423aeba11137d033dfe9bd4216c19b8360fd96d9',
}
ARMS = ['NATIVE_FP32', 'UNIFORM_8', 'UNIFORM_5', 'LOWRANK_4_8_R2', 'MIXED_5_6_BUDGET']
GRID = [32,48,64,80,96,112,128,160,192,256,512,1024,2048]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def file_sha(path):
    return sha(Path(path).read_bytes())

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)+'\n')

def load_cohort(name):
    manifest = json.loads((ROOT/'inputs/manifest.json').read_text())
    item = manifest['cohorts'][name]
    for key in ['tokens', 'gold', 'sample_ids']:
        path = ROOT/item[key+'_file']
        if file_sha(path) != item[key+'_file_sha256' if key != 'sample_ids' else 'sample_ids_sha256']:
            raise ValueError('Input identity mismatch: '+key)
    tokens=np.load(ROOT/item['tokens_file'],allow_pickle=False)
    gold=np.load(ROOT/item['gold_file'],allow_pickle=False)
    ids=json.loads((ROOT/item['sample_ids_file']).read_text())
    if tokens.shape != (item['sequences'],item['group_tokens']) or gold.shape != tokens.shape:
        raise ValueError('Input shape mismatch')
    if tokens.dtype != np.uint8 or gold.dtype != np.uint8 or np.any(tokens>5) or np.any(gold>5):
        raise ValueError('Invalid S3 input/gold bytes')
    if len(ids)!=len(tokens) or len(set(ids))!=len(ids):
        raise ValueError('Invalid sample identities')
    return tokens.astype(np.int64), gold, ids, item

def load_table(seed):
    folder=ROOT/'inputs/v1_calibration'/f'seed{seed}'
    config=(folder/'token_coefficients.json').read_bytes()
    raw=(folder/'token_coefficients.bin').read_bytes()
    cfg=json.loads(config)
    table={}
    for name, entry in cfg['entries'].items():
        array=np.frombuffer(raw[entry['offset']:entry['offset']+entry['bytes']],dtype=entry['dtype']).reshape(entry['shape']).copy()
        if array.dtype!=np.float32 or not np.isfinite(array).all():
            raise ValueError('Expected retained finite FP32 coefficients')
        array.setflags(write=False);table[name]=array
    return table,config,raw

def load_model(upstream, checkpoint, seed):
    from .reference import learned,v1
    if file_sha(checkpoint)!=CHECKPOINTS[seed]:
        raise ValueError('Checkpoint differs from retained seed identity')
    model=learned.create_model(learned.load_upstream(upstream),checkpoint=checkpoint).eval()
    table,config,raw=load_table(seed)
    actual=v1.table_bytes(v1.token_table(model))
    if actual != (config,raw):
        raise ValueError('Actual fixed-checkpoint coefficients differ from retained v1 bytes')
    return model,table

def runtime_config():
    return canonical({'format':'case010-v2-runtime-v1','coordinate_system':'physical',
        'coefficient_dtype':'float32','transition_dtype':'float32','readout':'original_FP32',
        'read_boundary':'after_every_write','bos_policy':'present_write0',
        'prediction_tie':'lowest_label_index','invalid_prediction':-1,
        'terminal_output':'always_invalid','finite_state_nonfinite_readout':'invalid_then_continue'})

def ledger_with_table(adapter,state,seed):
    _,config,raw=load_table(seed)
    ledger=adapter.ledger(state)
    ledger['shared_codec_bytes']=ledger['shared_bytes']
    ledger['shared_token_table_bytes']=len(config)+len(raw)
    ledger['shared_runtime_metadata_bytes']=len(runtime_config())
    ledger['shared_bytes']+=len(config)+len(raw)+len(runtime_config())
    ledger['total_bytes']={str(n):ledger['shared_bytes']+n*adapter.bytes_per_stream for n in (1,16,128)}
    ledger['scope']='actual serialized runtime cache/config/table, common checkpoint separate; evaluator history excluded and separately measured'
    return ledger

def verify_protocol():
    """Verify the pre-result design and all frozen coefficient/input/codec bytes."""
    protocol_path=ROOT/'protocol_v2.json'
    freeze=json.loads((ROOT/'provenance/protocol_freeze.json').read_text())
    if file_sha(protocol_path)!=freeze['protocol_sha256']:raise ValueError('Protocol changed after freeze')
    for path,digest in freeze['frozen_files'].items():
        if file_sha(ROOT/path)!=digest:raise ValueError('Frozen study artifact changed: '+path)
    return json.loads(protocol_path.read_text())
