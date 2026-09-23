"""Reuse v1 calibration; the new comparator is chosen by bytes, not predictions."""
import json
import numpy as np
from .common import ROOT,ledger_with_table,canonical,write_json,sha
from .online_v2 import OnlineAdapter,PackedCodec,NativeFloatCodec
SHAPE=(12,16,16)

def reference_arms(seed):
    with np.load(ROOT/'inputs/v1_calibration'/f'seed{seed}'/'calibration.npz',allow_pickle=False) as z:
        cal={k:z[k].copy() for k in z.files}
    arms={'NATIVE_FP32':OnlineAdapter(NativeFloatCodec(SHAPE)),
          'UNIFORM_8':OnlineAdapter(PackedCodec(SHAPE,bits=8)),
          'UNIFORM_5':OnlineAdapter(PackedCodec(SHAPE,bits=5)),
          'LOWRANK_4_8_R2':OnlineAdapter(PackedCodec(SHAPE,bits=4),PackedCodec((12,2,16),bits=8),cal['basis'][:,:,:2])}
    return arms,cal

def select_mixed_budget(seed):
    arms,cal=reference_arms(seed)
    zero=np.zeros((1,)+SHAPE,dtype=np.float32)
    target=arms['LOWRANK_4_8_R2'];cap=ledger_with_table(target,target.initial(zero),seed)['total_bytes']['128']
    scores=cal['mixed_scores'].reshape(-1)
    if scores.shape!=(192,) or not np.isfinite(scores).all():raise ValueError('Malformed retained CAL importance')
    order=np.argsort(-scores,kind='stable')
    bits=np.full((12,16),5,dtype=np.uint8);accepted=[];attempts=[]
    def candidate(mapping):return OnlineAdapter(PackedCodec(SHAPE,bits=5,mixed_bits=mapping))
    selected=candidate(bits)
    if ledger_with_table(selected,selected.initial(zero),seed)['total_bytes']['128']>cap:
        raise ValueError('Even all5-bit mapped comparator exceeds cap')
    for idx in order:
        next_bits=bits.copy();next_bits.flat[int(idx)]=6
        proposed=candidate(next_bits);ledger=ledger_with_table(proposed,proposed.initial(zero),seed)
        fits=ledger['total_bytes']['128']<=cap
        attempts.append({'flat_channel':int(idx),'head':int(idx//16),'key_channel':int(idx%16),
                         'CAL_score':float(scores[idx]),'total_bytes_N128':ledger['total_bytes']['128'],'accepted':fits})
        if not fits:break
        bits=next_bits;selected=proposed;accepted.append(int(idx))
    arms['MIXED_5_6_BUDGET']=selected
    return arms,{'rule':'global descending retained CAL mixed_scores; stable flattened head/key index ties; 5->6 bits until next real serialized allocation exceeds N128 rank2 cap',
        'seed':seed,'rank2_N128_cap':cap,'accepted_channels':accepted,'attempts':attempts,
        'bits':bits.tolist(),'no_promotions_duplicate_all5':not accepted,'outcomes_accessed':False,
        'ledgers':{k:ledger_with_table(v,v.initial(zero),seed) for k,v in arms.items()}}

def load_arms(seed):
    """Use pre-written frozen config bytes, never reselect during evaluation."""
    root=ROOT/'inputs/codecs'/f'seed{seed}'
    receipt=json.loads((root/'selection.json').read_text())
    result={}
    for name,item in receipt['codec_files'].items():
        config=(root/item['config']).read_bytes();basis=(root/item['basis']).read_bytes()
        if sha(config)!=item['config_sha256'] or sha(basis)!=item['basis_sha256']:
            raise ValueError('Frozen codec identity mismatch')
        result[name]=OnlineAdapter.from_shared(config,basis)
    return result,receipt

def freeze_arms(seed):
    arms,receipt=select_mixed_budget(seed)
    root=ROOT/'inputs/codecs'/f'seed{seed}';root.mkdir(parents=True,exist_ok=False)
    receipt['codec_files']={}
    for name,adapter in arms.items():
        (root/f'{name}.json').write_bytes(adapter.config_bytes);(root/f'{name}.basis.bin').write_bytes(adapter.basis_bytes)
        receipt['codec_files'][name]={'config':f'{name}.json','basis':f'{name}.basis.bin',
            'config_sha256':sha(adapter.config_bytes),'basis_sha256':sha(adapter.basis_bytes)}
    write_json(root/'selection.json',receipt)
    return receipt
