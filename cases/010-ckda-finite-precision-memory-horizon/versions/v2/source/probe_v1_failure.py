"""Independent synthetic reproduction of the v1 lost-failure-mask contract.

This is derived from the user-specified issue, not the unavailable reviewer
probe and not a measured CKDA numerical-failure frequency.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

if __package__ is None:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source.reference import v1
from source.common import write_json,sha

def probe():
    old_transition,old_logits=v1.torch_transition,v1.logits_from_numpy
    def injected(state,coeff):
        out=state+np.float32(1)
        out[coeff['trigger']==0]=np.nan
        return out
    def logits(model,state,coeff):
        return torch.zeros((len(state),6),dtype=torch.float32)
    try:
        v1.torch_transition=injected;v1.logits_from_numpy=logits
        adapter=v1.OnlineAdapter(v1.PackedCodec(v1.SHAPE,bits=4))
        tokens=np.array([[0,1,1,1],[1,1,1,1]],dtype=np.int64)
        table={'trigger':np.arange(7)}
        full,end,full_run=v1.sequence_call(None,table,tokens,adapter,include_bos=False,diagnostics=False)
        first,prefix,first_run=v1.sequence_call(None,table,tokens[:,:1],adapter,include_bos=False,diagnostics=False)
        restored=adapter.from_bytes(prefix.payload.tobytes(),batch_size=2)
        suffix,split,split_run=v1.sequence_call(None,table,tokens[:,1:],adapter,initial=restored,include_bos=False,diagnostics=False)
        return {'kind':'SYNTHETIC_CONTRACT_REPRODUCTION_FROM_REQUEST','reviewer_probe_available':False,
            'v1_failure_reproduced':bool(not np.array_equal(full[:,1:],suffix)),
            'uninterrupted_predictions':full.tolist(),'resumed_suffix_predictions':suffix.tolist(),
            'suffix_matches':bool(np.array_equal(full[:,1:],suffix)),
            'final_payload_matches':end.payload.tobytes()==split.payload.tobytes(),
            'healthy_row_predictions_match':bool(np.array_equal(full[1,1:],suffix[1])),
            'prefix_payload_sha256':sha(prefix.payload.tobytes()),
            'full_absorbed_sequences':full_run['absorbed_sequences'],
            'resumed_absorbed_sequences':split_run['absorbed_sequences'],
            'interpretation':'v1 sequence_call absorbed mask is call-local and absent from serialized OnlineState; not a CKDA model score'}
    finally:
        v1.torch_transition=old_transition;v1.logits_from_numpy=old_logits

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output must be new')
    result=probe();write_json(a.output,result);print(result)
