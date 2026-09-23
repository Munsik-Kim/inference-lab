"""Prediction-only sequence execution; gold and first-error history stay outside cache."""
import time
import numpy as np
import torch
from .reference import v1
from .online_v2 import OnlineAdapter

@torch.inference_mode()
def sequence_call(model,table,tokens,adapter,seeds=None,initial=None,include_bos=True,
                  diagnostics=False,readout=None):
    tokens=np.asarray(tokens)
    if tokens.ndim!=2 or tokens.dtype.kind not in 'iu' or np.any(tokens>5):
        raise ValueError('Expected nonnegative S3 group-token matrix')
    if np.any(tokens<0):raise ValueError('Negative token')
    if initial is not None and include_bos:raise ValueError('Resume must not prepend BOS')
    start=time.perf_counter()
    state=adapter.initial(np.zeros((len(tokens),)+adapter.shape,dtype=np.float32),seeds=seeds) if initial is None else initial
    if len(state.payload)!=len(tokens):raise ValueError('Batch shape changed on resume')
    offset=int(include_bos)
    predictions=np.full((len(tokens),tokens.shape[1]+offset),-1,dtype=np.int16)
    invalid_readout=np.zeros(len(tokens),dtype=np.int64)
    active_updates=terminal_noops=0
    read=v1.logits_from_numpy if readout is None else readout
    for pos in range(tokens.shape[1]+offset):
        ids=np.full(len(tokens),6,dtype=np.int64) if include_bos and pos==0 else tokens[:,pos-offset]
        coeff=v1.gather(table,ids)
        before=adapter.terminal_info(state)
        active_updates+=int(before['active'].sum())
        terminal_noops+=int((~before['active']).sum())
        state,represented,_=adapter.step(state,lambda x:v1.torch_transition(x,coeff),diagnostics=diagnostics)
        active=adapter.terminal_info(state)['active']
        if np.any(active):
            logits=read(model,represented,coeff)
            if not isinstance(logits,torch.Tensor) or tuple(logits.shape)!=(len(tokens),6):
                raise ValueError('Entire readout has unexpected type/shape')
            finite=torch.isfinite(logits).all(-1).numpy()
            invalid_readout+=(active & ~finite)
            pred=logits.argmax(-1).numpy()
            pred[~active | ~finite]=-1
            predictions[:,pos]=pred
    info=adapter.terminal_info(state)
    return predictions,state,{'elapsed_seconds':time.perf_counter()-start,
        'completed_writes':predictions.shape[1],
        'active_update_attempts':active_updates,'terminal_noop_steps':terminal_noops,
        'invalid_readout_counts':invalid_readout.tolist(),
        'terminal_codes':info['terminal_code'].tolist(),
        'first_terminal_write':[None if active else int(t) for active,t in zip(info['active'],info['first_terminal_write'])],
        'status':'COMPLETE_WITH_TERMINAL_STREAMS' if np.any(~info['active']) else 'COMPLETE',
        'finite_state_readout_failure_is_terminal':False,'bos_included':include_bos}
