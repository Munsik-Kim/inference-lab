"""Small synthetic CPU demonstration of active/terminal stochastic restart."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from verify_unified import CASE, verify_snapshots

def run(case=CASE):
    import numpy as np
    case=Path(case).resolve()
    verify_snapshots(case)
    root=case/'versions/v2'
    sys.path.insert(0,str(root))
    from source.online_v2 import OnlineAdapter, PackedCodec
    adapter=OnlineAdapter(PackedCodec((1,2,2),4,stochastic=True))
    initial=adapter.initial(np.zeros((2,1,2,2),np.float32),seeds=np.array([15,71],np.uint64))
    increments=np.full((6,2,1,2,2),.13,np.float32)
    increments[1,1]=np.inf  # Injected numerical fault; not a measured CKDA event.
    state=initial; reads=[]; split=None
    for step in range(6):
        state,read,_=adapter.step(state,lambda x:x*np.float32(.75)+increments[step],diagnostics=False)
        reads.append(read)
        if step==2: split=state
    with tempfile.TemporaryDirectory(prefix='case010-synthetic-restart-') as tmp:
        tmp=Path(tmp); src=tmp/'input';src.mkdir()
        (src/'config.json').write_bytes(adapter.config_bytes)
        (src/'basis.bin').write_bytes(adapter.basis_bytes)
        (src/'state.bin').write_bytes(split.payload.tobytes())
        np.savez(src/'operators.npz',increments=increments)
        child=subprocess.run([sys.executable,'-B',str(root/'tests/restart_online_v2_child.py'),'--input',str(src),'--output',str(tmp/'out'),'--stop','6'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
        if child.returncode:
            raise RuntimeError(f'synthetic child failed: {child.stderr}')
        same=(tmp/'out/state.bin').read_bytes()==state.payload.tobytes()
        suffix=np.array_equal(np.load(tmp/'out/reads.npy',allow_pickle=False),np.asarray(reads[3:]))
        child_receipt=json.loads((tmp/'out/receipt.json').read_text())
        if not same or not suffix or child_receipt['torch_loaded']:
            raise ValueError('fresh-process restart mismatch')
    info=adapter.terminal_info(state)
    return {'status':'PASS','evidence_kind':'SYNTHETIC_CONTRACT_TEST','trained_model_executed':False,
            'active_streams':int(info['active'].sum()),'terminal_streams':int((~info['active']).sum()),
            'cursor':info['cursor'].tolist(),'terminal_code':info['terminal_code'].tolist(),
            'first_terminal_write':[None if a else int(t) for a,t in zip(info['active'],info['first_terminal_write'])],
            'final_bytes_equal':same,'represented_suffix_equal':suffix,'torch_loaded_in_child':child_receipt['torch_loaded'],
            'stream_bytes':adapter.bytes_per_stream,'final_payload_sha256':hashlib.sha256(state.payload.tobytes()).hexdigest(),
            'note':'Zero scratch for a terminal stream is not a valid prediction. This demo executes synthetic transitions, not a learned model.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=CASE);p.add_argument('--output',type=Path)
    a=p.parse_args(); result=run(a.case)
    if a.output:
        with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
