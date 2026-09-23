"""Serial complete-call CPU cost; run only after other study processes stop."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,ARMS,CHECKPOINTS,load_model,load_cohort,verify_protocol,write_json,file_sha
from .arms import load_arms
from .evaluation import sequence_call

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['upstream','checkpoint','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
 torch.set_num_threads(2);verify_protocol();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 model,table=load_model(a.upstream,a.checkpoint,a.seed);tokens,_,ids,item=load_cohort('timing');arms,_=load_arms(a.seed)
 rows=[]
 # Loading is outside the timer. One declared short warmup per arm, original inputs.
 for name in ARMS:sequence_call(model,table,tokens[:,:4],arms[name],diagnostics=False)
 for repeat in range(3):
  for name in ARMS:
   predictions,state,e=sequence_call(model,table,tokens,arms[name],diagnostics=False)
   rows.append({'arm':name,'repeat':repeat,'seconds':e['elapsed_seconds'],
     'ms_per_group_token':e['elapsed_seconds']*1000/(16*128),'active_updates':e['active_update_attempts'],
     'terminal_noops':e['terminal_noop_steps'],'invalid_readout_tokens':sum(e['invalid_readout_counts']),
     'terminal_streams':sum(x!=0 for x in e['terminal_codes'])})
   print(json.dumps(rows[-1]),flush=True)
 write_json(out/'timing.json',{'kind':'CPU_COMPLETE_CALL_REMEASUREMENT','model_seed':a.seed,'checkpoint_sha256':CHECKPOINTS[a.seed],
  'protocol_sha256':file_sha(ROOT/'protocol_v2.json'),'input_hash':item['tokens_file_sha256'],'N':16,'T':128,'generator_seed':2002,'threads':2,
  'timed':'transition + reconstruction/projection + packing + failure status + original readout; BOS included in call, denominator group tokens only',
  'excluded':'model/table loading, gold/shadow/diagnostics, disk I/O','warmup':'one 4-group-token call per arm',
  'repeats':3,'rows':rows,'scope':'Python/NumPy/PyTorch CPU reference prototype; not single-request latency or GPU speed',
  'temporary_memory':'intermediate floating arrays are transient; no RAM peak measurement performed; not serialized cache savings'})
if __name__=='__main__':main()
