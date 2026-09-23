"""One predeclared seed/arm on all 1024 fresh sequences; checkpointed chunks."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,ARMS,CHECKPOINTS,load_model,load_cohort,file_sha,sha,write_json,ledger_with_table,verify_protocol
from .arms import load_arms
from .evaluation import sequence_call
from .checkpoint import save_checkpoint,load_checkpoint
from .metrics import save_result

def run(args):
    torch.set_num_threads(2)
    protocol=verify_protocol()
    tokens,gold,ids,item=load_cohort('fresh')
    model,table=load_model(args.upstream,args.checkpoint,args.seed)
    arms,receipt=load_arms(args.seed);adapter=arms[args.arm]
    identity={'kind':'FRESH_TEST_FIXED_CHECKPOINTS','model_seed':args.seed,'arm':args.arm,'cohort':'fresh',
      'checkpoint_sha256':CHECKPOINTS[args.seed],'input_manifest_sha256':file_sha(ROOT/'inputs/manifest.json'),
      'sample_ids_sha256':item['sample_ids_sha256'],'tokens_sha256':item['tokens_file_sha256'],
      'protocol_sha256':file_sha(ROOT/'protocol_v2.json'),'family_size':195,'codec_config_sha256':sha(adapter.config_bytes)}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Completed cell cannot be rerun/overwritten')
    private=Path(args.private_output);private.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();state=None;pred=np.empty((len(tokens),0),dtype=np.int16);offset=0
    counts={'active_update_attempts':0,'terminal_noop_steps':0,'invalid_readout_counts':[0]*len(tokens)}
    checkpoints=[];execution_seconds=0
    if args.resume:
        restored,state,pred,oldgold,rec=load_checkpoint(args.resume,expected_identity=identity,expected_sample_ids=ids)
        if restored.config_bytes!=adapter.config_bytes or restored.basis_bytes!=adapter.basis_bytes:raise ValueError('Resume codec differs from frozen candidate')
        offset=rec['scored_offset'];counts=rec['counts']
        if not np.array_equal(oldgold,gold[:,:offset]):raise ValueError('Resume input history mismatch')
    for end in [256,512,1024,1536,2048]:
        if end<=offset:continue
        if args.deadline_epoch and time.time()>=args.deadline_epoch:
            write_json(out/'partial.json',{'status':'PARTIAL_EXECUTION','completed_group_tokens':offset,'N':len(tokens),'identity':identity,'checkpoint_names':checkpoints});return
        p,state,e=sequence_call(model,table,tokens[:,offset:end],adapter,initial=state,include_bos=(state is None))
        pred=np.concatenate([pred,p],axis=1);execution_seconds+=e['elapsed_seconds']
        for key in ['active_update_attempts','terminal_noop_steps']:counts[key]+=e[key]
        counts['invalid_readout_counts']=(np.array(counts['invalid_readout_counts'])+e['invalid_readout_counts']).tolist()
        path=private/f'offset-{end:04d}'
        r=save_checkpoint(path,adapter,state,identity=identity,predictions=pred,gold=gold[:,:end],sample_ids=ids,counts=counts)
        restored,state,pred,_,_=load_checkpoint(path,expected_identity=identity,expected_sample_ids=ids)
        checkpoints.append(r);offset=end
        print(json.dumps({'arm':args.arm,'seed':args.seed,'offset':offset,'elapsed_seconds':time.perf_counter()-start}),flush=True)
    info=adapter.terminal_info(state)
    initial=adapter.initial(np.zeros((1,12,16,16),dtype=np.float32))
    (out/'initial_payload.bin').write_bytes(initial.payload.tobytes())
    cfg=ROOT/'inputs/codecs'/f'seed{args.seed}';cal=ROOT/'inputs/v1_calibration'/f'seed{args.seed}'
    files={'initial_payload':out/'initial_payload.bin','codec_config':cfg/f'{args.arm}.json','basis':cfg/f'{args.arm}.basis.bin',
      'token_config':cal/'token_coefficients.json','token_data':cal/'token_coefficients.bin','runtime_config':ROOT/'inputs/runtime.json'}
    shared={k:{'path':str(p.resolve().relative_to(ROOT.resolve())),'bytes':p.stat().st_size,'sha256':file_sha(p)} for k,p in files.items()}
    execution={**counts,'status':'COMPLETE_WITH_TERMINAL_STREAMS' if np.any(~info['active']) else 'COMPLETE',
      'elapsed_seconds':execution_seconds,'wall_seconds_including_evaluation_io':time.perf_counter()-start,
      'complete_group_tokens':offset,'completed_writes':offset+1,'resume_from_existing_job':bool(args.resume),'new_training_updates':0}
    first=[None if active else int(x) for active,x in zip(info['active'],info['first_terminal_write'])]
    summary=save_result(out,predictions=pred,gold=gold,identity=identity,execution=execution,
      ledger=ledger_with_table(adapter,state,args.seed),first_terminal=first,terminal_codes=info['terminal_code'].tolist(),
      extra={'shared_artifacts':shared,'evaluation_checkpoint_receipts':checkpoints,'final_payload_sha256':sha(state.payload.tobytes()),
        'runtime_payload_not_public':'Only initial zero payload and final hash retained publicly; actual resumable job checkpoints are private.'})
    print(json.dumps({'status':execution['status'],'arm':args.arm,'seed':args.seed,'RMST0':summary['RMST0'],'terminal_count':summary['terminal_count']}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--upstream',required=True);p.add_argument('--output',required=True);p.add_argument('--private-output',required=True)
    p.add_argument('--resume');p.add_argument('--deadline-epoch',type=float);run(p.parse_args())
if __name__=='__main__':main()
