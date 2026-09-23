"""Fixed historical v1/v2 and serialized-continuation diagnostics.

TEST work requires the frozen v2 protocol and an explicit launcher invocation.
--smoke-dev is restricted to seed0 native DEV2001 B16/T32. It is a technical
gate, not a TEST experiment or reproduction of the unavailable reviewer probe.
Only compact predictions and receipts are public; runtime bodies stay private.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__ is None:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source import common

HISTORICAL_ARMS = ("NATIVE_FP32","UNIFORM_8","UNIFORM_5","LOWRANK_4_8_R2")
EXTRA_ARMS = ("STOCHASTIC_8","FULL_RESIDUAL_4_4","LOWRANK_4_8_R1")
INVALID_CASES = {0:("UNIFORM_2","UNIFORM_3"),1:("UNIFORM_2",),2:("UNIFORM_2",)}
BOUNDARIES = (1,17,63,256,1024,2048)


class DeadlineReached(RuntimeError):
    pass


def check_deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise DeadlineReached("historical diagnostic deadline reached")


def deadline_value(text):
    if text is None:
        return None
    try:
        value = float(text)
        if not np.isfinite(value):
            raise ValueError('deadline must be finite')
        return value
    except ValueError:
        value=datetime.fromisoformat(text.replace("Z","+00:00"))
        if value.tzinfo is None:
            raise ValueError("deadline must be epoch seconds or timezone-aware ISO8601")
        return value.timestamp()


def boundaries(length):
    if type(length) is not int or length < 1:
        raise ValueError("expected a positive group-token horizon")
    return sorted(set([value for value in BOUNDARIES if value <= length]+[length]))


def first_prediction_difference(reference, candidate):
    a,b=np.asarray(reference),np.asarray(candidate)
    if a.shape != b.shape:
        return dict(kind="shape",reference_shape=list(a.shape),candidate_shape=list(b.shape))
    rows,writes=np.nonzero(a!=b)
    if not len(rows):
        return None
    write=int(writes.min());row=int(rows[writes==write].min())
    return dict(kind="prediction",row=row,write_index=write,group_token_index=None if write==0 else write,
                reference=int(a[row,write]),candidate=int(b[row,write]))


def payload_comparison(reference, candidate, *, reference_header=0, candidate_header=0):
    a=reference.payload[:,reference_header:];b=candidate.payload[:,candidate_header:]
    result=dict(match=a.shape==b.shape and np.array_equal(a,b),reference_sha256=common.sha(a.tobytes()),
                candidate_sha256=common.sha(b.tobytes()),first_difference=None)
    if a.shape != b.shape:
        result['first_difference']=dict(kind="shape",reference_shape=list(a.shape),candidate_shape=list(b.shape))
    elif not result['match']:
        row,byte=np.argwhere(a!=b)[0]
        result['first_difference']=dict(kind="final_payload_byte",row=int(row),byte_in_compared_region=int(byte),
            reference=int(a[row,byte]),candidate=int(b[row,byte]),time_scope="final state only; not a per-write state trace")
    return result


def empty_counts(batch):
    return dict(invalid_readout_counts=[0]*batch,active_update_attempts=0,terminal_noop_steps=0)


def add_counts(total, execution):
    for key in ('active_update_attempts','terminal_noop_steps'):
        total[key]+=execution[key]
    total['invalid_readout_counts']=(np.asarray(total['invalid_readout_counts'],dtype=np.int64)+
                                    np.asarray(execution['invalid_readout_counts'],dtype=np.int64)).tolist()


def chunked_call(call, model, table, tokens, adapter, seeds, *, deadline=None, checkpoint_hook=None, v2=False):
    """Shape stays B at every call. Only the first call prepends BOS."""
    parts=[];state=None;start=0;calls=[];trace=[];counts=empty_counts(len(tokens))
    for stop in boundaries(int(tokens.shape[1])):
        check_deadline(deadline)
        pred,state,execution=call(model,table,tokens[:,start:stop],adapter,seeds=seeds,initial=state,
                                   include_bos=start==0,diagnostics=False)
        parts.append(pred)
        if v2:
            add_counts(counts,execution)
        calls.append(dict(group_start=start,group_stop=stop,bos_included=start==0,execution=execution))
        joined=np.concatenate(parts,axis=1)
        trace.append(dict(group_stop=stop,prediction_sha256=common.sha(joined.astype('i1').tobytes()),
                          payload_sha256=common.sha(state.payload.tobytes()),payload_bytes=state.payload.nbytes))
        if checkpoint_hook is not None:
            adapter,state=checkpoint_hook(stop,adapter,state,joined,counts)
        start=stop
    return np.concatenate(parts,axis=1),state,dict(calls=calls,trace=trace,counts=counts)


def select_invalid_stream(record, sample_ids):
    """Prespecified earliest recorded nonfinite write, then original row index."""
    first=record['execution']['first_invalid_write_position']
    if len(first)!=len(sample_ids) or len(first)!=512:
        raise ValueError('Expected the complete original 512-stream invalid-position record')
    pairs=[(value,index) for index,value in enumerate(first) if value is not None]
    if not pairs:
        raise ValueError('Prespecified historical invalid case has no recorded invalid stream')
    if any(type(value) is not int or not 0<=value<=2048 for value,_ in pairs):
        raise ValueError('Malformed original invalid write position')
    write,index=min(pairs)
    return dict(original_row=index,sample_id=sample_ids[index],original_first_invalid_write=write,
                original_batch_size=512,replay_batch_size=1,
                selector='earliest non-null first_invalid_write_position, then lowest original row index')


def legacy_numeric_error(error):
    """Only reference-declared numeric failures may be retained as legacy errors."""
    from source.online_v2 import _V1Adapter
    module=sys.modules[_V1Adapter.__module__]
    return (isinstance(error,(module.NonfiniteStateError,FloatingPointError,OverflowError)) or
            (type(error) is ValueError and str(error) in {
                'state must contain only finite values',
                'per-head scale exceeds finite FP32 range',
                'nonzero per-head scale underflows FP32',
                'state exceeds native dtype finite range'}))


def source_identity():
    names=['source/run_historical.py','source/online_v2.py','source/evaluation.py','source/common.py',
           'source/checkpoint.py','source/arms.py','source/reference.py']
    return {name:common.file_sha(common.ROOT/name) for name in names}


def load_runtime(checkpoint,upstream,seed):
    from source.reference import v1
    from source.evaluation import sequence_call
    v1.torch.set_num_threads(2)
    model,table=common.load_model(upstream,checkpoint,seed)
    return model,table,v1,sequence_call


def verify_input_files(plan):
    path=Path(plan['input_file'])
    if common.file_sha(path)!=plan['input_file_sha256']:
        raise ValueError('Historical worker input bytes changed')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={'tokens','gold'}:
            raise ValueError('Unexpected historical worker arrays')
        tokens=archive['tokens'].copy();gold=archive['gold'].copy()
    if tokens.dtype!=np.uint8 or gold.dtype!=np.uint8 or tokens.shape!=gold.shape or tokens.ndim!=2:
        raise ValueError('Malformed historical worker input')
    if tokens.shape!=tuple(plan['input_shape']) or np.any(tokens>5) or np.any(gold>5):
        raise ValueError('Historical worker input shape/range mismatch')
    return tokens.astype(np.int64),gold


def worker(plan_path):
    from source.checkpoint import load_checkpoint,save_checkpoint
    plan=json.loads(Path(plan_path).read_text())
    if plan['source_sha256']!=source_identity():
        raise ValueError('Historical runtime changed after the worker plan was written')
    if plan['technical_dev_smoke']:
        if (plan['seed'],plan['identity']['arm'],plan['identity']['cohort'],plan['input_shape']) != (0,'NATIVE_FP32','TECHNICAL_DEV2001_B16_T32',[16,32]):
            raise ValueError('Pre-freeze worker is restricted to the approved native DEV smoke')
    else:
        common.verify_protocol()
        if common.file_sha(common.ROOT/'protocol_v2.json')!=plan['identity']['protocol_sha256']:
            raise ValueError('Historical worker protocol changed')
    check_deadline(plan['deadline'])
    tokens,gold=verify_input_files(plan)
    if plan['technical_dev_smoke']:
        from source.reference import groups
        expected=groups.frozen_sequences('S3','DEV',16,32,2001)
        if not np.array_equal(tokens,expected.tokens) or not np.array_equal(gold,expected.gold):
            raise ValueError('Technical smoke input differs from exact DEV2001 sample')
    adapter,state,pred,prefix_gold,record=load_checkpoint(plan['start_checkpoint'],expected_identity=plan['identity'],
                                                        expected_sample_ids=plan['sample_ids'])
    start=record['scored_offset']
    if start!=1 or not np.array_equal(prefix_gold,gold[:,:start]) or len(tokens)!=len(state.payload):
        raise ValueError('Fresh process must start at exact group-token1 boundary with unchanged B')
    model,table,_,call=load_runtime(Path(plan['checkpoint']),Path(plan['upstream']),plan['seed'])
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    counts=record['counts'];trace=[]
    for stop in boundaries(tokens.shape[1]):
        if stop<=start:
            continue
        check_deadline(plan['deadline'])
        suffix,state,execution=call(model,table,tokens[:,start:stop],adapter,initial=state,include_bos=False,diagnostics=False)
        pred=np.concatenate((pred,suffix),axis=1);add_counts(counts,execution)
        checkpoint=output/f'cut-{stop:04d}'
        save_checkpoint(checkpoint,adapter,state,identity=plan['identity'],predictions=pred,gold=gold[:,:stop],
                        sample_ids=plan['sample_ids'],counts=counts)
        adapter,state,pred,prefix_gold,record=load_checkpoint(checkpoint,expected_identity=plan['identity'],expected_sample_ids=plan['sample_ids'])
        counts=record['counts']
        trace.append(dict(group_stop=stop,prediction_sha256=common.sha(pred.astype('i1').tobytes()),
            payload_sha256=common.sha(state.payload.tobytes()),payload_bytes=state.payload.nbytes))
        start=stop
    (output/'final-runtime.bin').write_bytes(state.payload.tobytes())
    np.save(output/'predictions.npy',pred.astype('i1'))
    common.write_json(output/'receipt.json',dict(schema='case010-historical-fresh-worker-v1',status='COMPLETE',
        identity=plan['identity'],fresh_model_process=True,fresh_processes=1,
        first_group_offset=1,batch_size=len(tokens),boundaries=boundaries(tokens.shape[1]),trace=trace,counts=counts,
        prediction_sha256=common.sha(pred.astype('i1').tobytes()),final_payload_sha256=common.sha(state.payload.tobytes()),
        input_file_sha256=plan['input_file_sha256'],source_sha256=source_identity(),
        reviewer_probe_executed=False,serialization_at_each_remaining_boundary=True))


def run_cell(*,model,table,v1,v2_call,old,new,tokens,gold,ids,seed,input_seed,name,cohort,
             checkpoint,upstream,public,private,protocol_hash,technical=False,deadline=None,extra=None):
    from source.checkpoint import save_checkpoint,load_checkpoint
    public=Path(public);private=Path(private)
    public.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
    identity=dict(schema='case010-historical-parity-cell-v1',model_seed=seed,arm=name,cohort=cohort,
        checkpoint_sha256=common.CHECKPOINTS[seed],protocol_sha256=protocol_hash,
        tokens_sha256=common.sha(np.asarray(tokens,dtype='u1').tobytes()),gold_sha256=common.sha(np.asarray(gold,dtype='u1').tobytes()),
        sample_ids_sha256=common.sha(common.canonical(list(ids))),runtime_config_sha256=common.sha(common.runtime_config()),
        codec_config_sha256=common.sha(new.config_bytes),basis_sha256=common.sha(new.basis_bytes),
        legacy_codec_config_sha256=common.sha(old.config_bytes))
    seeds=v1.stream_seeds(input_seed,len(tokens))
    results={};predictions={};legacy_error=None
    check_deadline(deadline)
    try:
        pred,state,execution=v1.sequence_call(model,table,tokens,old,seeds,diagnostics=False)
        predictions['v1_full']=pred;results['v1_full']=(state,execution)
        check_deadline(deadline)
        def legacy_boundary(stop,adapter,state,pred,counts):
            folder=private/'legacy-checkpoints'/f'cut-{stop:04d}'
            folder.mkdir(parents=True,exist_ok=False)
            (folder/'codec.json').write_bytes(adapter.config_bytes)
            (folder/'basis.bin').write_bytes(adapter.basis_bytes)
            (folder/'runtime.bin').write_bytes(state.payload.tobytes())
            np.save(folder/'predictions.npy',pred.astype('i1'))
            restored=adapter.__class__.from_shared((folder/'codec.json').read_bytes(),(folder/'basis.bin').read_bytes())
            return restored,restored.from_bytes((folder/'runtime.bin').read_bytes(),batch_size=len(tokens))
        pred,state,execution=chunked_call(v1.sequence_call,model,table,tokens,old,seeds,deadline=deadline,
                                         checkpoint_hook=legacy_boundary)
        predictions['v1_split']=pred;results['v1_split']=(state,execution)
    except (ValueError,FloatingPointError,OverflowError) as exc:
        if not legacy_numeric_error(exc):
            raise
        legacy_error=dict(error_type=type(exc).__name__,status='LEGACY_EXECUTION_ERROR_NOT_CONVERTED_TO_TERMINAL',
                          stage='v1_full' if 'v1_full' not in results else 'v1_split')
    check_deadline(deadline)
    pred,state,execution=v2_call(model,table,tokens,new,seeds,diagnostics=False)
    predictions['v2_full']=pred;results['v2_full']=(state,execution)
    def save_boundary(stop,adapter,state,pred,counts):
        folder=private/'parent-checkpoints'/f'cut-{stop:04d}'
        save_checkpoint(folder,adapter,state,identity=identity,predictions=pred,gold=gold[:,:stop],sample_ids=ids,counts=counts)
        restored,restored_state,_,_,_=load_checkpoint(folder,expected_identity=identity,expected_sample_ids=ids)
        return restored,restored_state
    pred,state,execution=chunked_call(v2_call,model,table,tokens,new,seeds,deadline=deadline,checkpoint_hook=save_boundary,v2=True)
    predictions['v2_split']=pred;results['v2_split']=(state,execution)
    np.savez_compressed(private/'inputs.npz',tokens=np.asarray(tokens,dtype='u1'),gold=np.asarray(gold,dtype='u1'))
    plan=dict(schema='case010-historical-worker-plan-v1',seed=seed,identity=identity,sample_ids=list(ids),
        checkpoint=str(Path(checkpoint).resolve()),upstream=str(Path(upstream).resolve()),
        start_checkpoint=str((private/'parent-checkpoints/cut-0001').resolve()),
        input_file=str((private/'inputs.npz').resolve()),input_file_sha256=common.file_sha(private/'inputs.npz'),input_shape=list(tokens.shape),
        output=str((private/'fresh-worker').resolve()),technical_dev_smoke=technical,deadline=deadline,source_sha256=source_identity())
    common.write_json(private/'worker-plan.json',plan)
    check_deadline(deadline)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (private/'worker.log').open('w') as log:
        child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker-plan',str((private/'worker-plan.json').resolve())],
            env=env,stdout=log,stderr=subprocess.STDOUT)
    if child.returncode:
        common.write_json(public/'failure.json',dict(status='FRESH_WORKER_FAILED',exit_code=child.returncode,identity=identity,
                         full_payloads_public=False,reviewer_probe_executed=False))
        raise RuntimeError('Fresh historical worker failed; private log retained')
    fresh_dir=private/'fresh-worker';fresh=json.loads((fresh_dir/'receipt.json').read_text())
    fresh_state=new.from_bytes((fresh_dir/'final-runtime.bin').read_bytes(),batch_size=len(tokens))
    predictions['v2_fresh']=np.load(fresh_dir/'predictions.npy',allow_pickle=False)
    results['v2_fresh']=(fresh_state,fresh)
    comparisons={}
    for label,a,b,head_a,head_b in [('v1_full_vs_split','v1_full','v1_split',0,0),
        ('v1_vs_v2_full_body','v1_full','v2_full',8,17),('v2_full_vs_split','v2_full','v2_split',0,0),
        ('v2_full_vs_fresh','v2_full','v2_fresh',0,0)]:
        if a not in results or b not in results:
            comparisons[label]=dict(available=False)
            continue
        diff=first_prediction_difference(predictions[a],predictions[b])
        comparisons[label]=dict(available=True,predictions_match=diff is None,first_prediction_difference=diff,
                               payload=payload_comparison(results[a][0],results[b][0],reference_header=head_a,candidate_header=head_b))
    expected_trace=results['v2_split'][1]['trace'][1:]
    boundary_parity=fresh['trace']==expected_trace
    passed=all(comparisons[key]['predictions_match'] and comparisons[key]['payload']['match']
               for key in ('v2_full_vs_split','v2_full_vs_fresh')) and boundary_parity
    for key,(state,_) in results.items():
        (private/f'{key}.runtime.bin').write_bytes(state.payload.tobytes())
    np.savez_compressed(public/'predictions.npz',**{key:np.asarray(value,dtype='i1') for key,value in predictions.items()},
                        gold=np.asarray(gold,dtype='u1'))
    terminal=new.terminal_info(results['v2_full'][0])
    executions={key:value[1] for key,value in results.items() if key not in ('v2_fresh',)}
    receipt=dict(schema='case010-historical-parity-result-v1',status='PASS' if passed else 'V2_PARITY_FAILURE',identity=identity,
        diagnostic_only=True,fresh_primary_sample_count=0,reviewer_probe_available=False,reviewer_probe_executed=False,
        N=len(tokens),T=tokens.shape[1],sample_ids=list(ids),boundaries=boundaries(tokens.shape[1]),
        BOS='one initial BOS; all resumed calls include_bos=False; cache cursor records consumed writes',
        same_batch_shape=True,fresh_processes=1,fresh_process_version='v2',
        serialized_at_all_boundaries=['v1','v2'],fresh_serialized_boundary_trace_match=boundary_parity,
        full_split_match=passed,comparisons=comparisons,legacy_error=legacy_error,
        legacy_chunk_revival='Expected v1 contract defect if it occurs; not a v2 parity failure gate',
        v1_v2_body_equality_scope='Compare bytes after different version headers; terminal-body differences are intentional',
        predictions_sha256={key:common.sha(np.asarray(value,dtype='i1').tobytes()) for key,value in predictions.items()},
        final_payload_sha256={key:common.sha(value[0].payload.tobytes()) for key,value in results.items()},
        predictions_artifact=dict(file='predictions.npz',sha256=common.file_sha(public/'predictions.npz')),
        final_terminal_codes=terminal['terminal_code'].tolist(),
        first_terminal_write=[None if active else int(value) for active,value in zip(terminal['active'],terminal['first_terminal_write'])],
        execution=executions,full_payloads_public=False,source_sha256=source_identity(),extra=extra)
    if extra and 'original_first_invalid_write' in extra:
        observed=None if 'v1_full' not in results else results['v1_full'][1]['first_invalid_write_position'][0]
        receipt['original_invalid_case_replay']=dict(original_first_invalid_write=extra['original_first_invalid_write'],
            v1_B1_first_invalid_write=observed,reproduced_numerical_failure=observed is not None,
            same_write_as_original=observed==extra['original_first_invalid_write'],
            interpretation='B1 failure reproduced; compare recorded write positions' if observed is not None else
                'Original B512 invalid stream did not reproduce at B1; batch shape differs; no claim that the original failure was absent')
    common.write_json(public/'receipt.json',receipt)
    if not passed:
        raise RuntimeError('v2 historical full/split/fresh parity failed; evidence retained')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-plan',type=Path)
    parser.add_argument('--seed',type=int,choices=(0,1,2))
    for name in ('checkpoint','upstream','output','private-output','v1-case'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--deadline',help='epoch seconds or timezone-aware ISO8601')
    parser.add_argument('--smoke-dev',action='store_true')
    args=parser.parse_args(argv)
    if args.worker_plan:
        worker(args.worker_plan);return 0
    if args.seed is None or any(getattr(args,key) is None for key in ('checkpoint','upstream','output','private_output','v1_case')):
        parser.error('--seed, --checkpoint, --upstream, --output, --private-output and --v1-case are required')
    if args.output.resolve()==args.private_output.resolve() or args.private_output.resolve().is_relative_to(args.output.resolve()):
        raise ValueError('Private runtime output must be outside public result tree')
    deadline=deadline_value(args.deadline)
    if args.smoke_dev:
        if args.seed!=0:
            raise ValueError('Only seed0 native DEV B16/T32 is approved before freeze')
        protocol_hash=None
    else:
        common.verify_protocol()
        protocol_hash=common.file_sha(common.ROOT/'protocol_v2.json')
    check_deadline(deadline)
    model,table,v1,v2_call=load_runtime(args.checkpoint,args.upstream,args.seed)
    from source.online_v2 import OnlineAdapter,PackedCodec,NativeFloatCodec
    args.output.mkdir(parents=True,exist_ok=False);args.private_output.mkdir(parents=True,exist_ok=False)
    cells=[]
    if args.smoke_dev:
        batch=v1.frozen_sequences('S3','DEV',16,32,2001)
        cells.append(dict(name='NATIVE_FP32',cohort='TECHNICAL_DEV2001_B16_T32',tokens=batch.tokens,gold=batch.gold,
            ids=batch.sequence_ids,input_seed=2001,new=OnlineAdapter(NativeFloatCodec(v1.SHAPE)),extra=None))
    else:
        from source.arms import load_arms
        arms,_=load_arms(args.seed)
        tokens,gold,ids,_=common.load_cohort('historical')
        if tokens.shape!=(16,2048):raise ValueError('Historical parity is fixed at first16/T2048')
        for name in HISTORICAL_ARMS:
            cells.append(dict(name=name,cohort='HISTORICAL_FIRST16_TEST3001',tokens=tokens,gold=gold,ids=ids,
                              input_seed=3001,new=arms[name],extra=None))
        if args.seed==0:
            with np.load(common.ROOT/'inputs/v1_calibration/seed0/calibration.npz',allow_pickle=False) as data:cal={key:data[key] for key in data.files}
            batch=v1.frozen_sequences('S3','DEV',4,64,2001)
            extras={'STOCHASTIC_8':OnlineAdapter(PackedCodec(v1.SHAPE,8,stochastic=True)),
                    'FULL_RESIDUAL_4_4':OnlineAdapter(PackedCodec(v1.SHAPE,4),PackedCodec(v1.SHAPE,4)),
                    'LOWRANK_4_8_R1':OnlineAdapter(PackedCodec(v1.SHAPE,4),PackedCodec((12,1,16),8),cal['basis'][:,:,:1])}
            for name in EXTRA_ARMS:
                cells.append(dict(name=name,cohort='EXTRA_FIRST4_DEV2001_T64',tokens=batch.tokens,gold=batch.gold,ids=batch.sequence_ids,
                                  input_seed=2001,new=extras[name],extra=None))
        original=args.v1_case/'results'/f'learned-seed{args.seed}'
        original_inputs=json.loads((original/'TEST_inputs.json').read_text())
        full=v1.frozen_sequences('S3','TEST',512,2048,3001)
        if v1.input_identity(full)!=original_inputs:
            raise ValueError('Regenerated original TEST bytes/IDs differ from retained v1 identity')
        if not np.array_equal(tokens,full.tokens[:16]) or not np.array_equal(gold,full.gold[:16]) or list(ids)!=list(full.sequence_ids[:16]):
            raise ValueError('Retained historical cohort is not exact original first16')
        for name in INVALID_CASES[args.seed]:
            path=original/'TEST'/f'{name}.json';record=json.loads(path.read_text())
            selection=select_invalid_stream(record,full.sequence_ids);index=selection['original_row']
            selection['original_result_sha256']=common.file_sha(path)
            old=v1.OnlineAdapter.from_shared((original/'codecs'/f'{name}.json').read_bytes())
            new=OnlineAdapter(old.state_codec,old.residual_codec,old.basis,old.mode)
            cells.append(dict(name=name,cohort='ORIGINAL_INVALID_STREAM_B1_TEST3001',tokens=full.tokens[index:index+1],
                gold=full.gold[index:index+1],ids=[full.sequence_ids[index]],input_seed=3001,new=new,extra=selection))
    entries=[]
    for cell in cells:
        check_deadline(deadline)
        new=cell.pop('new');name=cell['name'];cohort=cell['cohort'];key=cohort+'--'+name
        old=v1.OnlineAdapter(new.state_codec,new.residual_codec,new.basis,new.mode)
        if cohort=='HISTORICAL_FIRST16_TEST3001':
            path=args.v1_case/'results'/f'learned-seed{args.seed}'/'codecs'/f'{name}.json'
            if old.config_bytes!=path.read_bytes():raise ValueError('Historical v1 codec differs from retained original arm')
        result=run_cell(model=model,table=table,v1=v1,v2_call=v2_call,old=old,new=new,seed=args.seed,
            checkpoint=args.checkpoint,upstream=args.upstream,public=args.output/key,private=args.private_output/key,
            protocol_hash=protocol_hash,technical=args.smoke_dev,deadline=deadline,**cell)
        entries.append(dict(file=key+'/receipt.json',sha256=common.file_sha(args.output/key/'receipt.json'),
                            arm=name,cohort=cohort,status=result['status']))
        print(json.dumps(dict(model_seed=args.seed,arm=name,cohort=cohort,status=result['status'])),flush=True)
    common.write_json(args.output/'index.json',dict(schema='case010-historical-parity-index-v1',status='COMPLETE',
        model_seed=args.seed,technical_dev_smoke=args.smoke_dev,protocol_sha256=protocol_hash,results=entries,
        reviewer_probe_executed=False,checkpoint_sha256=common.CHECKPOINTS[args.seed],source_sha256=source_identity()))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
