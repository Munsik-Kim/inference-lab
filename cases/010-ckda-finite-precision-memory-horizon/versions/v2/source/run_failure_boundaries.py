"""Technical continuation appendix on four already-observed historical failures.

No new memory trial or model/codec selection. The fixed main protocol and its
historical replay remain unchanged. Freeze this appendix before invoking runs.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import common
from source import run_historical as historical
from source.checkpoint import save_checkpoint, load_checkpoint
from source.online_v2 import OnlineAdapter

CASES = ((0, 'UNIFORM_2', 77, 838), (0, 'UNIFORM_3', 8, 1741),
         (1, 'UNIFORM_2', 31, 623), (2, 'UNIFORM_2', 5, 867))
COHORT = 'ORIGINAL_INVALID_STREAM_B1_TEST3001'
SCHEMA = 'case010-historical-failure-boundaries-appendix-v1'


def sources():
    return dict(historical.source_identity(), **{'source/run_failure_boundaries.py': common.file_sha(Path(__file__))})


def cuts(first, horizon):
    if type(first) is not int or type(horizon) is not int or not 1 < first < horizon:
        raise ValueError('Need a failure with both prior and subsequent group tokens')
    return sorted({first - 1, first, first + 1, horizon})


def freeze(path, historical_root):
    common.verify_protocol()
    path = Path(path)
    if path.exists() or path.with_suffix('.sha256').exists():
        raise ValueError('Never overwrite an existing appendix freeze')
    retained = []
    for seed, arm, row, write in CASES:
        receipt = Path(historical_root) / f'seed{seed}' / f'{COHORT}--{arm}' / 'receipt.json'
        result = json.loads(receipt.read_text())
        if (result['identity']['model_seed'], result['identity']['arm'], result['N'], result['T'],
                result['extra']['original_row'], result['extra']['original_first_invalid_write'],
                result['first_terminal_write']) != (seed, arm, 1, 2048, row, write, [write]):
            raise ValueError('Recorded fixed failure identity does not match the technical appendix')
        if result['status'] != 'PASS' or not result['original_invalid_case_replay']['same_write_as_original']:
            raise ValueError('Expected already-completed exact historical failure replay')
        prediction = receipt.with_name('predictions.npz')
        if common.file_sha(prediction) != result['predictions_artifact']['sha256']:
            raise ValueError('Historical predictions changed')
        retained.append(dict(seed=seed, arm=arm, original_row=row, first_terminal_write=write,
            group_token_cuts=cuts(write, 2048), fresh_process_resume_cut=write,
            receipt=receipt.relative_to(common.ROOT).as_posix(), receipt_sha256=common.file_sha(receipt),
            predictions=prediction.relative_to(common.ROOT).as_posix(), predictions_sha256=common.file_sha(prediction)))
    appendix = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        role='TECHNICAL_CONTINUATION_COVERAGE_ALREADY_OBSERVED_FAILURES', new_memory_trials=0,
        no_hyperparameter_or_model_selection=True, original_protocol_unchanged=True,
        reason='One fixed historical failure occurs after the final interior main-protocol cut; explicitly exercise continuation immediately before and after every selected actual failure.',
        protocol_sha256=common.file_sha(common.ROOT/'protocol_v2.json'), source_sha256=sources(), cases=retained,
        batch_size=1, horizon=2048, fresh_processes_per_case=1,
        assertions=['ACTIVE before failure', 'first terminal write equals fixed historical write',
                    'body and RNG bytes freeze while cursor advances',
                    'full/split/fresh predictions and final payload equal the retained full v2 run'],
        reviewer_probe_executed=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(path, appendix)
    path.with_suffix('.sha256').write_text(common.file_sha(path)+'\n')
    return appendix


def verify_appendix(path):
    common.verify_protocol()
    path = Path(path)
    if common.file_sha(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Technical appendix hash mismatch')
    result = json.loads(path.read_text())
    if result['schema'] != SCHEMA or result['source_sha256'] != sources():
        raise ValueError('Technical appendix runtime changed')
    if result['protocol_sha256'] != common.file_sha(common.ROOT/'protocol_v2.json'):
        raise ValueError('Main protocol identity changed')
    if [(r['seed'], r['arm'], r['original_row'], r['first_terminal_write']) for r in result['cases']] != list(CASES):
        raise ValueError('Technical appendix fixed cases changed')
    for entry in result['cases']:
        if entry['group_token_cuts'] != cuts(entry['first_terminal_write'], 2048):
            raise ValueError('Technical appendix cuts changed')
        for key in ['receipt', 'predictions']:
            candidate = (common.ROOT / entry[key]).resolve()
            if not candidate.is_relative_to(common.ROOT) or common.file_sha(candidate) != entry[key+'_sha256']:
                raise ValueError('Historical reference artifact identity changed')
    return result


def terminal_proof(adapter, state, stop, first, frozen_body=None):
    info = adapter.terminal_info(state)
    if len(state.payload) != 1 or int(info['cursor'][0]) != stop + 1:
        raise ValueError('Failure-boundary checkpoint lost batch shape or consumed cursor')
    if stop < first:
        if not info['active'][0]:
            raise ValueError('Actual failure occurred before its fixed historical position')
    elif info['active'][0] or int(info['first_terminal_write'][0]) != first:
        raise ValueError('Actual failure position or absorbing flag differs from historical run')
    body = state.payload[:, 17:].tobytes()
    if stop >= first and frozen_body is not None and body != frozen_body:
        raise ValueError('A failed or terminal write changed committed body/RNG bytes')
    return dict(group_stop=stop, cursor=int(info['cursor'][0]), terminal_code=int(info['terminal_code'][0]),
        first_terminal_write=None if info['active'][0] else int(info['first_terminal_write'][0]),
        body_sha256=common.sha(body), final_payload_sha256=common.sha(state.payload.tobytes()))


def run_chunks(call, model, table, tokens, gold, adapter, seeds, first, folder, identity, ids, *,
               initial=None, predictions=None, counts=None, start=0, deadline=None, frozen_body=None):
    folder = Path(folder)
    state = initial
    pred = predictions
    counts = historical.empty_counts(len(tokens)) if counts is None else counts
    trace = []
    for stop in cuts(first, int(tokens.shape[1])):
        if stop <= start:
            continue
        historical.check_deadline(deadline)
        suffix, state, execution = call(model, table, tokens[:, start:stop], adapter, seeds=seeds,
            initial=state, include_bos=state is None, diagnostics=False)
        pred = suffix if pred is None else np.concatenate((pred, suffix), axis=1)
        historical.add_counts(counts, execution)
        proof = terminal_proof(adapter, state, stop, first, frozen_body)
        if stop == first - 1:
            frozen_body = state.payload[:, 17:].tobytes()
        proof['prediction_sha256'] = common.sha(pred.astype('i1').tobytes())
        checkpoint = folder / f'cut-{stop:04d}'
        save_checkpoint(checkpoint, adapter, state, identity=identity, predictions=pred, gold=gold[:, :stop],
                        sample_ids=ids, counts=counts)
        adapter, state, pred, _, record = load_checkpoint(checkpoint, expected_identity=identity, expected_sample_ids=ids)
        counts = record['counts']
        trace.append(proof)
        start = stop
    return adapter, state, pred, counts, trace


def worker(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    verify_appendix(plan['appendix'])
    if common.file_sha(plan['appendix']) != plan['appendix_sha256'] or plan['source_sha256'] != sources():
        raise ValueError('Failure-boundary worker identity changed')
    tokens, gold = historical.verify_input_files(plan)
    adapter, state, pred, prefix_gold, record = load_checkpoint(plan['start_checkpoint'],
        expected_identity=plan['identity'], expected_sample_ids=plan['sample_ids'])
    first = plan['first_terminal_write']
    if record['scored_offset'] != first or not np.array_equal(prefix_gold, gold[:, :first]):
        raise ValueError('Fresh child must resume immediately after the actual failure')
    terminal_proof(adapter, state, first, first)
    model, table, _, call = historical.load_runtime(Path(plan['checkpoint']), Path(plan['upstream']), plan['seed'])
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    _, state, pred, counts, trace = run_chunks(call, model, table, tokens, gold, adapter, None, first,
        output/'checkpoints', plan['identity'], plan['sample_ids'], initial=state, predictions=pred,
        counts=record['counts'], start=first, deadline=plan['deadline'], frozen_body=state.payload[:, 17:].tobytes())
    (output/'final-runtime.bin').write_bytes(state.payload.tobytes())
    np.save(output/'predictions.npy', pred.astype('i1'))
    common.write_json(output/'receipt.json', dict(status='COMPLETE', trace=trace, counts=counts,
        final_payload_sha256=common.sha(state.payload.tobytes()), prediction_sha256=common.sha(pred.astype('i1').tobytes()),
        source_sha256=sources(), batch_size=1, fresh_process=True, resume_group_cut=first))


def run_case(entry, *, appendix, historical_private, public, private, checkpoint, upstream, model, table, v1, call, deadline):
    baseline_path = common.ROOT/entry['receipt']
    baseline = json.loads(baseline_path.read_text())
    original_private = Path(historical_private)/f"seed{entry['seed']}"/f"{COHORT}--{entry['arm']}"
    prior_plan = json.loads((original_private/'worker-plan.json').read_text())
    if prior_plan['identity'] != baseline['identity']:
        raise ValueError('Historical private input/cache identity differs from public reference')
    tokens, gold = historical.verify_input_files(prior_plan)
    if common.sha(tokens.astype('u1').tobytes()) != baseline['identity']['tokens_sha256'] or common.sha(gold.tobytes()) != baseline['identity']['gold_sha256']:
        raise ValueError('Actual replay input bytes differ from historical reference')
    adapter, _, _, _, _ = load_checkpoint(original_private/'parent-checkpoints/cut-0001',
        expected_identity=baseline['identity'], expected_sample_ids=baseline['sample_ids'])
    identity = dict(baseline['identity'], continuation_appendix_sha256=common.file_sha(appendix))
    public, private = Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=False); private.mkdir(parents=True, exist_ok=False)
    first = entry['first_terminal_write']
    adapter, state, pred, counts, trace = run_chunks(call, model, table, tokens, gold, adapter,
        v1.stream_seeds(3001, 1), first, private/'parent-checkpoints', identity, baseline['sample_ids'], deadline=deadline)
    np.savez_compressed(private/'inputs.npz', tokens=tokens.astype('u1'), gold=gold)
    plan = dict(seed=entry['seed'], identity=identity, sample_ids=baseline['sample_ids'], first_terminal_write=first,
        source_sha256=sources(), appendix=str(Path(appendix).resolve()), appendix_sha256=common.file_sha(appendix),
        input_file=str((private/'inputs.npz').resolve()), input_file_sha256=common.file_sha(private/'inputs.npz'), input_shape=list(tokens.shape),
        start_checkpoint=str((private/'parent-checkpoints'/f'cut-{first:04d}').resolve()), output=str((private/'fresh-worker').resolve()),
        checkpoint=str(Path(checkpoint).resolve()), upstream=str(Path(upstream).resolve()), deadline=deadline)
    common.write_json(private/'worker-plan.json', plan)
    historical.check_deadline(deadline)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (private/'worker.log').open('x') as log:
        child = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker-plan', str(private/'worker-plan.json')],
            env=env, stdout=log, stderr=subprocess.STDOUT)
    if child.returncode:
        raise RuntimeError('Failure-boundary child failed; private evidence retained')
    fresh = json.loads((private/'fresh-worker/receipt.json').read_text())
    fresh_pred = np.load(private/'fresh-worker/predictions.npy', allow_pickle=False)
    fresh_raw = (private/'fresh-worker/final-runtime.bin').read_bytes()
    with np.load(common.ROOT/entry['predictions'], allow_pickle=False) as data:
        reference = data['v2_full'].copy()
    comparisons = dict(parent_matches_retained_predictions=bool(np.array_equal(pred, reference)),
        fresh_matches_retained_predictions=bool(np.array_equal(fresh_pred, reference)),
        parent_matches_retained_final_payload=common.sha(state.payload.tobytes())==baseline['final_payload_sha256']['v2_full'],
        fresh_matches_retained_final_payload=common.sha(fresh_raw)==baseline['final_payload_sha256']['v2_full'],
        fresh_boundary_trace_matches_parent=fresh['trace']==[item for item in trace if item['group_stop'] > first],
        fresh_consumed_counts_match_parent=fresh['counts']==counts)
    (private/'parent-final-runtime.bin').write_bytes(state.payload.tobytes())
    np.savez_compressed(public/'predictions.npz', parent=pred.astype('i1'), fresh=fresh_pred.astype('i1'))
    receipt = dict(schema=SCHEMA, status='PASS' if all(comparisons.values()) else 'FAIL', seed=entry['seed'], arm=entry['arm'],
        role='TECHNICAL_CONTINUATION_COVERAGE_ALREADY_OBSERVED_FAILURES', new_memory_trials=0, original_protocol_unchanged=True,
        source_sha256=sources(), appendix_sha256=common.file_sha(appendix), original_receipt=entry['receipt'],
        original_receipt_sha256=entry['receipt_sha256'], first_terminal_write=first, original_row=entry['original_row'],
        N=1, T=2048, group_token_cuts=cuts(first,2048), fresh_process_resume_cut=first, fresh_processes=1,
        comparisons=comparisons, trace=trace, fresh_trace=fresh['trace'], counts=counts,
        predictions_artifact=dict(file='predictions.npz',sha256=common.file_sha(public/'predictions.npz')),
        first_prediction_difference=historical.first_prediction_difference(reference, fresh_pred),
        full_payloads_public=False, reviewer_probe_executed=False)
    common.write_json(public/'receipt.json', receipt)
    if receipt['status'] != 'PASS':
        raise RuntimeError('Failure-boundary parity failed; evidence retained')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-plan',type=Path)
    parser.add_argument('--appendix',type=Path)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--historical-root',type=Path,default=common.ROOT/'results/historical')
    for name in ['historical-private','output','private-output','checkpoint-root','upstream']:
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--deadline')
    args=parser.parse_args(argv)
    if args.worker_plan:
        worker(args.worker_plan); return 0
    if args.appendix is None: parser.error('--appendix is required')
    if args.freeze:
        freeze(args.appendix,args.historical_root); return 0
    if any(getattr(args,name) is None for name in ['historical_private','output','private_output','checkpoint_root','upstream']):
        parser.error('Run requires historical-private, output, private-output, checkpoint-root and upstream')
    if args.private_output.resolve().is_relative_to(args.output.resolve()):
        raise ValueError('Private bodies must stay outside public result tree')
    appendix=verify_appendix(args.appendix)
    deadline=historical.deadline_value(args.deadline)
    args.output.mkdir(parents=True,exist_ok=False);args.private_output.mkdir(parents=True,exist_ok=False)
    loaded_seed=None;results=[]
    for entry in appendix['cases']:
        historical.check_deadline(deadline)
        seed=entry['seed'];checkpoint=args.checkpoint_root/f'training-seed{seed}/final.pt'
        if seed!=loaded_seed:
            model,table,v1,call=historical.load_runtime(checkpoint,args.upstream,seed);loaded_seed=seed
        key=f"seed{seed}--{entry['arm']}"
        receipt=run_case(entry,appendix=args.appendix,historical_private=args.historical_private,
            public=args.output/key,private=args.private_output/key,checkpoint=checkpoint,upstream=args.upstream,
            model=model,table=table,v1=v1,call=call,deadline=deadline)
        results.append(dict(file=key+'/receipt.json',sha256=common.file_sha(args.output/key/'receipt.json'),status=receipt['status']))
        print(json.dumps(dict(cell=key,status=receipt['status'])),flush=True)
    common.write_json(args.output/'index.json',dict(schema=SCHEMA,status='COMPLETE',results=results,
        appendix_sha256=common.file_sha(args.appendix),source_sha256=sources(),new_memory_trials=0))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
