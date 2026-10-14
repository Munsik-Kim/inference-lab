"""Bounded same-history GPU diagnostics. Loads saved artifacts, never solves.

Model loading, projection replacement, greedy generation and fixture tool
functions come from the unchanged r2 modules. --help has no model imports.
"""
import argparse
import copy
import gc
import importlib
import math
from pathlib import Path
import sys
import time
import traceback

from contracts import (atomic, append, read, rows, sha, token_hash, tool_blocks,
                       tool_history, first_patch_correct, select_projection, extract,
                       PROJECTIONS, METHODS, common_length)


def execute(result, method, stage, worker_budget, attempt):
    started = time.monotonic()
    result = Path(result)
    runtime = read(result / 'runtime_manifest.json')
    registration = read(result / 'preregistration.json')
    if sha(result / 'input_manifest.json') != registration['input_manifest_sha256'] \
            or sha(result / 'runtime_manifest.json') != registration['runtime_manifest_sha256']:
        raise ValueError('registration input/runtime changed')
    source = Path(runtime['research_source'])
    for name, digest in runtime['source_hashes'].items():
        if sha(source / name) != digest:
            raise ValueError('changed original source: ' + name)
    sys.path.insert(0, str(source))
    original = importlib.import_module('paper_model_runner')
    quant = importlib.import_module('paper_int4')
    reference = importlib.import_module('reference_int4')
    for name, module in [('paper_model_runner', original), ('paper_int4', quant), ('reference_int4', reference)]:
        if Path(module.__file__).resolve() != (source / (name + '.py')).resolve():
            raise ValueError('wrong imported reference module: ' + str(module.__file__))
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import LogitsProcessor
    peaks = {'allocated': 0, 'reserved': 0}
    def observe_memory():
        peaks['allocated'] = max(peaks['allocated'], torch.cuda.max_memory_allocated())
        peaks['reserved'] = max(peaks['reserved'], torch.cuda.max_memory_reserved())

    out = result / 'workers' / stage / method / ('attempt' + str(attempt).zfill(2))
    if out.exists():
        raise FileExistsError('worker attempt directory exists: ' + str(out))
    out.mkdir(parents=True)
    def budget():
        if time.monotonic() - started >= worker_budget - 1:
            raise TimeoutError('registered conservative GPU worker wall budget')
    class Guard(LogitsProcessor):
        def __call__(self, input_ids, scores):
            budget()
            if not bool(torch.isfinite(scores).all()):
                raise ValueError('nonfinite generation logits')
            return scores
    class Diagnostic(original.Worker):
        @torch.inference_mode()
        def generate(self, ids, max_new):
            budget()
            ids_gpu = torch.tensor(ids, device='cuda', dtype=torch.long).unsqueeze(0)
            if len(ids) + max_new > registration['generation']['context_limit']:
                raise ValueError('CONTEXT_LIMIT; no truncate')
            torch.cuda.synchronize(); observe_memory(); torch.cuda.reset_peak_memory_stats(); t = time.monotonic()
            output = self.model.generate(ids_gpu, attention_mask=torch.ones_like(ids_gpu),
                do_sample=False, max_new_tokens=max_new, use_cache=True, logits_to_keep=1,
                pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=False,
                logits_processor=[Guard()])
            torch.cuda.synchronize()
            observe_memory()
            new = output[0, len(ids):].cpu().tolist()
            return {'text': self.tokenizer.decode(new, skip_special_tokens=True),
                    'raw_text': self.tokenizer.decode(new, skip_special_tokens=False),
                    'output_token_ids': new, 'input_tokens': len(ids), 'output_tokens': len(new),
                    'input_token_sha256': token_hash(ids), 'seconds': time.monotonic() - t,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'status': 'COMPLETE'}

        def apply_artifacts(self):
            raise RuntimeError('original optional-solver artifact loader forbidden in followup')

    w = Diagnostic(Path(runtime['original_result_root']), '4b', method, output_override=out)
    inputs = read(result / 'input_manifest.json')
    tasks = {t['id']: t for t in read(w.run / 'inputs/agent_tasks.json')}
    names = [n for n, m in w.model.named_modules()
             if isinstance(m, torch.nn.Linear) and n.split('.')[-1] in PROJECTIONS]
    if len(names) != 252:
        raise ValueError('projection coverage mismatch')
    original_modules = {}
    intervention_selection = None
    if stage == 'p2':
        intervention_selection = read(result / 'p2_selection.json')
        for name in (intervention_selection['target'], intervention_selection['control']):
            original_modules[name] = w.model.get_submodule(name)
    loaded = []
    if method != 'BF16':
        artifact_method = 'RTN' if method == 'RTN' else 'D16'
        directory = Path(runtime['artifact_root']) / artifact_method
        if sha(directory / 'manifest.json') != runtime['artifacts'][artifact_method]['manifest_sha256']:
            raise ValueError('changed original artifact manifest')
        entries = {x['layer']: x for x in runtime['artifacts'][artifact_method]['per_projection_identity']}
        if set(names) != set(entries) or set(p.stem for p in directory.glob('*.rpi4')) != set(names):
            raise ValueError('missing/extra original projection artifact')
        for i, name in enumerate(names):
            budget()
            layer = w.model.get_submodule(name)
            entry = entries[name]
            data = (directory / (name + '.rpi4')).read_bytes()
            import hashlib
            actual = hashlib.sha256(data).hexdigest()
            if len(data) != entry['bytes'] or actual != entry['sha256']:
                raise ValueError('changed saved artifact: ' + name)
            q = reference.QuantizedWeight.deserialize(data)
            if list(q.shape) != entry['shape'] or list(layer.weight.shape) != entry['shape']:
                raise ValueError('artifact/base shape mismatch')
            expected_method = 'rtn' if artifact_method == 'RTN' else 'row_column'
            if q.method != expected_method or q.dtype != torch.float32:
                raise ValueError('artifact contract/dtype mismatch')
            quant.replace_projection(w.model, name, q, method == 'D4')
            loaded.append({'layer': name, 'sha256': actual, 'bytes': len(data), 'shape': list(q.shape),
                           'action': 'READ_ONLY_LOAD', 'scale_dtype': str(q.dtype)})
            del data, q
            if i % 70 == 0:
                print({'stage': stage, 'method': method, 'loaded_projection': i + 1}, flush=True)
        # Original norm tensors were never replacement candidates. Recheck bytes
        # directly rather than invoking the old loader (which can run a solver).
        import hashlib
        for n, p in w.model.named_parameters():
            if n in w.original_small_hashes:
                if hashlib.sha256(p.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest() != w.original_small_hashes[n]:
                    raise ValueError('norm identity changed')
    now_tied = w.model.model.embed_tokens.weight.data_ptr() == w.model.lm_head.weight.data_ptr()
    if now_tied != w.base_shared_embedding:
        raise ValueError('embedding/lm-head tie changed')
    atomic(out / 'artifact_readback.json', {'status': 'PASS', 'method': method, 'artifacts': loaded,
          'new_solver_calls': 0, 'artifact_writes': 0, 'import_paths': {name: str(module.__file__) for name, module in
          [('paper_model_runner', original), ('paper_int4', quant), ('reference_int4', reference)]}})
    gc.collect(); w.model.cuda()
    atomic(out / 'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
        'threads': torch.get_num_threads(), 'TF32': torch.backends.cuda.matmul.allow_tf32,
        'BF16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'within_request_KV': True, 'across_turn_prefix_cache': False,
        'runtime': 'dequantized BF16 reference; no packed kernel',
        'device_memory_at_load': {'free_bytes': torch.cuda.mem_get_info()[0], 'total_bytes': torch.cuda.mem_get_info()[1]},
        'system_peak_memory': None, 'system_peak_reason': 'not sampled; Torch allocated/reserved only'})

    @torch.inference_mode()
    def hidden(ids):
        budget()
        x = torch.tensor(ids, device='cuda', dtype=torch.long).unsqueeze(0)
        value = w.model.model(x, attention_mask=torch.ones_like(x), use_cache=False).last_hidden_state[0]
        observe_memory()
        return value

    @torch.inference_mode()
    def short_parity():
        ids = inputs['natural'][0]['input_ids'][-64:]
        x = torch.tensor(ids, device='cuda', dtype=torch.long).unsqueeze(0)
        full = w.model(x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -16:]
        h = hidden(ids)
        reduced = w.model.lm_head(h[-16:])
        finite = bool(torch.isfinite(full).all() and torch.isfinite(reduced).all())
        difference = float((full.float() - reduced.float()).abs().max())
        flips = int((full.argmax(-1) != reduced.argmax(-1)).sum())
        cfg = registration['short_diagnostic_head_parity']
        passed = finite and difference <= cfg['max_abs_diff_at_most'] and flips == 0
        receipt = {'finite': finite, 'max_abs_difference': difference, 'argmax_mismatches': flips,
                   'registered_max_abs': cfg['max_abs_diff_at_most'], 'status': 'PASS' if passed else 'FAIL'}
        atomic(out / 'head_parity.json', receipt)
        if not passed:
            raise ValueError('limited head parity failed; tolerance not changed')
    short_parity()

    @torch.inference_mode()
    def score_next(ids, candidates):
        h = hidden(ids)
        logp = F.log_softmax(w.model.lm_head(h[-1:]).float()[0], dim=-1)
        if not bool(torch.isfinite(logp).all()):
            raise ValueError('nonfinite next-token logprob')
        top = torch.topk(logp, 2)
        scored = {str(k): float(logp[k]) for k in candidates}
        result_row = {'logprob': scored, 'top2_ids': top.indices.cpu().tolist(),
                      'top2_logprob': top.values.cpu().tolist(), 'finite': True}
        del h, logp
        return result_row

    @torch.inference_mode()
    def score_sequence(prefix, continuation):
        if not continuation:
            raise ValueError('empty name continuation')
        ids = prefix + continuation
        h = hidden(ids)
        values = []
        # At most one full-vocab FP32 vector lives at a time.
        for i, token in enumerate(continuation):
            lp = F.log_softmax(w.model.lm_head(h[len(prefix) + i - 1:len(prefix) + i]).float()[0], -1)
            if not bool(torch.isfinite(lp).all()):
                raise ValueError('nonfinite name sequence logprob')
            values.append(float(lp[token])); del lp
        del h
        return {'total_logprob': math.fsum(values), 'mean_logprob': math.fsum(values) / len(values),
                'length': len(values), 'token_logprobs': values}

    def capture_initial(item):
        captures = {}; handles = []
        def hook(name):
            def save(module, args, output):
                if name in captures:
                    raise ValueError('projection visited twice in one initial forward')
                captures[name] = output[0, -1].detach().float().cpu().numpy().copy()
            return save
        for name in names:
            handles.append(w.model.get_submodule(name).register_forward_hook(hook(name)))
        try:
            h = hidden(item['input_ids']); del h
        finally:
            for handle in handles:
                handle.remove()
        if set(captures) != set(names) or any(not np.isfinite(x).all() for x in captures.values()):
            raise ValueError('invalid last-position projection capture')
        private = result / '_private'
        private.mkdir(exist_ok=True)
        if method == 'BF16':
            np.savez(private / ('projection_BF16_' + item['task_id'] + '.npz'), **captures)
        else:
            with np.load(private / ('projection_BF16_' + item['task_id'] + '.npz'), allow_pickle=False) as baseline:
                errors = {name: {'difference_sse': float(np.square(captures[name].astype('float64') - baseline[name].astype('float64')).sum()),
                                 'baseline_energy': float(np.square(baseline[name].astype('float64')).sum())}
                          for name in names}
            atomic(out / ('projection_errors_' + item['task_id'] + '.json'), errors)

    if stage == 'first':
        for item in inputs['natural']:
            if method in ('BF16', 'D16') and item['task_id'] in ('AGENT-6', 'AGENT-7'):
                capture_initial(item)
            response = w.generate(item['input_ids'], 192)
            if response['input_token_sha256'] != item['input_hash']:
                raise ValueError('natural prompt identity mismatch')
            calls = tool_blocks(response['raw_text'])
            append(out / 'first_responses.jsonl', dict(response, task_id=item['task_id'], method=method,
                parsed_tool_calls=calls, first_patch_correct=first_patch_correct(response, tasks[item['task_id']]),
                role='DIAGNOSTIC_DEV_NATURAL_FIRST_RESPONSE'))
            print({'stage': 'natural', 'method': method, 'task': item['task_id'],
                   'tool_names': [c.get('name') for c in calls]}, flush=True)
        for probe in inputs['controlled']:
            ids = [x['first_different_token'] for x in probe['candidates']]
            next_score = score_next(probe['prefix_ids'], ids)
            sequences = {c['name']: score_sequence(probe['prefix_ids'], c['continuation_ids']) for c in probe['candidates']}
            a, b = probe['candidates']
            margin = next_score['logprob'][str(a['first_different_token'])] - next_score['logprob'][str(b['first_different_token'])]
            append(out / 'controlled_name_scores.jsonl', {'method': method, 'task_id': probe['task_id'],
                'correct_name': probe['correct_name'], 'wrong_name': probe['wrong_name'],
                'prefix_hash': probe['prefix_hash'], 'prefix_ids': probe['prefix_ids'],
                'candidates': probe['candidates'], 'first_token_correct_minus_wrong_margin': margin,
                'sequence_scores': sequences, 'next_token': next_score,
                'externally_supplied_prefix': True, 'name_boundary': probe['candidate_boundary']})
        for probe in inputs['recovery']:
            response = w.generate(probe['input_ids'], 192)
            calls = tool_blocks(response['raw_text'])
            append(out / 'error_recovery.jsonl', dict(response, method=method, task_id=probe['task_id'],
                parsed_tool_calls=calls, registered_names_only=bool(calls) and all(c.get('name') in
                    ('read_document', 'read_record', 'query_table', 'calculate', 'apply_fixture_patch', 'validate_fixture_test') for c in calls),
                repeats_original_wrong_name=any(c.get('name') in ('ApplyFixturePatch', 'ValidateFixtureTest') for c in calls),
                external_common_error_history=True))
            print({'stage': 'recovery', 'method': method, 'task': probe['task_id'],
                   'tool_names': [c.get('name') for c in calls]}, flush=True)
    elif stage == 'common':
        probes = read(result / 'common_prefix_input_manifest.json')
        for probe in probes:
            if probe['status'] != 'DIVERGENCE':
                continue
            next_score = score_next(probe['prefix_ids'], [probe['baseline_next_id'], probe['candidate_next_id']])
            append(out / 'common_prefix_scores.jsonl', {k: v for k, v in probe.items() if k != 'prefix_ids'} |
                   {'method': method, 'score': next_score,
                    'baseline_token_minus_candidate_margin': next_score['logprob'][str(probe['baseline_next_id'])]
                                                           - next_score['logprob'][str(probe['candidate_next_id'])]})
    elif stage == 'p2':
        gate = read(result / 'p1_gate.json')
        if not gate['P2_initial_gate']:
            raise ValueError('P2 registration gate not passed')
        initial_all = read(w.run / 'inputs/agent_initial_messages.json')
        tools = inputs['tool_schema']
        # Preserve each unchanged adapter object; reset target before control.
        adapter_modules = {n: w.model.get_submodule(n) for n in original_modules}
        original_ids = {n: id(w.model.get_submodule(n)) for n in names}

        def restore(condition):
            selected = intervention_selection['target'] if condition == 'I1' else intervention_selection['control'] if condition == 'I2' else None
            for name, adapter in adapter_modules.items():
                parent, leaf = name.rsplit('.', 1)
                setattr(w.model.get_submodule(parent), leaf, adapter)
            if selected:
                parent, leaf = selected.rsplit('.', 1)
                from intervention import restore_projection
                restore_projection(w.model, selected, original_modules[selected], 'cuda')
            changed = [n for n in names if id(w.model.get_submodule(n)) != original_ids[n]]
            if changed != ([selected] if selected else []):
                raise ValueError('intervention modified unexpected module set')
            receipt = {'condition': condition, 'changed_modules': changed, 'whole_nn_Linear_restored': bool(selected),
                       'all_other_artifacts_identical': True, 'no_beta_on_restored_module': bool(selected),
                       'original_weight_tensor_sha256': hashlib.sha256(original_modules[selected].weight.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest() if selected else None}
            atomic(out / (condition + '-module_identity.json'), receipt)

        def episode(task, condition):
            t = time.monotonic(); config = out / condition / (task['id'] + '-FREE-config.json')
            atomic(config, task['fixture']['config'], fresh=True)
            messages = copy.deepcopy(initial_all[task['id']]); events = []
            calls = tool_count = parse_errors = tool_errors = total_in = total_out = 0
            model_seconds = tool_seconds = 0.; success = False; status = 'CALL_LIMIT'
            for turn in range(8):
                budget()
                if time.monotonic() - t > 900:
                    status = 'TIMEOUT'; break
                ids = w.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                    return_dict=False, add_generation_prompt=True, enable_thinking=False)
                if len(ids) + 192 > 24576:
                    status = 'CONTEXT_LIMIT'; break
                response = w.generate(ids, 192); events.append(response); calls += 1
                total_in += len(ids); total_out += response['output_tokens']; model_seconds += response['seconds']
                raw = response['raw_text']; blocks = tool_blocks(raw)
                if '<tool_call>' in raw:
                    parse_errors += sum(extract(s.split('</tool_call>')[0]) is None for s in raw.split('<tool_call>')[1:])
                if blocks:
                    accepted = []; answers = []
                    for obj in blocks:
                        if tool_count >= 6:
                            status = 'TOOL_LIMIT'; break
                        tool_count += 1; tool_t = time.monotonic()
                        name, args = obj.get('name'), obj.get('arguments')
                        try:
                            if not isinstance(args, dict):
                                raise ValueError('arguments not an object')
                            answer = w.tool(name, args, task['fixture'], config)
                        except (ValueError, TypeError) as error:
                            answer = {'error': str(error)}; tool_errors += 1
                        tool_seconds += time.monotonic() - tool_t
                        accepted.append({'name': name, 'arguments': args}); answers.append(answer)
                        events.append({'tool': name, 'arguments': args, 'result': answer})
                    if not accepted:
                        break
                    messages = tool_history(messages, accepted, answers)
                    continue
                final = original.json_result(response['text'])
                if final is None:
                    parse_errors += 1; status = 'PARSE_ERROR'
                else:
                    success = all(final.get(k) == v for k, v in task['gold'].items())
                    if task['family'] == 'fixture_patch':
                        success = success and read(config) == task['fixture']['expected_config'] and any(
                            e.get('tool') == 'validate_fixture_test' and e.get('result', {}).get('valid') for e in events)
                    status = 'SUCCESS' if success else 'INCORRECT_FINAL'
                break
            record = {'id': task['id'], 'family': task['family'], 'mode': 'FREE', 'method': 'D16-' + condition,
                'success': success, 'status': status, 'model_calls': calls, 'tool_calls': tool_count,
                'parse_errors': parse_errors, 'tool_errors': tool_errors, 'retry_count': 0,
                'total_input_tokens': total_in, 'total_output_tokens': total_out,
                'model_seconds': model_seconds, 'tool_seconds': tool_seconds, 'wall_seconds': time.monotonic() - t,
                'gold': task['gold'], 'events': events, 'cache_policy': 'OFF across turns; KV within each request',
                'role': 'DIAGNOSTIC_DEV_INTERVENTION', 'condition': condition, 'final_config': read(config)}
            append(out / 'p2_interventions.jsonl', record)
            print({'stage': 'p2', 'condition': condition, 'id': task['id'], 'status': status}, flush=True)
            return record

        outcomes = []
        for condition in ('I0', 'I1', 'I2'):
            restore(condition)
            new_rows = [episode(task, condition) for task in tasks.values()]
            outcomes.extend(new_rows)
            # Assert the exact module set again after execution.
            restore(condition)
            if condition == 'I0':
                by_id = {x['id']: x for x in new_rows}
                reproduced = all(not by_id[key]['success'] and any(e.get('tool') not in
                    ('read_document', 'read_record', 'query_table', 'calculate', 'apply_fixture_patch', 'validate_fixture_test')
                    for e in by_id[key]['events'] if 'tool' in e) for key in gate['gated_task_ids'])
                atomic(out / 'I0_gate.json', {'passed': reproduced, 'gated_task_ids': gate['gated_task_ids']})
                if not reproduced:
                    atomic(out / 'p2_decision.json', {'P2': 'NOT_RUN_GATE', 'reason': 'BASELINE_EPISODE_NOT_REPRODUCED'})
                    break
        if len(outcomes) == 24:
            atomic(out / 'p2_decision.json', {'P2': 'COMPLETED', 'conditions': 3, 'episodes': 24, 'unique_original_DEV_tasks': 8})
    else:
        raise ValueError('unknown stage')
    torch.cuda.synchronize()
    observe_memory()
    atomic(out / 'receipt.json', {'status': 'COMPLETE', 'stage': stage, 'method': method,
        'wall_seconds': time.monotonic() - started, 'loaded_artifacts': len(loaded),
        'peak_allocated_bytes': peaks['allocated'], 'peak_reserved_bytes': peaks['reserved'],
        'new_solver_calls': 0, 'artifact_writes': 0, 'serving_benefit': 'NOT_ESTABLISHED'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--stage', choices=('first', 'common', 'p2'), required=True)
    p.add_argument('--worker-budget-seconds', type=float, required=True)
    p.add_argument('--attempt', type=int, default=1)
    a = p.parse_args()
    try:
        execute(a.result, a.method, a.stage, a.worker_budget_seconds, a.attempt)
    except Exception as error:
        traceback.print_exc()
        path = a.result / 'workers' / a.stage / a.method / ('attempt' + str(a.attempt).zfill(2))
        atomic(path / 'failure.json', {'status': 'FAILED', 'exception': type(error).__name__, 'message': str(error),
                                     'traceback': traceback.format_exc()}, fresh=True)
        raise SystemExit(1)
