"""Read-only saved-artifact loading and paired tool-contract mitigation DEV."""
import argparse
import copy
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
from guard import feedback, processor


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def append(p, x):
    with Path(p).open('a') as f:
        f.write(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n')


def execute(result, method, attempt, budget_seconds):
    start = time.monotonic()
    reg = read(result / 'preregistration.json'); runtime = read(result / 'runtime_manifest.json')
    if sha(result / 'runtime_manifest.json') != reg['runtime_sha256'] or sha(result / 'input_manifest.json') != reg['input_manifest_sha256']:
        raise ValueError('registered inputs changed')
    for n, h in reg['code_hashes'].items():
        if sha(Path(__file__).parent / n) != h:
            raise ValueError('registered code changed: ' + n)
    source = Path(runtime['research_source'])
    for n, h in runtime['source_hashes'].items():
        if sha(source / n) != h:
            raise ValueError('original source changed: ' + n)
    sys.path.insert(0, str(source))
    original = importlib.import_module('paper_model_runner')
    quant = importlib.import_module('paper_int4')
    reference = importlib.import_module('reference_int4')
    for name, m in [('paper_model_runner', original), ('paper_int4', quant), ('reference_int4', reference)]:
        if Path(m.__file__).resolve() != source / (name + '.py'):
            raise ValueError('wrong original module import')
    import torch
    from transformers import LogitsProcessor
    sys.path.insert(0, str(Path(__file__).parents[1] / 'agent_stability_20261013'))
    from contracts import token_hash, tool_blocks, tool_history
    out = result / 'workers' / method / ('attempt' + str(attempt).zfill(2))
    out.mkdir(parents=True, exist_ok=False)
    gen = reg['generation']; peaks = {'allocated': 0, 'reserved': 0}

    def budget():
        if time.monotonic() - start >= budget_seconds - 2:
            raise TimeoutError('registered worker wall budget')

    class FiniteBudget(LogitsProcessor):
        def __call__(self, ids, scores):
            budget()
            if not bool(torch.isfinite(scores).all()):
                raise ValueError('nonfinite unmasked generation logits')
            return scores

    class Worker(original.Worker):
        def apply_artifacts(self):
            raise RuntimeError('solver and artifact writing forbidden')

    w = Worker(Path(runtime['original_result_root']), '4b', method, output_override=out)
    names = [n for n, m in w.model.named_modules() if isinstance(m, torch.nn.Linear) and n.split('.')[-1] in quant.PROJECTIONS]
    if len(names) != 252:
        raise ValueError('expected 252 projection modules')
    loaded = []
    if method != 'BF16':
        am = 'RTN' if method == 'RTN' else 'D16'
        directory = Path(runtime['artifact_root']) / am
        identity = runtime['artifacts'][am]
        if sha(directory / 'manifest.json') != identity['manifest_sha256']:
            raise ValueError('original artifact manifest changed')
        entries = {x['layer']: x for x in identity['per_projection_identity']}
        if set(entries) != set(names) or {p.stem for p in directory.glob('*.rpi4')} != set(names):
            raise ValueError('missing/extra artifact')
        for i, name in enumerate(names):
            budget(); data = (directory / (name + '.rpi4')).read_bytes(); entry = entries[name]
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError('artifact identity mismatch: ' + name)
            q = reference.QuantizedWeight.deserialize(data)
            if list(q.shape) != entry['shape'] or q.dtype != torch.float32 or q.method != ('rtn' if am == 'RTN' else 'row_column'):
                raise ValueError('artifact precision/shape/method mismatch')
            quant.replace_projection(w.model, name, q, method == 'D4')
            loaded.append(entry); del data, q
            if i % 70 == 0:
                print({'method': method, 'projection_loaded': i + 1}, flush=True)
    for n, p in w.model.named_parameters():
        if n in w.original_small_hashes and hashlib.sha256(p.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest() != w.original_small_hashes[n]:
            raise ValueError('norm changed')
    if (w.model.model.embed_tokens.weight.data_ptr() == w.model.lm_head.weight.data_ptr()) != w.base_shared_embedding:
        raise ValueError('embedding tie changed')
    gc.collect(); w.model.cuda(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    parameter_identity = [(n, id(p), p._version) for n, p in w.model.named_parameters()]
    atomic(out / 'artifact_readback.json', {'status': 'PASS', 'loaded': loaded, 'solver_calls': 0, 'artifact_writes': 0})
    atomic(out / 'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
        'threads': torch.get_num_threads(), 'TF32': torch.backends.cuda.matmul.allow_tf32,
        'BF16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'runtime': 'dequantized BF16 reference, not native packed',
        'device_memory_at_load': list(torch.cuda.mem_get_info()), 'device_wide_peak': None})
    inputs = read(result / 'input_manifest.json')
    for n, entry in inputs.items():
        if sha(w.run / 'inputs' / n) != entry['sha256']:
            raise ValueError('frozen fixture changed')
    tasks = read(w.run / 'inputs/agent_tasks.json')
    initial = read(w.run / 'inputs/agent_initial_messages.json')
    tools = read(w.run / 'inputs/tool_schema.json')
    tool_names = tuple(t['function']['name'] for t in tools)
    old = [json.loads(line) for line in (w.run / 'raw/4b' / method / 'agent_dev.jsonl').read_text().splitlines()]
    old = {r['id']: r for r in old if r['mode'] == 'FREE'}
    enum_shared = {}

    @torch.inference_mode()
    def generate(ids, use_enum):
        budget()
        if len(ids) + gen['max_new_tokens'] > gen['context_limit']:
            raise ValueError('CONTEXT_LIMIT; no truncate')
        ids_gpu = torch.tensor(ids, device='cuda').unsqueeze(0)
        enum = processor(w.tokenizer, tool_names, len(ids), enum_shared) if use_enum else None
        processes = [FiniteBudget()] + ([enum] if enum else [])
        torch.cuda.synchronize(); t = time.monotonic()
        output = w.model.generate(ids_gpu, attention_mask=torch.ones_like(ids_gpu), do_sample=False,
            max_new_tokens=gen['max_new_tokens'], use_cache=True, logits_to_keep=1,
            pad_token_id=w.tokenizer.eos_token_id, return_dict_in_generate=False, logits_processor=processes)
        torch.cuda.synchronize()
        new = output[0, len(ids):].cpu().tolist()
        return {'text': w.tokenizer.decode(new, skip_special_tokens=True),
            'raw_text': w.tokenizer.decode(new, skip_special_tokens=False), 'output_token_ids': new,
            'input_token_sha256': token_hash(ids), 'input_tokens': len(ids), 'output_tokens': len(new),
            'seconds': time.monotonic() - t, 'status': 'COMPLETE',
            'enum_trace': enum.trace if enum else [], 'enum_applied': use_enum}

    for condition in reg['methods'][method]:
        for task in tasks:
            t = time.monotonic(); config = out / condition / (task['id'] + '-config.json')
            if config.exists():
                raise FileExistsError('preserve existing fixture attempt')
            atomic(config, task['fixture']['config'])
            messages = copy.deepcopy(initial[task['id']]); events = []
            calls = tc = parse_errors = tool_errors = unregistered = 0
            ti = to = 0; model_s = tool_s = 0.; success = False; status = 'CALL_LIMIT'
            first_equal = None; first_hash = None
            for turn in range(gen['max_model_calls']):
                budget()
                if time.monotonic() - t > gen['deadline_seconds']:
                    status = 'TIMEOUT'; break
                ids = w.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                    return_dict=False, add_generation_prompt=True, enable_thinking=False)
                response = generate(ids, condition in ('ENUM', 'BOTH'))
                if turn == 0:
                    first_hash = token_hash(ids)
                    first_equal = response['output_token_ids'] == old[task['id']]['events'][0].get('output_token_ids')
                    expected_hash = old[task['id']]['events'][0].get('input_token_sha256')
                    if expected_hash and first_hash != expected_hash:
                        raise ValueError('original first input mismatch')
                events.append(response); calls += 1; ti += len(ids); to += response['output_tokens']; model_s += response['seconds']
                raw = response['raw_text']; blocks = tool_blocks(raw)
                if '<tool_call>' in raw:
                    parse_errors += sum(original.json_result(s.split('</tool_call>')[0]) is None for s in raw.split('<tool_call>')[1:])
                if blocks:
                    accepted = []; answers = []
                    for obj in blocks:
                        if tc >= gen['max_tool_calls']:
                            status = 'TOOL_LIMIT'; break
                        tc += 1; tool_t = time.monotonic(); name = obj.get('name'); args = obj.get('arguments')
                        unregistered += name not in tool_names
                        if condition in ('ENUM', 'BOTH') and name not in tool_names:
                            raise ValueError('enum decoder emitted an unregistered name; no aliases applied')
                        try:
                            if not isinstance(args, dict):
                                raise ValueError('arguments not an object')
                            answer = w.tool(name, args, task['fixture'], config)
                        except (ValueError, TypeError) as e:
                            tool_errors += 1
                            answer = feedback(name, args, tools, e) if condition in ('FEEDBACK', 'BOTH') else {'error': str(e)}
                        tool_s += time.monotonic() - tool_t
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
            row = {'id': task['id'], 'family': task['family'], 'mode': 'FREE', 'method': method,
                'condition': condition, 'success': bool(success), 'status': status, 'model_calls': calls,
                'tool_calls': tc, 'tool_errors': tool_errors, 'unregistered_name_events': unregistered,
                'parse_errors': parse_errors, 'total_input_tokens': ti, 'total_output_tokens': to,
                'model_seconds': model_s, 'tool_seconds': tool_s, 'wall_seconds': time.monotonic() - t,
                'first_input_hash': first_hash, 'first_output_equals_original': first_equal,
                'gold': task['gold'], 'events': events, 'final_config': read(config),
                'role': 'MITIGATION_DEV', 'answer_correction': False, 'aliases': False}
            append(out / 'episodes.jsonl', row)
            print({'method': method, 'condition': condition, 'task': task['id'], 'success': success,
                   'tool_errors': tool_errors, 'unknown_names': unregistered}, flush=True)
        if parameter_identity != [(n, id(p), p._version) for n, p in w.model.named_parameters()]:
            raise ValueError('model parameters changed between mitigation conditions')
    peaks = {'allocated': torch.cuda.max_memory_allocated(), 'reserved': torch.cuda.max_memory_reserved()}
    atomic(out / 'receipt.json', {'status': 'COMPLETE', 'method': method, 'wall_seconds': time.monotonic() - start,
        'peak': peaks, 'parameters_unchanged': True, 'new_solver_calls': 0, 'serving_benefit': 'NOT_ESTABLISHED'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--method', choices=('BF16', 'RTN', 'D16', 'D4'), required=True)
    p.add_argument('--attempt', type=int, default=1)
    p.add_argument('--budget-seconds', type=float, required=True)
    a = p.parse_args()
    try:
        execute(a.result, a.method, a.attempt, a.budget_seconds)
    except Exception as e:
        traceback.print_exc()
        atomic(a.result / 'workers' / a.method / ('attempt' + str(a.attempt).zfill(2)) / 'failure.json',
               {'exception': type(e).__name__, 'message': str(e), 'traceback': traceback.format_exc()})
        raise SystemExit(1)
