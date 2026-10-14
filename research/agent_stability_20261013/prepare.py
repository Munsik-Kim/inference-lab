"""CPU-only preparation. Freeze inputs/contracts before the first GPU forward."""
import argparse
import datetime
from pathlib import Path
from contracts import atomic, read, rows, sha, token_hash, tool_blocks, tool_history, TASK_IDS, METHODS, SEED


def prepare(research, result, case):
    from transformers import AutoTokenizer
    research, result, case = map(Path, (research, result, case))
    r2 = research / 'results/paper_int4_reference_r2'
    original = read(r2 / 'experiment_manifest.json')
    model = original['models']['4b']
    model_path = Path(model['path'])
    if model['revision'] != 'cdbee75f17c01a7cc42f958dc650907174af0554':
        raise ValueError('unexpected base revision')
    for name, identity in model['tokenizer_files'].items():
        if sha(model_path / name) != identity['sha256']:
            raise ValueError('changed config/tokenizer: ' + name)
    for name in ('reference_int4.py', 'paper_int4.py', 'paper_model_runner.py'):
        if sha(research / 'source' / name) != sha(case / 'source' / name):
            raise ValueError('public/local reference source mismatch: ' + name)
    artifacts = {}
    for method in ('RTN', 'D16'):
        directory = r2 / 'artifacts/4b' / method
        meta = read(directory / 'manifest.json')
        entries = {x['layer']: x for x in meta['files']}
        if len(entries) != 252 or set(p.stem for p in directory.glob('*.rpi4')) != set(entries):
            raise ValueError('missing/extra saved projection artifacts')
        for name, entry in entries.items():
            path = directory / (name + '.rpi4')
            if path.stat().st_size != entry['bytes']:
                raise ValueError('changed artifact size: ' + name)
        # Every actual payload SHA is checked at read time, before application.
        artifacts[method] = {'manifest_sha256': sha(directory / 'manifest.json'),
                             'projection_count': 252, 'payload_bytes': meta['total_payload_bytes'],
                             'per_projection_identity': [{'layer': x['layer'], 'shape': x['shape'],
                                                          'bytes': x['bytes'], 'sha256': x['sha256']}
                                                         for x in meta['files']]}
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tools = read(r2 / 'inputs/tool_schema.json')
    initial = read(r2 / 'inputs/agent_initial_messages.json')
    tasks = {x['id']: x for x in read(r2 / 'inputs/agent_tasks.json')}
    frozen = {x['id']: x for x in read(r2 / 'inputs/agent_token_manifest.json')}
    saved = {x['id']: x for x in rows(r2 / 'raw/4b/D16/agent_dev.jsonl') if x['mode'] == 'FREE'}
    def tokens(messages):
        ids = tok.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=False,
                                      add_generation_prompt=True, enable_thinking=False)
        text = tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
        if tok(text, add_special_tokens=False)['input_ids'] != ids:
            raise ValueError('chat template string/token mismatch')
        return ids, text
    natural = []
    for key in TASK_IDS:
        ids, _ = tokens(initial[key])
        if token_hash(ids) != frozen[key]['initial_token_sha256']:
            raise ValueError('initial model input mismatch: ' + key)
        natural.append({'task_id': key, 'input_ids': ids, 'input_hash': token_hash(ids),
                        'tokens': len(ids), 'messages': initial[key]})
    controlled = []
    recovery = []
    for key in ('AGENT-6', 'AGENT-7'):
        task = tasks[key]
        common_patch_result = {'applied': True, 'config': task['fixture']['expected_config']}
        patched = tool_history(initial[key], [{'name': 'apply_fixture_patch',
                                               'arguments': task['fixture']['expected_config']}],
                               [common_patch_result])
        for correct, wrong, messages in (
                ('apply_fixture_patch', 'ApplyFixturePatch', initial[key]),
                ('validate_fixture_test', 'ValidateFixtureTest', patched)):
            ids, text = tokens(messages)
            start = text + '<tool_call>\n{"name": "'
            alternatives = {name: tok(start + name + '"', add_special_tokens=False)['input_ids']
                            for name in (correct, wrong)}
            from contracts import common_length
            n = common_length(alternatives[correct], alternatives[wrong])
            if n < len(ids) or any(a[:len(ids)] != ids for a in alternatives.values()):
                raise ValueError('candidate tokenization changes original prompt')
            prefix = alternatives[correct][:n]
            controlled.append({'task_id': key, 'correct_name': correct, 'wrong_name': wrong,
                               'prefix_ids': prefix, 'prefix_hash': token_hash(prefix),
                               'candidates': [{'name': name, 'continuation_ids': a[n:],
                                               'first_different_token': a[n]} for name, a in alternatives.items()],
                               'candidate_boundary': 'whole prompt+name+closing quote tokenized jointly; shared BPE prefix removed',
                               'messages': messages})
        first = next(e for e in saved[key]['events'] if 'text' in e)
        bad_calls = tool_blocks(first['raw_text'])
        actual_tools = [e for e in saved[key]['events'][1:1 + len(bad_calls)] if 'tool' in e]
        if len(actual_tools) != len(bad_calls) or not bad_calls:
            raise ValueError('cannot reconstruct original first error turn')
        for call, event in zip(bad_calls, actual_tools):
            if call['name'] != event['tool'] or call['arguments'] != event['arguments'] \
                    or event['result'] != {'error': 'unknown tool/incorrect argument keys'}:
                raise ValueError('original error feedback mismatch')
        history = tool_history(initial[key], bad_calls, [e['result'] for e in actual_tools])
        ids, _ = tokens(history)
        recovery.append({'task_id': key, 'input_ids': ids, 'input_hash': token_hash(ids),
                         'messages': history, 'original_wrong_tool_calls': bad_calls,
                         'original_error_results': [e['result'] for e in actual_tools]})
    inputs = {'natural': natural, 'controlled': controlled, 'recovery': recovery,
              'tool_schema': tools, 'task_ids': list(TASK_IDS)}
    atomic(result / 'input_manifest.json', inputs, fresh=True)
    atomic(result / 'runtime_manifest.json', {'model': model, 'research_source': str(research / 'source'),
        'original_result_root': str(r2), 'artifact_root': str(r2 / 'artifacts/4b'),
        'source_hashes': {name: sha(research / 'source' / name) for name in
                          ('reference_int4.py', 'paper_int4.py', 'paper_model_runner.py')},
        'reference_contract_hash': sha(r2 / 'reference_contract.json'), 'artifacts': artifacts}, fresh=True)
    registration = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'role': 'DIAGNOSTIC_DEV: purposefully selected previously observed DEV; no held-out/formal120 tasks',
        'methods': list(METHODS), 'tasks': list(TASK_IDS), 'input_manifest_sha256': sha(result / 'input_manifest.json'),
        'runtime_manifest_sha256': sha(result / 'runtime_manifest.json'), 'seed': SEED,
        'generation': original['generation'], 'cache': 'OFF across turns; ON within request',
        'dtype': 'BF16 reference weights; D16 input transport FP32; D4 existing activation variant',
        'reduction': {'TF32': False, 'cudnn_TF32': False, 'BF16_reduced_precision_reduction': False},
        'short_diagnostic_head_parity': {'input_tokens': 64, 'selected_last_positions': 16,
                                        'max_abs_diff_at_most': 0.25, 'argmax_mismatches': 0,
                                        'same_allowance_as_original_quality_runner': True},
        'logprob': 'FP32 log_softmax over full vocab at requested positions only; no two-name renormalization',
        'controlled_primary': 'total conditional logprob of jointly tokenized name remainder+closing quote; mean/length supplementary',
        'natural_divergence': 'BF16/RTN vs D16/D4, first different response ID; same preceding IDs teacher-forced in all methods; presentation divergence distinguished',
        'common_prefix_freeze': 'deterministic derivation from newly saved natural responses, frozen once before common-prefix score forwards',
        'P2_gate': ['input/artifact/dtype/head parity verified for all four methods',
                    'all required P1 measurements complete',
                    'AGENT-6 or7 first action: BF16+RTN call registered patch with exact arguments, D16 selects an unregistered first tool name',
                    'I0 original eight FREE tasks reproduce episode failure for the gated task(s)', 'worker budget remains'],
        'P2_selection': {'contexts': ['AGENT-6', 'AGENT-7'], 'position': 'last initial input token',
                         'metric': 'mean of two output SSE-difference/BF16 energy ratios; includes upstream input changes',
                         'target': 'largest metric, lexicographic full module name tie break',
                         'zero_energy': 'exclude module if either context energy is zero',
                         'control': 'other block, same type/shape/original D16 payload bytes, lexicographic candidate list + Random(610132).choice',
                         'seed': SEED, 'rollout_scores_used_for_selection': False},
        'P2_comparison': 'new I0 D16, target I1 and control I2 whole BF16 nn.Linear restored; original8 FREE tasks once each',
        'budget_worker_wall_seconds': 7200, 'priority': ['P0', 'P1', 'conditional P2'],
        'no_solver_or_artifact_writes': True, 'new_models_downloaded': 0, 'formal_serving_benefit': 'NOT_ESTABLISHED'}
    code = Path(__file__).resolve().parent
    registration['followup_code_hashes'] = {p.name: sha(p) for p in sorted(code.glob('*.py'))}
    for name in registration['followup_code_hashes']:
        target = result / 'registered_source' / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((code / name).read_bytes())
    atomic(result / 'preregistration.json', registration, fresh=True)
    print({'status': 'REGISTERED_BEFORE_GPU_FORWARD', 'natural': len(natural),
           'controlled': len(controlled), 'recovery': len(recovery), 'artifact_projections': 504})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--case', type=Path, required=True)
    a = p.parse_args()
    prepare(a.research, a.result, a.case)
