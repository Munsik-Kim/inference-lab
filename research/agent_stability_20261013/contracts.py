"""Small CPU contracts shared by registration, GPU diagnostics and readback."""
import hashlib
import json
import math
from pathlib import Path
import random
import struct

METHODS = ('BF16', 'RTN', 'D16', 'D4')
TASK_IDS = ('AGENT-6', 'AGENT-7', 'AGENT-2', 'AGENT-4')
SEED = 610132
PROJECTIONS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(2**20), b''):
            h.update(data)
    return h.hexdigest()


def token_hash(ids):
    return hashlib.sha256(struct.pack('<' + 'i' * len(ids), *ids)).hexdigest()


def atomic(path, value, fresh=False):
    path = Path(path)
    if fresh and path.exists():
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    partial.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def extract(text):
    d = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '{':
            try:
                obj, _ = d.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                pass
    return None


def tool_blocks(raw):
    return [obj for segment in raw.split('<tool_call>')[1:]
            if (obj := extract(segment.split('</tool_call>')[0])) is not None]


def tool_history(messages, calls, results):
    # This is the frozen harness representation, not the model's explanatory text.
    return list(messages) + [dict(role='assistant', content='', tool_calls=[
        {'type': 'function', 'function': {'name': c['name'], 'arguments': c['arguments']}}
        for c in calls])] + [dict(role='tool', name=c['name'], content=json.dumps(answer))
                              for c, answer in zip(calls, results)]


def common_length(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def first_patch_correct(record, task):
    calls = tool_blocks(record['raw_text'])
    return bool(calls and calls[0].get('name') == 'apply_fixture_patch'
                and calls[0].get('arguments') == task['fixture']['expected_config'])


def worker_dir(result, stage, method):
    parent = Path(result) / 'workers' / stage / method
    completed = [p for p in parent.glob('attempt*') if (p / 'receipt.json').is_file()
                 and read(p / 'receipt.json')['status'] == 'COMPLETE']
    if len(completed) != 1:
        raise ValueError('need exactly one completed worker attempt: ' + str(parent))
    return completed[0]


def select_projection(errors, metadata, seed=SEED):
    """Two-context mean SSE/energy; selection never reads rollout correctness."""
    usable = []
    excluded = []
    names = set(errors['AGENT-6'])
    if names != set(errors['AGENT-7']) or len(names) != 252:
        raise ValueError('projection/context coverage mismatch')
    for name in names:
        values = [errors[k][name] for k in ('AGENT-6', 'AGENT-7')]
        if any(v['baseline_energy'] == 0 for v in values):
            excluded.append(name)
            continue
        if any(v['baseline_energy'] < 0 or v['difference_sse'] < 0 or
               not all(math.isfinite(v[k]) for k in ('baseline_energy', 'difference_sse')) for v in values):
            raise ValueError('nonfinite/invalid selection metric')
        usable.append((sum(v['difference_sse'] / v['baseline_energy'] for v in values) / 2, name))
    if not usable:
        raise ValueError('no eligible nonzero-energy projection')
    score, target = sorted(usable, key=lambda x: (-x[0], x[1]))[0]
    entries = {r['layer']: r for r in metadata['files']}
    row = entries[target]
    candidates = sorted(n for n, r in entries.items() if n != target
                        and n.split('.')[-1] == target.split('.')[-1]
                        and n.split('.')[2] != target.split('.')[2]
                        and r['shape'] == row['shape'] and r['bytes'] == row['bytes'])
    control = random.Random(seed).choice(candidates) if candidates else None
    bf16_bytes = math.prod(row['shape']) * 2
    return {'target': target, 'control': control, 'mean_relative_output_error': score,
            'metric': 'mean over AGENT-6/7 of output difference SSE / BF16 output energy',
            'includes_upstream_input_difference': True, 'shape': row['shape'],
            'original_D16_payload_bytes': row['bytes'], 'nominal_BF16_weight_bytes': bf16_bytes,
            'nominal_storage_increase_bytes': bf16_bytes - row['bytes'],
            'matched_control_candidates': candidates, 'control_seed': seed,
            'zero_energy_excluded': sorted(excluded),
            'cost_scope': 'nominal projection replacement ledger, not packed artifact or measured VRAM'}
