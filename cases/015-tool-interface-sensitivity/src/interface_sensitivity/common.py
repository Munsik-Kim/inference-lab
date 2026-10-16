"""CPU-only I/O and canonical identities. No implicit repair or downloads."""
import datetime
import hashlib
import json
from pathlib import Path
import struct

CASE = Path(__file__).resolve().parents[2]
METHODS = ('BF16', 'RTN', 'D16')
INTERFACES = ('I0_ORIGINAL', 'I1_ORDER_ONLY', 'I2_STYLE_RENAME', 'I3_OPAQUE_RENAME')
HARNESS = 'case015-strict-functional-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def token_hash(ids):
    return hashlib.sha256(struct.pack('<' + 'i' * len(ids), *ids)).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, value, fresh=False):
    path = Path(path)
    if fresh and path.exists():
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    partial.replace(path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def extract(text):
    # Same first-valid-JSON-object parser as the pinned Case013 baseline.
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '{':
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                pass
    return None


def parse_calls(raw):
    calls, failures = [], []
    for i, segment in enumerate(raw.split('<tool_call>')[1:]):
        obj = extract(segment.split('</tool_call>')[0])
        if obj is None:
            failures.append({'index': i, 'label': 'PARSE_ERROR'})
        else:
            calls.append(obj)
    return calls, failures


def history(messages, calls, results):
    # Preserve original assistant/tool serialization; no explanatory text or hints.
    return list(messages) + [{'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': c.get('name'), 'arguments': c.get('arguments')}} for c in calls]}] + [
        {'role': 'tool', 'name': c.get('name'), 'content': json.dumps(result)} for c, result in zip(calls, results)]


def inputs(case=CASE):
    case = Path(case)
    return read(case / 'data/agent_tasks.json'), read(case / 'data/agent_initial_messages.json'), read(case / 'data/tool_schema.json')
