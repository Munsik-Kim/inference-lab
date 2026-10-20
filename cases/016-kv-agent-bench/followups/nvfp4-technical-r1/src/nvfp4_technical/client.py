"""Official completion SSE with original token IDs; CPU import is GPU-free."""
import asyncio
import hashlib
import json
import re
import time
from .common import digest, token_hash


PROMPT_ARRAY = re.compile(r'("prompt_token_ids"\s*:\s*)(\[[\d,\s]*\])')


def compact_sse(line, prompt):
    """Lossless content-addressed prompt-ID deduplication, including whitespace."""
    match = PROMPT_ARRAY.search(line)
    if not match:
        return {'line': line}
    array = match.group(2)
    if json.loads(array) != prompt['token_ids']:
        raise AssertionError('Server returned a different prompt')
    separator = ', ' if ', ' in array else ','
    if array != '[' + separator.join(str(i) for i in prompt['token_ids']) + ']':
        return {'line': line}  # Preserve any unusual formatting verbatim.
    return {'prefix': line[:match.start(2)], 'prompt_ref': prompt['prompt_id'], 'separator': separator, 'suffix': line[match.end(2):]}


def expand_sse(row, prompts):
    if 'line' in row:
        return row['line']
    return row['prefix'] + '[' + row['separator'].join(map(str, prompts[row['prompt_ref']]['token_ids'])) + ']' + row['suffix']


def parse_events(lines, prompt_ids):
    output, text, usage, finish, stop = [], '', None, None, None
    prompt_seen, done, times = False, False, []
    for elapsed, line in lines:
        if not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if payload == '[DONE]':
            done = True
            continue
        event = json.loads(payload)
        if 'error' in event:
            raise RuntimeError('Server SSE error: ' + str(event['error']))
        if event.get('usage'):
            usage = event['usage']
        for choice in event.get('choices', []):
            if choice['index'] != 0:
                raise AssertionError('Single-choice request received another choice')
            ids = choice.get('prompt_token_ids')
            if ids is not None:
                if ids != prompt_ids:
                    raise AssertionError('Actual prompt IDs mismatch')
                prompt_seen = True
            delta = choice.get('token_ids') or []
            if delta:
                output.extend(delta)
                times.append({'elapsed': elapsed, 'count': len(delta)})
            text += choice.get('text', '')
            if choice.get('finish_reason') is not None:
                finish, stop = choice['finish_reason'], choice.get('stop_reason')
    return {'output_token_ids': output, 'text': text, 'usage': usage, 'finish_reason': finish, 'stop_reason': stop,
            'prompt_ids_confirmed': prompt_seen, 'done': done, 'token_events': times}


def payload(prompt, output_tokens, ignore_eos):
    if len(prompt['token_ids']) + output_tokens > 28672:
        raise ValueError('Context exceeds registered limit; no truncation')
    return {'model': 'case016-tech-qwen3-4b', 'prompt': prompt['token_ids'], 'add_special_tokens': False,
            'temperature': 0, 'max_tokens': output_tokens, 'ignore_eos': ignore_eos, 'stop_token_ids': [],
            'stream': True, 'stream_interval': 1, 'stream_options': {'include_usage': True}, 'return_token_ids': True}


async def request(client, base, prompt, identity, semaphore, block_start, output_tokens=128, ignore_eos=True, timeout=180):
    row = {**identity, 'prompt_id': prompt['prompt_id'], 'input_token_sha256': prompt['input_token_sha256'],
           'model_request_attempts': 0, 'responses_received': 0, 'output_limit': output_tokens, 'ignore_eos': ignore_eos,
           'queued': time.monotonic() - block_start, 'sse': [], 'status': 'UNKNOWN'}
    body = payload(prompt, output_tokens, ignore_eos)
    row['request_sha256'] = digest(body)
    started = time.monotonic()
    raw = []
    try:
        async with asyncio.timeout(timeout):
            async with semaphore:
                row['slot_acquired'] = time.monotonic() - block_start
                row['http_dispatch'] = time.monotonic() - block_start
                row['model_request_attempts'] += 1
                async with client.stream('POST', base + '/v1/completions', json=body, timeout=timeout) as response:
                    row['http_status'] = response.status_code
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        elapsed = time.monotonic() - block_start
                        raw.append((elapsed, line))
                        row['sse'].append({'elapsed': elapsed, **compact_sse(line, prompt)})
                    row['responses_received'] += 1
        parsed = parse_events(raw, prompt['token_ids'])
        row.update(parsed)
        usage = parsed['usage'] or {}
        observed = len(parsed['output_token_ids'])
        valid = parsed['prompt_ids_confirmed'] and parsed['done'] and usage.get('completion_tokens') == observed and usage.get('prompt_tokens') == len(prompt['token_ids'])
        if ignore_eos:
            valid = valid and observed == output_tokens and parsed['finish_reason'] == 'length'
        row['status'] = 'SUCCESS' if valid else 'RAW_OR_LENGTH_MISMATCH'
        if parsed['token_events']:
            first, last = parsed['token_events'][0]['elapsed'], parsed['token_events'][-1]['elapsed']
            row['client_ttft_seconds'] = first - row['http_dispatch']
            row['client_decode_mean_seconds'] = (last - first) / (observed - 1) if observed > 1 else None
    except Exception as exc:
        row['status'] = 'TIMEOUT' if isinstance(exc, TimeoutError) else 'INFRA_ERROR'
        row['error_type'] = type(exc).__name__
        row['error'] = str(exc)
    row['terminal'] = time.monotonic() - block_start
    row['wall_seconds_including_slot_wait'] = time.monotonic() - started
    row['raw_sse_sha256'] = hashlib.sha256(('\n'.join(line for _, line in raw) + '\n').encode()).hexdigest()
    if 'output_token_ids' in row:
        row['output_token_sha256'] = token_hash(row['output_token_ids'])
    return row
