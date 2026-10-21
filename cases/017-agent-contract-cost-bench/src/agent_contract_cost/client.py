"""CPU-safe raw chat/SSE client; no native tool parser or automatic retry.

Adapted from Case016 StreamAssembly/chat, with a single content envelope shared
by every policy. Runtime imports of httpx and tokenizer helpers are explicit.
"""
import asyncio
import copy
import hashlib
import json
import time
import urllib.parse
from collections.abc import Mapping


MODEL_NAME = 'case017-qwen3-4b'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def loopback(base):
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.username or parsed.password or parsed.path not in ('', '/'):
        raise ValueError('Only an owned HTTP loopback base is allowed')
    return base.rstrip('/')


def build_payload(messages, schema=None, cap=512, request_id=None):
    """Public messages only. Tool declarations belong to shared system content."""
    if type(cap) is not int or not 1 <= cap <= 512:
        raise ValueError('Registered cap must be a positive integer <=512')
    if not isinstance(messages, list) or any(set(m) - {'role', 'content'} for m in messages):
        raise ValueError('Only role/content message fields are accepted')
    if any(m.get('role') not in ('system', 'user', 'assistant') or not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('Messages must contain text roles and content')
    result = {'model': MODEL_NAME, 'messages': copy.deepcopy(messages),
              'temperature': 0, 'top_p': 1, 'n': 1, 'max_tokens': cap,
              'ignore_eos': False, 'stream': True, 'stream_interval': 1,
              'stream_options': {'include_usage': True}, 'return_token_ids': True,
              'return_prompt_text': True, 'skip_special_tokens': True,
              'chat_template_kwargs': {'enable_thinking': False}}
    if request_id is not None:
        result['request_id'] = request_id
    if schema is not None:
        if not isinstance(schema, dict):
            raise TypeError('G schema must be an object')
        result['structured_outputs'] = {'json': copy.deepcopy(schema)}
    return result


def request_cap(configured, remaining_tokens, context_limit, prompt_tokens):
    limits = {'configured_cap': configured, 'remaining_episode_tokens': remaining_tokens,
              'remaining_context_tokens': context_limit - prompt_tokens}
    value = min(limits.values())
    return value, limits, [name for name, n in limits.items() if n == value]


class StreamAssembly:
    """One choice, content only. Every token delta is appended exactly once."""
    def __init__(self):
        self.events, self.token_ids, self.token_events = [], [], []
        self.content = ''
        self.prompt_token_ids = self.prompt_text = self.usage = None
        self.finish_reason = self.stop_reason = None
        self.first_role = self.first_token = None
        self.done = False
        self.response_id = None

    def add(self, obj, elapsed):
        if 'error' in obj:
            raise RuntimeError('Server SSE error: ' + str(obj['error']))
        self.events.append({'elapsed_seconds': elapsed, 'data': obj})
        if obj.get('id'):
            if self.response_id not in (None, obj['id']):
                raise ValueError('One HTTP response contains different response IDs')
            self.response_id = obj['id']
        if obj.get('usage') is not None:
            self.usage = obj['usage']
        if obj.get('prompt_token_ids') is not None:
            ids = obj['prompt_token_ids']
            if self.prompt_token_ids is not None and self.prompt_token_ids != ids:
                raise ValueError('Prompt token IDs changed within one response')
            self.prompt_token_ids = ids
        if obj.get('prompt_text') is not None:
            if self.prompt_text is not None and self.prompt_text != obj['prompt_text']:
                raise ValueError('Rendered prompt changed within one response')
            self.prompt_text = obj['prompt_text']
        for choice in obj.get('choices', []):
            if choice.get('index') != 0:
                raise ValueError('Only the registered single choice is supported')
            delta = choice.get('delta') or {}
            if delta.get('tool_calls') or delta.get('function_call'):
                raise ValueError('Native tool-call extraction is forbidden in Case017')
            if delta.get('reasoning') or delta.get('reasoning_content'):
                raise ValueError('Thinking must remain disabled for every arm')
            if delta.get('role') is not None and self.first_role is None:
                self.first_role = elapsed
            ids = choice.get('token_ids') or []
            if ids:
                if any(type(i) is not int for i in ids):
                    raise ValueError('Non-integer output token ID')
                self.token_ids.extend(ids)
                self.token_events.append({'elapsed_seconds': elapsed, 'count': len(ids)})
                if self.first_token is None:
                    self.first_token = elapsed
            if delta.get('content') is not None:
                self.content += delta['content']
            if choice.get('finish_reason') is not None:
                self.finish_reason, self.stop_reason = choice['finish_reason'], choice.get('stop_reason')

    def result(self):
        return {'content': self.content, 'output_token_ids': self.token_ids,
                'prompt_token_ids': self.prompt_token_ids, 'prompt_text': self.prompt_text,
                'usage': self.usage, 'server_finish_reason': self.finish_reason,
                'server_stop_reason': self.stop_reason, 'response_id': self.response_id,
                'sse_events': self.events, 'stream_done': self.done,
                'token_events': self.token_events, 'first_role_chunk_seconds': self.first_role,
                'first_generated_token_id_event_seconds': self.first_token}


def consume_sse_line(assembly, line, elapsed):
    if not line.startswith('data:'):
        return
    data = line[5:].strip()
    if data == '[DONE]':
        assembly.done = True
    elif data:
        assembly.add(json.loads(data), elapsed)


def parse_sse(lines):
    assembly = StreamAssembly()
    for row in lines:
        if isinstance(row, str):
            consume_sse_line(assembly, row, 0)
        else:
            consume_sse_line(assembly, row['line'], row['elapsed_seconds'])
    return assembly.result()


def decoded_raw(response, tokenizer):
    """Keep special-token-preserving decode separate from parser content."""
    return tokenizer.decode(response['output_token_ids'], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)


def tokenize_messages(messages, tokenizer):
    """Actual pinned chat template, identical to request enable_thinking=False."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       enable_thinking=False, return_dict=False)
    # Transformers 5.19 defaults to BatchEncoding; request a list explicitly and
    # handle mapping-shaped compatibility implementations without hashing an
    # opaque object or accepting a batched/padded input silently.
    if isinstance(ids, Mapping):
        if 'input_ids' not in ids:
            raise ValueError('Chat template mapping omitted input_ids')
        ids = ids['input_ids']
    if not isinstance(ids, (list, tuple)) or any(type(token) is not int for token in ids):
        raise ValueError('Chat template must return one unbatched integer token sequence')
    ids = list(ids)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
    if type(text) is not str:
        raise ValueError('Chat template must return one rendered prompt string')
    return {'token_ids': ids, 'input_token_sha256': digest(ids), 'prompt_text': text,
            'prompt_text_sha256': hashlib.sha256(text.encode()).hexdigest()}


async def chat(base, payload, timeout_seconds=180, tokenizer=None):
    """A single real HTTP attempt. Deadline includes queue and full streaming.

    Closing the stream on timeout propagates disconnect to vLLM; no invented
    explicit abort API is called. Drain must be checked before the next cell.
    """
    import httpx
    base = loopback(base)
    if 'tools' in payload or 'tool_choice' in payload or 'guided_json' in payload:
        raise ValueError('Native tools/legacy guided decoding are not the shared contract')
    assembly, started, lines = StreamAssembly(), time.monotonic(), []
    row = {'request_body': copy.deepcopy(payload), 'request_sha256': digest(payload),
           'model_request_attempts': 0, 'model_responses_received': 0,
           'server_abort_requests': 0, 'http_status': None, 'status': 'UNKNOWN',
           'stream_close_on_error': False}
    try:
        async with asyncio.timeout(timeout_seconds):
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds), trust_env=False) as client:
                row['model_request_attempts'] = 1
                async with client.stream('POST', base + '/v1/chat/completions', json=payload) as response:
                    row['http_status'] = response.status_code
                    if response.status_code != 200:
                        row['error_body'] = (await response.aread()).decode('utf-8', errors='replace')
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        elapsed = time.monotonic() - started
                        lines.append({'elapsed_seconds': elapsed, 'line': line})
                        consume_sse_line(assembly, line, elapsed)
                    row['model_responses_received'] = 1
        row['status'] = 'OK' if assembly.done else 'RAW_UNOBSERVABLE'
    except (TimeoutError, httpx.TimeoutException) as exc:
        row.update(status='REQUEST_TIMEOUT', error_type=type(exc).__name__, error=str(exc),
                   stream_close_on_error=True, drain_status='NOT_CONFIRMED')
    except Exception as exc:
        row.update(status='SERVER_ERROR' if row['http_status'] and row['http_status'] >= 400 else 'INFRA_ERROR',
                   error_type=type(exc).__name__, error=str(exc), stream_close_on_error=True)
    row.update(assembly.result())
    # Transport application content is the parser input. Special control tokens
    # remain visible in the separate raw-token decode, never silently stripped.
    row['raw_text'] = row['content']
    row['content_text'] = row['content']
    row['raw_utf8_sha256'] = hashlib.sha256(row['raw_text'].encode()).hexdigest()
    row['raw_sse_lines'] = lines
    row['raw_sse_sha256'] = digest([r['line'] for r in lines])
    row['http_end_seconds'] = time.monotonic() - started
    if tokenizer is not None:
        row['raw_token_text'] = decoded_raw(row, tokenizer)
        row['raw_token_text_sha256'] = hashlib.sha256(row['raw_token_text'].encode()).hexdigest()
    row['output_tokens'] = len(row['output_token_ids'])
    row['finish_reason'] = row['server_finish_reason']
    row['stop_reason'] = row['server_stop_reason']
    row['elapsed_seconds'] = row['http_end_seconds']
    row['request_id'] = payload.get('request_id')
    row['output_token_sha256'] = digest(row['output_token_ids'])
    row['input_token_sha256'] = digest(row['prompt_token_ids']) if row['prompt_token_ids'] is not None else None
    if row['status'] == 'OK':
        usage = row['usage'] or {}
        if row['prompt_token_ids'] is None or usage.get('completion_tokens') != len(row['output_token_ids']) or usage.get('prompt_tokens') != len(row['prompt_token_ids']):
            row['status'] = 'RAW_UNOBSERVABLE'
    return row
