"""Tool-name enum decoding only; never rewrites answers or arguments.

The lexer examines the top-level name string of an open <tool_call> object.
Other strings, nested argument fields and ordinary final answers remain free.
ASCII schema keys/names are used; escaped top-level keys/names are rejected.
"""
from dataclasses import dataclass, replace
import json


@dataclass(frozen=True)
class State:
    depth: int = 0
    string: str = ''
    buffer: str = ''
    escape: bool = False
    expect: str = 'key'
    key: str = ''
    ended: bool = False


def advance(state, text, names):
    s = state
    for c in text:
        if s.ended:
            break
        if s.string:
            if s.escape:
                if s.string in ('key', 'name'):
                    return None
                s = replace(s, escape=False)
            elif c == '\\':
                if s.string in ('key', 'name'):
                    return None
                s = replace(s, escape=True)
            elif c == '"':
                if s.string == 'name' and s.buffer not in names:
                    return None
                if s.string == 'key':
                    s = replace(s, string='', key=s.buffer, buffer='', expect='colon')
                else:
                    s = replace(s, string='', buffer='', expect='after' if s.depth == 1 else s.expect)
            elif s.string in ('key', 'name'):
                value = s.buffer + c
                if s.string == 'name' and not any(n.startswith(value) for n in names):
                    return None
                s = replace(s, buffer=value)
            continue
        if c.isspace():
            continue
        if s.depth == 1 and s.key == 'name' and s.expect == 'value' and c != '"':
            return None
        if c == '"':
            kind = 'key' if s.depth == 1 and s.expect == 'key' else 'name' if s.depth == 1 and s.key == 'name' and s.expect == 'value' else 'other'
            s = replace(s, string=kind, buffer='', escape=False)
        elif c in '{[':
            s = replace(s, depth=s.depth + 1)
        elif c in '}]':
            s = replace(s, depth=max(0, s.depth - 1), expect='after')
        elif c == ':' and s.depth == 1 and s.expect == 'colon':
            s = replace(s, expect='value')
        elif c == ',' and s.depth == 1:
            s = replace(s, expect='key', key='')
        elif c == '<' and s.depth == 0:
            s = replace(s, ended=True)
    return s


def active_state(text, names):
    start = text.rfind('<tool_call>')
    if start < 0 or text.rfind('</tool_call>') > start:
        return None
    state = advance(State(), text[start + len('<tool_call>'):], names)
    if state is None:
        raise ValueError('invalid tool-name prefix already generated; no silent correction')
    return None if state.ended else state


def feedback(name, args, schema, error):
    registry = {x['function']['name']: x['function']['parameters'] for x in schema}
    if name not in registry:
        return {'error': 'UNKNOWN_TOOL_NAME', 'rejected_name': name,
                'allowed_tool_names': sorted(registry),
                'instruction': 'Call an allowed tool using its exact case-sensitive name. No operation was performed.'}
    required = registry[name].get('required', [])
    return {'error': 'INVALID_TOOL_ARGUMENTS', 'tool': name, 'detail': str(error),
            'required_argument_keys': required,
            'instruction': 'Correct the arguments and call the registered tool. No failed operation is counted as success.'}


def processor(tokenizer, names, prompt_length, shared=None):
    """Create a Torch processor lazily; --help/tests do not import Torch."""
    import torch
    from transformers import LogitsProcessor
    shared = {} if shared is None else shared

    class EnumProcessor(LogitsProcessor):
        def __init__(self):
            self.trace = []
            self.cache = shared.setdefault('cache', {})
            self.token_text = shared.get('token_text')

        def __call__(self, input_ids, scores):
            if len(input_ids) != 1:
                raise ValueError('registered batch1 decoder')
            text = tokenizer.decode(input_ids[0, prompt_length:].tolist(), skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            state = active_state(text, names)
            if state is None:
                return scores
            if self.token_text is None:
                self.token_text = tokenizer.batch_decode([[i] for i in range(scores.shape[-1])],
                    skip_special_tokens=False, clean_up_tokenization_spaces=False)
                shared['token_text'] = self.token_text
                # A token cannot hide a whole new tool call and a name field
                # before this active-call scanner is entered.
                if any('<tool_call>' in t and '"name"' in t for t in self.token_text):
                    raise ValueError('unsupported tokenizer: combined tool marker/name token')
            if state not in self.cache:
                allowed = [bool(t) and advance(state, t, names) is not None for t in self.token_text]
                if not any(allowed):
                    raise ValueError('empty tool-name token mask')
                self.cache[state] = torch.tensor(allowed, device=scores.device, dtype=torch.bool)
            mask = self.cache[state]
            raw = int(scores[0].argmax())
            output = scores.masked_fill(~mask.unsqueeze(0), -torch.inf)
            chosen = int(output[0].argmax())
            self.trace.append({'position': input_ids.shape[-1] - prompt_length,
                'unconstrained_top_id': raw, 'selected_id': chosen,
                'forced_change': raw != chosen, 'name_prefix': state.buffer if state.string == 'name' else None,
                'allowed_tokens': int(mask.sum()), 'state': state.__dict__})
            return output
    return EnumProcessor()
