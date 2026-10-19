"""GPU-process copy of the registered CPU grammar; function AST unchanged.

Only the process-level CPU import guard is omitted in this explicit GPU module.
"""
import copy
import math



def schemas(tools):
    alternatives = []
    for tool in tools:
        fn = tool['function']
        alternatives.append({'type': 'object', 'properties': {
            'name': {'type': 'string', 'enum': [fn['name']]},
            'arguments': copy.deepcopy(fn['parameters'])},
            'required': ['name', 'arguments'], 'additionalProperties': False})
    if len(alternatives) != 4 or len({x['properties']['name']['enum'][0] for x in alternatives}) != 4:
        raise ValueError('four distinct declared tools required')
    final = {'type': 'object', 'properties': {'order_id': {'type': 'string'},
        'total_minor': {'type': 'integer'}, 'validated': {'type': 'boolean'}},
        'required': ['order_id', 'total_minor', 'validated'], 'additionalProperties': False}
    return {'tool_call': {'anyOf': alternatives}, 'final_answer': final}


def make_parser(tools):
    from lmformatenforcer import CharacterLevelParser, CharacterLevelParserConfig, JsonSchemaParser
    contract = schemas(tools)

    class Envelope(CharacterLevelParser):
        def __init__(self, phase='start', literal='', child=None, config=None):
            super().__init__(config or CharacterLevelParserConfig(force_json_field_order=False, max_consecutive_whitespaces=192))
            self.phase = phase; self.literal = literal; self.child = child

        def get_allowed_characters(self):
            if self.phase in ('start', 'between'):
                return ' \t\r\n<' + ('{' if self.phase == 'start' else '')
            if self.phase in ('open', 'close'):
                return self.literal[:1]
            chars = self.child.get_allowed_characters()
            if self.phase == 'call_json' and self.child.can_end():
                chars += '<'
            return chars

        def can_end(self):
            # A1 accepts a complete call dict even without its closing marker.
            return self.phase == 'between' or (self.phase in ('call_json', 'final_json') and self.child.can_end())

        def shortcut_key(self):
            return self.child.shortcut_key() if self.phase in ('call_json', 'final_json') else None

        def cache_key(self):
            if self.phase in ('start', 'between', 'open', 'close'):
                return ('envelope', self.phase, self.literal)
            key = self.child.cache_key()
            return ('envelope', self.phase, key) if key is not None else None

        def add_character(self, char):
            if char not in self.get_allowed_characters():
                raise ValueError('character outside registered grammar')
            if self.phase in ('start', 'between'):
                if char in ' \t\r\n':
                    return Envelope(self.phase, config=self.config)
                if char == '<':
                    return Envelope('open', 'tool_call>', config=self.config)
                child = JsonSchemaParser(contract['final_answer'], config=self.config).add_character(char)
                return Envelope('final_json', child=child, config=self.config)
            if self.phase in ('open', 'close'):
                left = self.literal[1:]
                if left:
                    return Envelope(self.phase, left, config=self.config)
                if self.phase == 'close':
                    return Envelope('between', config=self.config)
                return Envelope('call_json', child=JsonSchemaParser(contract['tool_call'], config=self.config), config=self.config)
            if self.phase == 'call_json' and self.child.can_end() and char == '<':
                return Envelope('close', '/tool_call>', config=self.config)
            return Envelope(self.phase, child=self.child.add_character(char), config=self.config)

    return Envelope()


def accepts_text(tools, text):
    parser = make_parser(tools)
    for char in text:
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


def token_enforcer(tokenizer_data, tools):
    from lmformatenforcer import TokenEnforcer

    class StrictEnforcer(TokenEnforcer):
        def get_allowed_tokens(self, token_sequence):
            result = super().get_allowed_tokens(token_sequence)
            state = self.prefix_states[tuple(token_sequence)]
            stop = self.eos_token_id if isinstance(self.eos_token_id, list) else [self.eos_token_id]
            # The library's error fallback can return EOS. Fail closed instead of
            # allowing that fallback to terminate an incomplete structure.
            if not state.parser.can_end() and any(x in result.allowed_tokens for x in stop):
                raise RuntimeError('grammar backend failed closed: premature EOS/fallback')
            if not result.allowed_tokens:
                raise RuntimeError('grammar backend returned an empty mask')
            return result

    return StrictEnforcer(tokenizer_data, make_parser(tools))


def finite_then_mask(unmasked_scores, allowed_ids):
    # Same ordering as the future GPU contract: FiniteBudget first, grammar next.
    if not all(math.isfinite(x) for x in unmasked_scores):
        raise ValueError('nonfinite UNMASKED logits')
    allowed = set(allowed_ids)
    if not allowed or min(allowed) < 0 or max(allowed) >= len(unmasked_scores):
        raise ValueError('invalid grammar token IDs')
    return [x if i in allowed else -math.inf for i, x in enumerate(unmasked_scores)]

