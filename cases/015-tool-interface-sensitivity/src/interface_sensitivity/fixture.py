"""Safe exact-name dispatch into the unchanged six local fixture functions."""
import copy
import math
from pathlib import Path
from .common import atomic, digest, read, require
from .fixture_original import OriginalFixture
from .registry import Registry
from .scoring import allowed_actions, argument_value_correct


class Fixture:
    def __init__(self, task, registry, root):
        self.task = copy.deepcopy(task)
        self.registry = registry
        root = Path(root)
        require(not root.exists(), 'episode fixture root must be new')
        root.mkdir(parents=True)
        require(not root.is_symlink(), 'fixture root must not be a symlink')
        self.path = root / 'config.json'
        atomic(self.path, self.task['fixture']['config'], fresh=True)
        self.original = OriginalFixture()
        self.events = []

    @property
    def state(self):
        return read(self.path)

    def execute(self, name, args, turn, index, call_id=None):
        before = self.state
        semantic = self.registry.resolve(name)
        allowed = allowed_actions(self.task, before, self.events)
        labels = []; schema = 'NOT_EVALUABLE'; value_ok = None
        # Only exact registered names reach this unchanged fixture implementation.
        # Baseline model-visible strings are deliberately kept unhelpful.
        try:
            if not isinstance(args, dict):
                labels.append('BAD_ARGUMENTS'); schema = 'NOT_OBJECT'
                raise ValueError('arguments not an object')
            if semantic is None:
                labels.append('UNKNOWN_TOOL')
                raise ValueError('unknown tool/incorrect argument keys')
            original = self.registry.original(name)
            params = {'document.read': [], 'record.read': [], 'updates.read': [],
                      'math.multiply': ['a', 'b'], 'fixture.patch': ['enabled', 'limit'], 'fixture.validate': []}
            if set(args) != set(params[semantic]):
                labels.append('BAD_ARGUMENTS'); schema = 'KEYS'
            elif semantic == 'math.multiply' and any(type(args[k]) not in (int, float) or not math.isfinite(args[k]) for k in ('a', 'b')):
                labels.append('BAD_ARGUMENTS'); schema = 'TYPE_OR_NONFINITE'
            elif semantic == 'fixture.patch' and (type(args['enabled']) is not bool or type(args['limit']) is not int):
                labels.append('BAD_ARGUMENTS'); schema = 'TYPE'
            else:
                schema = 'VALID'
                value_ok = argument_value_correct(semantic, args, self.task)
                if not value_ok:
                    labels.append('BAD_ARGUMENTS')
            answer = copy.deepcopy(self.original.tool(original, args, self.task['fixture'], self.path))
            status = 'SUCCESS'
            if semantic not in allowed:
                labels.append('WRONG_TOOL')
        except (ValueError, TypeError) as e:
            answer = {'error': str(e)}; status = 'ERROR'
        after = self.state
        signature = digest({'name': name, 'args': args})
        if any(e['signature'] == signature and (e['execution_status'] == 'ERROR' or e['error_labels']) for e in self.events):
            labels.append('REPEATED_FAILED_CALL')
        previous_error = next((e['event_id'] for e in reversed(self.events) if e['error_labels']), None)
        event = {'event_id': f't{turn}-c{index}', 'model_turn': turn, 'call_index': index,
                 'call_id': call_id or f't{turn}-c{index}', 'call_id_model_visible': False,
                 'raw_name': name, 'declared_names_hash': self.registry.names_hash,
                 'semantic_tool_id': semantic, 'raw_arguments': copy.deepcopy(args),
                 'schema_result': schema, 'argument_value_correct': value_ok,
                 'allowed_action_set': list(allowed), 'semantic_choice_correct': semantic in allowed if semantic else None,
                 'state_before': before, 'state_after': after, 'state_before_digest': digest(before), 'state_after_digest': digest(after),
                 'tool_result': answer, 'execution_status': status, 'error_labels': sorted(set(labels)),
                 'signature': signature, 'recovery_of_event_id': previous_error,
                 'fixture_changes': {'config': after} if before != after else None,
                 'model_feedback_policy': 'original_case013_error_strings; no hints, aliases or enum'}
        self.events.append(event)
        return event
