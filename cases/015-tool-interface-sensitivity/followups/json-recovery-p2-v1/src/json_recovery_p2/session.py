"""CPU state transitions from a frozen checkpoint. This module has no generator."""
import copy
import math
from dataclasses import dataclass
from .evidence import replay_prefix, observation
from .requests import FEEDBACK, initial, retry
from .upstream import Fixture, Registry, assess, digest, extract, history, independent_endpoint, parse_slots


@dataclass
class Budget:
    model_calls: int = 2
    tool_attempts: int = 2
    retries: int = 0
    elapsed_seconds: float = 0.0
    max_model_calls: int = 8
    max_tool_calls: int = 6
    max_new_tokens: int = 192
    deadline_seconds: float = 900.0

    def before_request(self):
        if self.model_calls >= self.max_model_calls:
            raise ValueError('MODEL_BUDGET_EXHAUSTED')
        if self.elapsed_seconds >= self.deadline_seconds:
            raise ValueError('TIMEOUT')
        return self.deadline_seconds - self.elapsed_seconds

    def check_input_tokens(self, count):
        if type(count) is not int or count < 0 or count + self.max_new_tokens > 24576:
            raise ValueError('CONTEXT_LIMIT; no automatic truncation')

    def debit_response(self, output_tokens, elapsed_seconds):
        self.before_request()
        if type(output_tokens) is not int or not (0 <= output_tokens <= self.max_new_tokens):
            raise ValueError('output token count outside registered cap')
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError('invalid measured worker wall debit')
        self.model_calls += 1
        self.elapsed_seconds += elapsed_seconds


class Session:
    """Feed scripted/stored responses to actual fixtures; never call a model.

    CPU tests use evidence_kind=synthetic_contract_test. A future model adapter
    must record genuine new-C0 provenance before enabling a measurement path.
    """
    def __init__(self, checkpoint, snapshot, scenario, root, *, method='D16', condition='C0'):
        if method not in ('BF16', 'RTN', 'D16') or condition not in ('C0', 'G'):
            raise ValueError('unsupported checkpoint condition')
        if digest(snapshot) != checkpoint['model_input_snapshot_sha256']:
            raise ValueError('checkpoint/snapshot identity differs')
        self.checkpoint = copy.deepcopy(checkpoint); self.snapshot = copy.deepcopy(snapshot)
        self.scenario = copy.deepcopy(scenario); self.method = method; self.condition = condition
        self.fixture = Fixture(scenario, Registry(checkpoint['api_version']), root)
        # Re-execute the exact two saved semantic operations, not gold replacements.
        for saved in checkpoint['prefix_events']:
            event = self.fixture.execute(saved['raw_name'], saved['raw_arguments'], saved['generated_in_turn'], saved['call_index'], saved['generated_after_event_ids'])
            if event != saved:
                raise ValueError('checkpoint prefix event replay differs')
        self.messages = initial(snapshot, condition)
        self.tools = self.messages['tools']; self.messages = self.messages['messages']
        self.available = list(checkpoint['available_result_event_ids'])
        if self.available != [e['event_id'] for e in self.fixture.events] or observation(self.fixture.events, self.available) != checkpoint['common_observation_sha256']:
            raise ValueError('checkpoint result-ID/observation mismatch')
        if digest(self.fixture.snapshot()) != checkpoint['state_sha256'] or self.fixture.version != checkpoint['state_version']:
            raise ValueError('checkpoint state mismatch')
        self.budget = Budget(elapsed_seconds=checkpoint['prefix_elapsed_seconds_debit'])
        self.responses = []; self.final = None; self.terminal = None; self.c0_observation = None
        self.evidence_kind = 'synthetic_contract_test'

    def next_request(self):
        if self.terminal is not None:
            raise ValueError('session has stopped')
        self.budget.before_request()
        payload = {'messages': copy.deepcopy(self.messages), 'tools': copy.deepcopy(self.tools)}
        return payload, {
            'condition': self.condition, 'messages_tools_sha256': digest(payload),
            'common_observation_sha256': observation(self.fixture.events, self.available),
            'state_sha256': digest(self.fixture.snapshot()), 'state_version': self.fixture.version,
            'available_result_event_ids': list(self.available),
            'model_calls_used': self.budget.model_calls, 'tool_attempts_used': self.budget.tool_attempts,
            'remaining_deadline_seconds': self.budget.before_request(),
            'grammar_active': self.condition == 'G' and self.budget.model_calls == 2,
            'intervention_sha256': digest({'condition': self.condition, 'feedback': FEEDBACK.get(self.condition),
                'grammar_active': self.condition == 'G' and self.budget.model_calls == 2}),
        }

    def consume(self, raw, *, output_tokens=1, elapsed_seconds=0.0):
        payload, sidecar = self.next_request()
        turn = self.budget.model_calls
        self.budget.debit_response(output_tokens, elapsed_seconds)
        response = {'raw_text': raw, 'text': raw.removesuffix('<|im_end|>').removesuffix('<|endoftext|>'),
            'model_request': payload, 'intervention_sha256': sidecar['intervention_sha256'],
            'generated_in_turn': turn, 'available_result_event_ids': list(self.available),
            'messages_tools_hash': digest(payload), 'pre_generation_state_digest': sidecar['state_sha256'],
            'pre_generation_history_hash': digest(self.messages), 'pre_generation_state_version': self.fixture.version,
            'canonical_observation_hash': sidecar['common_observation_sha256'], 'evidence_kind': self.evidence_kind,
            'output_tokens': output_tokens, 'elapsed_worker_seconds': elapsed_seconds}
        self.responses.append(response)
        if self.budget.elapsed_seconds >= self.budget.deadline_seconds:
            self.terminal = 'TIMEOUT'
            return response
        slots = parse_slots(raw); accepted = []; results = []; new_ids = []
        for slot in slots:
            if self.budget.tool_attempts >= self.budget.max_tool_calls:
                self.terminal = 'BUDGET_EXHAUSTED'; break
            self.budget.tool_attempts += 1
            obj = slot['parsed']
            if obj is None:
                continue
            event = self.fixture.execute(obj.get('name'), obj.get('arguments'), turn, slot['call_index'], response['available_result_event_ids'])
            accepted.append(obj); results.append(event['tool_result']); new_ids.append(event['event_id'])
        if slots:
            if accepted:
                self.messages = history(self.messages, accepted, results)
                self.available.extend(new_ids)
                # Original policy continues after accepted calls even when a
                # later marker exceeded the tool-attempt limit. A final answer
                # can still use the remaining model-call budget.
                self.terminal = None
            else:
                self.terminal = self.terminal or 'PARSE_ERROR'
                if self.condition == 'C0' and turn == 2:
                    self.c0_observation = {
                        'checkpoint_id': self.checkpoint['checkpoint_id'], 'method': self.method,
                        'parent_request_sha256': self.snapshot['messages_tools_sha256'],
                        'evidence_kind': self.evidence_kind, 'raw_text': raw,
                        'c0_observation_id': digest({'request': payload, 'raw': raw, 'method': self.method}),
                    }
        else:
            self.final = extract(response['text'])
            self.terminal = 'FINAL_ANSWER' if self.final is not None else 'PARSE_ERROR'
        if self.budget.elapsed_seconds >= self.budget.deadline_seconds:
            self.terminal = 'TIMEOUT'
        elif self.budget.model_calls == self.budget.max_model_calls and self.terminal is None:
            self.terminal = 'BUDGET_EXHAUSTED'
        return response

    def fork_retry(self, condition, root):
        if not self.c0_observation or self.condition != 'C0' or self.terminal != 'PARSE_ERROR':
            raise ValueError('only the same new malformed-only C0 can branch to N/E')
        self.budget.before_request()
        payload = retry(self.snapshot, condition, self.c0_observation,
            model_calls_used=self.budget.model_calls, tool_attempts_used=self.budget.tool_attempts,
            retries_used=self.budget.retries, allow_synthetic_contract_test=True)
        branch = Session(self.checkpoint, self.snapshot, self.scenario, root, method=self.method)
        branch.condition = condition; branch.messages = payload['messages']; branch.tools = payload['tools']
        branch.responses = copy.deepcopy(self.responses)
        branch.budget = copy.deepcopy(self.budget); branch.budget.retries += 1
        branch.c0_observation = copy.deepcopy(self.c0_observation)
        return branch

    def verdict(self):
        state = self.fixture.snapshot()
        result = assess(self.scenario, self.fixture.events, self.responses, self.final, state, self.fixture.version, self.fixture.integrity_violation)
        if self.terminal != 'FINAL_ANSWER':
            result['VERIFIED_WORKFLOW_SUCCESS'] = False
        row = {'final_answer': self.final, 'final_fixture': state, 'final_state_version': self.fixture.version,
               'tool_events': self.fixture.events, 'responses': self.responses}
        independent = independent_endpoint(self.scenario, row) if self.terminal == 'FINAL_ANSWER' else False
        if independent != result['VERIFIED_WORKFLOW_SUCCESS']:
            raise ValueError('original/independent endpoint mismatch')
        writes = [(e['raw_arguments']['order_id'], e['raw_arguments']['total_minor']) for e in self.fixture.events
                  if e['semantic_tool_id'] == 'order.write' and e['execution_status'] == 'SUCCESS']
        return {**result, 'independent_endpoint': independent, 'evidence_kind': self.evidence_kind,
                'terminal_reason': self.terminal, 'budget': self.budget.__dict__.copy(), 'new_model_forwards': 0,
                'auxiliary': {'duplicate_successful_write_events': len(writes) - len(set(writes)),
                    'parsed_tool_events_including_prefix': len(self.fixture.events),
                    'malformed_attempts_including_shared_C0': self.budget.tool_attempts - len(self.fixture.events)}}

