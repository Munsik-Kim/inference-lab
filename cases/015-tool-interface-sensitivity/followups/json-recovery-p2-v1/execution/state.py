"""CPU measurement orchestration around the byte-preserved P2 state machine."""
import copy
import hashlib
import json
from pathlib import Path
import sys

P2 = Path(__file__).resolve().parents[1]
CASE = P2.parents[1]
REPO = CASE.parents[1]
sys.path.insert(0, str(P2 / 'src'))
from json_recovery_p2.session import Session
from json_recovery_p2.requests import retry
from json_recovery_p2.upstream import A1, digest, parse_slots, token_hash


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_name(path.name + '.partial')
    with partial.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    partial.rename(path)


def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def inputs():
    points = read(P2 / 'data/checkpoints.json')
    snapshots = {s['checkpoint_id']: s for s in read(P2 / 'data/model_request_snapshots.json')}
    scenarios = {s['scenario_id']: s for s in read(A1 / 'data/scenarios.json')}
    old = {(r['checkpoint_id'], r['method']): r for r in read(P2 / 'data/historical_continuations.json')}
    return points, snapshots, scenarios, old


class MeasuredSession(Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evidence_kind = 'measurement_p2_new_c0' if self.condition == 'C0' else 'measurement_p2'
        self.observation_status = 'OBSERVED'
        self.infra_error = None

    def consume_measured(self, gpu, *, elapsed_worker_seconds):
        payload, sidecar = self.next_request()
        if gpu['messages_tools_hash'] != digest(payload):
            raise ValueError('GPU generated from a different messages/tools payload')
        if hashlib.sha256(gpu['raw_text'].encode()).hexdigest() != gpu['raw_utf8_sha256']:
            raise ValueError('GPU raw response hash mismatch')
        if gpu['output_tokens'] != len(gpu['output_token_ids']):
            raise ValueError('GPU output length mismatch')
        if len(gpu['input_token_ids']) != gpu['input_tokens'] or token_hash(gpu['input_token_ids']) != gpu['input_token_sha256']:
            raise ValueError('GPU input IDs/hash/length mismatch')
        if bool(gpu['grammar_applied']) != sidecar['grammar_active']:
            raise ValueError('grammar applied outside its registered focal response')
        self.budget.check_input_tokens(gpu['input_tokens'])
        first = self.budget.model_calls == 2
        if first and gpu['input_token_sha256'] != self.checkpoint['recorded_input_token_sha256']:
            raise ValueError('actual generation input-token hash differs from registered checkpoint')
        response = super().consume(gpu['raw_text'], output_tokens=gpu['output_tokens'], elapsed_seconds=elapsed_worker_seconds)
        # GPU fields contain no scorer/state/oracle; verify the overlapping request
        # and raw fields above, then retain actual IDs, decoding, costs and mask data.
        response.update(copy.deepcopy(gpu))
        response['elapsed_worker_seconds'] = elapsed_worker_seconds
        response['request_role'] = 'FOCAL' if first or (self.condition in ('N', 'E') and self.budget.model_calls == 4) else 'CONTINUATION'
        response['state_after_response_sha256'] = digest(self.fixture.snapshot())
        response['state_version_after_response'] = self.fixture.version
        return response

    def fork(self, condition, root):
        if self.condition != 'C0' or not self.c0_observation or self.terminal != 'PARSE_ERROR':
            raise ValueError('retry needs this actual newly observed malformed-only C0')
        self.budget.before_request()
        payload = retry(self.snapshot, condition, self.c0_observation,
                        model_calls_used=self.budget.model_calls, tool_attempts_used=self.budget.tool_attempts,
                        retries_used=self.budget.retries)
        branch = MeasuredSession(self.checkpoint, self.snapshot, self.scenario, root, method=self.method)
        branch.condition = condition
        branch.messages = payload['messages']; branch.tools = payload['tools']
        branch.responses = copy.deepcopy(self.responses)
        branch.budget = copy.deepcopy(self.budget); branch.budget.retries += 1
        branch.c0_observation = copy.deepcopy(self.c0_observation)
        branch.evidence_kind = 'measurement_p2'
        return branch

    def result(self, execution_identity):
        verdict = self.verdict()
        # The inherited class is explicitly a CPU state machine and reports 0
        # model forwards. Counts here come only from unique real request IDs.
        verdict.pop('new_model_forwards')
        if self.observation_status != 'OBSERVED':
            verdict['VERIFIED_WORKFLOW_SUCCESS'] = None
            verdict['independent_endpoint'] = None
        return {
            'stage': 'P2_SAME_OBSERVATION_RECOVERY', 'evidence_kind': 'measurement',
            'checkpoint_id': self.checkpoint['checkpoint_id'], 'scenario_id': self.checkpoint['scenario_id'],
            'api_version': self.checkpoint['api_version'], 'method': self.method, 'arm': self.condition,
            'model_revision': 'cdbee75f17c01a7cc42f958dc650907174af0554',
            **execution_identity, **verdict,
            'observation_status': self.observation_status, 'infra_error': self.infra_error,
            'responses': copy.deepcopy(self.responses), 'tool_events': copy.deepcopy(self.fixture.events),
            'final_answer': self.final, 'final_fixture': self.fixture.snapshot(), 'final_state_version': self.fixture.version,
            'available_result_event_ids': list(self.available),
            'prefix_model_calls_charged_not_generated': 2, 'prefix_tool_attempts_charged': 2,
            'common_observation_sha256': self.checkpoint['common_observation_sha256'],
            'checkpoint_state_sha256': self.checkpoint['state_sha256'],
            'shared_C0_request_id': self.responses[0]['request_id'] if self.condition in ('N', 'E') else None,
        }


def replay_cell(row, root):
    """Reparse raw and execute fresh fixtures; no model and no stored success used."""
    points, snapshots, scenarios, _ = inputs()
    point = next(p for p in points if p['checkpoint_id'] == row['checkpoint_id'])
    snap = snapshots[point['checkpoint_id']]; scenario = scenarios[point['scenario_id']]
    arm = row['arm']
    session = MeasuredSession(point, snap, scenario, Path(root) / 'base', method=row['method'], condition='G' if arm == 'G' else 'C0')
    for index, response in enumerate(row['responses']):
        if arm in ('N', 'E') and index == 1:
            session = session.fork(arm, Path(root) / 'retry')
        # Remove only state-machine-derived fields. All GPU evidence is preserved.
        gpu = {k: copy.deepcopy(v) for k, v in response.items() if k not in {
            'model_request', 'intervention_sha256', 'generated_in_turn', 'available_result_event_ids',
            'pre_generation_state_digest', 'pre_generation_history_hash', 'pre_generation_state_version',
            'canonical_observation_hash', 'evidence_kind', 'elapsed_worker_seconds', 'request_role',
            'state_after_response_sha256', 'state_version_after_response'}}
        replayed = session.consume_measured(gpu, elapsed_worker_seconds=response['elapsed_worker_seconds'])
        if replayed != response:
            raise ValueError('raw replay request/observation/state differs: ' + response['request_id'])
    if row['observation_status'] != 'OBSERVED':
        session.observation_status = row['observation_status']; session.infra_error = row['infra_error']
        session.terminal = row['terminal_reason']
    result = session.result({k: row[k] for k in ['execution_source_commit', 'execution_source_sha256', 'generation_config_hash']})
    for field in ['VERIFIED_WORKFLOW_SUCCESS', 'STATE_ONLY_SUCCESS', 'FINAL_ANSWER_CORRECT',
                  'REQUIRED_ACTIONS_DONE', 'OBSERVATION_ORDER_VALID', 'final_answer', 'final_fixture',
                  'final_state_version', 'tool_events', 'budget', 'terminal_reason', 'responses']:
        if result[field] != row[field]:
            raise ValueError('independent cell replay differs: ' + field)
    return result
