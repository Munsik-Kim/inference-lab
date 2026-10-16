"""Per-step oracle actions executed against real isolated fixture files."""
import tempfile
from pathlib import Path
from .common import inputs, INTERFACES, digest, require
from .fixture import Fixture
from .registry import Registry
from .transform import make_interface
from .scoring import score


def oracle_actions(task):
    f = task['family']
    if f == 'document':
        return [('document.read', {})]
    if f == 'updates':
        return [('updates.read', {})]
    if f == 'combine':
        r = task['fixture']['record']
        return [('record.read', {}), ('math.multiply', {'a': r['quantity'], 'b': r['unit_price']})]
    return [('fixture.patch', task['fixture']['expected_config']), ('fixture.validate', {})]


def oracle_check(case):
    tasks, messages, tools = inputs(case)
    records = []
    with tempfile.TemporaryDirectory(prefix='case015-oracle-') as tmp:
        for task in tasks:
            baseline = None
            for interface in INTERFACES:
                spec = make_interface(task, messages[task['id']], tools, interface)
                registry = Registry(spec['mapping'])
                fixture = Fixture(task, registry, Path(tmp)/task['id']/interface)
                for i, (semantic, args) in enumerate(oracle_actions(task)):
                    fixture.execute(registry.expose(semantic), args, i, 0)
                verdict = score(task, task['gold'], fixture.state, fixture.events)
                require(verdict['functional_success'], 'oracle functional verdict failed')
                normalized = [{k: e[k] for k in ('semantic_tool_id', 'raw_arguments', 'tool_result', 'execution_status', 'state_before', 'state_after', 'allowed_action_set', 'argument_value_correct')} for e in fixture.events]
                if baseline is None:
                    baseline = normalized
                require(normalized == baseline, 'stepwise observation/state mismatch')
                records.append({'task_id': task['id'], 'interface_id': interface, 'step_count': len(normalized),
                                'normalized_trace': normalized, 'normalized_trace_hash': digest(normalized), 'verdict': verdict})
    return {'status': 'PASS', 'oracle_episodes': len(records), 'oracle_steps': sum(r['step_count'] for r in records),
            'model_forwards': 0, 'records': records}
