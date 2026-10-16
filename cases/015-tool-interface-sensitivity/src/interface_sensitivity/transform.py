"""Typed declarations and explicitly known task API spans, never global replace."""
import copy
import hashlib
from .common import digest, require, INTERFACES
from .registry import Registry, SEMANTICS, ORIGINAL_NAMES


def stable(seed, task, purpose, value):
    return hashlib.sha256(f'case015-v1|{seed}|{task}|{purpose}|{value}'.encode()).hexdigest()


def lower_camel(name):
    first, *rest = name.split('_')
    return first + ''.join(s[:1].upper() + s[1:] for s in rest)


def make_interface(task, messages, tools, interface, seed=6101601):
    require(interface in INTERFACES, 'unknown interface')
    original_names = [x['function']['name'] for x in tools]
    require(set(original_names) == set(SEMANTICS) and len(original_names) == 6, 'six unique original tools required')
    mapping = dict(ORIGINAL_NAMES)
    if interface == 'I2_STYLE_RENAME':
        mapping = {s: lower_camel(n) for s, n in mapping.items()}
    if interface == 'I3_OPAQUE_RENAME':
        # Independently assign ranks by hashed semantic identity and numeric ID.
        # Neither task gold nor original declaration position enters allocation.
        semantics = sorted(mapping, key=lambda s: stable(seed, task['id'], 'semantic', s))
        numbers = sorted(range(10, 100), key=lambda v: stable(seed, task['id'], 'opaque-number', v))[:6]
        mapping = dict(zip(semantics, ('tool_' + str(n) for n in numbers)))
    registry = Registry(mapping)
    out_tools = copy.deepcopy(tools)
    out_messages = copy.deepcopy(messages)
    changes = []
    for i, tool in enumerate(out_tools):
        old = tool['function']['name']; new = registry.expose(SEMANTICS[old])
        tool['function']['name'] = new
        if old != new:
            changes.append({'field': f'tools[{i}].function.name', 'before': old, 'after': new, 'reason': 'typed API declaration'})
    # Only these task fields contain explicit API-name references. "Read document"
    # and payload text are not API identifiers and remain byte-for-byte unchanged.
    references = {'combine': ['calculate'], 'fixture_patch': ['validate_fixture_test']}.get(task['family'], [])
    for old in references:
        new = registry.expose(SEMANTICS[old])
        if old == new:
            continue
        marker = '\n\nTASK: '
        content = out_messages[-1]['content']
        start = content.rindex(marker) + len(marker)
        task_span = content[start:]
        require(task_span == task['task'], 'known task span mismatch')
        require(task_span.count(old) == 1, 'ambiguous explicit API span')
        pos = start + task_span.index(old)
        out_messages[-1]['content'] = content[:pos] + new + content[pos + len(old):]
        changes.append({'field': f'messages[{len(out_messages)-1}].content', 'start': pos, 'before': old, 'after': new, 'reason': 'verified task API reference'})
    permutation = list(range(6))
    if interface == 'I1_ORDER_ONLY':
        permutation = sorted(permutation, key=lambda i: stable(seed, task['id'], 'order', i))
        if permutation == list(range(6)):
            permutation = permutation[1:] + permutation[:1]
        out_tools = [out_tools[i] for i in permutation]
        changes.append({'field': 'tools.order', 'before': list(range(6)), 'after': permutation, 'reason': 'seed-only nonidentity permutation'})
    result = {'task_id': task['id'], 'interface_id': interface, 'canonical_task_hash': digest(task),
              'mapping': mapping, 'permutation': permutation, 'changes': changes,
              'tools': out_tools, 'messages': out_messages, 'rendered_prompt_hash': digest({'messages': out_messages, 'tools': out_tools})}
    require(reverse(result) == {'tools': tools, 'messages': messages}, 'transformation not reversible')
    # No model-visible API identifier from the old namespace may remain in these
    # known typed fields. Descriptions contain no literal API names in this fixture.
    require(all(t['function']['name'] in registry.inverse for t in out_tools), 'declaration mismatch')
    return result


def reverse(spec):
    tools = copy.deepcopy(spec['tools']); messages = copy.deepcopy(spec['messages'])
    if spec['interface_id'] == 'I1_ORDER_ONLY':
        ordered = [None] * len(tools)
        for i, original in enumerate(spec['permutation']):
            ordered[original] = tools[i]
        tools = ordered
    inverse = Registry(spec['mapping'])
    for t in tools:
        t['function']['name'] = inverse.original(t['function']['name'])
    for c in reversed(spec['changes']):
        if c['field'].startswith('messages['):
            pos = c['start']; content = messages[-1]['content']
            require(content[pos:pos+len(c['after'])] == c['after'], 'inverse span mismatch')
            messages[-1]['content'] = content[:pos] + c['before'] + content[pos+len(c['after']):]
    return {'tools': tools, 'messages': messages}
