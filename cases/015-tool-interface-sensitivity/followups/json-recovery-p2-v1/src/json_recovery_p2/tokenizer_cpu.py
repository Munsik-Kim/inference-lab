"""Optional local tokenizer/template and grammar checks, no Transformers/Torch."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from .cpu_only import install
from .grammar import token_enforcer
from .io import read, rows, sha, write
from .upstream import A1, CASE, ROOT, token_hash

install()


def local_tokenizer(snapshot):
    snapshot = Path(snapshot)
    original = read(CASE / 'provenance/model_identity.json')['files']
    checked = {}
    for name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'config.json']:
        p = snapshot / name
        if not p.is_file() or p.stat().st_size != original[name]['bytes'] or sha(p) != original[name]['sha256']:
            raise ValueError('local tokenizer/config identity mismatch: ' + name)
        checked[name] = original[name]
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(snapshot / 'tokenizer.json')), checked


def render(tokenizer_config, request):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
    # Match the inspected pinned Transformers filter, without importing it.
    environment.filters['tojson'] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    template = tokenizer_config['chat_template']
    if not isinstance(template, str) or '{% generation' in template or '{%- generation' in template:
        raise ValueError('unsupported template; do not silently approximate')
    return environment.from_string(template).render(messages=request['messages'], tools=request['tools'],
        add_generation_prompt=True, enable_thinking=False,
        eos_token=tokenizer_config.get('eos_token'), bos_token=tokenizer_config.get('bos_token'), pad_token=tokenizer_config.get('pad_token'))


def retokenize(snapshot, output, data_root=ROOT / 'data'):
    tokenizer, checked = local_tokenizer(snapshot)
    config = read(Path(snapshot) / 'tokenizer_config.json')
    points = {c['checkpoint_id']: c for c in read(data_root / 'checkpoints.json')}
    output_rows = []
    for request in read(data_root / 'model_request_snapshots.json'):
        text = render(config, request)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        point = points[request['checkpoint_id']]
        comparisons = {}
        for method, expected in point['per_method_identity'].items():
            if len(ids) != expected['recorded_input_tokens'] or token_hash(ids) != expected['recorded_input_token_sha256']:
                raise ValueError('retokenized C0 input differs: ' + request['checkpoint_id'] + ' ' + method)
            comparisons[method] = 'PASS'
        output_rows.append({'checkpoint_id': request['checkpoint_id'], 'input_token_ids': ids,
            'input_token_sha256': token_hash(ids), 'rendered_utf8_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'recorded_method_comparisons': comparisons})
    decoded = 0
    for episode in rows(A1 / 'results/a1_r1/episodes.jsonl'):
        for response in episode['responses']:
            if tokenizer.decode(response['output_token_ids'], skip_special_tokens=False) != response['raw_text']:
                raise ValueError('stored output IDs do not decode to original raw text')
            if tokenizer.decode(response['output_token_ids'], skip_special_tokens=True) != response['text']:
                raise ValueError('stored output IDs do not decode to original parser text')
            decoded += 1
    value = {'status': 'PASS', 'evidence_kind': 'CPU_LOCAL_TOKENIZATION_NOT_MODEL_INFERENCE',
        'unique_inputs_retokenized': len(output_rows), 'method_reference_hashes_matched': 3 * len(output_rows),
        'add_special_tokens': False, 'add_generation_prompt': True, 'enable_thinking': False,
        'local_files_hashed': checked, 'template_sha256': hashlib.sha256(config['chat_template'].encode()).hexdigest(),
        'tokenizers_version': importlib.metadata.version('tokenizers'), 'jinja2_version': importlib.metadata.version('Jinja2'),
        'transformers_imported': 'transformers' in sys.modules, 'torch_imported': 'torch' in sys.modules,
        'model_forwards': 0, 'rows': output_rows}
    value['stored_output_id_decodes_matched'] = decoded
    write(output, value)
    return value


def tokenizer_data(snapshot):
    tokenizer, checked = local_tokenizer(snapshot)
    from lmformatenforcer import TokenEnforcerTokenizerData
    tokenizer_json = read(Path(snapshot) / 'tokenizer.json')
    specials = {t['id'] for t in tokenizer_json['added_tokens'] if t['special']}
    vocabulary = tokenizer.get_vocab_size(with_added_tokens=True)
    token_zero = tokenizer.encode('0', add_special_tokens=False).ids[-1]
    regular = []
    for token in range(vocabulary):
        if token in specials:
            continue
        after_zero = tokenizer.decode([token_zero, token], skip_special_tokens=False)[1:]
        single = tokenizer.decode([token], skip_special_tokens=False)
        regular.append((token, after_zero, len(after_zero) > len(single)))
    decoder = lambda ids: tokenizer.decode(ids, skip_special_tokens=False).rstrip('�')
    stops = read(Path(snapshot) / 'generation_config.json')['eos_token_id']
    model_vocabulary = read(Path(snapshot) / 'config.json')['vocab_size']
    data = TokenEnforcerTokenizerData(regular, decoder, stops, use_bitmask=False, vocab_size=model_vocabulary)
    return tokenizer, data, {'tokenizer_vocabulary': vocabulary, 'model_vocabulary': model_vocabulary,
        'stop_token_ids': stops, 'regular_tokens': len(regular), 'special_tokens_excluded': sorted(specials),
        'padded_model_vocab_ids_masked': model_vocabulary - vocabulary, 'checked_files': checked}


def check_grammar(snapshot, output, data_root=ROOT / 'data', progress=None):
    tokenizer, data, identity = tokenizer_data(snapshot)
    snapshots = read(data_root / 'model_request_snapshots.json')
    reports = []
    for api in ['API_V1', 'API_V2']:
        tools = next(s['tools'] for s in snapshots if s['checkpoint_id'].startswith(api))
        names = [t['function']['name'] for t in tools]
        examples = [
            '<tool_call>\n' + json.dumps({'name': names[0], 'arguments': {'order_id': 'WRONG-ID'}}, ensure_ascii=False) + '\n</tool_call>',
            '<tool_call>\n' + json.dumps({'arguments': {'b': -7, 'a': 12345}, 'name': names[1]}, ensure_ascii=False) + '\n</tool_call>',
            '<tool_call>\n' + json.dumps({'name': names[2], 'arguments': {'order_id': 'NOT-A-TARGET', 'total_minor': -1}}) + '\n</tool_call>\n<tool_call>\n' + json.dumps({'name': names[3], 'arguments': {'order_id': 'NOT-A-TARGET'}}) + '\n</tool_call>',
            json.dumps({'validated': False, 'total_minor': -111, 'order_id': 'ANY-ID'}),
            '<tool_call>' + json.dumps({'name': names[2], 'arguments': {'order_id': 'WRONG-ID', 'total_minor': 987654321}}),
        ]
        for index, text in enumerate(examples):
            enforcer = token_enforcer(data, tools)
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            prefix = []
            for token in ids:
                allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
                if token not in allowed:
                    raise ValueError('actual tokenizer grammar mask excludes registered example ' + api + '/' + str(index) + ' at ' + str(len(prefix)))
                prefix.append(token)
            allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
            if not all(eos in allowed for eos in identity['stop_token_ids']):
                raise ValueError('complete example cannot end with original EOS')
            reports.append({'api': api, 'example': index, 'token_count': len(ids), 'all_tokens_allowed': True, 'complete_EOS_allowed': True,
                            'evidence_kind': 'synthetic_contract_test_not_model_output'})
            if progress:
                progress(api, index, len(ids))
        incomplete = '<tool_call>{"name": "' + names[2] + '", "arguments": {"order_id": "arbitrary", "total_minor": '
        enforcer = token_enforcer(data, tools); prefix = []
        for token in tokenizer.encode(incomplete, add_special_tokens=False).ids:
            if token not in enforcer.get_allowed_tokens(prefix).allowed_tokens:
                raise ValueError('incomplete example prefix rejected')
            prefix.append(token)
        if any(eos in enforcer.get_allowed_tokens(prefix).allowed_tokens for eos in identity['stop_token_ids']):
            raise ValueError('premature EOS permitted in incomplete JSON')
    value = {'status': 'PASS', 'evidence_kind': 'CPU_TOKEN_MASK_CONTRACT_CHECK', 'identity': identity,
        'lm_format_enforcer_version': importlib.metadata.version('lm-format-enforcer'), 'examples': reports,
        'incomplete_JSON_EOS_blocked_apis': 2, 'all_four_tools_selectable': True,
        'wrong_ID_and_wrong_amount_allowed': True, 'multiple_calls_and_write_validate_allowed': True,
        'model_forwards': 0, 'GPU_logits_processor_integration': 'NOT_RUN',
        'transformers_imported': 'transformers' in sys.modules, 'torch_imported': 'torch' in sys.modules,
        'scope': 'actual local tokenizer IDs and CPU allowed-token lists on registered synthetic examples, not exhaustive or GPU decoding proof'}
    write(output, value)
    return value

