"""Actual local tokenizer/compiler contract probes, no model load or forward."""
import copy
import hashlib
import json
import pathlib
import sys
import time


def run(snapshot):
    import xgrammar as xgr
    from transformers import AutoTokenizer
    from agent_contract_cost.contracts import validate_schema
    from agent_contract_cost.schemas import G_SCHEMA_SCOPE, server_schema
    from agent_contract_cost.tools import tool_definitions
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    vocab = json.loads((pathlib.Path(snapshot) / 'config.json').read_text())['vocab_size']
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab),
                                  max_threads=2, cache_enabled=False)
    from agent_contract_cost.client import tokenize_messages
    chat_input = tokenize_messages([{'role': 'system', 'content': 'Return one JSON object.'},
                                    {'role': 'user', 'content': 'Synthetic CPU template check.'}], tokenizer)
    independently_encoded = tokenizer.encode(chat_input['prompt_text'], add_special_tokens=False)
    if chat_input['token_ids'] != independently_encoded:
        raise AssertionError('Rendered prompt and actual chat-template token IDs differ')

    def sample(schema):
        if 'enum' in schema:
            return schema['enum'][0]
        kind = schema.get('type')
        if kind == 'object':
            return {key: sample(schema['properties'][key]) for key in schema.get('required', [])}
        if kind == 'array':
            return [sample(schema['items']) for _ in range(max(1, schema.get('minItems', 1)))]
        if kind == 'integer':
            return max(schema.get('minimum', 0), min(987, schema.get('maximum', 987)))
        if kind == 'string':
            return 'ARBITRARY-WRONG-ID'
        if kind == 'boolean':
            return False
        raise AssertionError(schema)

    rows, schemas = [], []
    for family in 'ABC':
        definitions = tool_definitions(family)
        schema = server_schema(definitions)
        compiled = compiler.compile_json_schema(schema)
        schemas.append({'family': family, 'schema': schema,
            'schema_sha256': hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()).hexdigest()})

        def probe(name, obj, allowed, token_path=False, host_schema=None):
            matcher = xgr.GrammarMatcher(compiled, override_stop_tokens=[151645])
            text = obj if isinstance(obj, str) else json.dumps(obj, separators=(',', ':'))
            observed = all(matcher.accept_token(i) for i in tokenizer.encode(text, add_special_tokens=False)) if token_path else matcher.accept_string(text)
            completed = matcher.accept_token(151645) if observed else False
            passed = (observed and completed) == allowed
            row = {'family': family, 'probe': name, 'expected_complete_allowed': allowed,
                   'observed_prefix_allowed': observed, 'EOS_after_text_allowed': completed,
                   'token_path': token_path, 'pass': passed, 'evidence_kind': 'SYNTHETIC_CONTRACT_TEST'}
            if host_schema is not None:
                row['host_rejects'] = validate_schema(obj['arguments'], host_schema) is not None
                row['pass'] = row['pass'] and row['host_rejects']
            rows.append(row)

        for definition in definitions:
            action = {'tool': definition['name'], 'arguments': sample(definition['parameters'])}
            probe('all_tool_and_final_token_path:' + definition['name'], action, True, token_path=True)
            if definition['name'] == 'final':
                true_final = copy.deepcopy(action)
                true_final['arguments']['completed'] = True
                probe('final_completed_true_also_allowed', true_final, True)
        first = definitions[0]
        valid = {'tool': first['name'], 'arguments': sample(first['parameters'])}
        for name, obj in [('extra_envelope', {**valid, 'extra': 1}),
                          ('missing_arguments', {'tool': first['name']}),
                          ('extra_arguments', {'tool': first['name'], 'arguments': {**valid['arguments'], 'extra': 1}}),
                          ('unknown_name', {'tool': 'not_registered', 'arguments': {}}),
                          ('trailing_prose', json.dumps(valid) + ' text'),
                          ('multiple_objects', json.dumps(valid) + json.dumps(valid))]:
            probe(name, obj, False)
        incomplete = xgr.GrammarMatcher(compiled, override_stop_tokens=[151645])
        premature = incomplete.accept_token(151645)
        rows.append({'family': family, 'probe': 'EOS_before_JSON', 'pass': not premature,
                     'observed_allowed': premature, 'evidence_kind': 'SYNTHETIC_CONTRACT_TEST'})
        if family == 'A':
            for label, amount in [('arbitrary_integer', 987654), ('arbitrary_integer_minus_one', 987653),
                                  ('arbitrary_integer_plus_one', 987655), ('negative', -1),
                                  ('bool', True), ('float', 1.0), ('numeric_string', '1')]:
                probe('total_type_or_bound:' + label,
                      {'tool': 'write_order_total', 'arguments': {'order_id': 'WRONG', 'total_minor': amount}},
                      type(amount) is int and amount >= 0)
            for label, lines, discount, expected in [('discount_maximum', [{'quantity': 1, 'unit_price_minor': 1}], 10000, True),
                    ('discount_over_maximum', [{'quantity': 1, 'unit_price_minor': 1}], 10001, False),
                    ('discount_negative', [{'quantity': 1, 'unit_price_minor': 1}], -1, False),
                    ('lines_empty', [], 0, False), ('lines_over_maximum', [{'quantity': 1, 'unit_price_minor': 1}] * 17, 0, False),
                    ('quantity_negative', [{'quantity': -1, 'unit_price_minor': 1}], 0, False)]:
                probe(label, {'tool': 'calculate_total', 'arguments': {'lines': lines, 'discount_bps': discount}}, expected)
        if family == 'B':
            transfer = {'tool': 'transfer_stock', 'arguments': sample(definitions[1]['parameters'])}
            for label, key, value, expected in [('quantity_minimum', 'quantity', 1, True),
                    ('quantity_below_minimum', 'quantity', 0, False), ('revision_negative', 'source_revision', -1, False)]:
                candidate = copy.deepcopy(transfer)
                candidate['arguments'][key] = value
                probe(label, candidate, expected)
        if family == 'C':
            update = {'tool': 'update_document', 'arguments': sample(definitions[1]['parameters'])}
            for label, key, value, expected in [('status_enum', 'status', 'invalid-status', False),
                    ('tags_empty_current_minimum_0', 'tags', [], True), ('tags_over_maximum', 'tags', ['a'] * 9, False),
                    ('revision_negative', 'expected_revision', -1, False)]:
                candidate = copy.deepcopy(update)
                candidate['arguments'][key] = value
                probe(label, candidate, expected)
            duplicate = copy.deepcopy(update)
            duplicate['arguments']['tags'] = ['duplicate', 'duplicate']
            probe('uniqueItems_not_in_server_grammar_host_rejects', duplicate, True,
                  host_schema=definitions[1]['parameters'])
    source = pathlib.Path(__file__).resolve().parents[1] / 'src/agent_contract_cost'
    return {'scope': 'CPU actual pinned tokenizer and xgrammar compiler/matcher; no model loaded or GPU forward',
            'G_SCHEMA_SCOPE': G_SCHEMA_SCOPE, 'rows': rows, 'schemas': schemas,
            'all_pass': all(row['pass'] for row in rows), 'probe_count': len(rows),
            'chat_template_validation': {'pass': True, 'token_ids_type': type(chat_input['token_ids']).__name__,
                'token_count': len(chat_input['token_ids']), 'input_token_sha256': chat_input['input_token_sha256'],
                'rendered_text_reencoding_matches': True, 'evidence_kind': 'SYNTHETIC_CONTRACT_TEST'},
            'source_files': [{'path': name, 'sha256': hashlib.sha256((source / name).read_bytes()).hexdigest()}
                             for name in ('tools.py', 'contracts.py', 'schemas.py', 'client.py', 'server_manager.py', 'p0_inspector.py')],
            'wall_seconds': time.monotonic() - started,
            'preparation_attempt_note': 'r2 tags_empty expected minItems1 while concurrently prepared tools.py had minItems0. r3 isolated minItems1 correctly rejects []; no backend defect inferred.'}


def main(argv=None):
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=pathlib.Path,
                        help='Existing local pinned Qwen snapshot; never downloaded')
    parser.add_argument('--output', required=True, type=pathlib.Path,
                        help='New JSON receipt path; existing files are not overwritten')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists; choose a new receipt path')
    if not (args.snapshot / 'config.json').is_file():
        parser.error('Existing local model config is required')
    # This explicit CPU subprocess owns these environment settings. Importing
    # this helper or requesting --help makes no heavy import and no GPU access.
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
    result = run(str(args.snapshot))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'all_pass': result['all_pass'], 'probe_count': result['probe_count'],
                      'failures': [r for r in result['rows'] if not r['pass']]}))
    return 0 if result['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
