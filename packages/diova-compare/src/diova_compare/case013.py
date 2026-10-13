"""Export frozen Case 013 records without importing a model or numeric library.

Correctness belongs to the historical parser/action contract. Canonical JSON
only normalizes representation; it does not repair identifiers or answers.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

from .core import require, validate

METHODS = ('BF16', 'RTN', 'D16', 'D4')
GOLD_VERSION = 'case013-r2-first-json-exact-and-required-fixture-actions-v1'
PARSE_FAILURE = '[PARSE_FAILURE:first-json-object-v1]'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON field: ' + key)
        result[key] = value
    return result


def extract_json(text):
    # Intentionally matches the frozen first-valid-object parser, including its
    # original JSON duplicate-key behavior inside a model response.
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '{':
            try:
                value, _ = decoder.raw_decode(text[i:])
                if isinstance(value, dict):
                    return value
            except ValueError:
                pass
    return None


def indexed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, 'duplicate source row: ' + str(key))
        result[key] = row
    return result


class Reader:
    """Verify each consumed public file against the closed original manifest."""
    def __init__(self, case):
        self.case = Path(case).resolve()
        path = self.case / 'provenance/PUBLIC_MANIFEST.json'
        require(path.is_file() and not path.is_symlink(), 'Case 013 public manifest required')
        self.inventory = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)
        self.used = {}
        self.bytes('provenance/PUBLIC_MANIFEST.json', manifest=True)

    def bytes(self, name, manifest=False):
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe source path')
        path = self.case / relative
        require(path.is_file() and not path.is_symlink(), 'missing source file: ' + name)
        require(path.resolve().is_relative_to(self.case), 'source escapes Case 013')
        data = path.read_bytes()
        actual = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        if not manifest:
            require(name in self.inventory['files'], 'unregistered source file: ' + name)
            require(actual == self.inventory['files'][name], 'changed source file: ' + name)
        self.used[name] = actual
        return data

    def json(self, name):
        return json.loads(self.bytes(name).decode('utf-8'), object_pairs_hook=unique_object)

    def rows(self, name):
        return [json.loads(line, object_pairs_hook=unique_object)
                for line in self.bytes(name).decode('utf-8').splitlines() if line.strip()]


def score_agent(record, task, final_config):
    calls = [event for event in record['events'] if 'text' in event]
    require(calls and len(calls) == record['model_calls'], 'model event count mismatch')
    require(sum(x['input_tokens'] for x in calls) == record['total_input_tokens']
            and sum(x['output_tokens'] for x in calls) == record['total_output_tokens'],
            'agent token denominator mismatch')
    parsed = extract_json(calls[-1]['text'])
    correct = bool(parsed is not None and all(parsed.get(k) == v for k, v in task['gold'].items()))
    execution = None
    if task['family'] == 'fixture_patch' and record['mode'] == 'FREE':
        require(final_config is not None, 'missing actual final fixture config')
        valid_event = any(e.get('tool') == 'validate_fixture_test'
                          and e.get('result', {}).get('valid') is True for e in record['events'])
        execution = {'expected_config_matches': final_config == task['fixture']['expected_config'],
                     'successful_validation_event': valid_event}
        correct = correct and all(execution.values())
    require(type(record['success']) is bool and correct == record['success'], 'agent source score mismatch')
    return parsed, correct, calls[0], execution


def export_case013(case, output):
    reader = Reader(case)
    output = Path(output)
    require(not output.exists(), 'export output must be a new directory')
    require(not output.resolve().is_relative_to(reader.case), 'export must be outside frozen Case 013')
    manifest = reader.json('records/experiment_manifest.json')
    contract = reader.json('records/reference_contract.json')
    identity = manifest['models']['4b']
    require(identity['id'] == 'Qwen/Qwen3-4B-Instruct-2507', 'unexpected model identity')
    require(contract['label'] == 'PAPER_INT4_REFERENCE_VARIANT', 'unsupported reference contract')
    model_id = identity['id'] + '@' + identity['revision']
    base_identity = {k: identity[k] for k in ('id', 'revision', 'tokenizer_files')}
    artifacts = {'BF16': digest({'base_checkpoint_identity': base_identity, 'method': 'BF16'})}
    for method in ('RTN', 'D16'):
        meta = reader.json('records/artifact_metadata/4b-' + method + '.json')
        require(meta['method'] == method and meta['projection_count'] == 252
                and len(indexed(meta['files'], ('layer',))) == 252, 'projection identity mismatch')
        artifacts[method] = digest({'base_checkpoint_identity': base_identity, 'method': method,
                                    'contract': contract, 'projection_metadata': meta})
    artifacts['D4'] = digest({'D16_identity': artifacts['D16'], 'activation_contract': contract['D4_runtime'],
                              'method': 'D4'})
    files = {}
    sidecars = {}
    exclusions = []
    pairing = {}
    generation = manifest['generation']
    commands = reader.json('records/attempts/suite_commands.json')
    process = {r['method']: r for r in commands if r['model'] == '4b'}
    require(set(process) == set(METHODS), 'missing original model process receipts')
    for method, receipt in process.items():
        # Explicit CLI model/method arguments link this exit to its original worker.
        command = receipt['command']
        require(command[command.index('--model') + 1] == '4b'
                and command[command.index('--method') + 1] == method
                and type(receipt['exit_code']) is int, 'unlinked source process exit')

    def add(cohort, method, sample_id, token_hash, gold, parsed, correct, sidecar, exit_code):
        prompt = digest({'initial_model_token_sha256': token_hash, 'cohort': cohort,
                         'tokenizer_files': identity['tokenizer_files'], 'generation': generation,
                         'tool_schema_sha256': sidecar.get('tool_schema_sha256'),
                         'gold_contract': GOLD_VERSION})
        pair_key = (cohort, sample_id)
        require(pair_key not in pairing or pairing[pair_key] == (prompt, canonical(gold)),
                'source input/gold pairing mismatch')
        pairing[pair_key] = (prompt, canonical(gold))
        name = cohort + '/' + method + '.jsonl'
        row = dict(schema_version=1, sample_id=sample_id, task='case013.' + cohort.replace('/', '.'),
                   task_version=GOLD_VERSION, prompt_hash=prompt, gold=[canonical(gold)],
                   gold_definition=GOLD_VERSION, model_id=model_id,
                   artifact_id='case013:' + method + ':' + artifacts[method],
                   output_type='extracted_answer', answer=canonical(parsed) if parsed is not None else PARSE_FAILURE,
                   correct=correct, evidence_kind='historical', process_exit_code=exit_code)
        files.setdefault(name, []).append(row)
        sidecars.setdefault(cohort + '/' + method + '.sidecar.jsonl', []).append(
            dict(sample_id=sample_id, task=row['task'], prompt_hash=prompt,
                 initial_model_token_sha256=token_hash, parsed=parsed, **sidecar))

    tasks = reader.json('records/inputs/long_tasks.json')
    taskmap = indexed(tasks, ('id',))
    require(len(taskmap) == 12, 'long task denominator changed')
    expected_long = {(task['id'], length) for task in tasks for length in (8192, 16384)}
    for method in METHODS:
        name = 'records/raw/4b/' + method + '/long_dev.jsonl'
        records = reader.rows(name)
        done = indexed([r for r in records if r['status'] == 'COMPLETE'], ('id', 'length'))
        expected = expected_long | ({(task['id'], 32768) for task in tasks} if method == 'BF16' else set())
        require(set(done) == expected, 'missing/extra long rows or method/length mismatch')
        skipped = [r for r in records if r['status'] != 'COMPLETE']
        require((not skipped) if method == 'BF16' else len(skipped) == 1, 'missing exclusion record')
        for row in skipped:
            require(row['status'] == 'NOT_RUN_MEMORY_POLICY' and row['length'] == 32768, 'unknown exclusion')
            exclusions.append({'method': method, 'length': 32768, 'status': row['status'],
                               'reason': 'Not executed under original memory policy; neither failure nor zero score',
                               'source': name, 'paired_candidate_rows': 0})
        for (sample_id, length), row in done.items():
            task = taskmap[(sample_id,)]
            require(row['method'] == method if 'method' in row else True, 'wrong long method')
            require(row['input_tokens'] == length and row['gold'] == task['answer']
                    and row['evidence'] == task['evidence'] and row['family'] == task['family'], 'long source contract mismatch')
            parsed = extract_json(row['text'])
            correct = bool(parsed and str(parsed.get('answer')) == task['answer']
                           and sorted(parsed.get('evidence', [])) == sorted(task['evidence']))
            require(type(row['exact_success']) is bool and correct == row['exact_success'], 'long score mismatch')
            gold = {'answer': task['answer'], 'evidence': sorted(task['evidence'])}
            add('long/' + str(length), method, sample_id, row['input_token_sha256'], gold, parsed,
                correct, {'source_file': name, 'source_key': [sample_id, length], 'raw_text': row['text'],
                          'record': row, 'scope': 'historical exact; strict whole JSON is not primary'},
                process[method]['exit_code'])

    for cohort, prefix, expected_n in (('base', 'records', 8),
                                       ('extension16k', 'records/agent16k_extension', 4)):
        tasks = reader.json(prefix + '/inputs/agent_tasks.json')
        taskmap = indexed(tasks, ('id',))
        require(len(taskmap) == expected_n, 'agent cohort denominator changed')
        tools = reader.json(prefix + '/inputs/tool_schema.json')
        tool_hash = digest(tools)
        if cohort == 'base':
            tokens = reader.json('records/inputs/agent_token_manifest.json')
            tokens = {r['id']: r['initial_token_sha256'] for r in tokens}
            exits = {m: process[m]['exit_code'] for m in METHODS}
        else:
            registration = reader.json(prefix + '/preregistration.json')
            require(registration['same8_unique_base_tasks_not_new_independent_tasks'] is True,
                    'extension independence contract changed')
            tokens = {r['id']: r['sha256'] for r in registration['input_manifest']}
            commands = reader.json(prefix + '/commands.json')
            exits = {}
            for receipt in commands:
                command = receipt['command']
                method = receipt['method']
                require(method not in exits and command[command.index('--method') + 1] == method
                        and type(receipt['exit_code']) is int, 'unlinked extension process')
                exits[method] = receipt['exit_code']
            require(set(exits) == set(METHODS), 'missing extension process receipts')
        require(set(tokens) == {t['id'] for t in tasks}, 'agent input manifest IDs mismatch')
        for method in METHODS:
            folder = prefix + ('/raw/4b/' if cohort == 'base' else '/raw/') + method
            name = folder + '/agent_dev.jsonl'
            records = indexed(reader.rows(name), ('id', 'mode'))
            require(set(records) == {(t['id'], mode) for t in tasks for mode in ('REPLAY', 'FREE')},
                    'missing/extra agent rows or accidental cohort/mode merge')
            for (sample_id, mode), record in records.items():
                task = taskmap[(sample_id,)]
                require(record['method'] == method and record['gold'] == task['gold']
                        and record['family'] == task['family'], 'agent method/gold contract mismatch')
                final_config = None
                if task['family'] == 'fixture_patch' and mode == 'FREE':
                    final_config = reader.json(folder + '/' + sample_id + '-FREE-config.json')
                parsed, correct, first, execution = score_agent(record, task, final_config)
                if mode == 'FREE':
                    require(first['input_token_sha256'] == tokens[sample_id], 'initial FREE input hash mismatch')
                add('agent/' + cohort + '/' + mode, method, sample_id, first['input_token_sha256'],
                    task['gold'], parsed, correct,
                    {'source_file': name, 'source_key': [sample_id, mode], 'raw_text': record['events'],
                     'final_config': final_config, 'required_execution': execution, 'record': record,
                     'tool_schema_sha256': tool_hash, 'base_task_id': sample_id,
                     'independent_extension_task': False}, exits[method])

    for rows in files.values():
        validate(rows)
    # PPL and saved KL are separate continuous summaries, not classification rows.
    continuous = []
    for model in ('06b', '4b'):
        data = reader.bytes('records/quality_' + model + '.csv').decode('utf-8')
        rows = list(csv.DictReader(data.splitlines()))
        require({r['method'] for r in rows} == set(METHODS) and len(rows) == 4, 'quality summary inventory')
        for row in rows:
            require(abs(math.exp(float(row['nll'])) - float(row['ppl'])) < 1e-8, 'PPL != exp(NLL)')
            continuous.append({'model_key': model, 'classification_conversion': False,
                               'KL_scope': 'saved selected-position full-vocabulary reduction aggregate', **row})
    report = {'adapter': 'case013-r2-v1', 'schema_version': 1, 'status': 'PASS',
              'evidence_kind': 'historical', 'model_id': model_id, 'artifact_ids': artifacts,
              'sources': reader.used, 'converter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'files': {name: {'records': len(rows), 'correct': sum(r['correct'] for r in rows)}
                        for name, rows in files.items()}, 'exclusions': exclusions,
              'independent_units': {'long_unique_tasks': 12, 'agent_unique_base_tasks': 8,
                                    'agent_16k_variants_of_existing_tasks': 4},
              'formal_eligibility': 'NOT_ESTABLISHED', 'new_gpu_runs': 0,
              'scope': 'One cohort/mode/length per file; never pool variants as independent tasks'}
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in {**files, **sidecars}.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(canonical(r) + '\n' for r in rows), encoding='utf-8')
    for name, value in [('adapter_manifest.json', report), ('continuous_summary.json', continuous)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
