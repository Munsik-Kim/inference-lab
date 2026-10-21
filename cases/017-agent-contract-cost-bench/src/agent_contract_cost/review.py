"""Posthoc bilingual review pages from audited scalar rows and stored cells.

This display module is outside execution freeze. It never imports an engine,
creates a task, executes a fixture, retokenizes a prompt, or claims a new run.
"""
import argparse
import copy
import hashlib
import html
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import quote


POLICIES = ('C0', 'G', 'N', 'E')
KVS = ('K0', 'K1')
RUNS = ('p1_r1', 'p2_r1')
DISPLAY_EPISODE_FIELDS = ('episode_id', 'task_id', 'family', 'stage', 'kv', 'policy',
    'canonical_task_sha256', 'initial_prompt_hash', 'initial_input_token_sha256', 'initial_prompt_tokens',
    'final_action', 'final_available_result_event_ids', 'final_state', 'state_version', 'terminal_reason',
    'observation_status', 'reported_endpoint', 'budget', 'first_failure', 'failures',
    'last_available_result_event_ids', 'last_observation_sha256', 'requires_server_drain_check', 'wall_seconds')
DISPLAY_REQUEST_FIELDS = ('generation_turn', 'messages', 'available_result_event_ids', 'observation_sha256',
    'state_digest', 'state_version', 'budget_before', 'budget_after', 'request_plan', 'parse', 'event',
    'feedback', 'retry_triggered', 'retry_consumed', 'model_request_attempted', 'input_token_sha256', 'local_error')
DISPLAY_RESPONSE_FIELDS = ('raw_text', 'raw_token_text', 'raw_utf8_sha256', 'raw_token_text_sha256',
    'output_token_ids', 'output_token_sha256', 'output_tokens', 'usage', 'status', 'transport_status',
    'finish_reason', 'stop_reason', 'elapsed_seconds', 'http_end_seconds', 'http_status',
    'first_generated_token_id_event_seconds', 'first_role_chunk_seconds', 'token_events',
    'model_request_attempts', 'model_responses_received', 'server_abort_requests', 'error_type', 'error',
    'drain_status', 'stream_close_on_error', 'request_id', 'input_token_sha256')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _read(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _file_record(path, root):
    data = path.read_bytes()
    return {'path': path.relative_to(root).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def _observed(row):
    return row.get('audit_pass') is True and row.get('observation_status') == 'OBSERVED' and type(row.get('verified')) is bool


def display_episode(raw):
    """An explicit display projection; lossless transport evidence stays in cell."""
    result = {key: copy.deepcopy(raw[key]) for key in DISPLAY_EPISODE_FIELDS if key in raw}
    requests = []
    for original in raw.get('requests', []):
        row = {key: copy.deepcopy(original[key]) for key in DISPLAY_REQUEST_FIELDS if key in original}
        response = original.get('response')
        row['response'] = {key: copy.deepcopy(response[key]) for key in DISPLAY_RESPONSE_FIELDS if key in response} if isinstance(response, dict) else None
        requests.append(row)
    result['requests'] = requests
    result['display_projection'] = {'lossless_raw': False,
        'omitted_from_display_copy': ['response.raw_sse_lines', 'response.sse_events',
            'response.prompt_token_ids', 'response.prompt_text', 'response.request_body',
            'request.input_token_ids', 'request.prompt_text', 'top-level duplicate events'],
        'original_evidence': 'Original cell file retained byte-for-byte; source_path/source_sha256 identify it'}
    return result


def intern_messages(episodes):
    """Content-address repeated message objects, without altering UTF8 content.

    Reconstructing a request's message_refs through this pool must exactly
    reproduce its former message list and canonical serialized bytes. This is
    display storage deduplication, never prompt truncation or rewriting.
    """
    pool = {}
    original_bytes = unique_bytes = request_count = 0
    for episode in episodes:
        for request in episode['raw'].get('requests', []):
            if 'messages' not in request:
                continue
            messages = request['messages']
            if not isinstance(messages, list):
                raise ValueError('Stored request messages must be an array')
            refs = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError('Stored message must be an object')
                encoded = json.dumps(message, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()
                reference = hashlib.sha256(encoded).hexdigest()
                original_bytes += len(encoded)
                if reference in pool:
                    if pool[reference] != message:
                        raise ValueError('Message pool hash collision')
                else:
                    pool[reference] = copy.deepcopy(message)
                    unique_bytes += len(encoded)
                refs.append(reference)
            reconstructed = [pool[ref] for ref in refs]
            if reconstructed != messages or digest(reconstructed) != digest(messages):
                raise ValueError('Message pool reconstruction changed model-visible input')
            del request['messages']
            request['message_refs'] = refs
            request['messages_sha256'] = digest(messages)
            request_count += 1
    return pool, {'pooled_request_count': request_count, 'unique_message_count': len(pool),
                  'message_canonical_bytes_before': original_bytes,
                  'message_canonical_bytes_after': unique_bytes,
                  'reconstruction_check': 'exact object values and canonical UTF8 bytes/hash',
                  'model_visible_content_changed': False}


def group_rows(rows, planned=None):
    """Compute display counts/costs from audited rows, never saved group scores."""
    groups = []
    for kv in KVS:
        for policy in POLICIES:
            present = [r for r in rows if r['kv'] == kv and r['policy'] == policy]
            observed = [r for r in present if _observed(r)]
            costs = {key: sum(r.get('counters', {}).get(key, 0) for r in present)
                     for key in ('model_request_attempts', 'model_responses_received', 'generated_output_tokens',
                                 'parsed_tool_attempts', 'contract_rejections', 'callable_executions',
                                 'side_effect_writes', 'final_attempts', 'retry_consumed')}
            walls = [r['wall_seconds'] for r in present if type(r.get('wall_seconds')) in (int, float)]
            groups.append({'kv': kv, 'policy': policy, 'planned': planned,
                           'present': len(present), 'observed': len(observed),
                           'verified_success': sum(r['verified'] for r in observed),
                           'state_correct': sum(r.get('state_correct') is True for r in observed),
                           'final_correct': sum(r.get('final_correct') is True for r in observed),
                           'wall_sum_seconds': sum(walls),
                           'terminal_counts': dict(Counter(r['terminal_reason'] for r in present)), **costs})
    return groups


def load_review(case_root):
    """Fail closed on duplicates, missing audit/raw identity, or stale summary."""
    case_root = Path(case_root).resolve()
    result = {'schema_version': 'case017-review-v2-message-pool', 'evidence_kind': 'POSTHOC_DISPLAY_FROM_STORED_AUDIT',
              'new_model_requests': 0, 'new_fixture_executions': 0, 'runs': [], 'episodes': [], 'sources': []}
    for run_id in RUNS:
        root = case_root / 'results' / run_id
        audited_path, summary_path = root / 'audit/audited_rows.json', root / 'analysis/summary.json'
        if not audited_path.is_file():
            stored = any((root / 'cells').glob('*.json'))
            result['runs'].append({'run_id': run_id, 'status': 'AWAITING_AUDIT' if stored else 'NOT_RUN',
                                   'groups': [], 'reason': 'Audited measurement rows are not available'})
            continue
        rows = _read(audited_path)
        if not isinstance(rows, list):
            raise ValueError('Audited rows must be an array')
        if any(r.get('evidence_kind') != 'measurement' for r in rows):
            raise ValueError('Synthetic contract tests cannot populate measurement result pages')
        keys = [(r['task_id'], r['kv'], r['policy']) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate audited condition cell')
        if any(r['kv'] not in KVS or r['policy'] not in POLICIES for r in rows):
            raise ValueError('Unregistered KV or policy')
        summary = _read(summary_path) if summary_path.is_file() else None
        if summary is not None and summary.get('evidence_scope') != 'model_measurement_stored_trace_audit':
            raise ValueError('Only stored measurement audit summaries may be displayed')
        planned = summary.get('registered_tasks') if summary else None
        groups = group_rows(rows, planned)
        if summary:
            expected = {(r['kv'], r['policy']): r for r in summary['groups']}
            for group in groups:
                saved = expected.get((group['kv'], group['policy']))
                if saved is None:
                    raise ValueError('Missing saved summary group')
                for field in ('planned', 'present', 'observed', 'verified_success', 'state_correct', 'final_correct',
                              'model_request_attempts', 'model_responses_received', 'generated_output_tokens',
                              'parsed_tool_attempts', 'contract_rejections', 'callable_executions'):
                    if group[field] != saved[field]:
                        raise ValueError('Summary disagrees with independently audited rows: ' + field)
        lookup = {key: row for key, row in zip(keys, rows)}
        used = set()
        raw_paths = sorted((root / 'cells').glob('*.json'))
        for source in raw_paths:
            if source.is_symlink():
                raise ValueError('Refuse symlink cell source')
            raw = _read(source)
            key = (raw.get('task_id'), raw.get('kv'), raw.get('policy'))
            if key not in lookup:
                raise ValueError('Raw cell has no independently audited row')
            if key in used:
                raise ValueError('Duplicate raw condition cell')
            row = lookup[key]
            if row.get('source_sha256') != digest(raw):
                raise ValueError('Raw cell disagrees with audited source hash')
            used.add(key)
            result['sources'].append(_file_record(source, case_root))
            result['episodes'].append({'run_id': run_id, 'episode_id': raw.get('episode_id'),
                'task_id': raw['task_id'], 'family': raw['family'], 'kv': raw['kv'], 'policy': raw['policy'],
                'source_path': source.relative_to(case_root).as_posix(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'audit': copy.deepcopy(row), 'raw': display_episode(raw)})
        if used != set(lookup):
            raise ValueError('Audited rows do not all have stored raw cells')
        result['sources'].append(_file_record(audited_path, case_root))
        if summary_path.is_file():
            result['sources'].append(_file_record(summary_path, case_root))
        result['runs'].append({'run_id': run_id, 'status': summary.get('status', 'PARTIAL') if summary else 'PARTIAL',
                               'planned_cells': summary.get('planned_cells') if summary else None,
                               'groups': groups, 'comparisons': summary.get('comparisons', []) if summary else [],
                               'cost_scope': 'Sum of episode walls including failures; not serving workload makespan'})
    result['message_pool'], result['message_pool_verification'] = intern_messages(result['episodes'])
    return result


def _safe_json(value):
    """Embedding data never lets a generated </script> become markup."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def _text(value):
    return html.escape(str(value), quote=True)


def _value(value):
    if value is None:
        return 'NOT_RECORDED'
    if type(value) is float:
        return f'{value:.3f}'
    return _text(value)


def _summary_html(data, lang):
    ko = lang == 'ko'
    fragments = []
    for run in data['runs']:
        fragments.append('<section><h2>' + _text(run['run_id']) + ' · ' + _text(run['status']) + '</h2>')
        if not run['groups']:
            fragments.append('<p>' + ('저장된 모델 측정과 감사 결과가 아직 없습니다. 0점이나 성공으로 표시하지 않습니다.' if ko else 'Stored model measurements and audit results are not available. Missing results are not zeros or successes.') + '</p></section>')
            continue
        headers = ['KV', '정책', '완주 / 관측 / 계획', '상태 정답', '최종 답 정답', '모델 요청', '출력 token', '계약 거절', 'episode wall 합계(s)'] if ko else ['KV', 'Policy', 'Verified / observed / planned', 'State correct', 'Final correct', 'Model attempts', 'Output tokens', 'Contract rejections', 'Episode wall sum (s)']
        fragments.append('<div class="scroll"><table><thead><tr>' + ''.join('<th scope="col">' + h + '</th>' for h in headers) + '</tr></thead><tbody>')
        for g in run['groups']:
            values = [g['kv'], g['policy'], f"{g['verified_success']} / {g['observed']} / {g['planned'] if g['planned'] is not None else 'NOT_RECORDED'}", g['state_correct'], g['final_correct'], g['model_request_attempts'], g['generated_output_tokens'], g['contract_rejections'], g['wall_sum_seconds']]
            fragments.append('<tr>' + ''.join('<td>' + _value(v) + '</td>' for v in values) + '</tr>')
        fragments.append('</tbody></table></div><p class="note">' + ('완주는 실제 상태·필수 실행·관측 순서·검증 관측·엄격한 final을 모두 만족한 경우입니다. 비용은 실패를 포함한 episode 시간 합계이며 서빙 처리량이 아닙니다.' if ko else 'Verified completion requires actual state, required actions, observation order, observed validation and the strict final action. Cost sums episode time including failures; it is not serving throughput.') + '</p>')
        if run['comparisons']:
            fragments.append('<details><summary>' + ('등록된 paired 비교' if ko else 'Registered paired comparisons') + '</summary><pre>' + _text(json.dumps(run['comparisons'], ensure_ascii=False, indent=2)) + '</pre></details>')
        fragments.append('</section>')
    return ''.join(fragments)


CSS = """body{font:16px/1.65 system-ui,sans-serif;margin:auto;max-width:1160px;padding:24px;color:#182537;background:#f7f9fc;overflow-wrap:anywhere}a{color:#0756a6}header{border-bottom:1px solid #b8c5d8;margin-bottom:24px}section{background:white;padding:20px;margin:20px 0;border:1px solid #d5deea;border-radius:8px}h1{line-height:1.25}h2,h3{line-height:1.35}label{display:inline-block;margin:8px 14px 8px 0}select{display:block;max-width:260px;padding:8px;font:inherit}button,a,select,summary{outline-offset:4px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f9;padding:14px;max-height:38rem;overflow:auto}.scroll{max-width:100%;overflow:auto}table{border-collapse:collapse;min-width:820px;width:100%}th,td{border-bottom:1px solid #d5deea;padding:8px;text-align:left;vertical-align:top}th{background:#eef3fa}.note{color:#43556d}.turn{border:1px solid #aabed8;border-left:5px solid #3473b1;padding:12px;margin:16px 0}.pill{font-weight:650;background:#eaf0f9;padding:3px 8px;border-radius:4px}details{margin:10px 0}summary{cursor:pointer}#episode-view h3{overflow-wrap:anywhere}@media(max-width:600px){body{padding:12px}section{padding:12px}h1{font-size:1.65rem}select{width:100%;max-width:100%}label{display:block;margin:10px 0}}"""


JS = r"""(() => {
'use strict';
const data=JSON.parse(document.getElementById('review-data').textContent), lang=document.documentElement.lang, ko=lang==='ko';
const view=document.getElementById('episode-view'), pick=document.getElementById('episode'), kv=document.getElementById('kv'), policy=document.getElementById('policy'), family=document.getElementById('family');
function messages(request){return (request.message_refs||[]).map(ref=>{if(!Object.prototype.hasOwnProperty.call(data.message_pool,ref))throw new Error('Missing message pool object');return data.message_pool[ref];});}
function node(tag,text,parent){const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(parent)parent.append(n);return n;}
function block(title,value,parent){const d=node('details',undefined,parent);node('summary',title,d);node('pre',JSON.stringify(value,null,2),d);}
function show(){view.replaceChildren();const e=data.episodes[Number(pick.value)];if(!e){node('p',ko?'선택 조건에 관측된 episode가 없습니다.':'No observed episode matches this selection.',view);return;}
node('h3',`${e.run_id} · ${e.task_id} · ${e.kv} · ${e.policy}`,view);
node('p',`${ko?'독립 감사':'Independent audit'}: ${e.audit.audit_pass} · VERIFIED=${e.audit.verified} · STATE=${e.audit.state_correct} · FINAL=${e.audit.final_correct}`,view);
node('p',ko?'다른 조건의 같은 번호 generation이 같은 관측 이력이라는 뜻은 아닙니다. 아래에는 이 episode가 생성 전에 실제 받은 정보만 표시합니다.':'The same generation number in another condition does not imply the same observation history. Below are the observations actually available before each generation.',view);
block(ko?'모델에 보인 초기 요청·공개 도구 정의':'Initial request and public action definitions',messages(e.raw.requests[0]||{}),view);
(e.raw.requests||[]).forEach((r,i)=>{const s=node('section',undefined,view);s.className='turn';node('h4',`Generation ${r.generation_turn===undefined?i:r.generation_turn}`,s);const response=r.response||{};
block(ko?'생성 전 관측 event ID':'Observed event IDs before generation',r.available_result_event_ids||[],s);
block(ko?'생성 전 상태·남은 예산':'State and budget before generation',{state_digest:r.state_digest,state_version:r.state_version,observation_sha256:r.observation_sha256,budget:r.budget_before,request_plan:r.request_plan},s);
block(ko?'원 생성 application text':'Raw generated application text',response.raw_text,s);
block(ko?'EOS를 보존한 token 문자열·ID·종료':'Token text retaining EOS, IDs and termination',{raw_token_text:response.raw_token_text,output_token_ids:response.output_token_ids,finish:response.finish_reason,stop:response.stop_reason,usage:response.usage},s);
block(ko?'엄격 parser 판정·호출':'Strict parser decision and action',r.parse,s);
if(r.feedback)block(ko?'host 재시도 안내':'Host retry feedback',r.feedback,s);
if(r.event){block(ko?'실제 도구 사건·반환값':'Actual tool event and result',{event_id:r.event.event_id,tool:r.event.tool,arguments:r.event.arguments,status:r.event.status,result:r.event.result,generated_in_turn:r.event.generated_in_turn,generated_after_event_ids:r.event.generated_after_event_ids,callable_executed:r.event.callable_executed},s);block(ko?'실행 전후 JSON':'JSON before and after execution',{before:r.event.state_before,after:r.event.state_after,version_before:r.event.version_before,version_after:r.event.version_after},s);}
block(ko?'전체 요청 메시지':'Full request messages',messages(r),s);});
block(ko?'최종 상태·final·독립 endpoint':'Final state, final action and independent endpoint',{state:e.raw.final_state,final:e.raw.final_action,final_available_result_event_ids:e.raw.final_available_result_event_ids,endpoint:e.audit.endpoint,terminal:e.raw.terminal_reason,failures:e.raw.failures,budget:e.raw.budget},view);
const source=node('p',undefined,view);const link=node('a',ko?'손실 없이 보존한 원 cell JSON':'Original lossless cell JSON',source);link.href=e.source_href;node('span',` · ${e.source_path} · SHA256 ${e.source_sha256}`,source);
node('p',ko?'이 표시용 사본은 원 SSE·입력 token IDs·request body의 중복 복사를 생략했습니다. 해당 정보는 위 원 cell 파일에 보존되어 있습니다.':'This display projection omits duplicate raw SSE, input token IDs and request bodies; those remain in the original cell file linked above.',view);}
function filter(){const old=pick.value;pick.replaceChildren();data.episodes.forEach((e,i)=>{if((!kv.value||e.kv===kv.value)&&(!policy.value||e.policy===policy.value)&&(!family.value||e.family===family.value)){const o=node('option',`${e.run_id} / ${e.task_id} / ${e.kv} / ${e.policy}`,pick);o.value=String(i);}});if([...pick.options].some(o=>o.value===old))pick.value=old;show();}
for(const s of[kv,policy,family])s.addEventListener('change',filter);pick.addEventListener('change',show);filter();
})();"""


def page(data, lang):
    ko = lang == 'ko'
    title = 'Case017 — 도구 실행 계약과 비용' if ko else 'Case017 — Agent Contract and Cost'
    intro = '같은 합성 업무에서 자유 생성, 문법 제약, 중립 재시도, 오류 설명 재시도를 비교합니다. 모델 가중치는 BF16으로 고정하고 KV 저장 정밀도를 비교합니다.' if ko else 'Compare free generation, grammar, neutral retry and error-feedback retry on the same synthetic workflows, with BF16 weights and two native KV storage precisions.'
    explorer = '사건 탐색' if ko else 'Episode explorer'
    model = '모델: Qwen3-4B-Instruct-2507 · BF16 weights' if ko else 'Model: Qwen3-4B-Instruct-2507 · BF16 weights'
    options = lambda values: '<option value="">' + ('전체' if ko else 'All') + '</option>' + ''.join('<option value="' + _text(v) + '">' + _text(v) + '</option>' for v in values)
    labels = [('kv', 'KV', KVS), ('policy', '정책' if ko else 'Policy', POLICIES), ('family', '업무' if ko else 'Family', ('A', 'B', 'C'))]
    controls = ''.join('<label>' + label + '<select id="' + id_ + '">' + options(values) + '</select></label>' for id_, label, values in labels)
    return '<!doctype html><html lang="' + lang + '"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>' + CSS + '</style></head><body><header><p><a href="' + ('en.html' if ko else 'ko.html') + '">' + ('English' if ko else '한국어') + '</a> · <a href="data.json">JSON</a></p><h1>' + title + '</h1><p>' + intro + '</p><p class="note">' + ('이 화면은 저장 원자료와 독립 CPU 감사의 표시본입니다. 새 모델 실행이나 복구 실험이 아닙니다.' if ko else 'This is a display of stored raw records and independent CPU audit, not a new model run or recovery experiment.') + '</p></header>' + _summary_html(data, lang) + '<section><h2>' + explorer + '</h2><label>' + ('모델' if ko else 'Model') + '<select id="model"><option>' + model + '</option></select></label><p>' + ('A는 여러 품목의 구매 기록 금액 계산, B는 창고 간 재고 이동, C는 문서 상태·태그·담당자 변경입니다. 전부 소유한 합성 JSON 자료입니다.' if ko else 'A computes a purchase-record total, B moves stock between warehouses, and C updates document status, tags and owner. All records are owned synthetic JSON fixtures.') + '</p>' + controls + '<label>' + ('과제·episode' if ko else 'Task and episode') + '<select id="episode"></select></label><noscript><p>' + ('JavaScript가 꺼져 있어도 위 핵심 결과표와 JSON 자료는 읽을 수 있습니다.' if ko else 'Core result tables above and the JSON data remain readable without JavaScript.') + '</p></noscript><div id="episode-view"></div></section><script id="review-data" type="application/json">' + _safe_json(data) + '</script><script>' + JS + '</script></body></html>\n'


def render(case_root, output):
    data = load_review(case_root)
    output = Path(output)
    source_root_href = Path(os.path.relpath(Path(case_root).resolve(), output.resolve())).as_posix()
    for episode in data['episodes']:
        episode['source_href'] = quote(source_root_href + '/' + episode['source_path'], safe='/.-_')
    data['original_link_scope'] = 'Relative links to existing original case cells; regenerate the display after relocating it'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    for lang in ('ko', 'en'):
        (output / (lang + '.html')).write_text(page(data, lang), encoding='utf-8')
    receipt = {'status': 'RENDERED', 'evidence_kind': data['evidence_kind'], 'episodes': len(data['episodes']),
               'run_statuses': {r['run_id']: r['status'] for r in data['runs']},
               'new_model_requests': 0, 'source_files': data['sources'],
               'message_pool_verification': data['message_pool_verification'],
               'files': [_file_record(p, output) for p in sorted(output.iterdir())]}
    (output / 'render_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New external output directory; never overwritten')
    parser.add_argument('--case-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    print(json.dumps(render(args.case_root, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
