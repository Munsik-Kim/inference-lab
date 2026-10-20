"""Static bilingual result tables and an escaped, lightweight local explorer."""
import html
import json
import pathlib
from .common import load,save

def render_demo(case):
    case=pathlib.Path(case);summary_path=case/'results/p1_r1/summary.json'
    summary=load(summary_path) if summary_path.exists() else {'status':'NOT_RUN','groups':[]}
    episodes=[]
    for p in sorted((case/'results/p1_r1/episodes').glob('*.json')):
        x=load(p);episodes.append({k:v for k,v in x.items() if k not in ('responses',) }|{
          'responses':[{k:r.get(k) for k in ('turn','available_result_event_ids','state_digest','state_version','raw_text','tool_calls','content','output_tokens','server_finish_reason','server_stop_reason','input_token_sha256','limiting_factors','http_end_seconds')} for r in x['responses']],
          'raw_record':'../results/p1_r1/episodes/'+p.name})
    tasks={t['task_id']:{k:t[k] for k in ('family','visible','dependency_edges')}for t in load(case/'data/tasks.json')if t['split']=='DEV'}
    display={'summary':summary,'episodes':episodes,'tasks':tasks,'model_forward_reproduced_in_browser':False}
    save(case/'demo/data.json',display)
    blob=json.dumps(display,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    for lang in ('ko','en'):
        ko=lang=='ko';title='작은 KV 캐시로 더 많은 업무를 정확히 끝낼 수 있을까?'if ko else 'Can a smaller KV cache complete more jobs correctly?'
        intro=('KV 캐시는 모델이 읽은 내용의 중간 계산 결과를 저장하는 공간입니다. 같은 BF16 가중치를 유지하고 이 저장 형식만 바꾸어, 긴 자료를 읽고 계산·기록·검증하는 합성 업무를 비교합니다.' if ko else 'A KV cache stores intermediate attention results for the text a model has read. We keep BF16 weights fixed and vary this cache format while testing synthetic jobs that read long evidence, calculate, update a record and verify it.')
        scope=('DEV12개에서 메뉴·응답 한도를 바꾼 P1입니다. 실제 업무 처리량을 검증하는 P2/P3는 미실행입니다. 같은 합성 생성기의 반복을 독립 회사 표본으로 세지 않습니다.'if ko else 'P1 varies tool menus and response caps on12 DEV fixtures. P2/P3 throughput experiments are not run. Generator instances are not independent companies.')
        header=['KV','메뉴','응답 cap','완료/예정','업무 성공/12','상태 정답/12','최종 답/12']if ko else ['KV','Menu','Response cap','Completed/planned','Workflow/12','State/12','Final answer/12']
        trs=[]
        for g in summary['groups']:
            cells=[g['kv'],g['menu'],g['cap'],f'{g["completed"]}/{g["planned"]}',g['verified'],g['state'],g['final']]
            trs.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in cells)+'</tr>')
        code_label='생성 전 관측과 실행 사건'if ko else 'Pre-generation observations and executed events'
        text=f'''<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Case016 — {title}</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;color:#e9edf4;background:#101723;margin:0}}main{{max-width:1100px;margin:auto;padding:24px}}a{{color:#8bc2ff}}nav{{display:flex;gap:18px;flex-wrap:wrap}}h1{{font-size:clamp(1.7rem,4vw,2.6rem)}}.scroll{{max-width:100%;overflow:auto}}table{{border-collapse:collapse;min-width:620px;width:100%}}th,td{{padding:10px;border-bottom:1px solid #475267;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#182333;padding:16px;max-height:640px;overflow:auto}}select{{font:inherit;max-width:100%;margin:8px;padding:6px}}:focus-visible{{outline:3px solid #ffcd70}}.scope{{border-left:3px solid #8bc2ff;padding:12px}}code{{overflow-wrap:anywhere}}</style>
<main><nav><a href="{'en'if ko else'ko'}.html">{'English'if ko else'한국어'}</a><a href="#results">{'결과'if ko else'Results'}</a><a href="#events">{'사건 탐색'if ko else'Event explorer'}</a><a href="../REPRODUCE.{'ko.md'if ko else'md'}">CPU reproduction</a></nav>
<h1>Case016 — {title}</h1><p>{intro}</p><p class="scope">{scope}</p><p>BF16 weights · Qwen3-4B · actual vLLM KV · <strong>{html.escape(summary['status'])}</strong></p>
<h2 id="results">{'성공과 실패'if ko else'Successes and failures'}</h2><div class="scroll"><table><thead><tr>{''.join('<th>'+x+'</th>'for x in header)}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>
<p>{'업무 성공은 올바른 파일 상태뿐 아니라 필요한 관측 순서와 최신 검증을 받은 최종 답까지 확인합니다. 미관측은 0점으로 채우지 않습니다.'if ko else'Workflow success requires correct state, observation order, current validation and a strict final answer. Unobserved cells are not imputed as failures.'}</p>
<h2 id="events">{code_label}</h2><label for="episode">Episode</label><select id="episode"><option value="">{'기록 선택'if ko else'Select a record'}</option></select><p id="recordlink"></p><pre id="viewer">{'JavaScript를 켜면 저장 사건을 선택해서 볼 수 있습니다. 위 표와 보고서는 JavaScript 없이 읽을 수 있습니다.'if ko else'Enable JavaScript to select recorded events. The table above and reports remain readable without JavaScript.'}</pre>
<p><a href="../REPORT.{'ko.md'if ko else'md'}">{'전체 보고서'if ko else'Full report'}</a> · <a href="../specs/protocol.json">Protocol</a> · <a href="data.json">Display data</a></p></main>
<script type="application/json" id="data">{blob}</script><script>const data=JSON.parse(document.getElementById('data').textContent);const select=document.getElementById('episode');for(const e of data.episodes){{const o=document.createElement('option');o.value=e.episode_id;o.textContent=e.episode_id+' — '+e.terminal_reason;select.appendChild(o)}}select.addEventListener('change',()=>{{const e=data.episodes.find(x=>x.episode_id===select.value);document.getElementById('viewer').textContent=e?JSON.stringify({{task:data.tasks[e.task_id],episode:e}},null,2):'';const p=document.getElementById('recordlink');p.replaceChildren();if(e){{const a=document.createElement('a');a.href=e.raw_record;a.textContent='Raw record / token and prompt hashes';p.appendChild(a)}}}});</script></html>'''
        (case/f'demo/{lang}.html').write_text(text)
