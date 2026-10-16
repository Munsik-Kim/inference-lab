"""Static bilingual explorer; response text is always rendered as text."""
import html
import json
from pathlib import Path
from .common import read,rows,atomic
from .analysis import summarize


def render(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=rows(root/'results/a1_r1/episodes.jsonl');tasks=read(root/'data/scenarios.json')
    value={'summary':summarize(records,read(root/'results/a1_r1/planned_cells.json')),'episodes':records,
        'scenarios':tasks,'api_mappings':read(root/'data/api_mappings.json')}
    atomic(output/'data.json',value,True)
    for lang in ['ko','en']:
        ko=lang=='ko'
        title='Case015 — 같은 도구, 다른 이름' if ko else 'Case015 — Same Tools, Different Interfaces'
        intro='API 이름 변경 전후의 조회·계산·기록 수정·검증을 비교합니다. 상태 정답과 생성 전에 실제로 받은 정보에 따른 워크플로 정답을 구분합니다.' if ko else 'Compare lookup, calculation, update and validation before and after an API rename. Correct final state and a workflow grounded in results received before generation are separate endpoints.'
        caution='실측 A1: 한 합성 업무, 4개 지시문 표현, 16개 fixture. 기존 P1 96개와 분모를 합치지 않습니다. INT4 저장본을 BF16으로 복원한 품질 평가입니다.' if ko else 'Measured A1: one synthetic workflow, four instruction templates, 16 fixtures. The denominator is separate from the original 96 P1 episodes. Dense BF16 quality evaluation of saved INT4 artifacts.'
        table=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['method'],r['api_version'],str(r['verified_success'])+'/'+str(r['observed']),r['STATE_ONLY_SUCCESS'],r['observation_violation_episodes']])+'</tr>' for r in value['summary']['success_table'])
        payload=json.dumps(value,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
        other='en' if ko else 'ko'
        document='''<!doctype html><html lang="LANG"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TITLE</title>
<style>body{max-width:1100px;margin:0 auto;padding:24px;font:16px/1.65 system-ui;color:#182f34;background:#f7faf8}a{color:#006a61}table{border-collapse:collapse;min-width:560px;width:100%}td,th{padding:9px;border-bottom:1px solid #bacfcb;text-align:left}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf1ef;padding:12px}select{font:inherit;max-width:100%;padding:8px}.turn{border:1px solid #a9c4bd;border-radius:10px;margin:16px 0;padding:14px}summary{cursor:pointer}a:focus,select:focus,summary:focus{outline:3px solid #c17516}code{overflow-wrap:anywhere}h2{scroll-margin-top:20px}</style>
<nav><a href="OTHER.html">LANG_SWITCH</a> · <a href="#results">RESULTS</a> · <a href="#explorer">EXPLORER</a> · <a href="data.json">JSON</a></nav>
<h1>TITLE</h1><p>INTRO</p><p>CAUTION</p><h2 id="results">RESULTS</h2><div class="scroll"><table><thead><tr><th>Model</th><th>API</th><th>Verified workflow</th><th>Correct state</th><th>Observation violations</th></tr></thead><tbody>TABLE</tbody></table></div>
<h2 id="explorer">EXPLORER</h2><label>Model <select id="method"><option>BF16</option><option>D16</option><option>RTN</option></select></label> <label>API <select id="api"><option>API_V1</option><option>API_V2</option></select></label> <label>Scenario <select id="scenario"></select></label><div id="episode"></div>
<noscript>Core measured results are in the table above. Complete episode records are available in data.json. Enable JavaScript for filtering.</noscript><script id="records" type="application/json">PAYLOAD</script><script>
'use strict';const data=JSON.parse(document.getElementById('records').textContent);const el=id=>document.getElementById(id);const add=(parent,tag,text)=>{const x=document.createElement(tag);x.textContent=text;parent.append(x);return x};const pretty=x=>JSON.stringify(x,null,2);
for(const s of data.scenarios.filter(s=>s.split==='primary')){const o=document.createElement('option');o.value=s.scenario_id;o.textContent=s.scenario_id+' / '+s.template_id;el('scenario').append(o)}
function show(){const row=data.episodes.find(r=>r.method===el('method').value&&r.api_version===el('api').value&&r.scenario_id===el('scenario').value);const target=el('episode');target.replaceChildren();if(!row){add(target,'p','NOT_RUN');return}const task=data.scenarios.find(s=>s.scenario_id===row.scenario_id);add(target,'h3',row.episode_id);add(target,'p','Verified workflow: '+row.VERIFIED_WORKFLOW_SUCCESS+' | Correct final state: '+row.STATE_ONLY_SUCCESS+' | Observation order: '+row.OBSERVATION_ORDER_VALID);add(target,'pre',task.messages.map(m=>m.role+': '+m.content).join('\n\n'));add(target,'h4','Declared API / 선언된 이름');add(target,'pre',pretty(data.api_mappings[row.api_version]));const initial=add(target,'details','');add(initial,'summary','Initial fixture / 초기 주문');add(initial,'pre',pretty(task.orders));
for(const response of row.responses){const turn=add(target,'section','');turn.className='turn';add(turn,'h4','Generation '+response.generated_in_turn+' — one immutable observation list for every call in this response');add(turn,'p','Results received before generation: '+pretty(response.available_result_event_ids));add(turn,'p','History hash: '+response.pre_generation_history_hash);add(turn,'pre',response.raw_text);const events=row.tool_events.filter(e=>e.generated_in_turn===response.generated_in_turn);for(const event of events){const d=add(turn,'details','');add(d,'summary',event.raw_name+' → '+(event.semantic_tool_id||'UNKNOWN_TOOL')+' / '+event.execution_status);add(d,'pre',pretty({arguments:event.raw_arguments,generated_after_event_ids:event.generated_after_event_ids,result:event.tool_result,state_before:event.state_before_execution,state_after:event.state_after_execution,state_version_before:event.state_version_before,state_version_after:event.state_version_after}))}}
add(target,'h4','Final answer and actual final state / 최종 답과 실제 상태');add(target,'pre',pretty({final_answer:row.final_answer,final_fixture:row.final_fixture,observation_violations:row.observation_violations,terminal_reason:row.terminal_reason}));}
for(const id of ['method','api','scenario'])el(id).addEventListener('change',show);show();</script></html>'''
        replacements={'LANG':lang,'TITLE':html.escape(title),'INTRO':html.escape(intro),'CAUTION':html.escape(caution),'OTHER':other,'LANG_SWITCH':'English' if ko else '한국어','RESULTS':'실제 결과' if ko else 'Measured results','EXPLORER':'사건 탐색' if ko else 'Episode explorer','TABLE':table,'PAYLOAD':payload}
        # Explicit tokens prevent accidental replacement inside saved model text.
        import re
        document=re.sub(r'\b(?:LANG_SWITCH|LANG|TITLE|INTRO|CAUTION|OTHER|RESULTS|EXPLORER|TABLE|PAYLOAD)\b',lambda m:replacements[m.group()],document)
        (output/(lang+'.html')).write_text(document,encoding='utf-8')
    return {'status':'PASS','measured_episodes':len(records),'languages':['ko','en'],'offline_file_support':True,'original_response_inserted_as':'textContent'}
