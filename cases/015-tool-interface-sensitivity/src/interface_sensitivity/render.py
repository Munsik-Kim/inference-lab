"""Static, escaped bilingual evidence explorer. No external assets or service."""
import html
import json
from pathlib import Path
from .common import inputs,read,rows,atomic,require
from .analysis import summarize
from .transform import make_interface


def render(case,records_path,output):
    case=Path(case);output=Path(output)
    require(not output.exists(),'use new explorer output directory');output.mkdir(parents=True)
    records=rows(records_path);summary=summarize(records,read(case/'results/p1_r1/planned_cells.json'))
    tasks,messages,tools=inputs(case);tasks={r['id']:r for r in tasks}
    atomic(output/'data.json',{'summary':summary,'episodes':records})
    esc=lambda x:html.escape(str(x),quote=True)
    css='''body{font:16px/1.6 system-ui,sans-serif;background:#f4f6fa;color:#142237;margin:0}main{max-width:1160px;margin:auto;padding:24px}a{color:#1550ad}nav{display:flex;gap:20px;flex-wrap:wrap}h1{line-height:1.25}section{margin:28px 0;padding:20px;background:white;border:1px solid #dbe1ea;border-radius:12px;min-width:0}table{border-collapse:collapse;min-width:580px;width:100%}td,th{padding:10px;border:1px solid #dbe1ea;text-align:left}div.scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f8;padding:12px}details{border-top:1px solid #dbe1ea;padding:10px 0}summary{cursor:pointer;font-weight:600}select{padding:8px;max-width:100%}label{display:inline-flex;gap:8px;margin:8px}*:focus-visible{outline:3px solid #ee9a25;outline-offset:3px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr));gap:12px}small{color:#435672}img{max-width:100%}@media(max-width:600px){main{padding:12px}section{padding:12px}h1{font-size:1.7rem}}'''
    for lang in ('ko','en'):
        ko=lang=='ko'; title='Case015 — 같은 도구, 다른 이름' if ko else 'Case015 — Same Tools, Different Interfaces'
        intro='같은 8개 과제에서 도구의 이름과 순서를 바꾸고 실제 실행·최종 상태·오류 후 행동을 비교했습니다.' if ko else 'The same eight tasks under changed tool names and ordering, tracking real execution, final state and behavior after errors.'
        rows_html=''
        for m in ('BF16','RTN','D16'):
            cells=[r for r in summary['success_table'] if r['method']==m]
            rows_html+='<tr><th>'+m+'</th>'+''.join('<td>'+str(r['success'])+'/'+str(r['observed_tasks'])+('<br><small>미관측 '+str(8-r['observed_tasks'])+'</small>' if r['observed_tasks']!=8 else '')+'</td>' for r in cells)+'</tr>'
        header='<th>Method</th>'+''.join('<th>'+x+'</th>' for x in ['I0 Original','I1 Order','I2 lowerCamelCase','I3 Opaque'])
        effects=summary['contrasts']['D16-minus-BF16']
        parts=[]
        for r in records:
            task=tasks[r['task_id']];spec=make_interface(task,messages[task['id']],tools,r['interface_id'])
            exposed={t['function']['name']:t['function']['description'] for t in spec['tools']}
            result='미관측' if r['functional_success'] is None else '성공' if r['functional_success'] else '실패'
            if not ko:result='unobserved' if r['functional_success'] is None else 'success' if r['functional_success'] else 'failure'
            body='<div class="grid"><div><h4>'+('선언한 도구' if ko else 'Declared tools')+'</h4><pre>'+esc(json.dumps(exposed,ensure_ascii=False,indent=2))+'</pre></div><div><h4>'+('최종 판정' if ko else 'Final verdict')+'</h4><pre>'+esc(json.dumps({k:r[k] for k in ('functional_success','legacy_success','first_error_type','final_failure_labels','final_answer','final_fixture','terminal_reason')},ensure_ascii=False,indent=2))+'</pre></div></div>'
            for response in r['responses']:
                turn=response['model_turn'];body+='<h4>Turn '+str(turn)+'</h4><pre>'+esc(response['raw_text'])+'</pre>'
                for event in r['tool_events']:
                    if event['model_turn']==turn:
                        view={k:event[k] for k in ('raw_name','semantic_tool_id','raw_arguments','allowed_action_set','error_labels','state_before','state_after','tool_result')}
                        body+='<pre>'+esc(json.dumps(view,ensure_ascii=False,indent=2))+'</pre>'
            label=r['task_id']+' · '+r['family_id']+' · '+r['method']+' · '+r['interface_id']+' · '+result
            parts.append('<details class="episode" data-task="'+esc(r['task_id'])+'" data-method="'+esc(r['method'])+'" data-interface="'+esc(r['interface_id'])+'"><summary>'+esc(label)+'</summary>'+body+'</details>')
        filters=''.join('<label>'+name+' <select id="'+key+'"><option value="">'+('전체' if ko else 'All')+'</option>'+''.join('<option>'+esc(v)+'</option>' for v in values)+'</select></label>' for key,name,values in [('task','Task',sorted(tasks)),('method','Method',['BF16','RTN','D16']),('interface','Interface',['I0_ORIGINAL','I1_ORDER_ONLY','I2_STYLE_RENAME','I3_OPAQUE_RENAME'])])
        script="""document.querySelectorAll('select').forEach(s=>s.addEventListener('change',()=>{document.querySelectorAll('.episode').forEach(e=>{e.hidden=['task','method','interface'].some(k=>document.getElementById(k).value&&e.dataset[k]!==document.getElementById(k).value);});}));"""
        links=('<a href="../README.ko.md">코드·CPU 사용법</a>' if ko else '<a href="../README.md">Code and CPU commands</a>')+'<a href="data.json">JSON</a>'
        scope='8개 기존 DEV, 4개 family, 자유 실행. ENUM·개선된 오류 안내·alias OFF. INT4 저장본의 BF16 복원 품질 경로.' if ko else 'Eight existing DEV tasks, four families, free episodes. ENUM, enhanced feedback and aliases OFF. INT4 storage with dense BF16 reference execution.'
        note='각 칸의 분모는 실제 관측된 task 수입니다. 이름 유효율과 실제 과제 성공은 다른 지표입니다. JavaScript 없이도 아래 기록을 열 수 있습니다.' if ko else 'Each denominator is the observed task count. Name validity and functional task success are separate. All records remain readable without JavaScript.'
        report_link='../REPORT.ko.md' if ko else '../REPORT.md'
        text=f'''<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{css}</style><main><nav><a href="{'en' if ko else 'ko'}.html">{'English' if ko else '한국어'}</a><a href="#results">{'결과' if ko else 'Results'}</a><a href="#episodes">Episodes</a><a href="{report_link}">{'상세 보고서' if ko else 'Report'}</a></nav><h1>{title}</h1><p>{intro}</p><p>{scope}</p><section id="results"><h2>{'기능적 성공' if ko else 'Functional success'}</h2><div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{rows_html}</tbody></table></div><p>{note}</p><p>Primary Δ(D16−BF16): {effects['estimate']}; exploratory task-paired95% CI {effects['CI95']}</p><p>{'음수는 변경 인터페이스에서 D16의 signed 변화가 BF16보다 나쁨을 뜻합니다. 손실과 회복은 별도로 셉니다.' if ko else 'Negative values mean a worse signed interface change for D16 than BF16. Losses and gains are counted separately.'}</p><pre>{esc(json.dumps(summary['paired_interface_counts'],ensure_ascii=False,indent=2))}</pre></section><section id="episodes"><h2>{'실제 사건 탐색' if ko else 'Episode evidence'}</h2><p>{'같은 task의 원형과 변경판을 선택해 비교할 수 있습니다. 모두 같은 초기 fixture에서 출발하지만 자유 실행의 이후 상태는 달라질 수 있습니다.' if ko else 'Compare original and changed interfaces for a shared task. Initial fixture state is identical; later free-rollout states may differ.'}</p>{filters}{''.join(parts)}</section><section><h2>{'직접 확인' if ko else 'Verify'}</h2><nav>{links}</nav></section><script>{script}</script></main></html>'''
        (output/(lang+'.html')).write_text(text,encoding='utf-8')
    return {'status':'COMPLETE','episodes':len(records),'html_languages':['ko','en'],'static_results_without_JS':True}
