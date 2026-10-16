"""Display-only comparison overlays; frozen GPU/research source remains unchanged."""
import argparse,html,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from interface_sensitivity.common import read,sha,atomic
from interface_sensitivity.render import render

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();case=Path(__file__).resolve().parents[1]
 render(case,case/'results/p1_r1/episodes.jsonl',a.output)
 comparisons=read(case/'results/p1_r1/common_event_prefixes.json');data=read(a.output/'data.json');data['common_event_prefix_comparisons']=comparisons;atomic(a.output/'data.json',data)
 for lang in ['ko','en']:
  f=a.output/(lang+'.html');s=f.read_text();ko=lang=='ko';esc=lambda x:html.escape(str(x),quote=True)
  for c in comparisons:
   start=s.index('<details class="episode" data-task="'+c['task_id']+'" data-method="'+c['method']+'" data-interface="'+c['changed_interface']+'">')
   pos=s.index('</summary>',start)+len('</summary>')
   title='원형 I0와 사건 비교' if ko else 'Event comparison with original I0'
   state='다음 행동 직전 같은 상태' if ko else 'Same state before next action'
   text=f'<div class="comparison"><h3>{title}</h3><p>Common event prefix: {c["common_event_prefix_length"]} · {state}: {c["next_actions_comparable_same_state"]}</p><div class="grid"><pre>I0\n'+esc(json.dumps(c['baseline_next_event'],ensure_ascii=False,indent=2))+'</pre><pre>'+esc(c['changed_interface'])+'\n'+esc(json.dumps(c['changed_next_event'],ensure_ascii=False,indent=2))+'</pre></div><small>'+('null은 다음 도구 사건이 없다는 뜻입니다. 다른 턴 번호를 임의로 대응시키지 않습니다.' if ko else 'null means no next tool event. Arbitrary turn numbers are not forced into alignment.')+'</small></div>'
   s=s[:pos]+text+s[pos:]
  tbody=''
  for m in ['BF16','RTN','D16']:
   tbody+='<tr><th>'+m+'</th>'
   for i in ['I0_ORIGINAL','I1_ORDER_ONLY','I2_STYLE_RENAME','I3_OPAQUE_RENAME']:
    r=next(r for r in data['summary']['success_table'] if r['method']==m and r['interface_id']==i)
    tbody+='<td><a class="matrix-link" href="#episodes" data-method="'+m+'" data-interface="'+i+'" aria-label="'+m+' '+i+' '+('8개 task 보기' if ko else 'view8tasks')+'">'+str(r['success'])+'/'+str(r['observed_tasks'])+'</a></td>'
   tbody+='</tr>'
  pos=s.index('<tbody>');end=s.index('</tbody>',pos);s=s[:pos]+'<tbody>'+tbody+s[end:]
  script="document.querySelectorAll('.matrix-link').forEach(a=>a.addEventListener('click',()=>{document.getElementById('task').value='';document.getElementById('method').value=a.dataset.method;document.getElementById('interface').value=a.dataset.interface;document.getElementById('method').dispatchEvent(new Event('change'));}));"
  effect=data['summary']['contrasts']['D16-minus-BF16']
  old=f"Primary Δ(D16−BF16): {effect['estimate']}; exploratory task-paired95% CI {effect['CI95']}"
  new=("주 signed 변화 차이" if ko else "Primary signed-change interaction")+f": {effect['estimate']*100:+.2f} pp · 95% CI [{effect['CI95'][0]*100:+.2f}, {effect['CI95'][1]*100:+.2f}] pp"
  s=s.replace(old,new)
  paired=data['summary']['paired_interface_counts']
  original_dump='<pre>'+esc(json.dumps(paired,ensure_ascii=False,indent=2))+'</pre>'
  table='<div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in (['모델','변경','손실 /8','회복 /8','둘 다 실패 /8'] if ko else ['Model','Changed','Loss /8','Gain /8','Both fail /8']))+'</tr></thead><tbody>'
  for r in paired:table+='<tr><td>'+r['method']+'</td><td>'+r['changed'].split('_')[0]+'</td>'+''.join('<td>'+str(r[k])+'</td>' for k in ['baseline_only','changed_only','both_fail'])+'</tr>'
  table+='</tbody></table></div>'
  observed=[]
  for r in paired:
   if r['baseline_only'] or r['changed_only']:
    observed.append(r['method']+' '+r['changed'].split('_')[0]+(': 손실 '+str(r['baseline_only'])+'개 · 회복 '+str(r['changed_only'])+'개' if ko else ': '+str(r['baseline_only'])+' loss, '+str(r['changed_only'])+' gain'))
  replacement='<p>'+esc('; '.join(observed))+'.</p><details><summary>'+('손실·회복 전체 비교' if ko else 'All paired losses and gains')+'</summary>'+table+'</details>'
  s=s.replace(original_dump,replacement)
  s=s.replace('</script>',script+'</script>')
  f.write_text(s)
 atomic(a.output/'display_manifest.json',{'source_records_sha256':sha(case/'results/p1_r1/episodes.jsonl'),'comparison_sha256':sha(case/'results/p1_r1/common_event_prefixes.json'),'display_generator_sha256':sha(Path(__file__)),'frozen_base_renderer_sha256':sha(case/'src/interface_sensitivity/render.py'),'comparison_panels':72,'matrix_task_links':12,'scope':'display layer only; raw/scorer/GPU source unchanged'},True)
 print(json.dumps({'episodes':96,'comparison_panels':72,'matrix_task_links':12}))
if __name__=='__main__':main()
