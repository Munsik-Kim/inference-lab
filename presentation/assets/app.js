/* Read-only UI. All record/prompt strings are rendered with textContent. */
'use strict';
(() => {
  const D=window.EVIDENCE, T=window.TEXT, $=id=>document.getElementById(id);
  const ko=document.documentElement.lang==='ko', human=(en,kr)=>ko?kr:en;
  const evidenceName=key=>({ORIGINAL_GPU_MEASUREMENT:human('Saved GPU measurement','저장된 GPU 실측'),POST_HOC_SAME_INPUT_READOUT_DIAGNOSTIC:human('Later output calculation on the same input','같은 입력으로 나중에 다시 계산한 출력 진단')})[key]||key;
  const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
  const fmt=x=>typeof x==='number'&&Number.isFinite(x)?x.toFixed(4):T.notRecorded;
  const letter=i=>'ABCD'[i];
  if(!D||!T||!Array.isArray(D.records)||!D.records.length||!D.prompts){$('count').textContent=T?.noData||'Missing display data';return;}
  const is7=D.case==='007';
  const S=window.ShowcaseState, defaults=S.defaults(is7), choices=S.choices(is7);
  const methodName=key=>T.methodLabels?.[key]||key;
  const labels={...(T.methodLabels||{}),RETRIEVAL:human('Find information in provided text','주어진 글에서 정보 찾기'),COMPARISON:human('Compare provided information','주어진 정보 비교'),CODE:human('Solve a short code problem','짧은 코드 문제 풀기'),ALL:T.all,HELD_OUT:human('192 questions not used to choose settings','설정을 고를 때 쓰지 않은 문제 192개'),standard:document.documentElement.lang==='ko'?'기본 평가 문제 192개':'Standard evaluation · 192 questions',boundary_pool:document.documentElement.lang==='ko'?'원래 선택지 점수 차가 작아 따로 고른 문제 46개':'Selected for close original answer scores · 46',H_NATIVE:document.documentElement.lang==='ko'?'원래 출력 계산 (native)':'Original output calculation (native)',H_FP32:document.documentElement.lang==='ko'?'같은 입력의 출력만 32비트로 재계산 (FP32)':'Recalculate only the output in 32-bit precision (FP32)','4':'4/16 · 25%','8':'8/16 · 50%'};
  let state={...defaults},visible=[],current=null;
  for(const key of Object.keys(choices)){if(!$(key))continue;for(const value of choices[key]){const o=node('option',labels[value]||T[value]||value);o.value=value;$(key).append(o);}$(key).addEventListener('change',()=>{state[key]=$(key).value;state.id='';save();render();});}
  $('query').addEventListener('input',()=>{state.query=$('query').value;state.id='';save();render();});
  $('items').addEventListener('change',()=>{state.id=$('items').value;save();detail();});
  $('reset').addEventListener('click',()=>{state={...defaults,set:choices.set.includes(state.set)?state.set:defaults.set};save();render();});
  function save(){history.replaceState(null,'',location.pathname+S.encode(state));syncLanguage();}
  function restore(){
    const anchor=location.hash.slice(1);
    if(S.isStudyAnchor(anchor)){
      // A contents link scrolls the explanation; it must not reset item filters.
      const saved=new URLSearchParams(location.search).get('view');
      if(saved)state=S.decode(saved,is7);
      render();syncLanguage();return;
    }
    state=S.decode(location.hash,is7);render();syncLanguage();
  }
  function syncLanguage(){
    const url=new URL($('language').href);
    url.hash=location.hash;
    url.search=S.isStudyAnchor(location.hash.slice(1))?'?'+new URLSearchParams({view:S.encode(state)}).toString():'';
    $('language').href=url.href;
  }
  for(const a of document.querySelectorAll('a[href^="#"]')){
    if(S.isStudyAnchor(a.hash.slice(1)))a.addEventListener('click',()=>{
      const url=new URL(location.href);url.search=new URLSearchParams({view:S.encode(state)}).toString();
      history.replaceState(null,'',url);syncLanguage();
    });
  }
  function selected(){return S.select(D.records,state,is7);}
  function render(){
    let invalid=!S.valid(state,is7);
    for(const key of Object.keys(choices)){if(!$(key))continue;$(key).value=state[key];if(!choices[key].includes(state[key]))invalid=true;}
    $('query').value=state.query;
    // Readout has no user control in Case007; reject crafted cross-view links.
    if(is7&&state.readout!=='H_NATIVE')invalid=true;
    $('language').hash=location.hash;
    $('view-note').textContent=is7?T.c7note:state.readout==='H_FP32'?T.readoutNote:T.nativeNote;
    $('view-note').className='note'+(!is7&&state.readout==='H_FP32'?' posthoc':'');
    visible=invalid?[]:selected();
    $('items').replaceChildren();
    for(const r of visible){const o=node('option',r.id+' · '+(T[r.cell]||r.cell));o.value=r.id;$('items').append(o);}
    if(state.id&&!visible.some(r=>r.id===state.id)){current=null;$('items').value='';}
    else {if(!state.id&&visible.length)state.id=visible[0].id;$('items').value=state.id;current=visible.find(r=>r.id===state.id)||null;}
    $('count').textContent=invalid?T.invalid:visible.length===0||!current?T.empty:visible.length+' '+T.count;
    $('count').className='status'+(invalid?' error':'');
    $('stats').replaceChildren();
    const n=visible.length, bc=visible.filter(r=>r.B.correct).length;
    const values=[[T.flip,visible.filter(r=>r.flip).length+'/'+n],[T.regression,bc?visible.filter(r=>r.cell==='regression').length+'/'+bc:human('Not defined: no originally correct answers','계산 대상 없음: 원래 맞힌 답 0개')],[T.gain,n-bc?visible.filter(r=>r.cell==='gain').length+'/'+(n-bc):human('Not defined: no originally wrong answers','계산 대상 없음: 원래 틀린 답 0개')],[T.delta,n?fmt(visible.reduce((s,r)=>s+r.delta_nll,0)/n):human('Not defined: no matching questions','계산 대상 없음: 해당 문제 0개')]];
    if(!invalid)for(const [label,value] of values){const e=node('div',undefined,'stat');e.append(node('strong',value),node('span',label));$('stats').append(e);}
    $('guided').replaceChildren();
    if(!invalid)for(const task of ['RETRIEVAL','COMPARISON','CODE']){const r=D.records.find(r=>r.set===state.set&&r.arm===state.arm&&r.readout===state.readout&&r.task===task&&(!is7||r.budget===state.budget));if(!r)continue;const b=node('button',labels[task]||T[task]||task,'secondary');b.addEventListener('click',()=>{state.task=task;state.outcome='ALL';state.query='';state.id=r.id;save();render();});$('guided').append(b);}
    detail();cost();
  }
  function table(container,columns,rows){const t=node('table');const head=node('thead'),hr=node('tr');for(const c of columns)hr.append(node('th',c));head.append(hr);t.append(head);const body=node('tbody');for(const row of rows){const tr=node('tr');row.forEach((cell,i)=>{const td=node(i===0?'th':'td',cell);if(i===0)td.scope='row';tr.append(td);});body.append(tr);}t.append(body);container.replaceChildren(t);}
  function detail(){
    current=visible.find(r=>r.id===state.id)||null;
    $('detail').classList.toggle('hidden',!current);
    if(!current)return;
    const siblings=D.records.filter(r=>r.id===current.id&&r.readout===current.readout&&r.set===current.set&&(!is7||r.budget===current.budget));
    siblings.sort((a,b)=>a.arm.localeCompare(b.arm));
    const scores=[current.B,...siblings.map(r=>r.candidate)],cols=[T.setting,methodName('B'),...siblings.map(r=>methodName(r.arm))];
    $('item-title').textContent=current.id+(is7?' · '+labels[current.budget]:' · '+labels[current.readout])+' · '+T.gold+': '+letter(current.B.gold)+' · '+evidenceName(current.evidence_kind);
    const rows=[[T.outcome,'—',...siblings.map(r=>(T[r.cell]||r.cell)+(r.wrong_to_wrong?' · '+T.wrong_to_wrong:''))],[human('Answer-probability score change (ΔNLL; changed − original)','정답 확률 점수 변화 (ΔNLL; 변경−원본)'),'0',...siblings.map(r=>fmt(r.delta_nll))],[T.prediction,...scores.map(s=>letter(s.prediction))],[T.top,...scores.map(s=>s.top.map(letter).join(', '))],[T.nll,...scores.map(s=>fmt(s.nll))],[T.mass,...scores.map(s=>fmt(s.label_mass))],[T.margin,...scores.map(s=>fmt(s.margin))],[T.local,'—',...siblings.map(r=>fmt(r.local_error*100))],[T.kl,'0',...siblings.map(r=>fmt(r.full_kl))]];
    if(is7)rows.push([T.groups,'—',...siblings.map(r=>r.selected_groups.join(', '))]);
    else rows.push([document.documentElement.lang==='ko'?'원래 계산에서 최고점이 같았던 답':'Equal top-scoring answers in the original calculation',current.original_B_top.map(letter).join(', '),...siblings.map(r=>r.original_B_top.map(letter).join(', '))]);
    table($('comparison'),cols,rows);
    $('probabilities').replaceChildren();scores.forEach((s,i)=>{const e=node('div',undefined,'prob-block');e.append(node('h3',cols[i+1]));s.q.forEach((v,j)=>{const row=node('div',undefined,'prob-row'),track=node('div',undefined,'track'),bar=node('div',undefined,'bar');bar.style.width=(v*100)+'%';track.append(bar);row.append(node('span',letter(j)),track,node('span',(v*100).toFixed(2)+'%'));e.append(row);});$('probabilities').append(e);});
    const p=D.prompts[current.id];if(!p){$('count').textContent=T.noData;$('detail').classList.add('hidden');return;}
    $('prompt').textContent=p.text;$('prompt-source').href=p.source;$('record-source').href=current.source;
    $('record').textContent=JSON.stringify({focus:current,paired_comparisons:siblings,prompt:p,units:{nll:'nats; conditional on four labels; lower is better',delta_nll:'candidate minus B',local_error:'relative fraction, not percent',full_kl:'KL(B || candidate), full-vocabulary retained scalar'},independent_unit:'base scenario',timing_scope:'separate fixed-input aggregate, no item timing'},null,2);
  }
  function cost(){if(!is7){$('cost').replaceChildren();return;}const methods=['INDEPENDENT_'+state.budget,'PAIRWISE_'+state.budget];const t=D.summary.timing;const rows=[];for(const scope of ['MLP','MODEL_PREFILL'])for(const m of methods){const v=t[scope]?.[m];if(v)rows.push([({MLP:human('One internal transformation block (MLP)','내부 정보 변환 블록 하나 (MLP)'),MODEL_PREFILL:human('Whole model reading its input','모델 전체가 입력을 읽는 구간')}[scope]||scope)+' · '+methodName(m.slice(0,m.lastIndexOf('_')))+' · '+labels[state.budget],v.speedup_median.toFixed(3)+'×','['+v.ci95.map(x=>x.toFixed(3)).join(', ')+']']);}table($('cost'),[T.setting,human('Original time / changed time','원본 시간 ÷ 변경 시간'),human('Estimate range (95% confidence interval)','추정 범위 (95% 신뢰구간)')],rows);}
  function download(data,name){const blob=new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=node('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  $('download').addEventListener('click',()=>{if(current)download(JSON.parse($('record').textContent),current.id+'-'+current.arm+'-'+current.readout+'.json');});
  $('download-all').addEventListener('click',()=>download(D,'case'+D.case+'-display.json'));
  window.addEventListener('hashchange',restore);restore();
})();
