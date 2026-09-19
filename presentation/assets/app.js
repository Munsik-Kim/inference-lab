/* Read-only UI. All record/prompt strings are rendered with textContent. */
'use strict';
(() => {
  const D=window.EVIDENCE, T=window.TEXT, $=id=>document.getElementById(id);
  const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
  const fmt=x=>typeof x==='number'&&Number.isFinite(x)?x.toFixed(4):T.notRecorded;
  const letter=i=>'ABCD'[i];
  if(!D||!T||!Array.isArray(D.records)||!D.records.length||!D.prompts){$('count').textContent=T?.noData||'Missing display data';return;}
  const is7=D.case==='007';
  const S=window.ShowcaseState, defaults=S.defaults(is7), choices=S.choices(is7);
  const labels={ALL:T.all,HELD_OUT:is7?(document.documentElement.lang==='ko'?'미관측 입력 192개':'Held-out · 192'):'HELD_OUT',standard:document.documentElement.lang==='ko'?'표준 192개':'Standard · 192',boundary_pool:document.documentElement.lang==='ko'?'선정 스트레스 46개':'Selected stress · 46',H_NATIVE:document.documentElement.lang==='ko'?'원래 계산 · native':'Original · native',H_FP32:document.documentElement.lang==='ko'?'사후 진단 · FP32':'Post-hoc diagnostic · FP32','4':'4/16 · 25%','8':'8/16 · 50%'};
  let state={...defaults},visible=[],current=null;
  for(const key of Object.keys(choices)){if(!$(key))continue;for(const value of choices[key]){const o=node('option',labels[value]||T[value]||value);o.value=value;$(key).append(o);}$(key).addEventListener('change',()=>{state[key]=$(key).value;state.id='';save();render();});}
  $('query').addEventListener('input',()=>{state.query=$('query').value;state.id='';save();render();});
  $('items').addEventListener('change',()=>{state.id=$('items').value;save();detail();});
  $('reset').addEventListener('click',()=>{state={...defaults,set:choices.set.includes(state.set)?state.set:defaults.set};save();render();});
  function save(){history.replaceState(null,'',S.encode(state));$('language').hash=location.hash;}
  function restore(){state=S.decode(location.hash,is7);render();}
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
    const values=[[T.flip,visible.filter(r=>r.flip).length+'/'+n],[T.regression,bc?visible.filter(r=>r.cell==='regression').length+'/'+bc:'UNDEFINED (0 B-correct)'],[T.gain,n-bc?visible.filter(r=>r.cell==='gain').length+'/'+(n-bc):'UNDEFINED (0 B-wrong)'],[T.delta,n?fmt(visible.reduce((s,r)=>s+r.delta_nll,0)/n):'UNDEFINED']];
    if(!invalid)for(const [label,value] of values){const e=node('div',undefined,'stat');e.append(node('strong',value),node('span',label));$('stats').append(e);}
    $('guided').replaceChildren();
    if(!invalid)for(const task of ['RETRIEVAL','COMPARISON','CODE']){const r=D.records.find(r=>r.set===state.set&&r.arm===state.arm&&r.readout===state.readout&&r.task===task&&(!is7||r.budget===state.budget));if(!r)continue;const b=node('button',task,'secondary');b.addEventListener('click',()=>{state.task=task;state.outcome='ALL';state.query='';state.id=r.id;save();render();});$('guided').append(b);}
    detail();cost();
  }
  function table(container,columns,rows){const t=node('table');const head=node('thead'),hr=node('tr');for(const c of columns)hr.append(node('th',c));head.append(hr);t.append(head);const body=node('tbody');for(const row of rows){const tr=node('tr');row.forEach((cell,i)=>{const td=node(i===0?'th':'td',cell);if(i===0)td.scope='row';tr.append(td);});body.append(tr);}t.append(body);container.replaceChildren(t);}
  function detail(){
    current=visible.find(r=>r.id===state.id)||null;
    $('detail').classList.toggle('hidden',!current);
    if(!current)return;
    const siblings=D.records.filter(r=>r.id===current.id&&r.readout===current.readout&&r.set===current.set&&(!is7||r.budget===current.budget));
    siblings.sort((a,b)=>a.arm.localeCompare(b.arm));
    const scores=[current.B,...siblings.map(r=>r.candidate)],cols=[T.setting,'B',...siblings.map(r=>r.arm)];
    $('item-title').textContent=current.id+(is7?' · '+labels[current.budget]:' · '+labels[current.readout])+' · '+T.gold+': '+letter(current.B.gold)+' · '+current.evidence_kind;
    const rows=[[T.outcome,'—',...siblings.map(r=>(T[r.cell]||r.cell)+(r.wrong_to_wrong?' · '+T.wrong_to_wrong:''))],['ΔNLL (candidate − B)','0',...siblings.map(r=>fmt(r.delta_nll))],[T.prediction,...scores.map(s=>letter(s.prediction))],[T.top,...scores.map(s=>s.top.map(letter).join(', '))],[T.nll,...scores.map(s=>fmt(s.nll))],[T.mass,...scores.map(s=>fmt(s.label_mass))],[T.margin,...scores.map(s=>fmt(s.margin))],[T.local,'—',...siblings.map(r=>fmt(r.local_error*100))],[T.kl,'0',...siblings.map(r=>fmt(r.full_kl))]];
    if(is7)rows.push([T.groups,'—',...siblings.map(r=>r.selected_groups.join(', '))]);
    else rows.push([document.documentElement.lang==='ko'?'원래 B 동률 집합':'Original native B top set',current.original_B_top.map(letter).join(', '),...siblings.map(r=>r.original_B_top.map(letter).join(', '))]);
    table($('comparison'),cols,rows);
    $('probabilities').replaceChildren();scores.forEach((s,i)=>{const e=node('div',undefined,'prob-block');e.append(node('h3',cols[i+1]));s.q.forEach((v,j)=>{const row=node('div',undefined,'prob-row'),track=node('div',undefined,'track'),bar=node('div',undefined,'bar');bar.style.width=(v*100)+'%';track.append(bar);row.append(node('span',letter(j)),track,node('span',(v*100).toFixed(2)+'%'));e.append(row);});$('probabilities').append(e);});
    const p=D.prompts[current.id];if(!p){$('count').textContent=T.noData;$('detail').classList.add('hidden');return;}
    $('prompt').textContent=p.text;$('prompt-source').href=p.source;$('record-source').href=current.source;
    $('record').textContent=JSON.stringify({focus:current,paired_comparisons:siblings,prompt:p,units:{nll:'nats; conditional on four labels; lower is better',delta_nll:'candidate minus B',local_error:'relative fraction, not percent',full_kl:'KL(B || candidate), full-vocabulary retained scalar'},independent_unit:'base scenario',timing_scope:'separate fixed-input aggregate, no item timing'},null,2);
  }
  function cost(){if(!is7){$('cost').replaceChildren();return;}const methods=['INDEPENDENT_'+state.budget,'PAIRWISE_'+state.budget];const t=D.summary.timing;const rows=[];for(const scope of ['MLP','MODEL_PREFILL'])for(const m of methods){const v=t[scope]?.[m];if(v)rows.push([scope+' · '+m,v.speedup_median.toFixed(3)+'×','['+v.ci95.map(x=>x.toFixed(3)).join(', ')+']']);}table($('cost'),[T.setting,'B / candidate','95% CI'],rows);}
  function download(data,name){const blob=new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=node('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  $('download').addEventListener('click',()=>{if(current)download(JSON.parse($('record').textContent),current.id+'-'+current.arm+'-'+current.readout+'.json');});
  $('download-all').addEventListener('click',()=>download(D,'case'+D.case+'-display.json'));
  window.addEventListener('hashchange',restore);restore();
})();
