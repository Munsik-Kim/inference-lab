/* Model-free human labels. No AI prefill, network requests or automatic upload. */
(function (global) {
  'use strict';
  const VALUES = new Set(['satisfied', 'not_satisfied', 'uncertain']);
  const fail = message => { throw new Error(message); };
  const object = x => x !== null && typeof x === 'object' && !Array.isArray(x);
  const same = (a, b) => JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());
  const groupAPI = typeof module !== 'undefined' && module.exports ? require('./case012-groups.js') : global.Case012Groups;

  function validateFile(file, data, copy) {
    if (!object(file) || !['case012-annotations-v1','case012-annotations-v2'].includes(file.schema) || file.scope !== 'MAIN_ONLY' ||
        file.rubric_sha256 !== data.rubric_sha256 || file.image_set_sha256 !== data.image_set_sha256)
      fail('Annotation schema / image set / rubric mismatch');
    if (!object(file.evaluator) || file.evaluator.type !== 'human' || typeof file.evaluator.id !== 'string' || !file.evaluator.id.trim())
      fail('Human evaluator ID required; AI labels are separate');
    if (!Array.isArray(file.rows) || !Array.isArray(file.draft_rows) || !Array.isArray(file.incomplete_image_ids))
      fail('Annotation rows and draft rows required');
    const known = new Map(data.items.map(i => [i.image_id, i])), seen = new Set(), answers = {};
    for (const [rows, complete] of [[file.rows, true], [file.draft_rows, false]]) {
      for (const row of rows) {
        if (!object(row) || !known.has(row.image_id) || seen.has(row.image_id)) fail('Unknown / duplicate image ID');
        const item = known.get(row.image_id), keys = item.constraints.map(c => c.id);
        if (row.image_sha256 !== item.image_sha256) fail('Image hash mismatch');
        if (!object(row.values) || Object.keys(row.values).some(k => !keys.includes(k)) ||
            Object.values(row.values).some(v => !VALUES.has(v))) fail('Constraint / value mismatch');
        const n = Object.keys(row.values).length;
        if ((complete && !same(Object.keys(row.values), keys)) || (!complete && (n === 0 || n >= keys.length)))
          fail('Complete / draft constraint coverage mismatch');
        answers[row.image_id] = {...row.values}; seen.add(row.image_id);
      }
    }
    const done = new Set(file.rows.map(r => r.image_id)), missing = data.items.filter(i => !done.has(i.image_id)).map(i => i.image_id);
    if (!same(file.incomplete_image_ids, missing) || new Set(file.incomplete_image_ids).size !== missing.length ||
        file.status !== (missing.length ? 'PARTIAL' : 'COMPLETE')) fail('Annotation completeness mismatch');
    if (!Number.isInteger(file.cursor) || file.cursor < 0 || file.cursor >= data.items.length ||
        file.current_image_id !== data.items[file.cursor].image_id) fail('Cursor / image identity mismatch');
    return {answers, cursor: file.cursor, evaluator: file.evaluator.id, ...groupAPI.restore(file,data,copy,answers)};
  }

  function exportFile(answers, cursor, evaluator, data, options) {
    if (typeof evaluator !== 'string' || !evaluator.trim()) fail('Enter a name or anonymous ID');
    const rows = [], draft_rows = [], incomplete_image_ids = [];
    for (const i of data.items) {
      const values = answers[i.image_id] || {}, row = {image_id: i.image_id, image_sha256: i.image_sha256, values: {...values}};
      if (i.constraints.every(c => VALUES.has(values[c.id]))) rows.push(row);
      else { incomplete_image_ids.push(i.image_id); if (Object.keys(values).length) draft_rows.push(row); }
    }
    let file = {schema: 'case012-annotations-v1', scope: 'MAIN_ONLY', rubric_sha256: data.rubric_sha256,
      image_set_sha256: data.image_set_sha256, evaluator: {type: 'human', id: evaluator.trim()},
      status: rows.length === data.items.length ? 'COMPLETE' : 'PARTIAL', rows, draft_rows,
      incomplete_image_ids, cursor, current_image_id: data.items[cursor]?.image_id};
    if(options)file=groupAPI.attach(file,options,data);
    validateFile(file, data, options?.copy); return file;
  }

  function groupedOrder(data, copy) {
    if (!copy) return data.items.map((_,i)=>i);
    const ids=new Set(copy.groups.map(g=>g.prompt_id));
    if (ids.size!==copy.groups.length) fail('Duplicate review group');
    const order=copy.groups.flatMap(g=>data.items.flatMap((i,n)=>copy.items[i.image_id]===g.prompt_id?[n]:[]));
    if (order.length!==data.items.length || new Set(order).size!==data.items.length) fail('Review group coverage mismatch');
    return order;
  }
  function nextUnfinished(answers,pos,data,order) {
    const current=order.indexOf(pos);
    for(let offset=1;offset<=order.length;offset++) {
      const n=order[(current+offset)%order.length], item=data.items[n];
      if(item.constraints.some(c=>!VALUES.has(answers[item.image_id]?.[c.id]))) return n;
    }
    return -1;
  }
  const api = {validateFile, exportFile, groupedOrder, nextUnfinished};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.Case012Human = api;
  if (!global.document || !global.CASE012_ANNOTATION) return;

  const data = global.CASE012_ANNOTATION, copy=global.CASE012_REVIEW_COPY, order=groupedOrder(data,copy), ko = document.documentElement.lang === 'ko';
  const el = id => document.getElementById(id), t = (en, korean) => ko ? korean : en;
  const key = 'case012-human-main-v1:' + data.image_set_sha256;
  let pos = 0, answers = {}, origins={}, separated=[], mode='similarity', storageProblem = false, restoredFromStorage=false;
  const options=()=>({copy,origins,separated,mode});
  function message(text) { el('message').textContent = text; }
  function readPosition() {
    const id = decodeURIComponent(location.hash.slice(1)), n = data.items.findIndex(i => i.image_id === id);
    if (n >= 0) pos = n;
  }
  try {
    const saved = localStorage.getItem(key);
    if (saved) {
      const restored = validateFile(JSON.parse(saved), data, copy);
      answers = restored.answers; origins=restored.origins;separated=restored.separated;mode=restored.mode;
      restoredFromStorage=true;
      pos = restored.cursor; el('evaluator').value = restored.evaluator;
    }
  } catch (error) {
    storageProblem = true;
    message(t('Could not restore browser data. Import your JSON backup; the stored data was not overwritten.', '브라우저 기록을 복원하지 못했습니다. 보관한 JSON을 가져오세요. 기존 저장 기록은 덮어쓰지 않았습니다.') + ' ' + error.message);
  }
  readPosition();
  if(!restoredFromStorage&&!data.items.some(i=>'#'+i.image_id===location.hash)){
    const first=groupAPI.units(copy,data,separated)[0];pos=data.items.findIndex(i=>i.image_id===first.representative);
  }
  function save() {
    if (storageProblem) return;
    try { localStorage.setItem(key, JSON.stringify(exportFile(answers, pos, el('evaluator').value.trim() || 'anonymous-local-draft', data,options()))); }
    catch (error) { message(t('Browser save failed. Download your JSON to keep it.', '브라우저 저장에 실패했습니다. JSON을 다운로드해 보관하세요.')); }
  }
  const done=id=>data.items.find(i=>i.image_id===id).constraints.every(c=>VALUES.has(answers[id]?.[c.id]));
  function reviewUnits(){
    return mode==='similarity'?groupAPI.units(copy,data,separated):order.map(n=>({group_id:'image-'+data.items[n].image_id,prompt_id:copy.items[data.items[n].image_id],members:[data.items[n].image_id],representative:data.items[n].image_id}));
  }
  function moveUnit(g){
    const reviewed=g.members.find(id=>done(id)&&Object.values(origins[id]||{}).every(o=>o.kind==='direct'));
    move(data.items.findIndex(i=>i.image_id===(reviewed||g.representative)));
  }
  function advanceUnfinished(){
    const all=reviewUnits(),current=all.findIndex(g=>g.members.includes(data.items[pos].image_id));
    for(let offset=1;offset<=all.length;offset++){const g=all[(current+offset)%all.length];if(g.members.some(id=>!done(id))){moveUnit(g);return;}}
    message(t('All groups have answers. Export your JSON.', '전체 묶음의 답안이 완료됐습니다. JSON을 내려받아 보관하세요.'));
  }
  function progress() {
    const n = data.items.filter(i => i.constraints.every(c => VALUES.has(answers[i.image_id]?.[c.id]))).length;
    const all=reviewUnits(),completed=all.filter(g=>g.members.every(done)).length;
    const reused=data.items.filter(i=>Object.values(origins[i.image_id]||{}).some(o=>o.kind==='similarity_reuse')).length;
    el('progress').textContent = mode==='similarity'?t(`Groups completed ${completed}/${all.length} · ${n} images covered (${reused} with reused labels)`, `사진 묶음 ${completed}/${all.length}개 완료 · 답안 적용 ${n}장 (묶음 적용 ${reused}장)`):t(`Complete images ${n}/${data.items.length}.`, `답안 완료 ${n}/${data.items.length}장 · 묶음 적용 ${reused}장`);
    if(copy&&el('request')) for(const option of el('request').options) {
      const index=copy.groups.findIndex(g=>g.prompt_id===option.value), g=copy.groups[index];
      const members=all.filter(u=>u.prompt_id===g.prompt_id);
      const complete=members.filter(u=>u.members.every(done)).length;
      option.textContent=`${index+1}. ${ko?g.ko.title:g.english.split('. ')[0]} · ${complete}/${members.length}`;
    }
  }
  function render() {
    const item = data.items[pos];
    const units=reviewUnits(),unit=units.find(g=>g.members.includes(item.image_id));el('review-mode').value=mode;
    const group=copy?.groups.find(g=>g.prompt_id===copy.items[item.image_id]);
    if (el('request') && copy) {
      el('request').replaceChildren();
      for (const [index,g] of copy.groups.entries()) {
        const members=units.filter(u=>u.prompt_id===g.prompt_id);
        const complete=members.filter(u=>u.members.every(done)).length;
        const option=document.createElement('option'); option.value=g.prompt_id;
        option.textContent=`${index+1}. ${ko?g.ko.title:g.english.split('. ')[0]} · ${complete}/${members.length}`;
        option.selected=g===group; el('request').append(option);
      }
    }
    el('image').src = item.path; el('original-image').href = item.path;
    el('prompt-text').textContent = ko ? (group?.ko.request || item.korean_display) : item.english;
    el('model-input').textContent = item.english;
    const members=units.filter(u=>u.prompt_id===group?.prompt_id);
    el('position').textContent=mode==='similarity'?t(`Group ${members.indexOf(unit)+1}/${members.length} in this request · ${unit.members.length} images`, `이 요청의 ${members.indexOf(unit)+1}/${members.length}번째 묶음 · 사진 ${unit.members.length}장`):t(`Image ${members.indexOf(unit)+1}/${members.length} in this request`, `이 요청의 ${members.indexOf(unit)+1}/${members.length}번째 이미지`);
    el('similar-members').replaceChildren();el('similar-group').hidden=mode!=='similarity'||unit.members.length<2;
    for(const id of unit.members){const member=data.items.find(i=>i.image_id===id),button=document.createElement('button'),thumb=document.createElement('img'),caption=document.createElement('span');button.type='button';button.className='similar-member';button.setAttribute('aria-pressed',String(id===item.image_id));thumb.src=member.path;thumb.alt=t('Group member image','묶음에 포함된 사진');thumb.width=100;thumb.height=100;caption.textContent=id===item.image_id?t('Reading this image','현재 체크할 사진'):t('Read this image','이 사진으로 체크');button.append(thumb,caption);button.onclick=()=>move(data.items.findIndex(i=>i.image_id===id));el('similar-members').append(button);}
    el('separate-image').hidden=mode!=='similarity'||unit.members.length<2;
    el('apply-group').hidden=mode!=='similarity'||unit.members.length<2;
    el('apply-group').disabled=!done(item.image_id);
    el('label-origin').textContent=Object.values(origins[item.image_id]||{}).some(o=>o.kind==='similarity_reuse')?t('This image uses answers copied from a group member. You can check it separately.', '이 사진에는 묶음에서 적용한 답이 있습니다. 따로 체크할 수 있습니다.'):'';
    el('constraints').replaceChildren();
    for (const constraint of item.constraints) {
      const f = document.createElement('fieldset'), g = document.createElement('legend');
      g.textContent = ko ? (group?.ko.constraints[constraint.id] || constraint.ko) : constraint.en; f.append(g);
      for (const [value, en, korean] of [['satisfied', 'Satisfied', '충족'], ['not_satisfied', 'Not satisfied', '미충족'], ['uncertain', 'Uncertain', '판단 불확실']]) {
        const label = document.createElement('label'), radio = document.createElement('input');
        radio.type = 'radio'; radio.name = constraint.id; radio.value = value;
        radio.checked = answers[item.image_id]?.[constraint.id] === value;
        radio.addEventListener('change', () => {
          const changed=groupAPI.edit(answers,origins,item.image_id,constraint.id,value);answers=changed.answers;origins=changed.origins;save();progress();el('apply-group').disabled=!done(item.image_id);
        });
        const text=document.createElement('span'); text.textContent=t(en,korean); label.append(radio,text); f.append(label);
      }
      el('constraints').append(f);
    }
    el('prev').disabled = units.indexOf(unit) === 0; el('next').disabled = units.indexOf(unit) === units.length-1;
    try { history.replaceState(null, '', '#'+item.image_id); } catch {}
    el('language').href = '../'+(ko ? 'en' : 'ko')+'/case012-annotate.html#'+item.image_id;
    progress();
  }
  function move(n) { pos = n; save(); render(); }
  el('prev').onclick = () => {const u=reviewUnits(),n=u.findIndex(g=>g.members.includes(data.items[pos].image_id));if(n>0)moveUnit(u[n-1]);};
  el('next').onclick = () => {const u=reviewUnits(),n=u.findIndex(g=>g.members.includes(data.items[pos].image_id));if(n<u.length-1)moveUnit(u[n+1]);};
  el('review-mode').onchange=()=>{mode=el('review-mode').value;save();render();};
  el('apply-group').onclick=()=>{const r=groupAPI.apply(answers,origins,data.items[pos].image_id,data,copy,separated);answers=r.answers;origins=r.origins;save();render();advanceUnfinished();message(t(`Group answers applied; ${r.preserved} prior direct choices kept.`, `같은 답을 묶음에 적용했습니다. 기존에 직접 체크한 ${r.preserved}개 답은 유지했습니다.`));};
  el('separate-image').onclick=()=>{const r=groupAPI.separate(answers,origins,data.items[pos].image_id,separated);answers=r.answers;origins=r.origins;separated=r.separated;save();render();message(t('Separated. Direct answers kept; borrowed answers cleared.', '이 사진을 묶음에서 분리했습니다. 직접 체크한 답은 유지하고 묶음에서 받은 답은 비웠습니다.'));};
  if(el('request')&&copy) el('request').onchange=()=>{
    const members=reviewUnits().filter(u=>u.prompt_id===el('request').value);
    moveUnit(members.find(u=>u.members.some(id=>!done(id)))||members[0]);
  };
  el('unrated').onclick = () => {
    advanceUnfinished();
  };
  el('evaluator').addEventListener('change', save);
  el('image').onerror = () => message(t('Image could not load; wait or reload before judging.', '이미지를 불러오지 못했습니다. 새로고침 후 이미지를 확인하고 판단하세요.'));
  el('export').onclick = () => {
    try {
      const file = exportFile(answers, pos, el('evaluator').value, data,options());
      const url = URL.createObjectURL(new Blob([JSON.stringify(file, null, 2)+'\n'], {type:'application/json'}));
      const link = document.createElement('a'); link.href = url; link.download = 'case012_human_annotations.json';
      document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      message(t('Downloaded. Keep this JSON file to resume or recalculate later.', '다운로드했습니다. 나중에 이어서 체크하거나 검산할 수 있도록 JSON 파일을 보관하세요.'));
    } catch (error) { message(t('Export failed: ', '내보내기 실패: ')+error.message); }
  };
  el('import').addEventListener('change', async () => {
    try {
      const f = el('import').files[0]; if (!f) return;
      if (f.size > 2*1024*1024) fail('JSON file exceeds 2 MiB');
      const restored = validateFile(JSON.parse(await f.text()), data,copy);
      answers = restored.answers;origins=restored.origins;separated=restored.separated;mode=restored.mode;
      pos = restored.cursor; el('evaluator').value = restored.evaluator;
      storageProblem = false; save(); render();
      message(t('Restored your complete and unfinished labels.', '완료·부분 체크 기록을 복원했습니다.'));
    } catch (error) { message(t('Import rejected; existing choices kept: ', '가져오기 거절 · 기존 체크 유지: ')+error.message); }
    finally { el('import').value = ''; }
  });
  global.addEventListener('hashchange', () => { readPosition(); render(); });
  render();
})(typeof window === 'undefined' ? globalThis : window);
