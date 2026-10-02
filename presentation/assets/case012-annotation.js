/* Model-free human labels. No AI prefill, network requests or automatic upload. */
(function (global) {
  'use strict';
  const VALUES = new Set(['satisfied', 'not_satisfied', 'uncertain']);
  const fail = message => { throw new Error(message); };
  const object = x => x !== null && typeof x === 'object' && !Array.isArray(x);
  const same = (a, b) => JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());

  function validateFile(file, data) {
    if (!object(file) || file.schema !== 'case012-annotations-v1' || file.scope !== 'MAIN_ONLY' ||
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
    return {answers, cursor: file.cursor, evaluator: file.evaluator.id};
  }

  function exportFile(answers, cursor, evaluator, data) {
    if (typeof evaluator !== 'string' || !evaluator.trim()) fail('Enter a name or anonymous ID');
    const rows = [], draft_rows = [], incomplete_image_ids = [];
    for (const i of data.items) {
      const values = answers[i.image_id] || {}, row = {image_id: i.image_id, image_sha256: i.image_sha256, values: {...values}};
      if (i.constraints.every(c => VALUES.has(values[c.id]))) rows.push(row);
      else { incomplete_image_ids.push(i.image_id); if (Object.keys(values).length) draft_rows.push(row); }
    }
    const file = {schema: 'case012-annotations-v1', scope: 'MAIN_ONLY', rubric_sha256: data.rubric_sha256,
      image_set_sha256: data.image_set_sha256, evaluator: {type: 'human', id: evaluator.trim()},
      status: rows.length === data.items.length ? 'COMPLETE' : 'PARTIAL', rows, draft_rows,
      incomplete_image_ids, cursor, current_image_id: data.items[cursor]?.image_id};
    validateFile(file, data); return file;
  }

  const api = {validateFile, exportFile};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.Case012Human = api;
  if (!global.document || !global.CASE012_ANNOTATION) return;

  const data = global.CASE012_ANNOTATION, ko = document.documentElement.lang === 'ko';
  const el = id => document.getElementById(id), t = (en, korean) => ko ? korean : en;
  const key = 'case012-human-main-v1:' + data.image_set_sha256;
  let pos = 0, answers = {}, storageProblem = false;
  function message(text) { el('message').textContent = text; }
  function readPosition() {
    const id = decodeURIComponent(location.hash.slice(1)), n = data.items.findIndex(i => i.image_id === id);
    if (n >= 0) pos = n;
  }
  try {
    const saved = localStorage.getItem(key);
    if (saved) {
      const restored = validateFile(JSON.parse(saved), data);
      answers = restored.answers; pos = restored.cursor; el('evaluator').value = restored.evaluator;
    }
  } catch (error) {
    storageProblem = true;
    message(t('Could not restore browser data. Import your JSON backup; the stored data was not overwritten.', '브라우저 기록을 복원하지 못했습니다. 보관한 JSON을 가져오세요. 기존 저장 기록은 덮어쓰지 않았습니다.') + ' ' + error.message);
  }
  readPosition();
  function save() {
    if (storageProblem) return;
    try { localStorage.setItem(key, JSON.stringify(exportFile(answers, pos, el('evaluator').value.trim() || 'anonymous-local-draft', data))); }
    catch (error) { message(t('Browser save failed. Download your JSON to keep it.', '브라우저 저장에 실패했습니다. JSON을 다운로드해 보관하세요.')); }
  }
  function progress() {
    const n = data.items.filter(i => i.constraints.every(c => VALUES.has(answers[i.image_id]?.[c.id]))).length;
    el('progress').textContent = t(`Complete images ${n}/${data.items.length}. Export JSON to keep a backup.`, `완료 이미지 ${n}/${data.items.length} · JSON을 내려받아 보관하세요.`);
  }
  function render() {
    const item = data.items[pos];
    el('image').src = item.path; el('original-image').href = item.path;
    el('prompt-text').textContent = ko ? item.korean_display : item.english;
    el('model-input').textContent = item.english;
    el('position').textContent = `${pos+1} / ${data.items.length}`;
    el('constraints').replaceChildren();
    for (const constraint of item.constraints) {
      const f = document.createElement('fieldset'), g = document.createElement('legend');
      g.textContent = constraint[ko ? 'ko' : 'en']; f.append(g);
      for (const [value, en, korean] of [['satisfied', 'Satisfied', '충족'], ['not_satisfied', 'Not satisfied', '미충족'], ['uncertain', 'Uncertain', '판단 불확실']]) {
        const label = document.createElement('label'), radio = document.createElement('input');
        radio.type = 'radio'; radio.name = constraint.id; radio.value = value;
        radio.checked = answers[item.image_id]?.[constraint.id] === value;
        radio.addEventListener('change', () => {
          (answers[item.image_id] ??= {})[constraint.id] = value; save(); progress();
        });
        label.append(radio, document.createTextNode(t(en, korean))); f.append(label);
      }
      el('constraints').append(f);
    }
    el('prev').disabled = pos === 0; el('next').disabled = pos === data.items.length-1;
    try { history.replaceState(null, '', '#'+item.image_id); } catch {}
    el('language').href = '../'+(ko ? 'en' : 'ko')+'/case012-annotate.html#'+item.image_id;
    progress();
  }
  function move(n) { pos = n; save(); render(); }
  el('prev').onclick = () => move(pos-1);
  el('next').onclick = () => move(pos+1);
  el('unrated').onclick = () => {
    const n = data.items.findIndex(i => i.constraints.some(c => !VALUES.has(answers[i.image_id]?.[c.id])));
    if (n >= 0) move(n);
  };
  el('evaluator').addEventListener('change', save);
  el('image').onerror = () => message(t('Image could not load; wait or reload before judging.', '이미지를 불러오지 못했습니다. 새로고침 후 이미지를 확인하고 판단하세요.'));
  el('export').onclick = () => {
    try {
      const file = exportFile(answers, pos, el('evaluator').value, data);
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
      const restored = validateFile(JSON.parse(await f.text()), data);
      answers = restored.answers; pos = restored.cursor; el('evaluator').value = restored.evaluator;
      storageProblem = false; save(); render();
      message(t('Restored your complete and unfinished labels.', '완료·부분 체크 기록을 복원했습니다.'));
    } catch (error) { message(t('Import rejected; existing choices kept: ', '가져오기 거절 · 기존 체크 유지: ')+error.message); }
    finally { el('import').value = ''; }
  });
  global.addEventListener('hashchange', () => { readPosition(); render(); });
  render();
})(typeof window === 'undefined' ? globalThis : window);
