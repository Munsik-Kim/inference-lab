/* Progressive navigation only. No network, analytics, models or stored evidence changes. */
(() => {
  'use strict';
  const language = document.getElementById('language');
  const originalLanguageHref = language && language.getAttribute('href');
  function revealHash() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = id && document.getElementById(id);
    if (target) {
      for (let p = target.parentElement; p; p = p.parentElement) {
        if (p.tagName === 'DETAILS') p.open = true;
      }
      requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
    }
    if (language) language.setAttribute('href', originalLanguageHref + location.hash);
  }
  addEventListener('hashchange', revealHash);
  if (location.hash) revealHash();
  const catalog = document.getElementById('case-catalog');
  if (!catalog) return;
  const controls = document.getElementById('case-controls');
  const query = document.getElementById('case-search');
  const topic = document.getElementById('case-topic');
  const status = document.getElementById('case-count');
  const empty = document.getElementById('case-empty');
  const rows = [...catalog.querySelectorAll('[data-case-id]')];
  const normalize = s => s.normalize('NFKC').toLocaleLowerCase().trim();
  function filter() {
    const terms = normalize(query.value).split(/\s+/).filter(Boolean);
    let count = 0;
    rows.forEach(row => {
      const haystack = normalize('case' + row.dataset.caseId + ' ' + row.textContent);
      row.hidden = !((topic.value === 'all' || row.dataset.topic === topic.value) && terms.every(t => haystack.includes(t)));
      if (!row.hidden) count += 1;
    });
    status.textContent = status.dataset.lang === 'ko' ? `${rows.length}개 중 ${count}개 사례` : `${count} of ${rows.length} cases`;
    empty.hidden = count !== 0;
  }
  query.addEventListener('input', filter);
  topic.addEventListener('change', filter);
  document.getElementById('case-reset').addEventListener('click', () => { query.value = ''; topic.value = 'all'; filter(); query.focus(); });
  controls.hidden = false;
  filter();
})();
