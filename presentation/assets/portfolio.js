/* Progressive navigation only. No network, analytics, models or stored evidence changes. */
(() => {
  'use strict';
  const ko = document.documentElement.lang === 'ko';
  const mobile = matchMedia('(max-width: 800px)');
  const menu = document.getElementById('menu-toggle');
  const navigation = document.getElementById('site-navigation');
  function closeMenu() {
    if (!menu) return;
    menu.setAttribute('aria-expanded', 'false');
    navigation.classList.remove('menu-open');
  }
  if (menu) {
    menu.hidden = false;
    document.documentElement.classList.add('reader-enhanced');
    menu.textContent = ko ? '메뉴' : 'Menu';
    menu.addEventListener('click', () => {
      const open = menu.getAttribute('aria-expanded') !== 'true';
      menu.setAttribute('aria-expanded', String(open));
      navigation.classList.toggle('menu-open', open);
    });
    navigation.addEventListener('click', event => { if (event.target.closest('a')) closeMenu(); });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') { closeMenu(); menu.focus(); }
    });
    mobile.addEventListener('change', closeMenu);
  }
  for (const toc of document.querySelectorAll('.reader-toc')) {
    toc.open = !mobile.matches;
    mobile.addEventListener('change', () => { toc.open = !mobile.matches; });
  }
  for (const pre of document.querySelectorAll('pre')) {
    const code = pre.querySelector('code');
    if (!code) continue;
    const wrapper = document.createElement('div'); wrapper.className = 'copy-block';
    pre.before(wrapper); wrapper.append(pre);
    const button = document.createElement('button'); button.type = 'button'; button.className = 'secondary copy-code';
    button.textContent = ko ? '코드 복사' : 'Copy code';
    const feedback = document.createElement('span'); feedback.className = 'copy-feedback'; feedback.setAttribute('role', 'status');
    wrapper.prepend(button); wrapper.append(feedback);
    button.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(code.textContent); feedback.textContent = ko ? '복사했습니다.' : 'Copied.'; }
      catch { feedback.textContent = ko ? '복사 권한이 없습니다. 코드 내용을 선택해 복사해 주세요.' : 'Copy unavailable. Select the code and copy it manually.'; }
    });
  }
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
      document.querySelectorAll('.reader-toc a').forEach(link => {
        if (link.hash === location.hash) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
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
