'use strict';
(() => {
  const controls = document.getElementById('case017-filters');
  const select = document.getElementById('case017-episode');
  if (!controls || !select) return;
  const episodes = [...document.querySelectorAll('.case017-episode')];
  const status = document.getElementById('case017-count');
  const language = document.documentElement.lang;
  function apply(value, updateHash) {
    if (value && !episodes.some(e => e.id === 'episode-' + value)) value = '';
    select.value = value;
    for (const e of episodes) {
      e.hidden = Boolean(value && e.id !== 'episode-' + value);
      if (value && !e.hidden) e.open = true;
    }
    status.textContent = (value ? '1' : String(episodes.length)) + (language === 'ko' ? '개 실제 사건' : ' recorded episodes');
    if (updateHash && value) location.hash = 'episode-' + value;
    const switcher = document.getElementById('language');
    if (switcher) switcher.href = '../' + (language === 'ko' ? 'en' : 'ko') + '/case017.html' + (value ? '#episode-' + value : '#episodes');
  }
  function fromHash() {
    const prefix = '#episode-';
    apply(location.hash.startsWith(prefix) ? location.hash.slice(prefix.length) : '', false);
  }
  controls.hidden = false;
  select.addEventListener('change', () => apply(select.value, true));
  document.getElementById('case017-reset').addEventListener('click', () => { apply('', false); location.hash = 'episodes'; });
  addEventListener('hashchange', fromHash);
  fromHash();
})();
