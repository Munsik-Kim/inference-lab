/* Progressive enhancement only: raw text stays in escaped static HTML. */
(() => {
  'use strict';
  const select = document.getElementById('case018-episode');
  if (!select) return;
  const cards = [...document.querySelectorAll('.case018-episode')];
  const controls = document.getElementById('case018-controls');
  const count = document.getElementById('case018-count');
  const apply = (id) => {
    if (!cards.some(c => c.id === 'episode-' + id)) id = '';
    select.value = id;
    cards.forEach(c => { c.hidden = !!id && c.id !== 'episode-' + id; if (id && !c.hidden) c.open = true; });
    count.textContent = `${cards.filter(c => !c.hidden).length} / ${cards.length}`;
  };
  const fromHash = () => apply(location.hash.startsWith('#episode-') ? decodeURIComponent(location.hash.slice(9)) : '');
  const choose = () => {
    const hash = select.value ? '#episode-' + encodeURIComponent(select.value) : '#episodes';
    history.pushState(null, '', hash); apply(select.value);
  };
  select.addEventListener('change', choose);
  document.getElementById('case018-reset').addEventListener('click', () => { select.value = ''; choose(); select.focus(); });
  addEventListener('popstate', fromHash); addEventListener('hashchange', fromHash);
  controls.hidden = false; fromHash();
})();
