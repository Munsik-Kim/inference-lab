'use strict';
(() => {
  const ids = ['a1-method', 'a1-api', 'a1-scenario'];
  const controls = ids.map(id => document.getElementById(id));
  const episodes = [...document.querySelectorAll('.a1-episode')];
  if (controls.some(x => !x)) return;
  const show = () => {
    const [method, api, scenario] = controls.map(x => x.value);
    const direct = episodes.find(x => '#' + x.id === location.hash);
    if (direct) {
      controls[0].value = direct.dataset.method;
      controls[1].value = direct.dataset.api;
      controls[2].value = direct.dataset.scenario;
    }
    let count = 0;
    for (const episode of episodes) {
      episode.hidden = !(episode.dataset.method === controls[0].value && episode.dataset.api === controls[1].value && episode.dataset.scenario === controls[2].value);
      if (!episode.hidden) { count++; episode.open = true; }
    }
    document.getElementById('a1-filter-status').textContent = `${count} / ${episodes.length} measured A1 episodes`;
  };
  controls.forEach(x => x.addEventListener('change', () => { history.replaceState(null, '', '#episodes'); show(); }));
  window.addEventListener('hashchange', show);
  show();
})();
