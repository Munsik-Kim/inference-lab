/* Preserve the same section when switching between the two static languages. */
'use strict';
(() => {
  const link = document.getElementById('language');
  if (!link) return;
  const base = link.getAttribute('href');
  const sync = () => { link.href = base + location.hash; };
  sync();
  addEventListener('hashchange', sync);
})();
