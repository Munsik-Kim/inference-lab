'use strict';
// Keep the same measured section when switching languages; no data fetch.
const languageLink = document.getElementById('language');
const languageBase = languageLink.getAttribute('href').split('#')[0];
function syncSection() {
  const section = ['#serving', '#quality', '#scope'].includes(location.hash) ? location.hash : '';
  languageLink.setAttribute('href', languageBase + section);
}
window.addEventListener('hashchange', syncSection);
syncSection();

// A file:// link may open JSON instead of downloading it. Reuse the retained
// display payload, without fetching files or contacting a model/service.
document.querySelector('a[download="case009.json"]').addEventListener('click', event => {
  event.preventDefault();
  const blob = new Blob([JSON.stringify(window.CASE009_EVIDENCE, null, 2) + '\n'],
    {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'case009.json';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
