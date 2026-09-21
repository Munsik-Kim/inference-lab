'use strict';
// Preserve only known track and evaluation anchors across language changes.
const languageLink = document.getElementById('language');
const languageBase = languageLink.getAttribute('href').split('#')[0];
function syncSection() {
  const section = ['#track-q', '#track-r', '#try-it', '#q-evaluation', '#q-timing', '#r-evaluation'].includes(location.hash) ? location.hash : '';
  languageLink.setAttribute('href', languageBase + section);
}
window.addEventListener('hashchange', syncSection);
syncSection();
