'use strict';
const select = document.getElementById('prompt-filter');
const sections = Array.from(document.querySelectorAll('.pair'));
function filter(value) {
  sections.forEach(section => { section.hidden = value !== 'all' && section.id !== value; });
  select.value = value;
}
select.addEventListener('change', () => {
  filter(select.value);
  history.replaceState(null, '', '#' + (select.value === 'all' ? 'images' : select.value));
  document.getElementById('language').hash = location.hash;
});
function openFragment() {
  const id = decodeURIComponent(location.hash.slice(1));
  if (sections.some(section => section.id === id)) filter(id);
  document.getElementById('language').hash = location.hash;
}
window.addEventListener('hashchange', openFragment);
openFragment();
