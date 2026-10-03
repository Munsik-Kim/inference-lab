/* A compact view of all preserved comparisons, including without JavaScript. */
'use strict';
const select = document.getElementById('prompt-filter');
const sections = Array.from(document.querySelectorAll('.pair'));
function openRequest(section, comparison) {
  sections.forEach(s => { s.open = s === section; });
  if (!section) return;
  const pairs = [...section.querySelectorAll('.comparison')];
  const target = comparison || pairs.find(p=>p.open) || pairs[0];
  pairs.forEach(p=>p.open=p===target);
}
function filter(value) {
  sections.forEach(s=>s.hidden=value!=='all'&&s.id!==value);
  select.value=value;
  openRequest(sections.find(s=>s.id===value)||sections.find(s=>s.open)||sections[0]);
}
select.addEventListener('change',()=>{
  filter(select.value);
  history.replaceState(null,'','#'+(select.value==='all'?'images':select.value));
  document.getElementById('language').hash=location.hash;
});
for (const section of sections) {
  section.addEventListener('toggle',()=>{
    if (!section.open) return;
    openRequest(section);
    history.replaceState(null,'','#'+section.id);
    document.getElementById('language').hash=location.hash;
  });
  for (const pair of section.querySelectorAll('.comparison')) pair.addEventListener('toggle',()=>{
    if (!pair.open) return;
    section.querySelectorAll('.comparison').forEach(p=>{if(p!==pair)p.open=false;});
    history.replaceState(null,'','#'+pair.id);
    document.getElementById('language').hash=location.hash;
  });
}
function openFragment() {
  const id=decodeURIComponent(location.hash.slice(1));
  const target=document.getElementById(id);
  const section=target?.matches('.pair')?target:target?.closest('.pair');
  if(section){filter(section.id);openRequest(section,target.matches('.comparison')?target:null);}
  document.getElementById('language').hash=location.hash;
}
window.addEventListener('hashchange',openFragment);
openFragment();
