/* A compact view of all preserved comparisons, including without JavaScript. */
'use strict';
const select = document.getElementById('prompt-filter');
const sections = Array.from(document.querySelectorAll('.pair'));
const copy=window.CASE012_REVIEW_COPY,representative=document.getElementById('representative-view'),ko=document.documentElement.lang==='ko';
const cards=[...document.querySelectorAll('.pair .card')];
for(const pair of document.querySelectorAll('.comparison'))pair.querySelector('summary').dataset.original=pair.querySelector('summary').textContent;
representative.checked=new URLSearchParams(location.search).get('view')!=='all';
function languageLink(){const link=document.getElementById('language');link.hash=location.hash;link.search='?view='+(representative.checked?'representatives':'all');}
for(const card of cards){
  const id=new URL(card.querySelector('img').src).pathname.split('/').pop().replace(/\.png$/,'');
  const group=copy.visual_groups.find(g=>g.members.includes(id));card.dataset.representative=String(group.representative===id);
  const note=document.createElement('p');note.className='visual-group-note';note.textContent=ko?`비슷한 사진 ${group.members.length}장 묶음`:`Visual review group: ${group.members.length} image(s)`;
  const link=document.createElement('a');link.href='case012-annotate.html#'+group.representative;link.textContent=ko?'이 묶음 대표로 체크':'Review this group';note.append(document.createElement('br'),link);card.append(note);
}
function representativeMode(){
  for(const card of cards)card.hidden=representative.checked&&card.dataset.representative!=='true';
  for(const pair of document.querySelectorAll('.comparison')){
    const visible=[...pair.querySelectorAll('.card')].filter(c=>!c.hidden),summary=pair.querySelector('summary');
    pair.hidden=!visible.length;
    const n=summary.dataset.original.match(/(?:비교|Comparison) (\d)\/4/)?.[1]||'';
    summary.textContent=representative.checked?(ko?`대표 사진 ${visible.length}장 · 비교 ${n}/4`:`${visible.length} representative image(s) · comparison ${n}/4`):summary.dataset.original;
  }
  document.getElementById('representative-progress').textContent=representative.checked?(ko?`원본 192장 중 비슷한 사진 ${copy.visual_groups.length}개 묶음의 대표를 보여 줍니다. 각 사진의 사람 답과 기존 AI 평가를 함께 보여 주며, 묶음에서 적용한 답은 구분해 표시합니다.`:`Showing ${copy.visual_groups.length} visual-group representatives of 192 originals. Human answers and original AI judgments are shown separately; visually reused human answers are marked.`):(ko?'원본 192장을 설정별로 모두 볼 수 있습니다.':'All 192 original images are available by setting.');
}
function openRequest(section, comparison) {
  sections.forEach(s => { s.open = s === section; });
  if (!section) return;
  const allPairs=[...section.querySelectorAll('.comparison')],pairs=allPairs.filter(p=>!p.hidden);
  const target = comparison || pairs.find(p=>p.open) || pairs[0];
  allPairs.forEach(p=>p.open=p===target);
}
function filter(value) {
  sections.forEach(s=>s.hidden=value!=='all'&&s.id!==value);
  select.value=value;
  openRequest(sections.find(s=>s.id===value)||sections.find(s=>s.open)||sections[0]);
}
select.addEventListener('change',()=>{
  filter(select.value);
  history.replaceState(null,'','#'+(select.value==='all'?'images':select.value));
  languageLink();
});
representative.addEventListener('change',()=>{const url=new URL(location.href);url.searchParams.set('view',representative.checked?'representatives':'all');history.replaceState(null,'',url);representativeMode();filter(select.value);languageLink();});
for (const section of sections) {
  // Native toggle events also follow programmatic opening during page load.
  // Only an explicit summary activation should choose an image URL fragment.
  section.querySelector(':scope > summary').addEventListener('click',()=>{
    if(section.open)return;
    history.replaceState(null,'','#'+section.id);
    languageLink();
  });
  section.addEventListener('toggle',()=>{
    if (!section.open) return;
    openRequest(section);
    languageLink();
  });
  for (const pair of section.querySelectorAll('.comparison')) {
    pair.querySelector(':scope > summary').addEventListener('click',()=>{
      if(pair.open||pair.hidden||!section.open)return;
      history.replaceState(null,'','#'+pair.id);
      languageLink();
    });
    pair.addEventListener('toggle',()=>{
      if (!pair.open||pair.hidden||!section.open) return;
      section.querySelectorAll('.comparison').forEach(p=>{if(p!==pair)p.open=false;});
      languageLink();
    });
  }
}
function openFragment() {
  const id=decodeURIComponent(location.hash.slice(1));
  const target=document.getElementById(id);
  const section=target?.matches('.pair')?target:target?.closest('.pair');
  if(section){if(target.matches('.comparison')&&new URLSearchParams(location.search).get('view')!=='representatives'){representative.checked=false;representativeMode();}filter(section.id);openRequest(section,target.matches('.comparison')?target:null);}
  languageLink();
}
window.addEventListener('hashchange',openFragment);
representativeMode();
openRequest(sections[0]);
openFragment();
