/* Case011 reads saved scalar projections only. No fetch, model execution or remote resources. */
'use strict';
(() => {
  const ko = document.documentElement.lang === 'ko';
  const text = (en, kr) => ko ? kr : en;
  const language = document.getElementById('language');
  const languageBase = language && language.getAttribute('href');
  const openAnchor = () => {
    if (language) language.href = languageBase + location.search + location.hash;
    const id = decodeURIComponent(location.hash.slice(1));
    const target = document.getElementById(id);
    if (target) {
      for (let node = target; node; node = node.parentElement) {
        if (node.tagName === 'DETAILS') node.open = true;
      }
    }
  };
  openAnchor();
  addEventListener('hashchange', openAnchor);
  const container = document.getElementById('case011-items');
  if (!container) return;
  const data = window.CASE011_ITEMS;
  if (!data || !Array.isArray(data.rows) || data.rows.length !== data.row_count) {
    container.hidden = false;
    container.textContent = text('Recorded item data are missing or malformed.', '입력별 기록이 없거나 형식이 올바르지 않습니다.');
    return;
  }
  const element = (tag, content, parent) => {
    const node = document.createElement(tag);
    if (content !== undefined && content !== null) node.textContent = String(content);
    if (parent) parent.appendChild(node);
    return node;
  };
  const controls = element('div', null, container); controls.className = 'item-controls';
  const definitions = [
    ['seed', 'Saved trained model', '학습된 모델 저장본', [['',text('All three saved models','세 모델 저장본 전체')], ['0','seed0'],['1','seed1'],['2','seed2']]],
    ['storage','Storage','저장 방식',[['',text('Both storage methods','두 저장 방식')],['NATIVE_FP32',text('Original 32-bit storage (FP32)','원래 32비트 저장 (FP32)')],['UNIFORM_8',text('8-bit integer storage (INT8)','8비트 정수 저장 (INT8)')]]],
    ['category','Prefix change','연속 정답 길이 변화',[['',text('All categories','모든 범주')],['longer',text('Longer','더 길어짐')],['shorter',text('Shorter','더 짧아짐')],['same',text('Unchanged','같음')]]]
  ];
  const inputs = {};
  const params = new URLSearchParams(location.search);
  let invalid = false;
  for (const [name,en,kr,options] of definitions) {
    const label = element('label',text(en,kr),controls);
    const select = element('select',null,label); select.id='case011-'+name;
    label.htmlFor=select.id;
    for (const [value,caption] of options) {const opt=element('option',caption,select);opt.value=value;}
    const value=params.get(name)||'';
    if (!options.some(row=>row[0]===value)) invalid=true;
    select.value=value;
    inputs[name]=select;
  }
  const qlabel=element('label',text('Sample ID','입력 ID'),controls);
  const query=element('input',null,qlabel);query.type='search';query.id='case011-query';qlabel.htmlFor=query.id;query.value=params.get('q')||'';inputs.q=query;
  const reset=element('button',text('Reset filters','필터 초기화'),container);reset.type='button';reset.className='secondary';
  const status=element('p',null,container);status.id='case011-item-count';status.setAttribute('role','status');status.setAttribute('aria-live','polite');
  const region=element('div',null,container);region.className='scroll';region.tabIndex=0;region.setAttribute('role','region');region.setAttribute('aria-label',text('First-error movement table','최초 오류 이동 표'));
  const table=element('table',null,region);const tr=element('tr',null,element('thead',null,table));
  for(const title of [text('Saved model','학습된 모델 저장본'),text('Storage','저장 방식'),text('Sample ID','입력 ID'),text('First error with original output rule','원래 출력 규칙의 첫 오류'),text('First error with corrected output rule','보정 출력 규칙의 첫 오류'),text('Change in consecutive correct symbol steps','연속 정답 길이 변화 (기호 단계)')]){const th=element('th',title,tr);th.scope='col';}
  const body=element('tbody',null,table);
  const navigation=element('div',null,container);navigation.className='item-pagination';
  const previous=element('button',text('Previous','이전'),navigation);previous.type='button';previous.className='secondary';
  const pageText=element('span',null,navigation);
  const next=element('button',text('Next','다음'),navigation);next.type='button';next.className='secondary';
  const pageSize=25;let page=Number(params.get('page')||1);
  if(!Number.isSafeInteger(page)||page<1){invalid=true;page=1;}
  const save = () => {
    const search=new URLSearchParams();
    for(const [key,input] of Object.entries(inputs))if(input.value)search.set(key,input.value);
    if(page>1)search.set('page',String(page));
    const newUrl=location.pathname+(search.toString()?'?'+search:'')+(location.hash||'#input-shifts');
    history.replaceState(null,'',newUrl);openAnchor();
  };
  const render = (update=false) => {
    const rows=invalid?[]:data.rows.filter(row=>(!inputs.seed.value||String(row.checkpoint_seed)===inputs.seed.value)&&(!inputs.storage.value||row.storage===inputs.storage.value)&&(!inputs.category.value||row.category===inputs.category.value)&&(!query.value||row.sample_id.includes(query.value)));
    const pages=Math.max(1,Math.ceil(rows.length/pageSize));
    if(page>pages)page=pages;
    body.replaceChildren();
    for(const row of rows.slice((page-1)*pageSize,page*pageSize)) {
      const tr=element('tr',null,body);
      for(const value of ['seed'+row.checkpoint_seed,row.storage==='UNIFORM_8'?text('8-bit integer (INT8)','8비트 정수 (INT8)'):text('Original 32-bit (FP32)','원래 32비트 (FP32)'),row.sample_id,row.tau_original??text('No first error observed within 2,048 symbol steps','검사한 2,048개 기호 단계에서 첫 오류 미관측'),row.tau_mixed??text('No first error observed within 2,048 symbol steps','검사한 2,048개 기호 단계에서 첫 오류 미관측'),row.rmst_delta>0?'+'+row.rmst_delta:String(row.rmst_delta)])element('td',value,tr);
    }
    status.textContent=invalid?text('Invalid filter or page value: 0 results. Reset filters to continue.','잘못된 필터 또는 페이지 값: 0건. 필터를 초기화할 수 있습니다.'):
      text(`${rows.length.toLocaleString()} saved same-question comparisons · ${data.row_count.toLocaleString()} total across saved-model and storage conditions; these are repeated conditions, not distinct questions`,`${rows.length.toLocaleString()}건의 기록 · 같은 문제를 모델 저장본·저장 방식별로 비교한 총 ${data.row_count.toLocaleString()}건 (독립 문제 수와 다름)`);
    pageText.textContent=text(`Page ${page} of ${pages}`,`${page} / ${pages} 페이지`);previous.disabled=page<=1;next.disabled=page>=pages;
    if(update)save();
  };
  for(const input of Object.values(inputs))input.addEventListener(input===query?'input':'change',()=>{invalid=false;page=1;render(true);});
  reset.addEventListener('click',()=>{for(const input of Object.values(inputs))input.value='';invalid=false;page=1;render(true);});
  previous.addEventListener('click',()=>{page--;render(true);});next.addEventListener('click',()=>{page++;render(true);});
  container.hidden=false;render();
})();
