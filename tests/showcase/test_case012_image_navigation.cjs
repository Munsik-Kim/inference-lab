'use strict';
// Synthetic DOM contract; actual Edge/file/HTTPS checks are separate.
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');

function viewer(){
 const queue=[],elements=new Map(),groups=[],sections=[],cards=[];
 function element(id){
  const listeners={},e={id,dataset:{},hidden:false,textContent:'',append(){},addEventListener(type,fn){(listeners[type]??=[]).push(fn)},fire(type){for(const fn of listeners[type]??[])fn()}};
  let open=false;Object.defineProperty(e,'open',{get:()=>open,set:v=>{if(open!==v){open=v;queue.push(()=>e.fire('toggle'))}}});
  e.summary={dataset:{original:''},textContent:'Comparison 1/4',addEventListener(type,fn){(this.listeners??={})[type]=fn},fire(type){this.listeners?.[type]?.()}};
  e.querySelector=()=>e.summary;e.matches=selector=>selector==='.'+e.kind;e.closest=()=>null;elements.set(id,e);return e;
 }
 for(let r=0;r<2;r++){
  const section=element('request-'+r);section.kind='pair';section.pairs=[];
  for(let n=0;n<2;n++){
   const id='image-'+r+'-'+n,card=element(id);card.querySelector=()=>({src:'https://example.invalid/assets/'+id+'.png'});cards.push(card);
   const pair=element('request-'+r+'-comparison-'+n);pair.kind='comparison';pair.cards=[card];pair.querySelectorAll=()=>pair.cards;pair.closest=()=>section;section.pairs.push(pair);
  }
  section.querySelectorAll=()=>section.pairs;sections.push(section);
  groups.push({members:section.pairs.map(p=>p.cards[0].id),representative:section.pairs[1].cards[0].id});
 }
 const select=element('prompt-filter'),representative=element('representative-view'),progress=element('representative-progress');select.value='all';element('results');
 const location=new URL('https://example.invalid/en/case012-images.html#results'),language=new URL('https://example.invalid/ko/case012-images.html');elements.set('language',language);
 const window={CASE012_REVIEW_COPY:{visual_groups:groups},addEventListener(){}};
 const document={documentElement:{lang:'en'},getElementById:id=>elements.get(id),createElement:()=>({append(){}}),
  querySelectorAll:selector=>selector==='.pair'?sections:selector==='.pair .card'?cards:selector==='.comparison'?sections.flatMap(s=>s.pairs):[]};
 const history={replaceState(a,b,url){location.href=new URL(url,location.href).href}};
 const context=vm.createContext({window,document,location,history,URL,URLSearchParams});
 const source=process.env.CASE012_VIEWER_TEST_SOURCE||path.join(__dirname,'../../presentation/assets/case012-images.js');
 vm.runInContext(fs.readFileSync(source,'utf8'),context);
 function flush(){for(let i=0;queue.length;i++){assert.ok(i<100,'native toggle loop');queue.shift()()}}
 flush();return {location,language,select,representative,sections,flush,activate(detail){detail.summary.fire('click');detail.open=!detail.open;flush()}};
}

test('Programmatic representative opening preserves the results fragment and language destination',()=>{
 const v=viewer();assert.equal(v.location.hash,'#results');assert.equal(v.language.hash,'#results');
 assert.ok(v.sections[0].pairs[1].open,'a visible representative still opens');
});
test('Switching representative display keeps the non-image fragment',()=>{
 const v=viewer();v.representative.checked=false;v.representative.fire('change');v.flush();
 assert.equal(v.location.hash,'#results');assert.equal(v.language.hash,'#results');
 assert.equal(v.language.search,'?view=all');
});
test('Explicit summary activation and request filtering still choose the corresponding deep link',()=>{
 const v=viewer();v.activate(v.sections[1]);assert.equal(v.location.hash,'#request-1');assert.equal(v.language.hash,'#request-1');
 v.representative.checked=false;v.representative.fire('change');v.flush();
 const other=v.sections[1].pairs.find(p=>!p.open);v.activate(other);assert.equal(v.location.hash,'#'+other.id);assert.equal(v.language.hash,'#'+other.id);
 v.select.value='request-0';v.select.fire('change');v.flush();assert.equal(v.location.hash,'#request-0');
});
