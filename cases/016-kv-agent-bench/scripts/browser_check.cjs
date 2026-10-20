'use strict';
// Existing Windows Node/Edge only. Fresh profile, CPU rendering, local files.
const fs=require('fs'),os=require('os'),path=require('path'),cp=require('child_process');
const {pathToFileURL}=require('url');
(async()=>{
 const folder=process.argv[2],out=process.argv[3];if(!folder||!out||fs.existsSync(out))throw Error('case folder and fresh output required');fs.mkdirSync(out,{recursive:true});
 const base=pathToFileURL(path.join(folder,'demo')+path.sep).href.replace(/\/$/,'');
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'case016-browser-'));const wait=ms=>new Promise(r=>setTimeout(r,ms));let ws,send;
 const checks=[],errors=[],captures=[];const proc=cp.spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{stdio:'ignore',windowsHide:true});
 try{
  let port;for(let i=0;i<100;i++){try{port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];break;}catch{}await wait(100);}if(!port)throw Error('No isolated Edge CDP port');
  const origin='http://127.0.0.1:'+port,version=await(await fetch(origin+'/json/version')).json();const target=(await(await fetch(origin+'/json/list')).json()).find(x=>x.type==='page');
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j});let seq=0;const pending=new Map();
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text)};
  send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
  const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value};
  const expect=(v,label)=>{if(!v)throw Error(label);checks.push(label)};
  const nav=async name=>{await send('Page.navigate',{url:base+'/'+name});for(let i=0;i<100;i++){if(await ev('document.readyState==="complete" && !!document.querySelector("main")'))break;await wait(80)}await wait(200)};
  for(const key of ['Page.enable','Runtime.enable','Network.enable'])await send(key);
  await send('Network.setBlockedURLs',{urls:['http://*','https://*']});
  for(const lang of ['ko','en']){
   await nav(lang+'.html');expect(await ev('document.documentElement.lang')===lang,lang+' language');
   expect(await ev('JSON.parse(document.getElementById("data").textContent).episodes.length')===96,lang+' all96 measured episodes');
   expect(await ev('document.querySelector("h1").textContent.includes("Case016")'),lang+' title');
   expect(await ev('getComputedStyle(document.body).backgroundColor')==='rgb(247, 246, 242)',lang+' DIOVA paper palette');
   for(const id of ['overview','business','cache','experiment','results','events','reproduce','sources'])expect(await ev('!!document.getElementById('+JSON.stringify(id)+')'),lang+' anchor '+id);
   expect(await ev('document.querySelectorAll("#raw-records a").length')===96,lang+' JS-off raw fallback links');
   for(const width of [390,768,1280]){
    await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await ev('window.scrollTo(0,0)');await wait(120);
    expect(await ev('document.documentElement.scrollWidth<=innerWidth+2'),lang+' no page overflow '+width);
    const image=await send('Page.captureScreenshot',{format:'png'});const name=lang+'-'+width+'.png';fs.writeFileSync(path.join(out,name),Buffer.from(image.data,'base64'));captures.push(name);
   }
   await ev('document.getElementById("filter-kv").value="F4";document.getElementById("filter-kv").dispatchEvent(new Event("change"))');
   expect(await ev('[...document.getElementById("episode").options].filter(x=>x.value).length')===48,lang+' KV filter48');
   await ev('const e=document.getElementById("episode");e.selectedIndex=1;e.dispatchEvent(new Event("change"))');
   expect(await ev('document.getElementById("viewer").textContent.includes("available_result_event_ids")'),lang+' generation observation boundary visible');
   expect(await ev('document.getElementById("viewer").textContent.includes("state_before")'),lang+' before/after state visible');
   const selected=await ev('document.getElementById("episode").value');await nav(lang+'.html#'+encodeURIComponent(selected));
   expect(await ev('document.getElementById("episode").value')===selected,lang+' direct episode anchor');
   const languageHref=await ev('document.querySelector("nav a").href');await send('Page.navigate',{url:languageHref});await wait(350);expect(await ev('document.documentElement.lang')===(lang==='ko'?'en':'ko'),lang+' language switch');
   await ev('history.back()');await wait(350);expect(await ev('document.documentElement.lang')===lang,lang+' browser back');
   await send('Emulation.setScriptExecutionDisabled',{value:true});await nav(lang+'.html');expect(await ev('document.getElementById("results").nextElementSibling.textContent.includes("F4")'),lang+' JS-off result table');expect(await ev('document.querySelectorAll("#raw-records a").length')===96,lang+' JS-off all raw links');await send('Emulation.setScriptExecutionDisabled',{value:false});
  }
  expect(errors.length===0,'no JS exceptions');
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({status:'PASS',browser:version.Browser,node:process.version,checks,viewports:[390,768,1280],captures,errors,mode:'file://; page HTTP blocked',GPU_rendering:'disabled',new_model_requests:0,real_iPad:'NOT_TESTED',computed_body_font:await ev('getComputedStyle(document.body).fontFamily'),computed_p_font:await ev('getComputedStyle(document.querySelector("main p")).fontFamily'),profile:'isolated task-created temporary profile; user profile untouched'},null,2));console.log('PASS',checks.length,'browser checks');
 }catch(e){fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({status:'FAIL',error:e.message,checks,errors},null,2));throw e}
 finally{if(send)try{await send('Browser.close')}catch{}if(ws)ws.close();await wait(400);if(proc.exitCode!==null)try{fs.rmSync(profile,{recursive:true,force:false})}catch{}}
})().catch(e=>{console.error(e.stack);process.exitCode=1});
