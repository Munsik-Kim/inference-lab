'use strict';
// Adapted from Case016's owned Node/Edge CDP checker. No packages or live model endpoint.
const fs=require('fs'),os=require('os'),path=require('path'),cp=require('child_process');
const {fileURLToPath}=require('url');
(async()=>{
 const base=process.argv[2]?.replace(/\/$/,''),out=process.argv[3];
 if(!base||!out||fs.existsSync(out))throw Error('file base URL and fresh output required');
 fs.mkdirSync(out,{recursive:true});
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'case017-public-browser-'));
 const delay=ms=>new Promise(r=>setTimeout(r,ms));
 const edge=process.env.CASE017_EDGE_PATH||'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
 cp.spawn(edge,['--headless=new','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
 let ws,send;const checks=[],errors=[],captures=[];
 try{
  let port;for(let i=0;i<100;i++){try{port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];break;}catch{}await delay(100);}
  if(!port)throw Error('Owned browser CDP endpoint unavailable');
  const origin='http://127.0.0.1:'+port;
  const version=await(await fetch(origin+'/json/version')).json();
  const target=(await(await fetch(origin+'/json/list')).json()).find(x=>x.type==='page');
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
  let id=0;const pending=new Map();
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);};
  send=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
  const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value;};
  const expect=(condition,label)=>{if(!condition)throw Error(label);checks.push(label);};
  for(const method of ['Page.enable','Runtime.enable','Network.enable'])await send(method);
  await send('Network.setBlockedURLs',{urls:['http://*','https://*']});
  const nav=async name=>{await send('Page.navigate',{url:base+'/'+name});for(let i=0;i<100;i++){if(await ev('document.readyState==="complete"&&!!document.querySelector("main")'))break;await delay(80);}await delay(150);};
  for(const lang of ['ko','en']){
   for(const page of ['index','case017']){
    const name=lang+'/'+page+'.html';await nav(name);
    expect(await ev('document.documentElement.lang')===lang,name+' language');
    if(page==='index'){
     expect(await ev('document.querySelectorAll("#contract-cost-study").length===1'),name+' one Case017 entry');
     expect(await ev('document.querySelectorAll("#featured-stories article.story").length===3'),name+' previous featured stories');
     expect(await ev('!!document.getElementById("kv-engineering")&&!!document.getElementById("interface-study")'),name+' Case015/016 retained');
    }else{
     expect(await ev('["overview","workflow","policies","results","cost","episodes","protocol-deviation","reproduce","references"].every(x=>document.getElementById(x))'),name+' anchors');
     expect(await ev('document.getElementById("overview").innerText.includes("15/24")&&document.getElementById("overview").innerText.includes("5/8")&&document.getElementById("overview").innerText.includes("6/8")'),name+' results and prominent deviation');
     expect(await ev('document.querySelectorAll(".case017-episode").length===7'),name+' seven recorded episodes');
    }
    for(const width of [390,768,1280]){
     await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await ev('scrollTo(0,0)');await delay(100);
     expect(await ev('document.documentElement.scrollWidth<=innerWidth+2'),name+' page fits '+width);
     const shot=await send('Page.captureScreenshot',{format:'png'});const file=lang+'-'+page+'-'+width+'.png';fs.writeFileSync(path.join(out,file),Buffer.from(shot.data,'base64'));captures.push(file);
    }
    if(page==='case017'){
     await nav(name+'#episode-HOLDOUT-B03_K1_N');
     expect(await ev('document.getElementById("case017-episode").value==="HOLDOUT-B03_K1_N"&&document.getElementById("episode-HOLDOUT-B03_K1_N").open'),name+' direct event link/filter');
     await ev('document.getElementById("case017-episode").focus()');expect(await ev('document.activeElement.id==="case017-episode"'),name+' keyboard selector');
     await ev('document.getElementById("case017-reset").click()');
     expect(await ev('Array.from(document.querySelectorAll(".case017-episode")).every(e=>!e.hidden)'),name+' show all episodes');
     await ev('document.getElementById("episode-HOLDOUT-A00_K1_C0").open=true;document.querySelector("#episode-HOLDOUT-A00_K1_C0 .case017-turn details summary").focus()');
     await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',text:'\r',windowsVirtualKeyCode:13});
     await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await delay(100);
     expect(await ev('document.querySelector("#episode-HOLDOUT-A00_K1_C0 .case017-raw").parentElement.open'),name+' keyboard opens raw');
     expect(await ev('document.querySelector("#episode-HOLDOUT-A00_K1_C0 .case017-raw").textContent.endsWith("}}}")'),name+' unmodified malformed raw visible');
     await nav(name+'#results');
     expect(await ev('document.getElementById("results").getBoundingClientRect().top>=document.querySelector(".site-header").getBoundingClientRect().bottom-3'),name+' anchor not covered');
     await ev('document.images[0].scrollIntoView()');await delay(200);
     expect(await ev('Array.from(document.images).every(x=>x.complete&&x.naturalWidth>0)'),name+' generated SVG loads');
     const href=await ev('document.getElementById("language").href');await send('Page.navigate',{url:href});await delay(350);
     expect(await ev('document.documentElement.lang')===(lang==='ko'?'en':'ko'),name+' language switch');
     await ev('history.back()');await delay(350);expect(await ev('document.documentElement.lang')===lang,name+' back navigation');
     await send('Emulation.setScriptExecutionDisabled',{value:true});await nav(name);
     expect(await ev('document.getElementById("overview").innerText.includes("15/24")&&document.querySelectorAll(".case017-episode").length===7'),name+' no-JS results and episodes');
     await send('Emulation.setScriptExecutionDisabled',{value:false});
    }
   }
  }
  expect(errors.length===0,'no runtime exceptions');
  const manifest=JSON.parse(fs.readFileSync(fileURLToPath(base+'/build_manifest.json'),'utf8'));
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({status:'PASS',browser:version.Browser,node:process.version,
   checks,errors,captures,viewports:[390,768,1280],mode:'local file:// isolated Edge; GPU rendering disabled',
   repository_build_revision:manifest.repository_build_revision,presentation_identity:manifest.presentation_identity,
   new_model_requests:0,Pages_deployment:'NOT_RUN',physical_mobile_device:'NOT_TESTED'},null,2));
  console.log('PASS',checks.length,'browser checks');
 }catch(e){fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({error:e.message,checks,errors},null,2));throw e;}
 finally{if(send)try{await send('Browser.close');}catch{}if(ws)ws.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
