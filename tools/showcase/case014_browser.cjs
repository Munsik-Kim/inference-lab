'use strict';
// Existing Windows Node/Edge CDP. No packages, user profile, GPU rendering or model calls.
const fs=require('fs'),os=require('os'),path=require('path'),cp=require('child_process');
const {fileURLToPath}=require('url');
(async()=>{
 const base=process.argv[2]?.replace(/\/$/,''),out=process.argv[3];if(!base||!out||fs.existsSync(out))throw Error('base URL and new output required');fs.mkdirSync(out,{recursive:true});
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'case014-publication-browser-'));const wait=ms=>new Promise(r=>setTimeout(r,ms));let ws,send;const checks=[],errors=[],requests=[];const captures=[];
 cp.spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
 try{
  let port;for(let i=0;i<80;i++){try{port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];break;}catch{}await wait(150);}if(!port)throw Error('No isolated Edge CDP port');
  const origin='http://127.0.0.1:'+port;const version=await(await fetch(origin+'/json/version')).json();const target=(await(await fetch(origin+'/json/list')).json()).find(x=>x.type==='page');ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});let seq=0;const pending=new Map();
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);else if(m.method==='Network.requestWillBeSent')requests.push(m.params.request.url);};
  send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value;};
  const expect=(v,label)=>{if(!v)throw Error(label);checks.push(label);};
  const nav=async name=>{await send('Page.navigate',{url:base+'/'+name});for(let i=0;i<100;i++){if(await ev('document.readyState==="complete" && !!document.querySelector("main")'))break;await wait(80);}await wait(90);};
  const key=async()=>{await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',text:'\r',unmodifiedText:'\r',windowsVirtualKeyCode:13});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await wait(80);};
  for(const method of ['Page.enable','Runtime.enable','Network.enable'])await send(method);
  if(base.startsWith('file:'))await send('Network.setBlockedURLs',{urls:['http://*','https://*']});
  for(const lang of ['ko','en']){
   for(const page of ['index','case014']){
    const name=lang+'/'+page+'.html';await nav(name);expect(await ev('document.documentElement.lang')===lang,name+' language');
    if(page==='index'){
     expect(await ev('document.querySelectorAll("#compensation-study").length')===1,name+' one latest Case014 card');expect(await ev('document.querySelectorAll("#featured-stories article.story").length')===3,name+' existing results-first stories retained');expect(await ev('document.querySelector("#compensation-study").innerText.includes("12.92%") && document.querySelector("#compensation-study").innerText.includes("64/64")'),name+' source-backed headline');
    }else{
     expect(await ev('document.querySelectorAll("script").length')===0,name+' static JS-free essentials');
     expect(await ev('["overview","implementation","results","method","journey","reproduce","technical","references"].every(x=>!!document.getElementById(x))'),name+' eight stable sections');
     expect(await ev('document.getElementById("results").innerText.includes("57.600") && document.getElementById("results").innerText.includes("50.156") && document.getElementById("results").innerText.includes("45.405") && document.getElementById("results").innerText.includes("12.687")'),name+' PG19/ArXiv/BF16 present without tabs');
     expect(await ev('document.getElementById("overview").innerText.includes("BF16")'),name+' runtime scope next to gain');
    }
    for(const width of [390,768,1280]){
     await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await ev('window.scrollTo(0,0)');await wait(80);expect(await ev('document.documentElement.scrollWidth<=innerWidth+2'),name+' no horizontal page overflow '+width);
     const shot=await send('Page.captureScreenshot',{format:'png'});const file=lang+'-'+page+'-'+width+'.png';fs.writeFileSync(path.join(out,file),Buffer.from(shot.data,'base64'));captures.push(file);
    }
    if(page==='index'){
     for(const width of [390,1280]){await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await ev('document.getElementById("compensation-study").scrollIntoView()');await wait(90);const shot=await send('Page.captureScreenshot',{format:'png'});const file=lang+'-case014-home-card-'+width+'.png';fs.writeFileSync(path.join(out,file),Buffer.from(shot.data,'base64'));captures.push(file);}
    }
    if(page==='case014'){
     await ev('document.querySelector("summary").focus()');await key();expect(await ev('document.querySelector("details").open'),name+' keyboard disclosure');await ev('document.querySelectorAll("details").forEach(x=>x.open=true)');
     const count=await ev('document.images.length');for(let i=0;i<count;i++){await ev('document.images['+i+'].scrollIntoView()');await wait(90);}const imageState=()=>ev('Array.from(document.images).map(x=>({src:x.currentSrc||x.src,complete:x.complete,naturalWidth:x.naturalWidth}))');const imageBefore=await imageState();const imageStart=Date.now();for(let attempt=0;attempt<150;attempt++){if(await ev('Array.from(document.images).every(x=>x.complete&&x.naturalWidth>0)'))break;await wait(100);}const imageAfter=await imageState();fs.writeFileSync(path.join(out,lang+'-image-load.json'),JSON.stringify({before:imageBefore,after:imageAfter,additional_wait_ms:Date.now()-imageStart},null,2));expect(await ev('Array.from(document.images).every(x=>x.complete&&x.naturalWidth>0)'),name+' all scientific figures load');
     await send('Emulation.setDeviceMetricsOverride',{width:390,height:1000,deviceScaleFactor:1,mobile:false});await nav(name+'#results');expect(await ev('location.hash==="#results" && document.getElementById("results").getBoundingClientRect().top>=document.querySelector(".site-header").getBoundingClientRect().bottom-2'),name+' direct anchor clears sticky header');
     for(const width of [390,1280]){await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await ev('document.getElementById("results").scrollIntoView()');await wait(90);const shot=await send('Page.captureScreenshot',{format:'png'});const file=lang+'-case014-results-'+width+'.png';fs.writeFileSync(path.join(out,file),Buffer.from(shot.data,'base64'));captures.push(file);}
     const href=await ev('document.getElementById("language").href');await send('Page.navigate',{url:href});await wait(200);expect(await ev('document.documentElement.lang')===(lang==='ko'?'en':'ko'),name+' language switch');await ev('history.back()');await wait(200);expect(await ev('document.documentElement.lang')===lang,name+' browser back');
     await send('Emulation.setScriptExecutionDisabled',{value:true});await nav(name);expect(await ev('document.getElementById("results").innerText.includes("50.156")'),name+' JS-off core results');await send('Emulation.setScriptExecutionDisabled',{value:false});
     const links=await ev('Array.from(document.querySelectorAll("#implementation a,#reproduce a")).map(x=>x.href)');expect(links.every(x=>/case014|download|source/.test(x)),name+' concrete implementation/reproduce/download links');
    }
   }
   for(const page of ['case008','case009','case010','case011','case012','case013','qwen3-4b','llm-guide']){
    await nav(lang+'/'+page+'.html');for(const width of [390,1280]){await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await wait(70);expect(await ev('document.documentElement.scrollWidth<=innerWidth+2'),lang+'/'+page+' regression no overflow '+width);}
    expect(await ev('document.querySelectorAll("h1").length')===1,lang+'/'+page+' regression content');
   }
  }
  expect(errors.length===0,'no page JS exceptions');
  if(base.startsWith('file:'))expect(requests.filter(x=>/^https?:/.test(x)).length===0,'file-mode pages have no HTTP runtime dependency');
  const manifest=base.startsWith('file:')?JSON.parse(fs.readFileSync(fileURLToPath(base+'/build_manifest.json'),'utf8')):await(await fetch(base+'/build_manifest.json')).json();
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({status:'PASS',browser:version.Browser,node:process.version,mode:base.startsWith('file:')?'file':'hosted HTTPS',viewports:[390,768,1280],checks,errors,captures,repository_build_revision:manifest.repository_build_revision,presentation_identity:manifest.presentation_identity,new_model_forwards:0,GPU_rendering:'disabled',real_iPad:'NOT_TESTED',profile:'isolated task-created temporary profile, user profile untouched'},null,2));console.log('PASS',checks.length,'browser checks');
 }catch(e){fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({error:e.message,checks,errors},null,2));throw e;}finally{if(send)try{await send('Browser.close');}catch{}if(ws)ws.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
