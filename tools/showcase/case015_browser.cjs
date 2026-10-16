'use strict';
// Existing Windows Node/Edge CDP. No packages, user profile, GPU rendering or model calls.
const fs=require('fs'),os=require('os'),path=require('path'),cp=require('child_process');
const {fileURLToPath}=require('url');
(async()=>{
 const base=process.argv[2]?.replace(/\/$/,''),out=process.argv[3];if(!base||!out||fs.existsSync(out))throw Error('base URL and new output required');fs.mkdirSync(out,{recursive:true});
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'case015-publication-browser-'));const wait=ms=>new Promise(r=>setTimeout(r,ms));let ws,send;const checks=[],errors=[],requests=[];const captures=[];
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
   for(const page of ['index','case015']){
    const name=lang+'/'+page+'.html';await nav(name);expect(await ev('document.documentElement.lang')===lang,name+' language');
    if(page==='index'){
     expect(await ev('document.querySelectorAll("#interface-study").length')===1,name+' one Case015 application card');
     expect(await ev('document.querySelectorAll("#featured-stories article.story").length')===3,name+' existing three results-first stories retained');
     expect(await ev('document.querySelector("#interface-study").innerText.includes("15/16") && document.querySelector("#interface-study").innerText.includes("12/16")'),name+' raw-backed success counts');
     expect(await ev('!!document.getElementById("compensation-study")'),name+' Case014 retained');
    }else{
     expect(await ev('["overview","workflow","interfaces","results","episodes","reproduce","technical","references"].every(x=>!!document.getElementById(x))'),name+' stable sections');
     expect(await ev('document.getElementById("results").innerText.includes("15/16") || document.getElementById("results").innerText.includes("15")'),name+' measured results');
     expect(await ev('document.querySelectorAll(".a1-episode").length')===96,name+' all96 episodes retained');
     expect(await ev('document.querySelectorAll(".a1-episode:not([hidden])").length')===1,name+' one selected episode');
    }
    for(const width of [390,768,1280]){
     await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await ev('window.scrollTo(0,0)');await wait(100);
     expect(await ev('document.documentElement.scrollWidth<=innerWidth+2'),name+' no horizontal page overflow '+width);
     const shot=await send('Page.captureScreenshot',{format:'png'});const file=lang+'-'+page+'-'+width+'.png';fs.writeFileSync(path.join(out,file),Buffer.from(shot.data,'base64'));captures.push(file);
    }
    if(page==='index'){
     for(const width of [390,1280]){await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await ev('document.getElementById("interface-study").scrollIntoView()');await wait(100);const shot=await send('Page.captureScreenshot',{format:'png'});const file=lang+'-case015-home-'+width+'.png';fs.writeFileSync(path.join(out,file),Buffer.from(shot.data,'base64'));captures.push(file);}
    }else{
     await ev('document.getElementById("a1-method").value="D16";document.getElementById("a1-scenario").value="A1-00";document.getElementById("a1-method").dispatchEvent(new Event("change"))');
     expect(await ev('document.querySelectorAll(".a1-episode:not([hidden])").length===1 && document.querySelector(".a1-episode:not([hidden])").id==="D16__API_V1__A1-00"'),name+' filters select recorded failure');
     expect(await ev('document.querySelector(".a1-episode:not([hidden])").innerText.includes("write_order_total") && document.querySelector(".a1-episode:not([hidden])").innerText.includes("g0-c0")'),name+' generated calls and immutable receipt visible');
     await ev('document.querySelector(".a1-episode:not([hidden])>summary").focus()');await key();expect(await ev('!document.querySelector(".a1-episode:not([hidden])").open'),name+' keyboard disclosure');await key();
     await nav(name+'#BF16__API_V2__A1-06');expect(await ev('document.querySelector(".a1-episode:not([hidden])").id')==='BF16__API_V2__A1-06',name+' direct episode anchor');
     for(const width of [390,1280]){await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await ev('document.getElementById("results").scrollIntoView()');await wait(100);const shot=await send('Page.captureScreenshot',{format:'png'});const file=lang+'-case015-results-'+width+'.png';fs.writeFileSync(path.join(out,file),Buffer.from(shot.data,'base64'));captures.push(file);}
     const href=await ev('document.getElementById("language").href');await send('Page.navigate',{url:href});await wait(300);expect(await ev('document.documentElement.lang')===(lang==='ko'?'en':'ko'),name+' language switch');await ev('history.back()');await wait(300);expect(await ev('document.documentElement.lang')===lang,name+' browser back');
     await send('Emulation.setScriptExecutionDisabled',{value:true});await nav(name);expect(await ev('document.getElementById("results").innerText.includes("D16") && document.getElementById("results").innerText.includes("8.33")'),name+' JS-off A1 and historical P1 remain readable');expect(await ev('document.querySelectorAll(".a1-episode[hidden]").length')===0,name+' JS-off all episodes accessible');await send('Emulation.setScriptExecutionDisabled',{value:false});
    }
   }
   for(const page of ['case008','case009','case010','case011','case012','case013','case014','qwen3-4b','llm-guide','compare']){
    await nav(lang+'/'+page+'.html');for(const width of [390,1280]){await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await wait(70);expect(await ev('document.documentElement.scrollWidth<=innerWidth+2'),lang+'/'+page+' regression no overflow '+width);}expect(await ev('document.querySelectorAll("h1").length')===1,lang+'/'+page+' regression content');
   }
  }
  expect(errors.length===0,'no page JS exceptions');
  if(base.startsWith('file:'))expect(requests.filter(x=>/^https?:/.test(x)).length===0,'file-mode pages have no HTTP runtime dependency');
  const manifest=base.startsWith('file:')?JSON.parse(fs.readFileSync(fileURLToPath(base+'/build_manifest.json'),'utf8')):await(await fetch(base+'/build_manifest.json')).json();
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({status:'PASS',browser:version.Browser,node:process.version,mode:base.startsWith('file:')?'file':'hosted HTTPS',viewports:[390,768,1280],checks,errors,captures,repository_build_revision:manifest.repository_build_revision,presentation_identity:manifest.presentation_identity,new_model_forwards:0,GPU_rendering:'disabled',real_iPad:'NOT_TESTED',profile:'isolated task-created temporary profile, user profile untouched'},null,2));console.log('PASS',checks.length,'browser checks');
 }catch(e){fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({error:e.message,checks,errors},null,2));throw e;}finally{if(send)try{await send('Browser.close');}catch{}if(ws)ws.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
