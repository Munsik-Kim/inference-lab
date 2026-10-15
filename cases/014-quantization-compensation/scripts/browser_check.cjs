'use strict';
// Existing Windows Node + Edge CDP; new isolated profile, no installed packages.
const fs=require('fs'),os=require('os'),path=require('path'),cp=require('child_process');
(async()=>{
 const root=process.argv[2],out=process.argv[3];if(!root?.startsWith('file:')||!out||fs.existsSync(out))throw Error('file URL and fresh output required');fs.mkdirSync(out,{recursive:true});
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'case014-review-'));const wait=ms=>new Promise(r=>setTimeout(r,ms));
 try{await fetch('http://127.0.0.1:9256/json/version',{signal:AbortSignal.timeout(400)});throw Error('Port occupied');}catch(e){if(e.message==='Port occupied')throw e;}
 cp.spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=9256','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
 let ws,send;const checks=[],errors=[],requests=[];let version;
 try{
  let target;for(let i=0;i<80;i++){try{version=await(await fetch('http://127.0.0.1:9256/json/version')).json();target=(await(await fetch('http://127.0.0.1:9256/json/list')).json()).find(x=>x.type==='page');if(target)break;}catch{}await wait(150);}
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});let seq=0;const pending=new Map();
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);else if(m.method==='Network.requestWillBeSent')requests.push(m.params.request.url);};
  send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value;};
  const expect=(v,label)=>{if(!v)throw Error(label);checks.push(label);};
  for(const m of ['Page.enable','Runtime.enable','Network.enable'])await send(m);await send('Network.setBlockedURLs',{urls:['http://*','https://*']});
  for(const lang of ['ko','en']){
   await send('Page.navigate',{url:root+'/'+lang+'/index.html'});for(let i=0;i<100;i++){if(await ev('document.readyState==="complete" && document.querySelectorAll("nav a").length===6'))break;await wait(80);}
   expect(await ev('document.documentElement.lang')===lang,lang+' lang');expect(await ev('document.querySelectorAll("#p1-grid tbody tr").length')===30,lang+' all30 pending cells');
   expect(await ev('Array.from(document.querySelectorAll("img")).every(x=>x.complete&&x.naturalWidth>0)'),lang+' figures loaded');
   for(const width of [390,768,1280]){
    await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await wait(90);
    expect(await ev('document.documentElement.scrollWidth<=innerWidth+2'),lang+' no page overflow '+width);
    const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(out,lang+'-'+width+'.png'),Buffer.from(shot.data,'base64'));
   }
   expect(await ev('Array.from(document.querySelectorAll("nav a")).every(a=>document.querySelector(a.getAttribute("href")))'),lang+' anchors exist');
   await send('Page.bringToFront');await ev('document.querySelector("summary").focus()');await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',text:'\r',unmodifiedText:'\r',windowsVirtualKeyCode:13});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await wait(120);expect(await ev('document.querySelector("details").open'),lang+' keyboard disclosure');
   await send('Emulation.setScriptExecutionDisabled',{value:true});await send('Page.reload');await wait(500);expect(await ev('document.querySelectorAll("#p1-grid tbody tr").length')===30,lang+' JS-off results');await send('Emulation.setScriptExecutionDisabled',{value:false});
   const link=await ev('document.querySelector("main>a").href');await send('Page.navigate',{url:link});await wait(500);expect(await ev('document.documentElement.lang')===(lang==='ko'?'en':'ko'),lang+' language switch');
  }
  expect(errors.length===0,'no JS exceptions');expect(requests.filter(u=>/^https?:/.test(u)).length===0,'no external assets');
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({status:'PASS',browser:version.Browser,node:process.version,protocol:'file:',viewports:[390,768,1280],checks,errors,real_iPad:'NOT_TESTED',human_comprehension_test:'NOT_PERFORMED',method:'existing Windows Node/Edge headless CDP; isolated profile; GPU rendering disabled'},null,2));console.log('PASS',checks.length);
 }catch(e){fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({error:e.message,checks,errors},null,2));throw e;}finally{if(send)try{await send('Browser.close');}catch{}if(ws)ws.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
