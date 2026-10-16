'use strict';
// Existing Windows Node + isolated Edge CDP. No new dependencies or hosting.
const fs=require('fs'),path=require('path'),os=require('os'),cp=require('child_process');
(async()=>{
 const base=process.argv[2]?.replace(/\/$/,''),out=process.argv[3];if(!base||!out||fs.existsSync(out))throw Error('new output and base URL required');fs.mkdirSync(out,{recursive:true});
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'case015-review-'));const pause=ms=>new Promise(r=>setTimeout(r,ms));let send,ws;const checks=[],errors=[],captures=[];
 cp.spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
 try{
  let port;for(let i=0;i<80;i++){try{port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];break;}catch{}await pause(150);}if(!port)throw Error('isolated CDP unavailable');
  const origin='http://127.0.0.1:'+port;const version=await(await fetch(origin+'/json/version')).json();const target=(await(await fetch(origin+'/json/list')).json()).find(x=>x.type==='page');ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});let seq=0;const pending=new Map();
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);};
  send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value;};
  const expect=(ok,label)=>{if(!ok)throw Error(label);checks.push(label);};
  const nav=async file=>{await send('Page.navigate',{url:base+'/'+file});for(let i=0;i<100;i++){if(await ev('document.readyState==="complete" && !!document.querySelector("main")'))break;await pause(80);}await pause(90);};
  for(const method of ['Page.enable','Runtime.enable','Network.enable'])await send(method);
  await send('Network.setBlockedURLs',{urls:['http://*','https://*']});
  for(const lang of ['ko','en']){
   await nav(lang+'.html');expect(await ev('document.documentElement.lang')===lang,lang+' language');expect(await ev('document.querySelectorAll(".episode").length')===96,lang+' all96 episodes');expect(await ev('document.querySelectorAll(".comparison").length')===72,lang+' common-state comparison panels');expect(await ev('document.querySelectorAll(".matrix-link").length')===12,lang+' matrix task links');
   expect(await ev('document.querySelector("table").innerText.includes("D16") && document.querySelector("table").innerText.includes("5/8") && document.querySelector("table").innerText.includes("7/8")'),lang+' complete success/failure matrix');
   for(const width of [390,768,1280]){
    await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await ev('window.scrollTo(0,0)');await pause(80);expect(await ev('document.documentElement.scrollWidth<=innerWidth+2'),lang+' no page overflow '+width);
    const shot=await send('Page.captureScreenshot',{format:'png'});const f=lang+'-'+width+'.png';fs.writeFileSync(path.join(out,f),Buffer.from(shot.data,'base64'));captures.push(f);
   }
   await ev('document.querySelector(".matrix-link[data-method=D16][data-interface=I2_STYLE_RENAME]").click()');expect(await ev('Array.from(document.querySelectorAll(".episode")).filter(e=>!e.hidden).length')===8,lang+' matrix link selects8 tasks');
   await ev('document.getElementById("method").value="D16";document.getElementById("interface").value="I0_ORIGINAL";document.getElementById("task").value="AGENT-6";document.getElementById("task").dispatchEvent(new Event("change"))');
   expect(await ev('Array.from(document.querySelectorAll(".episode")).filter(e=>!e.hidden).length')===1,lang+' method/interface/task filter');
   await ev('Array.from(document.querySelectorAll(".episode")).find(e=>!e.hidden).querySelector("summary").focus()');
   await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',text:'\r',windowsVirtualKeyCode:13});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await pause(90);
   expect(await ev('Array.from(document.querySelectorAll(".episode")).find(e=>!e.hidden).open'),lang+' keyboard disclosure');
   expect(await ev('Array.from(document.querySelectorAll(".episode")).find(e=>!e.hidden).innerText.includes("ApplyFixturePatch")'),lang+' raw unknown name visible');
   await ev('Array.from(document.querySelectorAll(".episode")).find(e=>!e.hidden).scrollIntoView()');const detail=await send('Page.captureScreenshot',{format:'png'});const df=lang+'-D16-I0-AGENT6.png';fs.writeFileSync(path.join(out,df),Buffer.from(detail.data,'base64'));captures.push(df);
   await nav(lang+'.html#episodes');expect(await ev('location.hash')==='#episodes',lang+' direct anchor');
   await ev('document.querySelector("nav a").click()');await pause(200);expect(await ev('document.documentElement.lang')===(lang==='ko'?'en':'ko'),lang+' language switch');await ev('history.back()');await pause(200);expect(await ev('document.documentElement.lang')===lang,lang+' history back');
   await send('Emulation.setScriptExecutionDisabled',{value:true});await nav(lang+'.html');expect(await ev('document.querySelectorAll(".episode").length')===96,lang+' JS-off records');expect(await ev('document.querySelector("table").innerText.includes("5/8")'),lang+' JS-off scores');await send('Emulation.setScriptExecutionDisabled',{value:false});
  }
  expect(errors.length===0,'no JS exceptions');fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({status:'PASS',browser:version.Browser,node:process.version,checks,captures,viewports:[390,768,1280],errors,mode:'file://, external HTTP blocked',GPU_rendering:'disabled',real_iPad:'NOT_TESTED',model_forward:0},null,2));console.log('PASS',checks.length,'checks');
 }catch(e){fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({error:e.message,checks,errors},null,2));throw e;}finally{if(send)try{await send('Browser.close');}catch{}if(ws)ws.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
