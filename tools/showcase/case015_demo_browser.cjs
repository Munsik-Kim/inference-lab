'use strict';
// Existing Windows Node/Edge CDP. No packages, user profile, GPU rendering or model calls.
const fs=require('fs'),os=require('os'),path=require('path'),cp=require('child_process');
const {fileURLToPath}=require('url');
(async()=>{
 const base=process.argv[2]?.replace(/\/$/,''),out=process.argv[3];if(!base||!out||fs.existsSync(out))throw Error('base URL and new output required');fs.mkdirSync(out,{recursive:true});
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'case015-standalone-browser-'));const wait=ms=>new Promise(r=>setTimeout(r,ms));let ws,send;const checks=[],errors=[],requests=[];const captures=[];
 cp.spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
 try{
  let port;for(let i=0;i<80;i++){try{port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];break;}catch{}await wait(150);}if(!port)throw Error('No isolated Edge CDP port');
  const origin='http://127.0.0.1:'+port;const version=await(await fetch(origin+'/json/version')).json();const target=(await(await fetch(origin+'/json/list')).json()).find(x=>x.type==='page');ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});let seq=0;const pending=new Map();
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);else if(m.method==='Network.requestWillBeSent')requests.push(m.params.request.url);};
  send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value;};
  const expect=(v,label)=>{if(!v)throw Error(label);checks.push(label);};
  const nav=async name=>{await send('Page.navigate',{url:base+'/'+name});for(let i=0;i<100;i++){if(await ev('document.readyState==="complete" && !!document.querySelector("h1")'))break;await wait(80);}await wait(90);};
  const key=async()=>{await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',text:'\r',unmodifiedText:'\r',windowsVirtualKeyCode:13});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await wait(80);};
  for(const method of ['Page.enable','Runtime.enable','Network.enable'])await send(method);
  if(base.startsWith('file:'))await send('Network.setBlockedURLs',{urls:['http://*','https://*']});
  for(const lang of ['ko','en']){
   await nav(lang+'.html');expect(await ev('document.documentElement.lang')===lang,lang+' language');
   expect(await ev('document.querySelectorAll("#scenario option").length')===16,lang+'16 scenario choices');
   expect(await ev('document.querySelector("#episode h3").textContent')==='BF16__API_V1__A1-00',lang+' initial measured episode rendered');
   for(const width of [390,768,1280]){await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await wait(80);const fits=await ev('document.documentElement.scrollWidth<=innerWidth+2');if(!fits){const diagnosis=await ev('({viewport:innerWidth,document:document.documentElement.scrollWidth,bodyWidth:document.body.clientWidth,bodyRect:document.body.getBoundingClientRect().width,computedBody:getComputedStyle(document.body).width,overflow:Array.from(document.querySelectorAll("body *")).filter(x=>x.getBoundingClientRect().right>innerWidth+2).map(x=>({tag:x.tagName,id:x.id,class:x.className,right:x.getBoundingClientRect().right,text:x.innerText?.slice(0,120)})).slice(0,12)})');fs.writeFileSync(path.join(out,'overflow.json'),JSON.stringify(diagnosis,null,2));const failureShot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,'overflow.png'),Buffer.from(failureShot.data,'base64'));}expect(fits,lang+' standalone no overflow '+width);const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,lang+'-'+width+'.png'),Buffer.from(shot.data,'base64'));}
   await ev('document.getElementById("method").value="D16";document.getElementById("method").dispatchEvent(new Event("change"))');expect(await ev('document.querySelector("#episode h3").textContent')==='D16__API_V1__A1-00',lang+' model filter executes actual JS');
   await ev('document.getElementById("scenario").value="A1-06";document.getElementById("api").value="API_V2";document.getElementById("api").dispatchEvent(new Event("change"))');expect(await ev('document.querySelector("#episode h3").textContent')==='D16__API_V2__A1-06',lang+' API/scenario filter');
   expect(await ev('document.querySelectorAll(".turn").length')===3,lang+' three actual response groups for recorded parse failure');
   expect(await ev('document.getElementById("episode").innerText.includes("Results received before generation")'),lang+' receipt visibility present');
  }
  expect(errors.length===0,'no standalone JS exceptions');expect(requests.filter(x=>/^https?:/.test(x)).length===0,'no HTTP runtime dependency');
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({status:'PASS',browser:version.Browser,node:process.version,mode:'standalone file',viewports:[390,768,1280],checks,errors,new_model_forwards:0,real_iPad:'NOT_TESTED'},null,2));console.log('PASS',checks.length,'standalone browser checks');
 }catch(e){fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({error:e.message,checks,errors},null,2));throw e;}finally{if(send)try{await send('Browser.close');}catch{}if(ws)ws.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
