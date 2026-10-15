'use strict';
// Existing Windows Node + Edge, isolated headless profile, GPU rendering disabled.
const fs=require('fs'),os=require('os'),path=require('path'),cp=require('child_process');
(async()=>{
 const url=process.argv[2],out=process.argv[3];if(!url?.startsWith('file:')||!out||fs.existsSync(out))throw Error('local file URL and fresh output required');fs.mkdirSync(out,{recursive:true});
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'case014-p2-review-'));const wait=ms=>new Promise(r=>setTimeout(r,ms));let send,ws;const checks=[],errors=[],requests=[];
 cp.spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
 try{
  let port;for(let i=0;i<80;i++){try{port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];break;}catch{}await wait(150);}if(!port)throw Error('Edge profile did not expose CDP');
  const origin='http://127.0.0.1:'+port;const version=await(await fetch(origin+'/json/version')).json();const target=(await(await fetch(origin+'/json/list')).json()).find(x=>x.type==='page');ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});let seq=0;const pending=new Map();
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);else if(m.method==='Network.requestWillBeSent')requests.push(m.params.request.url);};
  send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value;};
  const expect=(v,label)=>{if(!v)throw Error(label);checks.push(label);};
  for(const m of ['Page.enable','Runtime.enable','Network.enable'])await send(m);await send('Network.setBlockedURLs',{urls:['http://*','https://*']});await send('Page.navigate',{url});
  for(let i=0;i<80;i++){if(await ev('document.readyState==="complete" && document.querySelectorAll("tbody tr").length===8'))break;await wait(100);}
  expect(await ev('document.querySelectorAll("tbody tr").length')===8,'all8 model/cohort summary rows rendered');expect(await ev('Array.from(document.images).every(x=>x.complete&&x.naturalWidth>0)'),'both scientific figures loaded');
  expect(await ev('document.body.innerText.includes("LOWER_NLL") || document.body.innerText.includes("−RTN")'),'primary comparison text present');
  expect(await ev('Array.from(document.querySelectorAll("a")).every(x=>x.protocol==="file:")'),'all links local; no external assets');
  for(const width of [390,1280]){await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await wait(100);expect(await ev('document.documentElement.scrollWidth<=innerWidth+2'),'no page overflow '+width);const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,'results-'+width+'.png'),Buffer.from(shot.data,'base64'));}
  expect(errors.length===0,'no JS exceptions');expect(requests.filter(u=>/^https?:/.test(u)).length===0,'no HTTP assets requested');fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({status:'PASS',browser:version.Browser,node:process.version,protocol:'local file via Windows Node/Edge CDP',GPU_rendering:'disabled',viewports:[390,1280],checks,errors,actual_iPad:'NOT_TESTED',human_comprehension:'NOT_TESTED'},null,2));console.log('Browser PASS',checks.length);
 }catch(e){fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({error:e.message,checks,errors},null,2));throw e;}finally{if(send)try{await send('Browser.close');}catch{}if(ws)ws.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
