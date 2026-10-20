'use strict';
// Existing Node24/Edge CDP; isolated task profile, disabled GPU, no package install.
const fs=require('fs'),os=require('os'),path=require('path'),cp=require('child_process');
const {fileURLToPath}=require('url');
(async()=>{
 const base=process.argv[2]?.replace(/\/$/,''),out=process.argv[3];
 if(!base||!out||fs.existsSync(out))throw Error('file base URL and fresh output directory required');
 fs.mkdirSync(out,{recursive:true});
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'case016-browser-'));const delay=ms=>new Promise(r=>setTimeout(r,ms));
 const edge=process.env.CASE016_EDGE_PATH||'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
 cp.spawn(edge,['--headless=new','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
 const checks=[],errors=[],captures=[];let ws,send;
 try {
  let port;for(let i=0;i<100;i++){try{port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];break;}catch{}await delay(100);}
  if(!port)throw Error('No owned browser CDP endpoint');
  const origin='http://127.0.0.1:'+port,version=await(await fetch(origin+'/json/version')).json();
  const target=(await(await fetch(origin+'/json/list')).json()).find(x=>x.type==='page');
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
  let id=0;const pending=new Map();
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);};
  send=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
  const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value;};
  const expect=(x,label)=>{if(!x)throw Error(label);checks.push(label);};
  for(const method of ['Page.enable','Runtime.enable','Network.enable'])await send(method);
  await send('Network.setBlockedURLs',{urls:['http://*','https://*']});
  const nav=async name=>{await send('Page.navigate',{url:base+'/'+name});for(let i=0;i<100;i++){if(await ev('document.readyState==="complete"&&!!document.querySelector("main")'))break;await delay(80);}await delay(150);};
  for(const lang of ['ko','en']){
   for(const page of ['index','case016']){
    const name=lang+'/'+page+'.html';await nav(name);expect(await ev('document.documentElement.lang')===lang,name+' language');
    if(page==='index'){
     expect(await ev('document.querySelectorAll("#kv-engineering").length===1'),name+' one Case016 card');
     expect(await ev('document.querySelectorAll("#featured-stories article.story").length===3'),name+' previous featured stories');
     expect(await ev('!!document.getElementById("interface-study")&&!!document.getElementById("compensation-study")'),name+' Case014/015 retained');
    }else{
     expect(await ev('["overview","kv-cache","storage","capacity","serving","numeric","limits","reproduce","references"].every(x=>document.getElementById(x))'),name+' anchors');
     expect(await ev('document.getElementById("capacity").innerText.includes("103552")&&document.getElementById("capacity").innerText.includes("29120")'),name+' capacity');
     expect(await ev('document.getElementById("limits").innerText.includes("0/48")&&document.getElementById("limits").innerText.includes("6/48")'),name+' original negative result');
     expect(await ev('document.querySelectorAll("script").length===0'),name+' no JavaScript requirement');
    }
    for(const width of [390,768,1280]){
     await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await ev('window.scrollTo(0,0)');await delay(100);
     expect(await ev('document.documentElement.scrollWidth<=innerWidth+2'),name+' no page overflow '+width);
     const shot=await send('Page.captureScreenshot',{format:'png'}),file=lang+'-'+page+'-'+width+'.png';fs.writeFileSync(path.join(out,file),Buffer.from(shot.data,'base64'));captures.push(file);
    }
    if(page==='case016'){
     await ev('document.querySelector("summary").focus()');await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',text:'\r',unmodifiedText:'\r',windowsVirtualKeyCode:13});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await delay(120);
     expect(await ev('document.querySelector("details").open'),name+' keyboard details');
     for(let i=0;i<4;i++){await ev('document.images['+i+'].scrollIntoView()');await delay(120);}
     for(let i=0;i<100;i++){if(await ev('Array.from(document.images).every(x=>x.complete&&x.naturalWidth>0)'))break;await delay(100);}
     expect(await ev('Array.from(document.images).every(x=>x.complete&&x.naturalWidth>0)'),name+' all SVGs load');
     await nav(name+'#serving');expect(await ev('location.hash==="#serving"&&document.getElementById("serving").getBoundingClientRect().top>=document.querySelector(".site-header").getBoundingClientRect().bottom-3'),name+' direct anchor');
     const href=await ev('document.getElementById("language").href');await send('Page.navigate',{url:href});await delay(300);expect(await ev('document.documentElement.lang')===(lang==='ko'?'en':'ko'),name+' language switch');await ev('history.back()');await delay(300);expect(await ev('document.documentElement.lang')===lang,name+' back');
     await send('Emulation.setScriptExecutionDisabled',{value:true});await nav(name);expect(await ev('document.getElementById("serving").innerText.includes("req/s")'),name+' JS disabled results');await send('Emulation.setScriptExecutionDisabled',{value:false});
    }
   }
  }
  for(const page of ['case008','case009','case014','case015','qwen3-4b']){
   await nav('ko/'+page+'.html');for(const width of [390,1280]){await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});await delay(90);expect(await ev('document.documentElement.scrollWidth<=innerWidth+2'),'regression '+page+' '+width);}
  }
  expect(errors.length===0,'no page JavaScript exceptions');
  const manifest=JSON.parse(fs.readFileSync(fileURLToPath(base+'/build_manifest.json'),'utf8'));
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({status:'PASS',browser:version.Browser,node:process.version,checks,errors,captures,viewports:[390,768,1280],mode:'file:// isolated Edge',repository_build_revision:manifest.repository_build_revision,presentation_identity:manifest.presentation_identity,new_gpu_model_requests:0,real_iPad:'NOT_TESTED',GPU_rendering:'disabled'},null,2));
  console.log('PASS',checks.length,'browser checks');
 }catch(e){fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({error:e.message,checks,errors},null,2));throw e;}finally{if(send)try{await send('Browser.close');}catch{}if(ws)ws.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
