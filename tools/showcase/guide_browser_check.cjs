/* No packages or browser install. Use Node >=22 with an already installed Edge.
 * node guide_browser_check.cjs <file-or-loopback-base-url/> <NEW-external-report-dir> [edge-executable]
 * On Windows the default executable matches the existing browser_check.cjs.
 * Desktop viewport emulation, not a physical phone/iPad test or reader study.
 */
'use strict';
const fs=require('fs'),os=require('os'),path=require('path'),cp=require('child_process');
(async()=>{
  const [base,out,edge='C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe']=process.argv.slice(2);
  if(!base||!out||fs.existsSync(out))throw Error('Supply a base URL and a new external report directory');
  const u=new URL(base);
  if(u.protocol!=='file:'&&!(u.protocol==='http:'&&['localhost','127.0.0.1'].includes(u.hostname)))throw Error('Only file or loopback previews');
  const endpoint='http://127.0.0.1:9239';
  try{await fetch(endpoint+'/json/version',{signal:AbortSignal.timeout(500)});throw Error('CDP port already occupied');}
  catch(e){if(e.message==='CDP port already occupied')throw e;}
  fs.mkdirSync(out,{recursive:true});
  const profile=fs.mkdtempSync(path.join(os.tmpdir(),'diova-guide-'));
  const child=cp.spawn(edge,['--headless=new','--disable-gpu','--disable-background-networking','--no-first-run',
    '--remote-debugging-port=9239','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
  let spawnError;child.on('error',e=>spawnError=e);
  const checks=[],errors=[],requests=[],responseErrors=[];let send,ws,version;
  const wait=ms=>new Promise(r=>setTimeout(r,ms));
  const expect=(v,label)=>{if(!v)throw Error(label);checks.push(label);};
  try{
    let targets;
    for(let i=0;i<80;i++){
      if(spawnError)throw spawnError;
      try{version=await(await fetch(endpoint+'/json/version',{signal:AbortSignal.timeout(500)})).json();
        targets=await(await fetch(endpoint+'/json/list')).json();if(targets.some(t=>t.type==='page'))break;
      }catch{} await wait(150);
    }
    if(!targets?.some(t=>t.type==='page'))throw Error('Owned Edge did not expose CDP');
    ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
    await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
    let seq=0;const pending=new Map();
    ws.onmessage=e=>{
      const m=JSON.parse(e.data);
      if(m.id){const p=pending.get(m.id);if(!p)return;clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}
      else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);
      else if(m.method==='Runtime.consoleAPICalled'&&m.params.type==='error')errors.push('console.error');
      else if(m.method==='Network.requestWillBeSent')requests.push(m.params.request.url);
      else if(m.method==='Network.responseReceived'&&m.params.response.status>=400)responseErrors.push(m.params.response.url);
    };
    send=(method,params={})=>new Promise((resolve,reject)=>{
      const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout: '+method));},10000);
      pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));
    });
    const ev=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
    for(const method of ['Page.enable','Runtime.enable','Network.enable'])await send(method);
    if(u.protocol==='file:')await send('Network.setBlockedURLs',{urls:['http://*','https://*']});
    const navigate=async url=>{
      await send('Page.navigate',{url});
      for(let i=0;i<100;i++){
        if(await ev('document.readyState==="complete"&&location.href==='+JSON.stringify(url)))return;
        await wait(60);
      }throw Error('Navigation incomplete: '+url);
    };
    const shot=async name=>{const r=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(out,name+'.png'),Buffer.from(r.data,'base64'));};
    for(const lang of ['ko','en'])for(const page of ['index','llm-guide','qwen3-4b','measurements','guide'])for(const width of [390,768,1280]){
      await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
      await navigate(new URL(lang+'/'+page+'.html',base).href);await wait(120);
      const label=lang+'/'+page+' '+width;
      expect(await ev('document.documentElement.scrollWidth<=innerWidth+1'),label+' no page overflow');
      expect(await ev('document.querySelectorAll("h1").length===1'),label+' single h1');
      expect(await ev('[...document.querySelectorAll("a")].every(a=>a.textContent.trim().length>0)'),label+' named links');
      expect(await ev('[...document.querySelectorAll(".guide-table")].every(e=>e.tabIndex===0&&e.getAttribute("aria-label")&&getComputedStyle(e).overflowX==="auto")'),label+' accessible table scrolling');
      expect(await ev('[...document.querySelectorAll(".guide-table th,.guide-table td")].every(e=>parseFloat(getComputedStyle(e).fontSize)>=15)'),label+' readable table text at least 15px');
      expect(await ev('document.documentElement.lang==='+JSON.stringify(lang)),label+' document language');
      expect(await ev('[...document.querySelectorAll(".site-header nav a")].map(a=>a.getAttribute("href")).join("|")==="index.html|llm-guide.html|index.html#cases|guide.html"'),label+' common role navigation');
      await ev('document.body.tabIndex=-1;document.body.focus()');
      await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Tab',code:'Tab',windowsVirtualKeyCode:9});
      await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Tab',code:'Tab',windowsVirtualKeyCode:9});
      expect(await ev('document.activeElement.classList.contains("skip")&&getComputedStyle(document.activeElement).outlineStyle!=="none"'),label+' visible keyboard skip focus');
      await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Tab',code:'Tab',windowsVirtualKeyCode:9});
      await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Tab',code:'Tab',windowsVirtualKeyCode:9});
      expect(await ev('document.activeElement.classList.contains("brand")'),label+' keyboard reaches home');
      await ev('document.activeElement.blur();scrollTo(0,0)');
      if(lang==='ko')await shot(page+'-'+width);
      if(page==='index'){
        expect(await ev('(()=>{const nodes=["featured-stories","model-guides","contributions","capabilities","tech-stack","cases"].map(id=>document.getElementById(id));return nodes.every((node,i)=>node&&(!i||nodes[i-1].compareDocumentPosition(node)&Node.DOCUMENT_POSITION_FOLLOWING))})()'),label+' reader order');
        if(lang==='ko')for(const id of ['featured-stories','model-guides']){await ev('document.getElementById('+JSON.stringify(id)+').scrollIntoView()');await shot(id+'-'+width);}
      }
      if(page==='qwen3-4b'&&lang==='ko'){
        await ev('document.querySelector("#configurations details summary").focus()');
        await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r',unmodifiedText:'\r'});
        await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
        await wait(80);
        expect(await ev('document.querySelector("#configurations details").open'),label+' disclosure opens by keyboard');
        await ev('document.querySelector("#configurations details").scrollIntoView()');await shot('qwen3-4b-disclosure-'+width);
        await ev('document.querySelector("#device-memory").scrollIntoView()');await shot('qwen3-4b-memory-'+width);
        await ev('document.querySelector("#quality").scrollIntoView()');await shot('qwen3-4b-quality-'+width);
        const scrolled=await ev('(()=>{const e=document.querySelector("#quality .scroll");e.scrollLeft=150;return e.scrollWidth>e.clientWidth?e.scrollLeft>0:true})()');
        expect(scrolled,label+' quality table scrolls within its region');
      }
      const target=await ev('document.getElementById("language").href');await navigate(target);
      expect(await ev('document.documentElement.lang')===(lang==='ko'?'en':'ko'),label+' language destination loads');
    }
    expect(errors.length===0,'No browser runtime/console errors');
    expect(responseErrors.length===0,'No HTTP response errors');
    expect(requests.every(r=>r.startsWith(base)||r==='about:blank'||r.startsWith('data:')),'Only local runtime resources requested');
    fs.writeFileSync(path.join(out,'browser-report.json'),JSON.stringify({status:'PASS',browser:version.Browser,base,
      viewport_widths:[390,768,1280],checks,errors,responseErrors,requests:[...new Set(requests)],
      mode:'desktop viewport emulation',real_iPad:'NOT_TESTED',reader_study:'NOT_RUN',visual_review:'SEPARATE_MANUAL_REVIEW'},null,2)+'\n');
    console.log(JSON.stringify({status:'PASS',checks:checks.length,screenshots:fs.readdirSync(out).filter(x=>x.endsWith('.png')).length,browser:version.Browser}));
  }catch(e){fs.writeFileSync(path.join(out,'browser-report.json'),JSON.stringify({status:'FAIL',error:String(e),checks,errors,responseErrors},null,2)+'\n');throw e;}
  finally{if(send){try{await send('Browser.close');}catch{}}if(ws)ws.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
