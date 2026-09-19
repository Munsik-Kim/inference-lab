/* Optional check with an already installed Windows Edge and Node >=22. No packages. */
'use strict';
const fs=require('fs'),os=require('os'),path=require('path'),cp=require('child_process');
(async()=>{
  const base=process.argv[2],out=process.argv[3];
  if(!base||!out||fs.existsSync(out))throw Error('Supply site base URL and a NEW external output directory');
  const u=new URL(base);if(u.protocol!=='file:'&&!(u.protocol==='http:'&&['127.0.0.1','localhost'].includes(u.hostname)))throw Error('Only file or loopback preview');
  fs.mkdirSync(out,{recursive:true});
  try{await fetch('http://127.0.0.1:9238/json/version',{signal:AbortSignal.timeout(500)});throw Error('Existing CDP port occupied');}catch(e){if(e.message==='Existing CDP port occupied')throw e;}
  const profile=fs.mkdtempSync(path.join(os.tmpdir(),'inference-showcase-')),downloads=path.join(profile,'downloads');fs.mkdirSync(downloads);
  cp.spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=9238','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
  let send,ws;const errors=[],requests=[],checks=[],responseErrors=[];let version;
  const wait=ms=>new Promise(r=>setTimeout(r,ms));
  try{
    let targets;for(let i=0;i<80;i++){try{version=await(await fetch('http://127.0.0.1:9238/json/version')).json();targets=await(await fetch('http://127.0.0.1:9238/json/list')).json();if(targets.some(t=>t.type==='page'))break;}catch{}await wait(150);}
    ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
    let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);else if(m.method==='Network.requestWillBeSent')requests.push(m.params.request.url);else if(m.method==='Network.responseReceived' && m.params.response.status>=400)responseErrors.push({url:m.params.response.url,status:m.params.response.status});else if(m.method==='Runtime.consoleAPICalled'&&m.params.type==='error')errors.push('console.error');};
    send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
    const ev=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
    const expect=(value,message)=>{if(!value)throw Error(message);checks.push(message);};
    for(const method of ['Page.enable','Runtime.enable','Network.enable'])await send(method);
    if(u.protocol==='file:')await send('Network.setBlockedURLs',{urls:['http://*','https://*']});
    await send('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloads});
    const navigate=async(url)=>{await send('Page.navigate',{url});for(let i=0;i<100;i++){if(await ev('document.readyState==="complete" && !!document.querySelector("#items")?.options.length'))return;await wait(80);}throw Error('Load failed: '+url);};
    for(const lang of ['ko','en'])for(const c of ['case007','case006']){
      await navigate(new URL(lang+'/'+c+'.html',base).href);
      expect(await ev('document.getElementById("items").options.length')===192,lang+c+' initial 192');
      expect(await ev('document.querySelectorAll(".study-section").length')===8,'Static study loaded '+lang+c);
      expect(await ev('[...document.querySelectorAll("#arm option")].every(o=>o.textContent.includes(" · "))'),'Descriptive method names '+lang+c);
      for(const width of [390,768,1280]){
        await send('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:false});await wait(70);
        const dimensions=await ev('({width:innerWidth,scroll:document.documentElement.scrollWidth,labels:[...document.querySelectorAll("input,select")].every(e=>e.labels.length>0)})');
        expect(dimensions.scroll<=dimensions.width+2&&dimensions.labels,lang+c+' '+width+' layout and labels');
        if(lang==='ko'){await ev('scrollTo(0,0)');const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,c+'-intro-'+width+'.png'),Buffer.from(shot.data,'base64'));}
        if(lang==='ko'){await ev('document.querySelector(".study-table").scrollIntoView()');const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,c+'-methods-'+width+'.png'),Buffer.from(shot.data,'base64'));await ev('scrollTo(0,0)');}
      }
      await ev('document.getElementById("query").value="not-an-input";document.getElementById("query").dispatchEvent(new Event("input"))');
      expect(await ev('document.getElementById("items").options.length')===0,'Empty filter '+lang+c);
      await ev('document.getElementById("reset").click();document.getElementById("task").value="CODE";document.getElementById("task").dispatchEvent(new Event("change"))');
      expect(await ev('document.getElementById("items").options.length')===64,'Task filter '+lang+c);
      await ev('document.getElementById("items").selectedIndex=2;document.getElementById("items").dispatchEvent(new Event("change"))');
      const id=await ev('document.getElementById("items").value'),hash=await ev('location.hash');
      await send('Page.reload');await wait(350);expect(await ev('document.getElementById("items").value')===id,'Reload ID '+lang+c);
      await navigate(await ev('document.getElementById("language").href'));expect(await ev('document.getElementById("items").value')===id,'Language retains ID '+lang+c);
      await navigate(new URL(lang+'/'+c+'.html'+hash,base).href);
      const order=await ev('[...document.querySelectorAll(".study-section")].map(e=>e.id)');
      expect(JSON.stringify(order)===JSON.stringify(['objective','data','theory','design','validation','results','interpretation','conclusion']),'Eight study sections '+lang+c);
      await ev('document.querySelector(".study-toc").querySelectorAll("a")[5].click()');await wait(100);
      expect(await ev('document.getElementById("items").value')===id,'Contents link preserves item '+lang+c);
      expect(await ev('location.hash')==='#results','Results anchor '+lang+c);
      await send('Page.reload');await wait(350);
      expect(await ev('document.getElementById("items").value')===id,'Section reload preserves item '+lang+c);
      await navigate(await ev('document.getElementById("language").href'));
      expect(await ev('location.hash')==='#results'&&await ev('document.getElementById("items").value')===id,'Language retains section and item '+lang+c);
      await navigate(new URL(lang+'/'+c+'.html'+hash,base).href);
      // Synthetic XSS probe in page memory only; never written to source or measurements.
      await ev('window.__probe=undefined;window.EVIDENCE.prompts[document.getElementById("items").value].text="<script>window.__probe=1</script>";document.getElementById("items").dispatchEvent(new Event("change"))');
      expect(await ev('window.__probe===undefined && document.getElementById("prompt").textContent.includes("<script>") && document.getElementById("prompt").querySelector("script")===null'),'Text rendering XSS '+lang+c);
      await navigate(new URL(lang+'/'+c+'.html'+hash,base).href);
      await ev('document.getElementById("query").focus()');await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Tab',code:'Tab',windowsVirtualKeyCode:9});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Tab',code:'Tab',windowsVirtualKeyCode:9});
      expect(await ev('document.activeElement.id')==='reset','Keyboard Tab '+lang+c);
      if(c==='case006'){
        await ev('document.getElementById("readout").value="H_FP32";document.getElementById("readout").dispatchEvent(new Event("change"))');
        expect(await ev('JSON.parse(document.getElementById("record").textContent).focus.readout')==='H_FP32','Shadow view '+lang);
        expect(await ev('document.getElementById("view-note").classList.contains("posthoc")'),'Persistent posthoc label '+lang);
        await ev('document.getElementById("set").value="boundary_pool";document.getElementById("set").dispatchEvent(new Event("change"));document.getElementById("task").value="ALL";document.getElementById("task").dispatchEvent(new Event("change"))');
        expect(await ev('document.getElementById("items").options.length')===46,'Stress separate 46 '+lang);
      }else{
        await ev('document.getElementById("budget").value="8";document.getElementById("budget").dispatchEvent(new Event("change"))');
        const records=await ev('JSON.parse(document.getElementById("record").textContent).paired_comparisons');
        expect(records[0].local_error===records[1].local_error&&JSON.stringify(records[0].selected_groups)===JSON.stringify(records[1].selected_groups),'50% identical sets '+lang);
      }
      await ev('location.hash="set=invalid"');await wait(50);expect(await ev('document.getElementById("items").options.length')===0,'Invalid set rejected '+lang+c);
      await navigate(new URL(lang+'/'+c+'.html',base).href);
      if(lang==='ko'){
        if(c==='case006')await ev('document.getElementById("readout").value="H_FP32";document.getElementById("readout").dispatchEvent(new Event("change"))');
        // First fixed comparison item; screenshots are not chosen for a dramatic result.
        await ev('document.getElementById("results").scrollIntoView()');await wait(80);const resultShot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,c+'-results.png'),Buffer.from(resultShot.data,'base64'));
        await ev('document.getElementById("detail").scrollIntoView()');
        const screenshot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(out,c+'.png'),Buffer.from(screenshot.data,'base64'));
      }
    }
    const before=new Set(fs.readdirSync(downloads));
    const e=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
    const expected=await e('JSON.parse(document.getElementById("record").textContent).focus.id');await e('document.getElementById("download").click()');
    let downloaded;for(let i=0;i<80;i++){downloaded=fs.readdirSync(downloads).find(n=>n.endsWith('.json')&&!before.has(n));if(downloaded)break;await wait(100);}
    expect(!!downloaded&&JSON.parse(fs.readFileSync(path.join(downloads,downloaded),'utf8')).focus.id===expected,'Measured record JSON download');
    for(const lang of ['en','ko'])for(const page of ['index','guide']){
      await send('Page.navigate',{url:new URL(lang+'/'+page+'.html',base).href});await wait(200);
      expect(await e('document.readyState==="complete" && document.querySelector("h1").textContent.length>0'),'Home/guide loaded '+lang+page);
      for(const width of [390,768,1280]){
        await send('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:false});await wait(30);
        expect(await e('document.documentElement.scrollWidth<=innerWidth+2'),'Home/guide layout '+lang+page+width);
      }
    }
    const forbidden=requests.filter(x=>/^https?:/.test(x)&&(u.protocol==='file:'||new URL(x).origin!==u.origin));
    expect(errors.length===0,'No console exceptions/errors');expect(responseErrors.length===0,'No HTTP resource failures');expect(forbidden.length===0,'No external runtime requests');
    fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({status:'PASS',browser:version.Browser,node:process.version,os:process.platform,protocol:u.protocol,method:'existing Edge, CDP, fresh temporary profile',viewports:[390,768,1280],checks,console_errors:errors,external_requests:forbidden,http_resource_failures:responseErrors,screenshots:fs.readdirSync(out).filter(name=>name.endsWith('.png')).sort(),real_ipad:'NOT_TESTED'},null,2));
    console.log('PASS '+checks.length+' browser assertions');
  }catch(e){fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({status:'FAIL',error:e.message,checks,console_errors:errors,http_resource_failures:responseErrors},null,2));throw e;}
  finally{if(send)try{await send('Browser.close');}catch{}if(ws)ws.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
