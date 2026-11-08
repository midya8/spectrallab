'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
const {waitForDebugger}=require('./chrome_support.js');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
test('real Chrome sensitivity28states, 168rowstates, finite tie/refusal, physical keyboard, clipboard, Blob and mobile', {timeout:90000}, async()=>{
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'spectrallab-'));
 const browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-extensions','--disable-dev-shm-usage','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});let ws,startupError,diagnostics='';browser.on('error',e=>startupError=e);browser.stderr.on('data',d=>diagnostics=(diagnostics+d).slice(-4000));
 try{
  for(let i=0;i<1200&&!startupError&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);
  assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),`Chrome debugger did not start: ${startupError?.message||diagnostics||browser.exitCode}`);const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];
  const tab=await waitForDebugger(`http://127.0.0.1:${port}/json/list`,{alive:()=>browser.exitCode===null&&!startupError});ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});
  let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const {res,rej}=pending.get(m.id);pending.delete(m.id);m.error?rej(new Error(JSON.stringify(m.error))):res(m.result);}};
  const call=(method,params={})=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});ws.send(JSON.stringify({id,method,params}));});
  const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  await call('Page.enable');await call('Page.navigate',{url:process.env.SENSITIVITY_URL||'file://'+path.resolve(__dirname,'../docs/sensitivity.html')});
  for(let i=0;i<250;i++){if(await js(`document.body?.dataset.ready==='true'`))break;await delay(20);}



  assert.equal(await js(`document.body.dataset.ready`),'true');
  assert.equal(await js(`(()=>{let count=0;for(const c of SPECTRAL_SENSITIVITY.cases){
   document.getElementById('case').value=c.id;document.getElementById('case').dispatchEvent(new Event('input'));
   if(c.status!=='ok'){const w=JSON.parse(document.getElementById('witness').textContent);if(w.projector!==null||document.querySelectorAll('#matrix-row tbody tr').length||document.querySelectorAll('#actual-matrix rect').length||document.querySelectorAll('#linear-matrix rect').length)throw Error('stale refusal');continue;}
   for(const state of c.states){document.getElementById('step').value=String(state.step);document.getElementById('step').dispatchEvent(new Event('input'));
    for(let row=0;row<c.weights.length;row++){document.getElementById('row').value=String(row);document.getElementById('row').dispatchEvent(new Event('input'));const w=JSON.parse(document.getElementById('witness').textContent);if(w.case_id!==c.id||w.step!==state.step||w.row!==row||w.status!==state.fit.status)throw Error('identity');if(w.actual_delta===null&&document.querySelectorAll('#actual-matrix rect').length)throw Error('finite refusal stale heatmap');if(document.querySelectorAll('#changed-graph circle').length!==c.weights.length)throw Error('nodes');count++;}
   }
  }return count;})()`),168);
  await js(`document.getElementById('case').value='path';document.getElementById('case').dispatchEvent(new Event('input'));document.getElementById('step').value='0.001';document.getElementById('step').dispatchEvent(new Event('input'))`);
  assert.equal(await js(`JSON.parse(document.getElementById('witness').textContent).state.changed_labels_up_to_swap`),1);
  assert.ok(await js(`JSON.parse(document.getElementById('witness').textContent).scalar_checks.projector_distance<.001`));
  await js(`document.getElementById('step').value='1';document.getElementById('step').dispatchEvent(new Event('input'))`);
  assert.match(await js(`document.getElementById('status').textContent`),/FINITE REFIT REFUSED/);
  assert.equal(await js(`document.querySelectorAll('#actual-matrix rect').length`),0);
  assert.ok(await js(`document.querySelectorAll('#linear-matrix rect').length>0`));
  await js(`document.getElementById('step').value='0.001';document.getElementById('step').dispatchEvent(new Event('input'));document.getElementById('step').focus()`);
  await call('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowDown',code:'ArrowDown',windowsVirtualKeyCode:40});await call('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowDown',code:'ArrowDown',windowsVirtualKeyCode:40});
  assert.equal(await js(`JSON.parse(document.getElementById('witness').textContent).step`),.01);
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async()=>{throw Error('denied')}},configurable:true});document.getElementById('copy').click()`);await delay(10);assert.match(await js(`document.getElementById('copy-status').textContent`),/Clipboard unavailable/);
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async t=>globalThis.copied=t},configurable:true});document.getElementById('copy').click()`);await delay(10);assert.deepEqual(await js(`JSON.parse(globalThis.copied)`),await js(`JSON.parse(document.getElementById('witness').textContent)`));
  await js(`HTMLAnchorElement.prototype.click=function(){globalThis.downloaded={name:this.download,url:this.href}};document.getElementById('download').click()`);assert.equal(await js(`globalThis.downloaded.name`),'spectrallab-sensitivity.json');assert.deepEqual(await js(`fetch(globalThis.downloaded.url).then(r=>r.json())`),await js(`JSON.parse(document.getElementById('witness').textContent)`));
  for(const width of [320,390,1280]){await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});await delay(20);assert.ok(await js(`document.documentElement.scrollWidth<=innerWidth`),'overflow at '+width);assert.ok(await js(`Array.from(document.querySelectorAll('button,select')).every(e=>e.getBoundingClientRect().height>=44)`),'touch at '+width);}
  if(process.env.SENSITIVITY_SCREENSHOT){const width=Number(process.env.SENSITIVITY_SCREENSHOT_WIDTH||390);await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});await delay(20);const layout=await call('Page.getLayoutMetrics');const h=layout.cssContentSize?.height||layout.contentSize.height;const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:h,scale:1}});fs.writeFileSync(process.env.SENSITIVITY_SCREENSHOT,Buffer.from(shot.data,'base64'));}
 }finally{if(ws)ws.close();if(browser.pid&&browser.exitCode===null){browser.kill('SIGTERM');await new Promise(resolve=>browser.once('exit',resolve));}await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});}
});
