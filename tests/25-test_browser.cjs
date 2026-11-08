'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
const {waitForDebugger}=require('./chrome_support.js');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
test('real Chrome18graphstates1728pointwitnesses, physical keyboard, refusal, clipboard, realBlob and mobile', {timeout:90000}, async()=>{
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'spectrallab-'));
 const browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-extensions','--disable-dev-shm-usage','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});let ws,startupError,diagnostics='';browser.on('error',e=>startupError=e);browser.stderr.on('data',d=>diagnostics=(diagnostics+d).slice(-4000));
 try{
  for(let i=0;i<1200&&!startupError&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);
  assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),`Chrome debugger did not start: ${startupError?.message||diagnostics||browser.exitCode}`);const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];
  const tab=await waitForDebugger(`http://127.0.0.1:${port}/json/list`,{alive:()=>browser.exitCode===null&&!startupError});ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});
  let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const {res,rej}=pending.get(m.id);pending.delete(m.id);m.error?rej(new Error(JSON.stringify(m.error))):res(m.result);}};
  const call=(method,params={})=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});ws.send(JSON.stringify({id,method,params}));});
  const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  await call('Page.enable');await call('Page.navigate',{url:process.env.SPECTRAL_URL||'file://'+path.resolve(__dirname,'../docs/index.html')});
  for(let i=0;i<250;i++){if(await js(`document.body?.dataset.ready==='true'`))break;await delay(20);}


  assert.equal(await js(`document.body.dataset.ready`),'true');
  assert.equal(await js(`(()=>{let count=0;for(const name of ['moons','circles'])for(const c of SPECTRAL_META.configurations){
   document.getElementById('dataset').value=name;document.getElementById('configuration').value=c.id;document.getElementById('dataset').dispatchEvent(new Event('input'));
   const d=globalThis['SPECTRAL_'+name],f=d.fits.find(x=>x.id===c.id);
   for(let i=0;i<96;i++){document.getElementById('node').value=String(i);document.getElementById('node').dispatchEvent(new Event('input'));const w=JSON.parse(document.getElementById('witness').textContent);if(w.dataset!==name||w.configuration.id!==c.id||w.point.node!==i||w.point.status!==f.status)throw Error('identity');if(w.point.modes.some(m=>Math.abs(m.residual)>1e-9))throw Error('eigenpair');if(f.status!=='ok'&&(w.point.cluster!==null||w.cut!==null))throw Error('refusal');count++;}
   }return count;})()`),1728);
  await js(`document.getElementById('dataset').value='moons';document.getElementById('configuration').value='mutual-3';document.getElementById('dataset').dispatchEvent(new Event('input'))`);
  assert.match(await js(`document.getElementById('status').textContent`),/refused-components/);assert.equal(await js(`document.querySelectorAll('#embedding circle').length`),0);
  await js(`document.getElementById('configuration').value='union-6';document.getElementById('configuration').dispatchEvent(new Event('input'));document.getElementById('node').value='0';document.getElementById('node').dispatchEvent(new Event('input'));document.getElementById('threshold').focus()`);
  const before=await js(`JSON.parse(document.getElementById('witness').textContent)`);
  await call('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});await call('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});
  const after=await js(`JSON.parse(document.getElementById('witness').textContent)`);assert.notEqual(after.display_threshold_only,before.display_threshold_only);assert.deepEqual(after.point,before.point);assert.deepEqual(after.cut,before.cut);
  await js(`document.querySelector('#graph circle[data-node="7"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))`);assert.equal(await js(`JSON.parse(document.getElementById('witness').textContent).point.node`),7);
  for(const color of ['truth','raw','spectral']){await js(`document.getElementById('colors').value='${color}';document.getElementById('colors').dispatchEvent(new Event('input'))`);assert.equal(await js(`document.querySelectorAll('#graph circle').length`),96);}
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async()=>{throw Error('denied')}},configurable:true});document.getElementById('copy').click()`);await delay(10);assert.match(await js(`document.getElementById('copy-status').textContent`),/Clipboard unavailable/);
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async t=>globalThis.copied=t},configurable:true});document.getElementById('copy').click()`);await delay(10);assert.deepEqual(await js(`JSON.parse(globalThis.copied)`),await js(`JSON.parse(document.getElementById('witness').textContent)`));
  await js(`HTMLAnchorElement.prototype.click=function(){globalThis.downloaded={name:this.download,url:this.href}};document.getElementById('download').click()`);assert.equal(await js(`globalThis.downloaded.name`),'spectrallab-witness.json');assert.deepEqual(await js(`fetch(globalThis.downloaded.url).then(r=>r.json())`),await js(`JSON.parse(document.getElementById('witness').textContent)`));
  for(const width of [320,390,1280]){await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});await delay(20);assert.ok(await js(`document.documentElement.scrollWidth<=innerWidth`),'overflow at '+width);assert.ok(await js(`Array.from(document.querySelectorAll('button,select,input')).every(e=>e.getBoundingClientRect().height>=44)`),'touch at '+width);}
  if(process.env.SPECTRAL_SCREENSHOT){const width=Number(process.env.SPECTRAL_SCREENSHOT_WIDTH||390);await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});await delay(20);const layout=await call('Page.getLayoutMetrics');const h=layout.cssContentSize?.height||layout.contentSize.height;const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:h,scale:1}});fs.writeFileSync(process.env.SPECTRAL_SCREENSHOT,Buffer.from(shot.data,'base64'));}
 }finally{if(ws)ws.close();if(browser.pid&&browser.exitCode===null){browser.kill('SIGTERM');await new Promise(resolve=>browser.once('exit',resolve));}await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});}
});
