'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const S=require('../docs/inference.js'),R=JSON.parse(fs.readFileSync(path.join(__dirname,'../examples/report.json')));
const close=(a,b,tol=2e-10)=>assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} != ${b}`);
test('all18 scalar graphs/degrees/components,1632eigenpairrows,18rawcuts and17spectralcuts',()=>{
 let graphs=0,rows=0,cuts=0;
 for(const d of Object.values(R.datasets))for(let f=0;f<d.fits.length;f++){
  const fit=d.fits[f],w=S.graph(d.points,R.configurations[f]);
  assert.deepEqual(S.components(w),fit.components);const degrees=w.map(r=>r.reduce((s,v)=>s+v,0));degrees.forEach((v,i)=>close(v,fit.degrees[i]));
  const raw=S.cut(w,d.raw.labels);close(raw.normalized_cut,fit.raw_cut.normalized_cut);close(S.ari(d.truth,d.raw.labels),d.raw.ari);
  if(fit.eigenvectors)for(let i=0;i<d.points.length;i++){
   const witness=S.witness(w,fit,i);for(const m of witness.modes){assert.ok(Math.abs(m.residual)<1e-9);rows++;}
   close(witness.degree,fit.degrees[i]);
  }
  if(fit.status==='ok'){const cut=S.cut(w,fit.fit.labels);close(cut.normalized_cut,fit.cut.normalized_cut);close(cut.cut_weight,fit.cut.cut_weight);close(S.ari(d.truth,fit.fit.labels),fit.ari);cuts++;}
  graphs++;
 }
 assert.equal(graphs,18);assert.equal(rows,4896);assert.equal(cuts,17);
});
test('knn tied-distance rank and union/mutual distinction independent rational integer control',()=>{
 const p=[[0,0],[1,0],[-1,0],[0,1],[0,-1]];
 assert.deepEqual(S.graph(p,{kind:'union',neighbors:1}),[[0,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]]);
 assert.deepEqual(S.graph(p,{kind:'mutual',neighbors:1}),[[0,1,0,0,0],[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]);
});
test('explicit zero-degree refusal has no invented normalized weights',()=>{
 const w=[[0,0,0],[0,0,1],[0,1,0]],f={status:'refused-isolated'};
 assert.deepEqual(S.components(w),[[0],[1,2]]);assert.equal(S.witness(w,f,0).degree,0);assert.equal(S.witness(w,f,0).cluster,null);assert.deepEqual(S.witness(w,f,0).modes,[]);
});
test('all downloaded dataset scripts exactly match checked Python archive',()=>{
 for(const [name,d]of Object.entries(R.datasets)){require('../docs/'+name+'.js');assert.deepEqual(globalThis['SPECTRAL_'+name],d);}
 require('../docs/data.js');assert.deepEqual(globalThis.SPECTRAL_META.configurations,R.configurations);
});
test('seven-vertex rounding counterexample:63independentBigIntfractioncuts, not spectraloptimality',()=>{
 const r=JSON.parse(fs.readFileSync(path.join(__dirname,'../examples/rounding.json'))),w=r.weights;let best=null,labels=null,count=0;
 const cmp=(a,b)=>a[0]*b[1]-b[0]*a[1];
 for(let mask=1;mask<64;mask++){const y=[0,...Array.from({length:6},(_,i)=>(mask>>(5-i))&1)];let cut=0n,v=[0n,0n];for(let i=0;i<7;i++)for(let j=0;j<7;j++){v[y[i]]+=BigInt(w[i][j]);if(i<j&&y[i]!==y[j])cut+=BigInt(w[i][j]);}const score=[cut*(v[0]+v[1]),v[0]*v[1]];if(best===null||cmp(score,best)<0){best=score;labels=y;}count++;}
 const [num,den]=r.exact.normalized_cut.split('/').map(BigInt);assert.equal(best[0]*den,best[1]*num);assert.deepEqual(labels,r.exact.labels);assert.equal(count,63);
 close(S.cut(w,r.spectral.fit.labels).normalized_cut,r.spectral.cut.normalized_cut);assert.ok(r.spectral.cut.normalized_cut>Number(num)/Number(den));require('../docs/rounding.js');assert.deepEqual(globalThis.SPECTRAL_ROUNDING,r);
});
