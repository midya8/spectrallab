'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const S=require('../docs/sensitivity-scalar.js');
const D=JSON.parse(fs.readFileSync(path.join(__dirname,'../examples/sensitivity.json')));
const near=(a,b,tol=2e-9)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const matrixNear=(a,b)=>{assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)for(let j=0;j<a.length;j++)near(a[i][j],b[i][j]);};
test('scalar degree response and all cross-boundary contractions, projector identities and full eigenpairs',()=>{
 let equations=0,entries=0;
 for(const c of D.cases){if(c.status!=='ok')continue;
  const n=c.weights.length,q=c.witness.eigenvectors,v=c.witness.eigenvalues,lap=S.laplacian(c.weights),t=S.tangent(c);
  matrixNear(t.laplacian_tangent,c.laplacian_tangent);matrixNear(t.projector,c.witness.projector);matrixNear(t.projector_tangent,c.witness.projector_tangent);
  for(let i=0;i<n;i++)for(let k=0;k<n;k++){let left=0;for(let j=0;j<n;j++)left+=lap[i][j]*q[j][k];near(left,v[k]*q[i][k]);equations++;}
  const pp=S.multiply(t.projector,t.projector),pdp=S.multiply(t.projector,t.projector_tangent),dpp=S.multiply(t.projector_tangent,t.projector),adp=S.multiply(lap,t.projector_tangent),dpa=S.multiply(t.projector_tangent,lap),pe=S.multiply(t.projector,t.laplacian_tangent),ep=S.multiply(t.laplacian_tangent,t.projector);
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){near(pp[i][j],t.projector[i][j]);near(pdp[i][j]+dpp[i][j],t.projector_tangent[i][j]);near(adp[i][j]-dpa[i][j],pe[i][j]-ep[i][j]);entries++;}
  near(S.fixedCutTangent(c),c.fixed_cut_tangent.normalized_cut_derivative);
 }
 assert.equal(equations,146);assert.equal(entries,146);
});
test('all28actualstates, scalar matrix rows/cuts/distances and exact archived script',()=>{
 let states=0,rows=0;
 for(const c of D.cases){if(c.status!=='ok'){assert.equal(S.witness(c,0,0).projector,null);continue;}
  for(const state of c.states){
   for(let row=0;row<c.weights.length;row++){
    const w=S.witness(c,state.step,row);near(w.scalar_checks.fixed_cut,state.fixed_cut);near(w.scalar_checks.fixed_cut_linear,state.fixed_cut_linear);
    for(const key of ['projector_distance','linear_projector_distance','linearization_error'])if(state[key]===null)assert.equal(w.scalar_checks[key],null);else near(w.scalar_checks[key],state[key]);
    for(const entry of w.matrix_row){near(entry.base,c.witness.projector[row][entry.column]);near(entry.tangent,c.witness.projector_tangent[row][entry.column]);if(state.projector)near(entry.actual,state.projector[row][entry.column]-entry.base);else assert.equal(entry.actual,null);rows++;}
    if(state.fit.status==='ok'){const u=state.fit.eigenvectors,p=u.map(a=>u.map(b=>a[0]*b[0]+a[1]*b[1]));matrixNear(p,state.projector);near(S.fixedCut(w.weights,state.fit.fit.labels).value,state.fit.cut.normalized_cut);}
   }states++;
  }
 }
 assert.equal(states,28);assert.equal(rows,1022);
 const raw=fs.readFileSync(path.join(__dirname,'../examples/sensitivity.json'),'utf8').trim();assert.equal(fs.readFileSync(path.join(__dirname,'../docs/sensitivity-data.js'),'utf8'),'globalThis.SPECTRAL_SENSITIVITY='+raw+';\n');
});
