'use strict';
(function(root){
const zeros=n=>Array.from({length:n},()=>Array(n).fill(0));
const transpose=a=>a[0].map((_,i)=>a.map(r=>r[i]));
const multiply=(a,b)=>a.map(r=>b[0].map((_,j)=>r.reduce((s,x,k)=>s+x*b[k][j],0)));
const norm=a=>Math.sqrt(a.reduce((s,r)=>s+r.reduce((t,x)=>t+x*x,0),0));
const subtract=(a,b)=>a.map((r,i)=>r.map((x,j)=>x-b[i][j]));
function graph(c,t){const w=c.weights.map(r=>r.slice()),[a,b]=c.edge;w[a][b]+=t;w[b][a]+=t;return w;}
function laplacian(w){const d=w.map(r=>r.reduce((s,x)=>s+x,0));if(d.some(x=>x<=0))return null;return w.map((r,i)=>r.map((x,j)=>(i===j?1:0)-x/Math.sqrt(d[i]*d[j])));}
function tangent(c){
 const w=c.weights,n=w.length,d=w.map(r=>r.reduce((s,x)=>s+x,0)),[a,b]=c.edge,s=Array(n).fill(0);s[a]=s[b]=1;
 if(d.some(x=>x<=0))throw Error('isolated vertex');
 const dl=w.map((r,i)=>r.map((x,j)=>{const edge=(i===a&&j===b)||(i===b&&j===a)?1:0;return -edge/Math.sqrt(d[i]*d[j])+.5*x/Math.sqrt(d[i]*d[j])*(s[i]/d[i]+s[j]/d[j]);}));
 const q=c.witness.eigenvectors,v=c.witness.eigenvalues,p=zeros(n),dp=zeros(n);
 for(let i=0;i<n;i++)for(let j=0;j<n;j++)for(let k=0;k<2;k++)p[i][j]+=q[i][k]*q[j][k];
 for(let i=0;i<2;i++)for(let j=2;j<n;j++){
  let coupling=0;for(let x=0;x<n;x++)for(let y=0;y<n;y++)coupling+=q[x][j]*dl[x][y]*q[y][i];
  const coefficient=coupling/(v[i]-v[j]);
  for(let x=0;x<n;x++)for(let y=0;y<n;y++)dp[x][y]+=coefficient*(q[x][j]*q[y][i]+q[x][i]*q[y][j]);
 }
 return {laplacian_tangent:dl,projector:p,projector_tangent:dp};
}
function fixedCut(w,labels){const d=w.map(r=>r.reduce((s,x)=>s+x,0)),v=[0,0];for(let i=0;i<w.length;i++)v[labels[i]]+=d[i];let cut=0;for(let i=0;i<w.length;i++)for(let j=0;j<w.length;j++)if(labels[i]===0&&labels[j]===1)cut+=w[i][j];return {cut,volumes:v,value:cut*(1/v[0]+1/v[1])};}
function fixedCutTangent(c){const y=c.base_fit.fit.labels,b=fixedCut(c.weights,y),[a,d]=c.edge,dv=[0,0];dv[y[a]]++;dv[y[d]]++;const cross=y[a]!==y[d]?1:0;return cross*(1/b.volumes[0]+1/b.volumes[1])-b.cut*(dv[0]/b.volumes[0]**2+dv[1]/b.volumes[1]**2);}
function witness(c,step,row){
 const w=graph(c,step),base={case_id:c.id,edge:c.edge,step,row,weights:w,status:c.status,scope:root.SPECTRAL_SENSITIVITY?.scope||'Local fixed graph-weight response, not a cluster-label derivative.'};
 if(c.status!=='ok')return {...base,reason:c.reason,projector:null,projector_tangent:null,actual_delta:null,linear_delta:null,matrix_row:[],state:null};
 const state=c.states.find(s=>s.step===step);if(!state)throw Error('unarchived step');
 const calculated=tangent(c),p=calculated.projector,dp=calculated.projector_tangent;
 const linear=dp.map(r=>r.map(x=>step*x)),actual=state.projector?subtract(state.projector,p):null;
 const values=p[row].map((x,j)=>({column:j,base:x,tangent:dp[row][j],actual:actual?actual[row][j]:null,linear:linear[row][j],error:actual?actual[row][j]-linear[row][j]:null}));
 const scalarCut=fixedCut(w,c.base_fit.fit.labels),dc=fixedCutTangent(c);
 const nearTies=state.fit.status==='ok'?new Set(state.fit.fit.restarts.filter(r=>r.status==='ok'&&Math.abs(r.inertia-state.fit.fit.inertia)<=1e-12*(1+Math.abs(state.fit.fit.inertia))).map(r=>r.labels.join(','))).size:null;
 return {...base,status:state.fit.status,projector:p,projector_tangent:dp,laplacian_tangent:calculated.laplacian_tangent,
  actual_delta:actual,linear_delta:linear,matrix_row:values,state,near_tied_tested_partitions:nearTies,
  scalar_checks:{projector_distance:actual?norm(actual):null,linear_projector_distance:norm(linear),
   linearization_error:actual?norm(subtract(actual,linear)):null,fixed_cut:scalarCut.value,fixed_cut_tangent:dc,
   fixed_cut_linear:fixedCut(c.weights,c.base_fit.fit.labels).value+step*dc}};
}
const api={zeros,transpose,multiply,norm,subtract,graph,laplacian,tangent,fixedCut,fixedCutTangent,witness};
if(typeof module==='object'&&module.exports)module.exports=api;root.SpectralSensitivity=api;
})(globalThis);
