/* Scalar reconstruction of all affinities, normalized rows, eigenpair residuals and cut witnesses. */
(function(root){
'use strict';
function graph(points,config){
 const n=points.length,w=Array.from({length:n},()=>Array(n).fill(0));
 if(config.kind==='rbf'){
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){
   const d=points[i].reduce((s,v,k)=>s+(v-points[j][k])**2,0);
   w[i][j]=w[j][i]=Math.exp(-d/(2*config.sigma**2));
  }
 }else{
  const near=points.map((p,i)=>new Set(points.map((q,j)=>[j,p.reduce((s,v,k)=>s+(v-q[k])**2,0)])
   .filter(([j])=>j!==i).sort((a,b)=>a[1]-b[1]||a[0]-b[0]).slice(0,config.neighbors).map(a=>a[0])));
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)w[i][j]=w[j][i]=Number(config.kind==='union'?(near[i].has(j)||near[j].has(i)):(near[i].has(j)&&near[j].has(i)));
 }
 return w;
}
function components(w){
 const unseen=new Set(w.map((_,i)=>i)),out=[];
 while(unseen.size){const start=Math.min(...unseen),todo=[start],group=[];unseen.delete(start);
  while(todo.length){const i=todo.pop();group.push(i);for(const j of unseen)if(w[i][j]>0){unseen.delete(j);todo.push(j);}}
  out.push(group.sort((a,b)=>a-b));
 }return out;
}
function cut(w,labels){
 const d=w.map(row=>row.reduce((s,v)=>s+v,0)),volumes=[0,0];let weight=0;
 for(let i=0;i<w.length;i++){volumes[labels[i]]+=d[i];for(let j=i+1;j<w.length;j++)if(labels[i]!==labels[j])weight+=w[i][j];}
 return {cut_weight:weight,volumes,normalized_cut:weight*(1/volumes[0]+1/volumes[1])};
}
function witness(w,fit,node){
 const degrees=w.map(row=>row.reduce((s,v)=>s+v,0));
 const neighbors=w[node].map((weight,j)=>({node:j,weight,normalized:degrees[node]>0&&degrees[j]>0?weight/Math.sqrt(degrees[node]*degrees[j]):null,
  crosses:fit.status==='ok'?fit.fit.labels[node]!==fit.fit.labels[j]:null})).filter(v=>v.weight>0).sort((a,b)=>b.weight-a.weight||a.node-b.node);
 const modes=fit.eigenvectors?fit.eigenvalues.slice(0,3).map((value,c)=>{
  let lhs=fit.eigenvectors[node][c];
  for(let j=0;j<w.length;j++)if(w[node][j]>0)lhs-=w[node][j]/Math.sqrt(degrees[node]*degrees[j])*fit.eigenvectors[j][c];
  const rhs=value*fit.eigenvectors[node][c];return {mode:c,eigenvalue:value,lhs,rhs,residual:lhs-rhs};
 }):[];
 return {node,degree:degrees[node],neighbors,modes,status:fit.status,cluster:fit.status==='ok'?fit.fit.labels[node]:null};
}
function ari(truth,labels){
 let a=0,b=0,both=0,total=0;for(let i=0;i<labels.length;i++)for(let j=i+1;j<labels.length;j++){
 const ta=truth[i]===truth[j],tb=labels[i]===labels[j];a+=Number(ta);b+=Number(tb);both+=Number(ta&&tb);total++;}
 const e=a*b/total,den=(a+b)/2-e;return den===0?1:(both-e)/den;
}
const api={graph,components,cut,witness,ari};root.Spectral=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
