'use strict';
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const META=globalThis.SPECTRAL_META,DATA={moons:globalThis.SPECTRAL_moons,circles:globalThis.SPECTRAL_circles};
let current=null;
const fmt=x=>Number.isFinite(x)?x.toFixed(6):'not fitted';
for(const c of META.configurations){const o=new Option(c.id,c.id);$('configuration').append(o);}
$('configuration').value='union-6';
for(let i=0;i<96;i++)$('node').append(new Option('Point '+i,String(i)));
for(const [name,d] of Object.entries(DATA))for(const f of d.fits){
 const tr=document.createElement('tr');for(const text of [name+' / '+f.id,String(f.components.length),f.eigengap===undefined?'not fitted':fmt(f.eigengap),f.ari===undefined?'not fitted':fmt(f.ari),fmt(d.raw.ari),f.status]){
 const td=document.createElement('td');td.textContent=text;tr.append(td);} $('allfits').tBodies[0].append(tr);
}
function element(tag,attrs,parent){const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,String(v));parent.append(e);return e;}
function plot(svg,points,labels,weights,selected,threshold){
 svg.replaceChildren();if(!points){element('text',{x:24,y:175},svg).textContent='No unique two-dimensional fit. See refusal below.';return {shown:0,total:0};}
 const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);let loX=Math.min(...xs),hiX=Math.max(...xs),loY=Math.min(...ys),hiY=Math.max(...ys);
 const scale=Math.min(410/Math.max(hiX-loX,.1),280/Math.max(hiY-loY,.1)),cx=(loX+hiX)/2,cy=(loY+hiY)/2;
 const xy=points.map(([x,y])=>[240+(x-cx)*scale,175-(y-cy)*scale]);let shown=0,total=0;
 if(weights)for(let i=0;i<points.length;i++)for(let j=i+1;j<points.length;j++)if(weights[i][j]>0){total++;if(weights[i][j]>=threshold){shown++;element('line',{x1:xy[i][0],y1:xy[i][1],x2:xy[j][0],y2:xy[j][1],stroke:i===selected||j===selected?'#b86c38':'#bbc9be','stroke-width':i===selected||j===selected?1.5:.65,'stroke-opacity':i===selected||j===selected?.8:.35},svg);}}
 const colors=['#24736c','#b86c38'];
 for(let i=0;i<points.length;i++){const circle=element('circle',{cx:xy[i][0],cy:xy[i][1],r:i===selected?7:4.2,fill:labels?colors[labels[i]]:'#64706a',stroke:i===selected?'#233432':'#fffdf7','stroke-width':i===selected?2:1,'data-node':i},svg);element('title',{},circle).textContent='Point '+i+(labels?', cluster '+labels[i]:'');circle.addEventListener('click',()=>{$('node').value=String(i);render();});}
 return {shown,total};
}
function render(){
 const name=$('dataset').value,d=DATA[name],config=META.configurations.find(c=>c.id===$('configuration').value),fit=d.fits.find(f=>f.id===config.id),node=Number($('node').value),threshold=Number($('threshold').value);
 const w=Spectral.graph(d.points,config),wit=Spectral.witness(w,fit,node),colors=$('colors').value,labels=colors==='truth'?d.truth:colors==='raw'?d.raw.labels:fit.status==='ok'?fit.fit.labels:null;
 const drawn=plot($('graph'),d.points,labels,w,node,threshold);plot($('embedding'),fit.embedding||null,labels,null,node,threshold);
 $('fixture-description').textContent=d.description;$('threshold-value').textContent=threshold.toFixed(2);$('edge-summary').textContent=`Drawing ${drawn.shown} of ${drawn.total} positive-weight edges. Full graph has ${fit.components.length} connected components.`;
 $('status').textContent=fit.status==='ok'?`Fitted ${config.id}: both retained eigenvectors pass the archived numerical checks. The graph, not labels, supplied this geometry.`:`${fit.status}: ${fit.reason}`;
 $('metrics').replaceChildren();for(const [value,label]of [[fmt(fit.ari),'spectral ARI (generator audit only)'],[fmt(d.raw.ari),'raw-coordinate k-means ARI'],[fit.cut?fmt(Spectral.cut(w,fit.fit.labels).normalized_cut):'not fitted','normalized cut on this graph']]){const div=document.createElement('div'),b=document.createElement('strong'),s=document.createElement('span');b.textContent=value;s.textContent=label;div.append(b,s);$('metrics').append(div);}
 $('point-title').textContent='Follow point '+node+'.';$('point-summary').textContent=`Original coordinates [${d.points[node].map(fmt).join(', ')}]. Degree ${fmt(wit.degree)}. ${wit.neighbors.length} positive neighbors. `+(wit.cluster===null?'No fitted cluster.':`Fitted cluster ${wit.cluster}. Generator label ${d.truth[node]} is only an audit label.`);
 for(const [id,rows]of [['modes',wit.modes.map(m=>[m.mode,fmt(m.eigenvalue),fmt(m.lhs),fmt(m.rhs),m.residual.toExponential(3)])],['neighbors',wit.neighbors.slice(0,12).map(n=>[n.node,fmt(n.weight),fmt(n.normalized),n.crosses===null?'not fitted':n.crosses?'yes':'no'])]]){
  $(id).tBodies[0].replaceChildren();for(const row of rows){const tr=document.createElement('tr');for(const text of row){const td=document.createElement('td');td.textContent=String(text);tr.append(td);}$(id).tBodies[0].append(tr);}
 }
 current={schema:'spectrallab-witness-1',dataset:name,configuration:config,points_sha256:d.points_sha256,graph_input_sha256:fit.graph_sha256,display_threshold_only:threshold,point:wit,cut:fit.status==='ok'?Spectral.cut(w,fit.fit.labels):null,scope:META.scope};
 $('witness').textContent=JSON.stringify(current,null,2);$('copy-status').textContent='';document.body.dataset.ready='true';
}
for(const id of ['dataset','configuration','node','colors','threshold'])$(id).addEventListener('input',render);
$('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(JSON.stringify(current,null,2));$('copy-status').textContent='Witness JSON copied.';}catch{$('copy-status').textContent='Clipboard unavailable. Use Download JSON or the expanded witness.';}});
$('download').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(current,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='spectrallab-witness.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});render();

const rounding=globalThis.SPECTRAL_ROUNDING,exactParts=rounding.exact.normalized_cut.split('/').map(Number),exactValue=exactParts[0]/(exactParts[1]||1);
for(const row of rounding.weights){const tr=document.createElement('tr');for(const value of row){const td=document.createElement('td');td.textContent=value;tr.append(td);}$('rounding').tBodies[0].append(tr);}
for(const [value,label]of [[fmt(rounding.spectral.cut.normalized_cut),'spectral relaxation + k-means cut'],[fmt(exactValue),'exact discrete optimum'],[rounding.exact.enumerated_partitions,'all nonempty cuts, symmetry removed']]){const d=document.createElement('div'),b=document.createElement('strong'),s=document.createElement('span');b.textContent=value;s.textContent=label;d.append(b,s);$('rounding-metrics').append(d);}
$('rounding-labels').textContent='Spectral clusters: ['+rounding.spectral.fit.labels.join(', ')+']. Exact reference: ['+rounding.exact.labels.join(', ')+']. Exact normalized cut = '+rounding.exact.normalized_cut+'.';
