/* Display existing saved grids only. No new rho/g, simulation or interpolation. */
(()=>{'use strict';const p=window.CASE019?.projection;if(!p)return;
const ko=document.documentElement.lang==='ko',ns='http://www.w3.org/2000/svg';
const el=id=>document.getElementById(id),label=(k,e)=>ko?k:e;
const colors=['#245b89','#ab4c20','#18845d','#8056a3'],dashes=['','9 4','2 4','10 3 2 3'];
function svgNode(tag,attrs={},text){const n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function plot(id,lines,xName,yName){const svg=el(id);svg.replaceChildren();const w=Math.max(320,Math.round(svg.getBoundingClientRect().width)),h=w<500?320:280,l=48,r=16,t=lines.length>2&&w<600?80:64,b=46,maxX=Math.max(...lines[0].times),maxY=1;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
const xx=x=>l+(w-l-r)*x/maxX,yy=y=>h-b-(h-b-t)*y/maxY;
for(const v of [0,.25,.5,.75,1]){svg.append(svgNode('line',{x1:l,x2:w-r,y1:yy(v),y2:yy(v),stroke:'#e4ddd5'}));svg.append(svgNode('text',{x:l-8,y:yy(v)+4,'text-anchor':'end'},String(v)));}
for(const v of [0,maxX/4,maxX/2,3*maxX/4,maxX])svg.append(svgNode('text',{x:xx(v),y:h-26,'text-anchor':'middle'},v.toFixed(v%1?1:0)));
svg.append(svgNode('line',{x1:l,x2:w-r,y1:h-b,y2:h-b,stroke:'#4c5055'}));svg.append(svgNode('text',{x:w/2,y:h-7,'text-anchor':'middle'},xName));svg.append(svgNode('text',{x:10,y:15},yName));
lines.forEach((line,i)=>{const points=line.times.map((x,j)=>`${xx(x)},${yy(line.values[j])}`).join(' ');svg.append(svgNode('polyline',{points,fill:'none',stroke:colors[i], 'stroke-width':2.3,'stroke-dasharray':dashes[i]}));svg.append(svgNode('text',{x:l+8+(i%(w<600?2:4))*((w-l-r)/(w<600?2:4)),y:40+Math.floor(i/(w<600?2:4))*20,fill:colors[i]},line.name));});}
function initial(){const saved=p.rho[el('c19-rho').value],indices=saved.times.map((t,i)=>t<=20?i:-1).filter(i=>i>=0),a={times:indices.map(i=>saved.times[i]),fields:Object.fromEntries(Object.entries(saved.fields).map(([k,v])=>[k,indices.map(i=>v[i])]))};plot('c19-rho-plot',[
{times:a.times,values:a.fields.learnable_residual,name:label('손실 residual (solid)','Loss residual (solid)')},
{times:a.times,values:a.fields.right_distance,name:label('mean 오차 (dashed)','Mean error (dashed)')}],'t · gradient-flow','R8 pointwise median · dimensionless');}
function gap(){const a=p.gap[el('c19-gap').value][el('c19-gap-summary').value];plot('c19-gap-plot',[
{times:a.times,values:a.fields.right_distance,name:label('mean 오차 (solid)','Mean error (solid)')},
{times:a.times,values:a.fields.right_worst,name:label('worst 오차 (dashed)','Worst error (dashed)')}],'t · gradient-flow',a.aggregation+' · squared error');}
function arms(){const key=el('c19-parent').value,arms=p.arms[key],names=['CONTINUE','GAUGE','ALIGN','ROTATE'];for(const [id,field]of [['c19-arm-loss','learnable_residual'],['c19-arm-worst','right_worst']])plot(id,names.map(n=>({times:arms[n].times,values:arms[n].fields[field],name:n})),'τ · gradient-flow','R8 pointwise median · '+field);
const [g,st]=key.split('/').map(Number),tau=Number(el('c19-tau').value),rows=p.effects.filter(x=>x.g===g&&x.t_star===st&&x.tau===tau);
const div=el('c19-effects');div.replaceChildren();div.className='scroll';div.tabIndex=0;div.setAttribute('role','region');div.setAttribute('aria-label',label('같은 R8 짝 차이 평균','Means of same-R8 paired effects'));
const table=document.createElement('table'),head=document.createElement('thead'),tr=document.createElement('tr');for(const text of ['arm','mean J/D','mean G/D','mean A/D']){const th=document.createElement('th');th.scope='col';th.textContent=text;tr.append(th);}head.append(tr);table.append(head);const body=document.createElement('tbody');for(const n of names){const x=rows.find(x=>x.arm===n),r=document.createElement('tr');for(const value of [n,x.J_normalized.mean,x.G_normalized.mean,x.A_normalized.mean]){const td=document.createElement('td');td.textContent=typeof value==='number'?value.toPrecision(5):value;r.append(td);}body.append(r);}table.append(body);div.append(table);}
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{initial();gap();arms();},100);});
for(const [id,fn]of [['c19-rho',initial],['c19-gap',gap],['c19-gap-summary',gap],['c19-parent',arms],['c19-tau',arms]])el(id)?.addEventListener('change',fn);
document.querySelectorAll('.c19-dynamic').forEach(n=>n.hidden=false);initial();gap();arms();
})();
