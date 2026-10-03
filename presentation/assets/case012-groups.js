/* Visual-group reuse is explicit and traceable; it is not a new image judgment. */
(function(global){
 'use strict';
 const fail=m=>{throw Error(m)},clone=x=>JSON.parse(JSON.stringify(x));
 const same=(a,b)=>JSON.stringify([...a].sort())===JSON.stringify([...b].sort());
 function direct(answers){return Object.fromEntries(Object.entries(answers).map(([id,v])=>[id,Object.fromEntries(Object.keys(v).map(c=>[c,{kind:'direct'}]))]));}
 function units(copy,data,separated=[]){
  const known=new Set(data.items.map(i=>i.image_id)),split=new Set(separated),groups=copy.visual_groups;
  if(!Array.isArray(groups)||!Array.isArray(separated)||split.size!==separated.length||separated.some(i=>!known.has(i)))fail('Invalid visual grouping / separated IDs');
  const seen=groups.flatMap(g=>g.members);
  if(seen.length!==known.size||new Set(seen).size!==known.size||seen.some(i=>!known.has(i)))fail('Visual group coverage mismatch');
  return copy.groups.flatMap(p=>groups.filter(g=>g.prompt_id===p.prompt_id).flatMap(g=>{
   if(!g.members.includes(g.representative)||g.members.some(i=>copy.items[i]!==p.prompt_id))fail('Visual group prompt / representative mismatch');
   const members=g.members.filter(i=>!split.has(i));
   return [...(members.length?[{...g,members,representative:members.includes(g.representative)?g.representative:members[0]}]:[]),
    ...g.members.filter(i=>split.has(i)).map(i=>({group_id:'separate-'+i,prompt_id:p.prompt_id,members:[i],representative:i}))];
  }));
 }
 function restore(file,data,copy,answers){
  if(file.schema==='case012-annotations-v1')return {origins:direct(answers),separated:[],mode:'similarity'};
  const review=file.review;
  if(!copy||!review||review.grouping_sha256!==copy.visual_group_manifest_sha256||!['similarity','individual'].includes(review.mode))fail('Visual grouping identity / review mode mismatch');
  const groups=units(copy,data,review.separated_image_ids),origins={};
  for(const row of [...file.rows,...file.draft_rows]){
   if(!row.origins||!same(Object.keys(row.origins),Object.keys(row.values)))fail('Label origin coverage mismatch');
   origins[row.image_id]=clone(row.origins);
  }
  for(const [id,v] of Object.entries(answers))for(const c of Object.keys(v)){
   const origin=origins[id][c];
   if(!origin||!['direct','similarity_reuse'].includes(origin.kind))fail('Unknown label origin');
   if(origin.kind==='direct'){
    if(!same(Object.keys(origin),['kind']))fail('Invalid direct label origin');
    continue;
   }
   if(!same(Object.keys(origin),['kind','source_image_id','group_id']))fail('Invalid reused label origin');
   const group=groups.find(g=>g.group_id===origin.group_id),source=origin.source_image_id;
   if(!group||source===id||!group.members.includes(id)||!group.members.includes(source)||
      answers[source]?.[c]!==v[c]||origins[source]?.[c]?.kind!=='direct')fail('Reused label source / membership mismatch');
  }
  return {origins,separated:[...review.separated_image_ids],mode:review.mode};
 }
 function attach(file,options,data){
  const f=clone(file);f.schema='case012-annotations-v2';
  f.review={mode:options.mode,grouping_sha256:options.copy.visual_group_manifest_sha256,separated_image_ids:[...options.separated]};
  for(const row of [...f.rows,...f.draft_rows])row.origins=Object.fromEntries(Object.keys(row.values).map(c=>[c,clone(options.origins[row.image_id]?.[c]||{kind:'direct'})]));
  return f;
 }
 function edit(answers,origins,id,c,value){
  const a=clone(answers),o=clone(origins);
  for(const target of Object.keys(o))if(o[target][c]?.kind==='similarity_reuse'&&o[target][c].source_image_id===id){delete a[target][c];delete o[target][c];}
  (a[id]??={})[c]=value;(o[id]??={})[c]={kind:'direct'};
  return {answers:a,origins:o};
 }
 function apply(answers,origins,source,data,copy,separated=[]){
  const group=units(copy,data,separated).find(g=>g.members.includes(source)),item=data.items.find(i=>i.image_id===source);
  if(!group||!item||item.constraints.some(c=>!['satisfied','not_satisfied','uncertain'].includes(answers[source]?.[c.id])))fail('Answer every representative question before applying the group');
  const a=clone(answers),o=clone(origins);let copied=0,preserved=0;
  for(const id of group.members)if(id!==source)for(const c of item.constraints){
   const cid=c.id;
   if(a[id]?.[cid]!==undefined&&(!o[id]?.[cid]||o[id][cid].kind==='direct')){preserved++;continue;}
   const original=o[source]?.[cid]?.kind==='similarity_reuse'?o[source][cid].source_image_id:source;
   if(id===original)continue;
   (a[id]??={})[cid]=a[source][cid];(o[id]??={})[cid]={kind:'similarity_reuse',source_image_id:original,group_id:group.group_id};copied++;
  }
  return {answers:a,origins:o,copied,preserved};
 }
 function separate(answers,origins,id,separated){
  const a=clone(answers),o=clone(origins);
  for(const target of Object.keys(o))for(const c of Object.keys(o[target]))if(o[target][c].kind==='similarity_reuse'&&(target===id||o[target][c].source_image_id===id)){delete a[target][c];delete o[target][c];}
  return {answers:a,origins:o,separated:[...new Set([...separated,id])].sort()};
 }
 const api={direct,units,restore,attach,edit,apply,separate};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
 global.Case012Groups=api;
})(typeof window==='undefined'?globalThis:window);
