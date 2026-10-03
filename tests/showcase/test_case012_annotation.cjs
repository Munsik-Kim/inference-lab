'use strict';
require('./test_case012_image_navigation.cjs');
const test = require('node:test'), assert = require('node:assert/strict');
const api = require('../../presentation/assets/case012-annotation.js');
const data = {rubric_sha256:'rubric', image_set_sha256:'cohort', items:[
  {image_id:'a', image_sha256:'ha', constraints:[{id:'c1'},{id:'c2'}]},
  {image_id:'b', image_sha256:'hb', constraints:[{id:'c1'}]}
]};
const clone = x => structuredClone(x);
function partial() { return api.exportFile({a:{c1:'satisfied'},b:{c1:'uncertain'}},0,'synthetic-test-rater',data); }

test('Partial constraint choices, cursor and human identity survive JSON round trip', () => {
  const file = partial(), restored = api.validateFile(JSON.parse(JSON.stringify(file)),data);
  assert.equal(file.status,'PARTIAL'); assert.equal(file.rows.length,1); assert.equal(file.draft_rows.length,1);
  assert.deepEqual(restored.answers,{b:{c1:'uncertain'},a:{c1:'satisfied'}});
  assert.equal(restored.cursor,0); assert.equal(restored.evaluator,'synthetic-test-rater');
});
test('All choices are blank until the person supplies them', () => {
  const file = api.exportFile({},1,'test',data);
  assert.deepEqual(file.rows,[]); assert.deepEqual(file.draft_rows,[]);
  assert.deepEqual(api.validateFile(file,data).answers,{}); assert.equal(file.current_image_id,'b');
});
test('Complete export preserves not-satisfied and uncertain, rather than treating them as unanswered', () => {
  const file = api.exportFile({a:{c1:'not_satisfied',c2:'satisfied'},b:{c1:'uncertain'}},1,'test',data);
  assert.equal(file.status,'COMPLETE'); assert.equal(file.rows.length,2); assert.deepEqual(file.incomplete_image_ids,[]);
  assert.equal(api.validateFile(file,data).answers.b.c1,'uncertain');
});
test('Key and row ordering do not alter labels', () => {
  const file = api.exportFile({a:{c2:'satisfied',c1:'uncertain'},b:{c1:'not_satisfied'}},1,'test',data);
  file.rows.reverse(); assert.equal(api.validateFile(file,data).answers.a.c1,'uncertain');
});
for (const [name, edit, reason] of [
  ['AI import', f=>f.evaluator.type='model-assisted', /Human evaluator/],
  ['rubric mismatch', f=>f.rubric_sha256='other', /rubric mismatch/],
  ['cohort mismatch', f=>f.image_set_sha256='other', /image set/],
  ['unknown ID', f=>f.rows[0].image_id='unknown', /Unknown/],
  ['duplicate ID', f=>f.rows.push(clone(f.rows[0])), /duplicate/],
  ['image bytes mismatch', f=>f.rows[0].image_sha256='other', /Image hash/],
  ['unknown value', f=>f.rows[0].values.c1='yes', /value mismatch/],
  ['missing constraints', f=>delete f.draft_rows[0].values.c1, /coverage mismatch/],
  ['extra constraint', f=>f.rows[0].values.c2='satisfied', /Constraint/],
  ['false complete status', f=>f.status='COMPLETE', /completeness/],
  ['incorrect missing list', f=>f.incomplete_image_ids=[], /completeness/],
  ['invalid cursor', f=>f.cursor=20, /Cursor/],
  ['cursor/image mismatch', f=>f.current_image_id='b', /Cursor/]
]) test(name+' is rejected atomically', () => {
  const file = partial(); edit(file); const before = clone(file);
  assert.throws(()=>api.validateFile(file,data),reason); assert.deepEqual(file,before);
});
test('Empty evaluator does not export success JSON', () => assert.throws(()=>api.exportFile({},0,' ',data),/anonymous ID/));

const groupedData={rubric_sha256:'rubric',image_set_sha256:'unchanged-cohort',items:[
 {image_id:'a',image_sha256:'ha',constraints:[{id:'c1'}]},
 {image_id:'b',image_sha256:'hb',constraints:[{id:'c1'}]},
 {image_id:'c',image_sha256:'hc',constraints:[{id:'c1'}]}
]};
const displayCopy={groups:[{prompt_id:'first'},{prompt_id:'second'}],items:{a:'second',b:'first',c:'second'}};
test('Grouping changes navigation only; scientific item order and identity are unchanged',()=>{
 const before=clone(groupedData);
 assert.deepEqual(api.groupedOrder(groupedData,displayCopy),[1,0,2]);
 assert.deepEqual(groupedData,before);
});
test('An existing partial JSON resumes at the exact same image after grouped navigation',()=>{
 const file=api.exportFile({a:{c1:'uncertain'}},0,'existing-rater',groupedData);
 const restored=api.validateFile(file,groupedData),order=api.groupedOrder(groupedData,displayCopy);
 assert.equal(groupedData.items[restored.cursor].image_id,'a');
 assert.equal(order[order.indexOf(restored.cursor)+1],2);
 assert.equal(restored.answers.a.c1,'uncertain');
});
test('Next unfinished scans forward instead of repeatedly jumping to the first skipped image',()=>{
 assert.equal(api.nextUnfinished({},0,groupedData,[1,0,2]),2);
 assert.equal(api.nextUnfinished({c:{c1:'satisfied'}},0,groupedData,[1,0,2]),1);
});
test('Grouping rejects missing and duplicate membership instead of dropping survey rows',()=>{
 assert.throws(()=>api.groupedOrder(groupedData,{groups:[{prompt_id:'first'}],items:displayCopy.items}),/coverage/);
 assert.throws(()=>api.groupedOrder(groupedData,{groups:[{prompt_id:'first'},{prompt_id:'first'}],items:displayCopy.items}),/Duplicate/);
});
test('Fully answered uncertain images are not revisited as unfinished',()=>{
 const answers={a:{c1:'uncertain'},b:{c1:'satisfied'},c:{c1:'not_satisfied'}};
 assert.equal(api.nextUnfinished(answers,0,groupedData,[1,0,2]),-1);
});

const grouping=require('../../presentation/assets/case012-groups.js');
const similarData={rubric_sha256:'rubric',image_set_sha256:'cohort',items:['a','b','c','d'].map(id=>({image_id:id,image_sha256:'hash-'+id,constraints:[{id:'c1'},{id:'c2'}]}))};
const similarCopy={groups:[{prompt_id:'p'},{prompt_id:'q'}],items:{a:'p',b:'p',c:'p',d:'q'},visual_group_manifest_sha256:'visual-config',visual_groups:[
 {group_id:'g1',prompt_id:'p',members:['a','b','c'],representative:'a'},
 {group_id:'g2',prompt_id:'q',members:['d'],representative:'d'}]};
function reused(){const a={a:{c1:'satisfied',c2:'uncertain'}},o=grouping.direct(a);return grouping.apply(a,o,'a',similarData,similarCopy);}
function groupedFile(){const r=reused();return api.exportFile(r.answers,0,'SYNTHETIC_TEST_ONLY',similarData,{copy:similarCopy,origins:r.origins,separated:[],mode:'similarity'});}
test('Representative answers cover a group with explicit per-constraint provenance',()=>{
 const file=groupedFile(),r=api.validateFile(JSON.parse(JSON.stringify(file)),similarData,similarCopy);
 assert.equal(file.schema,'case012-annotations-v2');assert.equal(r.answers.c.c2,'uncertain');
 assert.deepEqual(r.origins.b.c1,{kind:'similarity_reuse',source_image_id:'a',group_id:'g1'});
 assert.equal(r.origins.a.c1.kind,'direct');assert.equal(file.rows.length,3);
 assert.equal(file.status,'PARTIAL');assert.deepEqual(file.incomplete_image_ids,['d']);
});
test('Reuse is never applied before all representative questions have answers',()=>{
 const a={a:{c1:'satisfied'}},before=clone(a);
 assert.throws(()=>grouping.apply(a,grouping.direct(a),'a',similarData,similarCopy),/every representative/);assert.deepEqual(a,before);
});
test('Existing direct answers are preserved even when the representative disagrees',()=>{
 const a={a:{c1:'satisfied',c2:'uncertain'},b:{c1:'not_satisfied'}},before=clone(a);
 const r=grouping.apply(a,grouping.direct(a),'a',similarData,similarCopy);
 assert.equal(r.answers.b.c1,'not_satisfied');assert.equal(r.origins.b.c1.kind,'direct');
 assert.equal(r.answers.b.c2,'uncertain');assert.equal(r.preserved,1);assert.deepEqual(a,before);
});
test('Editing a source clears only its stale borrowed constraint, without changing another direct answer',()=>{
 const r=reused(),x=grouping.edit(r.answers,r.origins,'a','c1','not_satisfied');
 assert.equal(x.answers.a.c1,'not_satisfied');assert.equal(x.answers.b.c1,undefined);
 assert.equal(x.answers.b.c2,'uncertain');assert.equal(x.origins.b.c1,undefined);
});
test('A directly overridden group member is retained on a later reuse operation',()=>{
 const r=reused(),x=grouping.edit(r.answers,r.origins,'b','c1','not_satisfied');
 const y=grouping.apply(x.answers,x.origins,'a',similarData,similarCopy);
 assert.equal(y.answers.b.c1,'not_satisfied');assert.equal(y.origins.b.c1.kind,'direct');
});
test('Separating an image clears borrowed answers and retains direct judgments',()=>{
 const r=reused(),x=grouping.edit(r.answers,r.origins,'b','c1','not_satisfied');
 const y=grouping.separate(x.answers,x.origins,'b',[]);
 assert.equal(y.answers.b.c1,'not_satisfied');assert.equal(y.answers.b.c2,undefined);
 assert.deepEqual(y.separated,['b']);assert.equal(grouping.units(similarCopy,similarData,y.separated).length,3);
});
test('Separating the source invalidates its dependent reused labels rather than leaving stale completion',()=>{
 const r=reused(),x=grouping.separate(r.answers,r.origins,'a',[]);
 assert.deepEqual(x.answers.b,{});assert.deepEqual(x.answers.c,{});assert.equal(x.answers.a.c1,'satisfied');
});
test('Reusing an already borrowed member retains the original direct source, without chains',()=>{
 const r=reused(),x=grouping.apply(r.answers,r.origins,'b',similarData,similarCopy);
 assert.equal(x.origins.c.c1.source_image_id,'a');assert.equal(x.origins.a.c1.kind,'direct');
});
for(const [name,edit,reason] of [
 ['group config mismatch',f=>f.review.grouping_sha256='changed',/identity/],
 ['unknown separation',f=>f.review.separated_image_ids=['x'],/separated/],
 ['missing origins',f=>delete f.rows[0].origins,/coverage/],
 ['unknown source',f=>f.rows.find(r=>r.image_id==='b').origins.c1.source_image_id='x',/source/],
 ['self source',f=>f.rows.find(r=>r.image_id==='b').origins.c1.source_image_id='b',/source/],
 ['reused value mismatch',f=>f.rows.find(r=>r.image_id==='b').values.c1='not_satisfied',/source/],
 ['cross group',f=>f.rows.find(r=>r.image_id==='b').origins.c1.group_id='g2',/membership/],
 ['borrowed source chain',f=>f.rows.find(r=>r.image_id==='c').origins.c1.source_image_id='b',/source/]
])test('Grouped import rejects '+name+' without mutating its input',()=>{
 const f=groupedFile();edit(f);const before=clone(f);assert.throws(()=>api.validateFile(f,similarData,similarCopy),reason);assert.deepEqual(f,before);
});
test('Old JSON preserves cursor and direct labels without auto-applying a similarity group',()=>{
 const f=api.exportFile({b:{c1:'not_satisfied'}},1,'old-rater',similarData);
 const r=api.validateFile(f,similarData,similarCopy);assert.equal(r.cursor,1);assert.deepEqual(r.answers,{b:{c1:'not_satisfied'}});
 assert.equal(r.origins.b.c1.kind,'direct');assert.equal(r.mode,'similarity');
});

test('The last group wraps to an earlier skipped group rather than ending a partial review',()=>{
 const units=grouping.units(similarCopy,similarData),answers={d:{c1:'satisfied',c2:'satisfied'}},before=clone(answers);
 const action=api.nextReviewAction(answers,3,similarData,units);
 assert.equal(action.kind,'remaining');assert.equal(action.unit.group_id,'g1');assert.equal(action.remaining,1);
 assert.deepEqual(answers,before);
});
test('A checked representative with missing group members cannot report completion',()=>{
 const units=grouping.units(similarCopy,similarData),answers={a:{c1:'satisfied',c2:'uncertain'},d:{c1:'satisfied',c2:'satisfied'}};
 const action=api.nextReviewAction(answers,3,similarData,units);
 assert.equal(action.kind,'remaining');assert.equal(action.unit.group_id,'g1');
});
test('Completion requires all images and retains explicit uncertain and negative answers',()=>{
 const r=reused();r.answers.d={c1:'not_satisfied',c2:'uncertain'};
 assert.deepEqual(api.nextReviewAction(r.answers,3,similarData,grouping.units(similarCopy,similarData)),{kind:'complete',remaining:0});
});
test('Only the current unfinished group remains available at the end',()=>{
 const r=reused(),action=api.nextReviewAction(r.answers,3,similarData,grouping.units(similarCopy,similarData));
 assert.equal(action.kind,'remaining');assert.equal(action.unit.group_id,'g2');assert.equal(action.remaining,1);
});
test('Earlier groups preserve sequential navigation without inventing any answers',()=>{
 const answers={},before=clone(answers),action=api.nextReviewAction(answers,0,similarData,grouping.units(similarCopy,similarData));
 assert.equal(action.kind,'next');assert.equal(action.unit.group_id,'g2');assert.equal(action.remaining,2);assert.deepEqual(answers,before);
});
test('Separated unfinished images remain in the completion requirement',()=>{
 const r=reused(),split=grouping.separate(r.answers,r.origins,'b',[]);split.answers.d={c1:'satisfied',c2:'satisfied'};
 const action=api.nextReviewAction(split.answers,3,similarData,grouping.units(similarCopy,similarData,split.separated));
 assert.equal(action.kind,'remaining');assert.equal(action.unit.group_id,'separate-b');
});
