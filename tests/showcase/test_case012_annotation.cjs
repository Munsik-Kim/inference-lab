'use strict';
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
