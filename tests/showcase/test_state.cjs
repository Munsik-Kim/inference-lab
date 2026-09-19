'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const S=require('../../presentation/assets/state.js');
const rows=[{id:'fixture-1',set:'standard',arm:'A_PUBLIC',readout:'H_NATIVE',task:'CODE',budget:'NA',cell:'both_wrong',flip:true,wrong_to_wrong:true},
{id:'fixture-1',set:'standard',arm:'A_PUBLIC',readout:'H_FP32',task:'CODE',budget:'NA',cell:'both_correct',flip:false,wrong_to_wrong:false},
{id:'fixture-2',set:'boundary_pool',arm:'A_PUBLIC',readout:'H_NATIVE',task:'CODE',budget:'NA',cell:'regression',flip:true,wrong_to_wrong:false}];
test('native, shadow and stress never pooled',()=>{assert.equal(S.select(rows,S.defaults(false),false).length,1);assert.equal(S.select(rows,{...S.defaults(false),readout:'H_FP32'},false)[0].cell,'both_correct');assert.equal(S.select(rows,{...S.defaults(false),set:'boundary_pool'},false)[0].id,'fixture-2');});
test('empty IDs and groups stay empty',()=>{assert.deepEqual(S.select(rows,{...S.defaults(false),query:'absent'},false),[]);assert.deepEqual(S.select(rows,{...S.defaults(false),task:'RETRIEVAL'},false),[]);});
test('malformed filter is invalid',()=>{for(const v of ['invalid','<script>','../standard'])assert.equal(S.valid({...S.defaults(false),set:v},false),false);});
test('state roundtrip with Unicode and exact ID',()=>{const x={...S.defaults(false),id:'한글 & = ?',query:'<script>'};assert.deepEqual(S.decode(S.encode(x),false),x);});
test('same URL fragment works across languages',()=>{const s={...S.defaults(false),id:'fixture-1',task:'CODE'};const url=new URL('https://example.invalid/inference-lab/en/case006.html'+S.encode(s));url.pathname=url.pathname.replace('/en/','/ko/');assert.deepEqual(S.decode(url.hash,false),s);});
test('Case007 allows only native and frozen budgets',()=>{assert.equal(S.valid({...S.defaults(true),readout:'H_FP32'},true),false);assert.equal(S.valid({...S.defaults(true),budget:'6'},true),false);});
test('wrong-to-wrong is a flip without being a regression',()=>{assert.equal(S.select(rows,{...S.defaults(false),outcome:'wrong_to_wrong'},false).length,1);assert.equal(S.select(rows,{...S.defaults(false),outcome:'regression'},false).length,0);});
