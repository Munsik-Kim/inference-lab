/* Pure filter/deep-link functions shared by browser UI and CPU Node tests. */
'use strict';
(function(root){
  const defaults=is7=>({set:is7?'HELD_OUT':'standard',task:'ALL',arm:is7?'PAIRWISE':'A_PUBLIC',budget:'4',readout:'H_NATIVE',outcome:'ALL',query:'',id:''});
  const choices=is7=>({set:is7?['HELD_OUT']:['standard','boundary_pool'],task:['ALL','RETRIEVAL','COMPARISON','CODE'],arm:is7?['INDEPENDENT','PAIRWISE']:['A_PUBLIC','V4'],budget:['4','8'],readout:is7?['H_NATIVE']:['H_NATIVE','H_FP32'],outcome:['ALL','flip','both_correct','regression','gain','both_wrong','wrong_to_wrong']});
  function decode(hash,is7){const s=defaults(is7);for(const [k,v] of new URLSearchParams(hash.replace(/^#/,'')))if(k in s)s[k]=v;return s;}
  function encode(s){return '#'+new URLSearchParams(Object.entries(s).filter(([,v])=>v)).toString();}
  function valid(s,is7){return Object.entries(choices(is7)).every(([k,values])=>values.includes(s[k]));}
  function select(rows,s,is7){if(!valid(s,is7))return [];return rows.filter(r=>r.set===s.set&&r.arm===s.arm&&r.readout===s.readout&&(!is7||r.budget===s.budget)&&(s.task==='ALL'||r.task===s.task)&&(!s.query||r.id.toLowerCase().includes(s.query.toLowerCase()))&&(s.outcome==='ALL'||(s.outcome==='flip'?r.flip:s.outcome==='wrong_to_wrong'?r.wrong_to_wrong:r.cell===s.outcome)));}
  const api={defaults,choices,decode,encode,valid,select};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ShowcaseState=api;
})(typeof globalThis!=='undefined'?globalThis:this);
