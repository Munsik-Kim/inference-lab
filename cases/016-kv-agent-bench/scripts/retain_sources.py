#!/usr/bin/env python3
"""Retain byte-exact intermediate sources, checked against original freezes.

Some versions were copied before editing. Others below are reconstructed from
the preserved version plus the recorded small diff and accepted ONLY if their
bytes hash to the already-frozen identity. No freeze is changed.
"""
import hashlib
import json
import pathlib
import sys

C=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(C/'src'))
from kv_agent_bench.common import load,save,sha_file

def replace_once(s,old,new):
    if s.count(old)!=1:raise AssertionError('recorded diff no longer uniquely applies')
    return s.replace(old,new,1)

def main():
    retained=C/'provenance/retained_sources';retained.mkdir(exist_ok=True);made=[]
    def keep(text,expected,description):
        b=text.encode();actual=hashlib.sha256(b).hexdigest()
        if actual!=expected:raise AssertionError('reconstructed version differs from original freeze: '+description+' '+actual)
        path=retained/(expected+'.py')
        if path.exists() and path.read_bytes()!=b:raise AssertionError('existing retained source mismatch')
        if not path.exists():path.write_bytes(b)
        made.append({'path':str(path.relative_to(C)),'sha256':actual,'method':'reconstructed from preserved bytes and recorded diff; matched pre-existing freeze','description':description})
    s=(C/'src/kv_agent_bench/metrics.py').read_text();start=s.index('def workload_cost(');end=s.index('def summarize(')
    keep(s[:start]+s[end:],'b77dc009bd969cac92edb327ef02ecd8c8b23d85003827f577bb48c698c169f4','P0 metrics before added workload-cost helper')
    s=(C/'src/kv_agent_bench/audit.py').read_text();s=replace_once(s,"        if episode.get('slot_order_policy')=='raw_hermes_slots_v2':\n            from .observations import ordered_slots\n            ordered=ordered_slots(r['raw_text'],built['tool_calls'])\n            slots=[(s['parsed']['name'],s['parsed']['arguments']) if s['parsed']is not None else (None,None) for s in ordered]\n",'')
    keep(s,'919a15a3b58a6a454b00e5ad664a7c838e1c67e747537e60ee7d909f66aed0b5','P0 auditor before explicit raw-slot-order branch')
    s=(C/'src/kv_agent_bench/episode_runner.py').read_text();s=replace_once(s,"        if len(out)>limit:\n            terminal='SERVER_ERROR';availability='SERVER_BUDGET_VIOLATION';errors.append({'type':'SERVER_BUDGET_VIOLATION','requested':limit,'returned':len(out)})\n",'')
    keep(s,'c28d3aee3b0b955e51ca014889966d4c27953255201572444b6cd9d9226bb115','P0 diagnostics2 episode runner before over-cap check')
    s=replace_once(s,"'evidence_kind':'model_measurement','slot_order_policy':'raw_hermes_slots_v2','availability'","'evidence_kind':'model_measurement','availability'")
    keep(s,'0a3331628c5a0c410a2d570ac7ba1a32090443afb11682d020302fef1e4afc0f','P0 post-token-accounting runner before slot-policy tag')
    s=(C/'provenance/attempt_sources/workload_runner_p0_r2.py').read_text();s=replace_once(s,"    summaries=[];planned=[];kvs=['B16','F8','F4'] if stage=='p0' else runtime['p1_kvs']\n","    summaries=[];planned=[];kvs=runtime.get('p0_kvs',['B16','F8','F4']) if stage=='p0' else runtime['p1_kvs']\n    if stage=='p0':\n        for old_root in runtime.get('p0_reuse_results',[]):\n            old=load(case/'results'/old_root/'decision.json')\n            summaries.extend([{**s,'reused_from':old_root} for s in old['statuses'] if s['status']=='COMPLETED' and s['kv']not in kvs])\n")
    keep(s,'b453b538b1f1c4eb9861a556cf2f8fc4daa9cc8e32e9dde3144ee9a71b4eeb4f','P0 focused-F4 worker, before P1 supplemental gates/cost accounting')
    s=(C/'src/kv_agent_bench/observations.py').read_text();s=replace_once(s,"                name=server[i]['server_call']['function']['name']\n                if name and re.search(r'\"name\"\\s*:\\s*'+re.escape(json.dumps(name)),body):chosen=i;break\n","                candidate=server[i]['parsed']\n                if candidate is not None and re.search(r'\"name\"\\s*:\\s*'+re.escape(json.dumps(candidate['name'])),body):chosen=i;break\n")
    keep(s,'0845dd46de8deaeea7a35a2f8741d9cc390b1ad172e2b0060db7b9e76d44150a','Initial raw-slot helper before alignment of partial server calls')
    candidates=[p for p in C.rglob('*')if p.is_file() and p.suffix in('.py','.json') and'results'not in p.relative_to(C).parts and'demo'not in p.relative_to(C).parts]
    by_hash={}
    for p in sorted(candidates):by_hash.setdefault(sha_file(p),[]).append(p)
    freezes=[C/'provenance/pre_gpu_freeze.json',C/'provenance/pre_model_freeze_r2.json']+sorted((C/'results').glob('*/code_freeze.json'));mapping=[]
    for freeze in freezes:
        f=load(freeze);rows=[]
        for r in f['files']:
            options=by_hash.get(r['sha256'],[])
            if not options:raise AssertionError('missing original source: '+r['path'])
            direct=C/r['path'];chosen=direct if direct in options else options[0]
            rows.append({**r,'retained_path':str(chosen.relative_to(C)),'current_path_still_matches':chosen==direct})
        mapping.append({'freeze':str(freeze.relative_to(C)),'original_source_sha256':f['source_sha256'],'files':rows})
    save(C/'provenance/execution_source_map.json',{'status':'PASS','frozen_sets':mapping,'reconstructed_exact_versions':made,'original_freezes_modified':False,'p1_execution_files_unchanged':all(r['current_path_still_matches']for m in mapping if m['freeze']=='results/p1_r1/code_freeze.json'for r in m['files'])})
    print(json.dumps({'status':'PASS','freeze_sets':len(mapping),'reconstructed_exact_versions':len(made),'p1_execution_files_unchanged':True}))

if __name__=='__main__':main()
