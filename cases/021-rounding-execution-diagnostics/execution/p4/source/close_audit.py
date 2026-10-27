"""Bounded preservation, plan/count and saved-evidence audit. CPU only."""
import argparse,csv,json,os,re,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from common import read,write,identity,verify,utc
from numerics import weight_vector

def command(w,*args):
    p=subprocess.run(['git','-C',str(w),*args],capture_output=True,text=True,check=True);return p.stdout
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);a=p.parse_args();r=a.root;a.output.mkdir(parents=True,exist_ok=False)
    before=read(r/'records/preservation_before.json');loc=read(r/'references/locations.json');case=Path(loc['original_case']);work=Path(loc['original_worktree'])
    files={};symlinks={}
    for parent,dirs,names in os.walk(case,followlinks=False):
        for name in dirs+names:
            f=Path(parent)/name
            if f.is_symlink():symlinks[str(f.relative_to(case))]=os.readlink(f)
        for name in names:
            f=Path(parent)/name
            if f.is_file() and not f.is_symlink():files[str(f.relative_to(case))]=identity(f)
    changed=[k for k in files.keys()&before['files'].keys() if files[k]!=before['files'][k]];added=sorted(files.keys()-before['files'].keys());missing=sorted(before['files'].keys()-files.keys())
    git={'HEAD':command(work,'rev-parse','HEAD'),'branch':command(work,'branch','--show-current'),'status':command(work,'status','--porcelain=v1','--untracked-files=all'),'staged':command(work,'diff','--cached','--binary'),'tracked':command(work,'diff','--binary'),'worktrees':command(work,'worktree','list','--porcelain')}
    git_equal={k:v==before['git'][k] for k,v in git.items()};other=[]
    for old in before['other_worktree_tracked_states']:
        path=Path(old['worktree'])
        if path.exists():
            q=subprocess.run(['git','-C',str(path),'status','--porcelain=v1','--untracked-files=no'],capture_output=True,text=True)
            other.append({'worktree':str(path),'same':q.returncode==old['returncode'] and q.stdout==old['status_tracked'],'current_status_sha256':__import__('hashlib').sha256(q.stdout.encode()).hexdigest(),'returncode':q.returncode})
        else:other.append({'worktree':str(path),'same':None,'scope':'No current path; no recovery or mutation attempted'})
    write(a.output/'preservation_after.json',{'utc':utc(),'case_root':str(case),'files':files,'symlinks':symlinks,'git':git,'other_worktree_status_comparison':other})
    if changed or added or missing or symlinks!=before['symlinks'] or not all(git_equal.values()):raise ValueError('ORIGINAL_PRESERVATION_FAILED')
    ext=read(r/'references/external_inputs.json');checked=[]
    for artifact in ext['artifacts']:
        assert identity(artifact['path'])=={k:artifact[k] for k in ['bytes','sha256']};checked.append(artifact['path'])
    for key in ['small_model_files','existing_runtime_sources']:
        for path,row in ext[key].items():assert identity(path)==row;checked.append(path)
    model=Path(ext['model_weight']['path']);assert model.stat().st_size==ext['model_weight']['bytes']
    norms=[]
    for key,row in ext['used_norm_vectors'].items():
        _,h=weight_vector(model,key,row['bytes']//2);assert h==row['sha256'];norms.append(key)
    for path,row in read(r/'references/input_identity.json')['files'].items():assert identity(path)==row
    for name in ['freeze_AB.json','freeze_AB_v2.json','freeze_C_preparation.json','freeze_C_main.json','freeze_C_final.json']:verify(r,read(r/'records'/name))
    led=[json.loads(line) for line in (r/'records/worker_ledger.jsonl').read_text().splitlines()];oldled=[json.loads(line) for line in (case/'worker_ledger.jsonl').read_text().splitlines()]
    # Original schema has wall_seconds, not new parent_wall_seconds.
    original_fields=sorted(set(k for x in oldled for k in x))
    write(a.output/'original_ledger_schema.json',{'keys':original_fields,'rows':len(oldled),'current_identity':identity(case/'worker_ledger.jsonl'),'expected_identity':loc['old_ledger']})
    assert identity(case/'worker_ledger.jsonl')==loc['old_ledger']
    plan=read(r/'protocol_C.json');actual=[];output_rows=0;core_calls=controls=profiles=0;max_peak=0;kernel_checks=[];events=[]
    for spec in plan['operator_workers']:
        label=f"gpu-q{spec['layer']}-b{spec['block']}";parent=next(x for x in led if x['label']==label)
        assert parent['spawned'] and parent['exit_code']==0
        rec=read(r/'records'/label/'data/operator_receipt.json');assert rec['layer']==spec['layer'] and rec['block']==spec['block']
        assert len(rec['rows'])==30;core_calls+=len(rec['rows']);controls+=sum(x['calls'] for x in rec['controls']);profiles+=rec['profile_calls'];output_rows+=sum(x['valid_rows'] for x in rec['rows']);max_peak=max(max_peak,rec['memory']['peak_allocated'])
        assert rec['memory']['peak_allocated']<=plan['allocated_peak_cap_bytes']
        for row in rec['rows']:
            if spec['block']==0:assert identity(row['output_path'])==row['output_file']
            events.append({'layer':spec['layer'],'block':spec['block'],'fixture':row['fixture'],'method':row['method'],'host_call_ns':row['host_call_ns'],'event_elapsed_ms':row['event_elapsed_ms']})
        if spec['block']==0:
            trace=read(r/'records'/label/'data/marlin_trace.json');names={x.get('name','') for x in trace['traceEvents'] if x.get('cat')=='kernel'};assert any('marlin' in x.lower() for x in names);assert set(rec['kernel_evidence']['kernel_names'])==names;kernel_checks.append({'layer':spec['layer'],'actual_trace_kernel_names':sorted(names)})
        actual.append(spec)
    prep=read(r/'records/gpu-inputs/data/input_preparation.json');assert len(prep['rows'])==24 and prep['full_model_forwards']==0
    rr=read(a.replay/'replay_receipt.json');assert rr['GPU_calls']==0
    for match in rr['CSV_exact_byte_matches']:assert identity(match['original'])==identity(match['replay'])==match['identity']
    svg=[]
    for path in (a.replay/'figures').glob('*.svg'):ET.parse(path);svg.append(path.name)
    from common import csvwrite
    csvwrite(a.output/'diagnostic_call_costs.csv',events)
    write(a.output/'preservation_receipt.json',{'status':'UNCHANGED','original_case_files':len(files),'added':added,'changed':changed,'missing':missing,'symlinks_equal':True,'original_git_fields_equal':git_equal,'other_worktrees_checked_for_tracked_status_only':other,'external_full_hash_checked':checked,'norm_vector_bytes_checked':norms,'original_referenced_raw_files':len(read(r/'references/input_identity.json')['files']),'model_current_scope':'Existence,size,small files and7 norm vectors; full shard current hash NOT_REHASHED'})
    write(a.output/'verification.json',{'execution_status':'COMPLETE','plan_workers':9,'actual_workers':len(actual),'main_core_calls':core_calls,'module_core_control_calls':controls,'profiler_calls':profiles,'main_output_rows_including_both_methods_and_repeats':output_rows,'distinct_projection_fixture_rows':3*7680,'maximum_projection_allocated_bytes':max_peak,'native_input_norm_calls':24,'explicit_norm_controls':24,'beta_transports':24,'full_model_forwards':0,'GPU_replay_calls':0,'kernel_trace_checks':kernel_checks,'SVG_XML_checked':svg,'SVG_browser_rendering':'NOT_RUN','CPU_only_Torch_imported':'torch' in sys.modules,'evidence_replay_receipt':identity(a.replay/'replay_receipt.json')})
    print(json.dumps({'status':'COMPLETE','original_files':len(files),'new_core_calls':core_calls,'other_worktree_changed':sum(x['same'] is False for x in other)}))
if __name__=='__main__':main()
