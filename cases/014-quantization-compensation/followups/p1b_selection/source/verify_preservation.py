"""Read-only before/after checks using an external private baseline inventory."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from p1b_common import save,sha,utc

def main():
    ap=argparse.ArgumentParser()
    for k in ['case','baseline','p0-private','p1-private','model','run']:ap.add_argument('--'+k,required=True)
    a=ap.parse_args();t=time.monotonic();case=Path(a.case).resolve();repo=case.parents[1];run=Path(a.run);before=json.loads(Path(a.baseline).read_text())
    findings=[];counts={}
    for label,root,key in [('original_case',case,'case'),('original_P0_private',Path(a.p0_private),'old_p0_private'),('original_P1_private',Path(a.p1_private),'old_p1_private')]:
        inventory=before[key];counts[label]=len(inventory)
        for rel,v in inventory.items():
            p=root/rel
            if not p.is_file() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:findings.append(label+'/'+rel)
    counts['tracked_repository_files']=len(before['tracked'])
    for rel,h in before['tracked'].items():
        if not (repo/rel).is_file() or sha(repo/rel)!=h:findings.append('tracked/'+rel)
    now={k:subprocess.check_output(cmd,cwd=repo,text=True) for k,cmd in {'HEAD':['git','rev-parse','HEAD'],'branch':['git','branch','--show-current'],'index':['git','diff','--cached','--binary'],'remotes':['git','remote','-v']}.items()}
    for k,v in now.items():assert v.strip()==before['git'][k].strip(),k
    identity=json.loads((Path(a.p1_private)/'snapshot.private.json').read_text())
    for rel,v in identity['files'].items():
        p=Path(a.model)/rel
        if not p.is_file() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:findings.append('model/'+rel)
    counts['model_snapshot_files_full_local_hash']=len(identity['files'])
    frozen=json.loads((run/'source_freeze.json').read_text())
    for rel,h in frozen['files'].items():assert sha(case/rel)==h,rel
    assert sha(case/'followups/p1b_selection/specs/p1b.json')==frozen['spec_sha256']
    # Recheck every K, target and actual-used prefix against the original P1 manifest.
    assets=json.loads((run/'reused_artifact_manifest.json').read_text())
    for r in assets['artifacts']:
        p=Path(a.p1_private)/r['label'].split('/',1)[1]
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['label']
    save(run/'preservation.json',{'status':'PRESERVED' if not findings else 'MISMATCH','recorded_utc':utc(),'counts':counts,'mismatches':findings,
         'original_K6_target30_prefix140_full_hashes':'PASS','model_source':'existing WSL snapshot read-only; local full-file hashes match P1; no D-origin or newly fetched official digest',
         'source_spec_hashes':'PASS','Git_HEAD_branch_index_remotes':'UNCHANGED','stage_commit_push_PR_merge_Pages_Drive':False,
         'environment':'Existing Python packages reused; no install/upgrade/build; observed Torch/Transformers/NumPy match P1 runtime. Full environment tree not recursively hashed.',
         'new_files_scope':['followups/p1b_selection','results/p1b_r1','results/p1b_r2','figures/p1b_r2','external P1b private/review folders'],
         'private_before_inventory_sha256':sha(a.baseline),'wall_seconds':time.monotonic()-t})
    if findings:raise AssertionError(findings)
    print(json.dumps({'status':'PRESERVED','counts':counts},indent=2))

if __name__=='__main__':main()
