"""Verify P1 artifacts against original manifests; never recreate missing files."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from p1b_common import ALPHAS, TARGETS, save, sha, utc

def main():
    ap=argparse.ArgumentParser()
    for k in ['case','old-private','private','output']:ap.add_argument('--'+k,required=True)
    a=ap.parse_args();start=time.monotonic();case=Path(a.case);old=Path(a.old_private);out=Path(a.output)
    hist=json.loads((old/'PRIVATE_MANIFEST.json').read_text())['files']
    fit=json.loads((case/'results/p1_r2/fit_frozen.json').read_text());fitmap={r['target']:r for r in fit['selected']}
    artifact=json.loads((case/'results/p1_r2/artifact_manifest.json').read_text())['candidates']
    amap={(r['target'],r['alpha']):r for r in artifact};verified=[]
    required=[f'artifacts/{n}_K.npy' for n in TARGETS]
    required += [f'artifacts/candidates/{n}.alpha{a:g}.int4' for n in TARGETS for a in ALPHAS]
    required += [str(p.relative_to(old)) for p in sorted((old/'artifacts/prefix').glob('*.int4'))]
    missing=[p for p in required if not (old/p).is_file() or p not in hist]
    if missing:raise FileNotFoundError(json.dumps({'missing':missing}))
    for rel in required:
        p=old/rel;h=sha(p);size=p.stat().st_size
        assert h==hist[rel]['sha256'] and size==hist[rel]['bytes'],rel
        verified.append({'label':'P1_PRIVATE/'+rel,'bytes':size,'sha256':h})
    for n in TARGETS:
        assert sha(old/f'artifacts/{n}_K.npy')==fitmap[n]['K_sha256']
        assert fitmap[n]['selected_artifact_sha256']==amap[n,fitmap[n]['selected_alpha']]['sha256']
        for alpha in ALPHAS:
            p=old/f'artifacts/candidates/{n}.alpha{alpha:g}.int4';assert sha(p)==amap[n,alpha]['sha256']
            assert p.stat().st_size==(1179684 if n.endswith('o_proj') else 1769508)
    assert sum('/prefix/' in r for r in required)==140
    snapshot=json.loads((old/'snapshot.private.json').read_text());model=Path(snapshot['model_path'])
    model_checks=[]
    for name,identity in snapshot['files'].items():
        p=model/name
        assert p.is_file() and p.stat().st_size==identity['bytes'] and sha(p)==identity['sha256'],name
        model_checks.append(dict(identity,label='MODEL_SNAPSHOT/'+name))
    git={k:subprocess.check_output(cmd,cwd=case,text=True).strip() for k,cmd in {
        'branch':['git','branch','--show-current'],'HEAD':['git','rev-parse','HEAD'],'root':['git','rev-parse','--show-toplevel']}.items()}
    save(out/'reused_artifact_manifest.json',{'recorded_utc':utc(),'status':'PASS','K_count':6,'candidate_count':30,'prefix_count':140,
         'artifacts':verified,'model':model_checks,'model_revision':model.name,'model_identity':'LOCAL_IDENTITY_VERIFIED; full weight SHA compared to P1 local manifest, no new external digest/D-origin comparison',
         'old_fit':fit['selected'],'old_fit_file_sha256':sha(case/'results/p1_r2/fit_frozen.json'),
         'old_artifact_manifest_sha256':sha(case/'results/p1_r2/artifact_manifest.json'),
         'old_smoke_tolerance':json.loads((case/'results/p1_r2/smoke.json').read_text())['tolerance'],'validation_seconds':time.monotonic()-start})
    save(Path(a.private)/'identity.private.json',dict(git,model_path=str(model),old_private=str(old),private_root=a.private,python=snapshot['python'],approval='User P1b instruction 2026-10-15: existing WSL snapshot read-only, no D-origin prerequisite, reuse K/artifacts; no refit/requantization or remote publication.',approval_recorded_utc=utc()))
    save(out/'identity.json',{'recorded_utc':utc(),'git':{k:v for k,v in git.items() if k!='root'},'workspace_label':'EXISTING_CASE014_WORKTREE','private_label':'P1B_PRIVATE_R1','model_source':'APPROVED_EXISTING_WSL_SNAPSHOT_READ_ONLY','D_origin_required':False,'new_download_or_install':False,'historical_P1_code_hash':fit['code_freeze_sha256']})
    print(json.dumps({'status':'PASS','K':6,'target_INT4':30,'prefix_INT4':140,'model_local_file_hashes':len(model_checks),'seconds':time.monotonic()-start}))

if __name__=='__main__':main()
