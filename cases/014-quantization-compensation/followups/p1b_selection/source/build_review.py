"""Explicit local review export and extracted-bundle CPU verification. No uploads."""
import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath
from p1b_common import save,sha,utc

def links(root):
    checked=0;bad=[]
    for p in root.rglob('*.md'):
        for ref in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if re.match(r'https?://|mailto:',ref):continue
            path,_,anchor=ref.partition('#');dest=(p.parent/path).resolve() if path else p
            checked+=1
            if not dest.exists():bad.append([str(p.relative_to(root)),ref])
            elif anchor and dest.is_file() and dest.suffix=='.md' and anchor.startswith('section-'):
                if f'id="{anchor}"' not in dest.read_text():bad.append([str(p.relative_to(root)),ref])
    assert not bad,bad
    return checked

def main():
    ap=argparse.ArgumentParser()
    for k in ['case','review','cpu-python','reference-python']:ap.add_argument('--'+k,required=True)
    a=ap.parse_args();case=Path(a.case).resolve();review=Path(a.review);review.mkdir(parents=True,exist_ok=False)
    run=case/'results/p1b_r2'
    freeze={str(p.relative_to(case)):sha(p) for p in sorted((case/'followups/p1b_selection/source').glob('*.py'))}
    post=run/'postprocessing_freeze.json';i=2
    while post.exists():post=run/f'postprocessing_freeze.v{i}.json';i+=1
    save(post,{'recorded_utc':utc(),'files':freeze,'aggregate_sha256':__import__('hashlib').sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest(),
         'scope':'current postprocessing source; distinct from GPU execution source/commit/model/input identity. GPU-freeze file entries remain unchanged.'})
    export=review/'bundle/case014_p1b';export.mkdir(parents=True)
    selected=[]
    for d in ['followups/p1b_selection','results/p1b_r1','results/p1b_r2','figures/p1b_r2']:
        selected += [p for p in (case/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    selected += [case/p for p in ['LICENSE','NOTICE.md','references/source_audit.json','scripts/codec.py','scripts/common.py','scripts/decompose.py','scripts/fit.py','scripts/fixtures.py','tests/test_core.py',
            'results/p1_r2/fit_frozen.json','results/p1_r2/artifact_manifest.json','results/p1_r2/smoke.json','results/p1_r2/summary.csv']]
    for p in sorted(set(selected)):
        assert p.suffix not in ['.int4','.npy','.npz','.safetensors','.arrow','.pt','.bin','.pdf'] and not p.is_symlink(),p.name
        dst=export/p.relative_to(case);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    (export/'START_HERE.ko.md').write_text('# Case014 P1b 검토본\n\n[한국어 보고서](followups/p1b_selection/REPORT.ko.md) · [English report](followups/p1b_selection/REPORT.md) · [CPU 재계산](followups/p1b_selection/REPRODUCE.md)\n\n새 SELECT32 / EVAL64 / SHIFT16, 고정 K·INT4 후보의 선택 전이를 평가했습니다. 기존 P1 결과와 합치지 않습니다. 모델·K·INT4 payload·원문·전체 배열은 제외했습니다. 이 ZIP은 로컬 검토용이며 업로드하지 않았습니다.\n')
    privacy=[]
    for p in export.rglob('*'):
        if p.is_file() and p.suffix!='.png':
            text=p.read_text()
            if re.search(r'/home/[A-Za-z0-9_.-]+/|/mnt/[a-z]/Users/[^/\s]+/|\b(?:sk-proj-|hf_)[A-Za-z0-9]{20,}',text):privacy.append(str(p.relative_to(export)))
    assert not privacy,privacy
    link_count=links(export)
    files={str(p.relative_to(export)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(export.rglob('*')) if p.is_file()}
    save(export/'REVIEW_MANIFEST.json',{'recorded_utc':utc(),'files':files,'scope':'closed review file set; manifest excludes itself; private/external identities not claimed as bundled assets',
         'privacy_scan':'PASS','relative_links_checked':link_count,'source_transform':'source/records copied byte-for-byte; only raw log copies were path-redacted before export; original private logs retained'})
    z=review/'case014_P1b_selection_transfer_review_v1.zip'
    with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as f:
        for p in sorted(export.rglob('*')):
            if p.is_file():f.write(p,'case014_p1b/'+str(p.relative_to(export)))
    extracted=review/'extracted';extracted.mkdir()
    with zipfile.ZipFile(z) as f:
        infos=f.infolist();names=[i.filename for i in infos];assert len(names)==len(set(names)) and f.testzip() is None
        for info in infos:
            p=PurePosixPath(info.filename)
            assert not p.is_absolute() and '..' not in p.parts and not info.filename.startswith('\\') and not stat.S_ISLNK(info.external_attr>>16)
        f.extractall(extracted)
    root=extracted/'case014_p1b';mf=json.loads((root/'REVIEW_MANIFEST.json').read_text())
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert actual==set(mf['files'])|{'REVIEW_MANIFEST.json'}
    for rel,v in mf['files'].items():assert (root/rel).stat().st_size==v['bytes'] and sha(root/rel)==v['sha256'],rel
    extracted_links=links(root)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    commands=[
      ('old15',[a.reference_python,'-m','unittest','discover','-s','tests','-v']),
      ('new10',[a.reference_python,'-m','unittest','discover','-s','followups/p1b_selection/tests','-v']),
      ('saved_audit',[a.cpu_python,'followups/p1b_selection/source/audit_saved.py','--run','results/p1b_r2','--output',str(review/'extracted_saved_audit.json')]),
      ('copy_raw',[a.cpu_python,'followups/p1b_selection/source/copy_for_reanalysis.py','--run','results/p1b_r2','--output',str(review/'recomputed')]),
      ('rebuild_tables',[a.cpu_python,'followups/p1b_selection/source/analyze_saved.py','--case','.', '--run',str(review/'recomputed')]),
      ('rebuilt_audit',[a.cpu_python,'followups/p1b_selection/source/audit_saved.py','--run',str(review/'recomputed'),'--output',str(review/'recomputed/independent_audit.json')])]
    checks=[]
    for label,cmd in commands:
        start=time.monotonic()
        with (review/(label+'.log')).open('x') as log:proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        checks.append({'check':label,'exit_code':proc.returncode,'wall_seconds':time.monotonic()-start,'log_sha256':sha(review/(label+'.log'))})
        assert proc.returncode==0,(label,review/(label+'.log'))
    original=json.loads((root/'results/p1b_r2/summary.json').read_text());again=json.loads((review/'recomputed/summary.json').read_text())
    for key in ['macro_contrasts','target_contrasts','classification_counts','continuous_diagnostics','choices']:
        assert original[key]==again[key],key
    # Tests run without bytecode creation; extracted evidence stays identical after validation.
    for rel,v in mf['files'].items():assert sha(root/rel)==v['sha256'],rel
    receipt={'status':'PASS','recorded_utc':utc(),'ZIP_name':z.name,'ZIP_bytes':z.stat().st_size,'ZIP_sha256':sha(z),'CRC':'PASS','path_traversal_duplicates_symlinks':'PASS',
       'closed_file_hashes_checked':len(mf['files']),'relative_links_checked':extracted_links,'privacy_scan':'PASS','commands':checks,
       'raw_to_tables_and462_intervals':'REBUILT_AND_INDEPENDENTLY_AUDITED','recomputed_core_results_equal':True,'GPU_model_reexecution':False,'unique_test_counts':'15 existing +10 new; extracted reruns are repeats',
       'publication':'NOT_UPLOADED; no stage/commit/push/PR/merge/Pages/Drive'}
    save(review/'ZIP_VERIFICATION.json',receipt)
    (review/(z.name+'.sha256')).write_text(sha(z)+'  '+z.name+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
