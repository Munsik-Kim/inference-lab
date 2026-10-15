"""Explicit P2 review allowlist, closed hashes, safe extraction and actual CPU replay."""
import argparse,hashlib,json,os,re,shutil,stat,subprocess,time,zipfile
from pathlib import Path,PurePosixPath
from html.parser import HTMLParser

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def scan(root):
    files=[p for p in root.rglob('*') if p.is_file()];broken=[];links=0
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.refs=[]
        def handle_starttag(self,tag,attrs):
            self.refs += [v for k,v in attrs if k in ['href','src']]
    for p in files:
        if p.suffix in ['.png']:continue
        text=p.read_text()
        if re.search(r'/home/[A-Za-z0-9_.-]+/|/mnt/[cde]/|C:\\\\Users\\|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{24,}',text):raise ValueError('private path/credential candidate in '+str(p.relative_to(root)))
        refs=[]
        if p.suffix=='.md':refs=re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)',text)
        if p.suffix=='.html':parser=Links();parser.feed(text);refs=parser.refs
        for ref in refs:
            if ref.startswith(('https:','http:','mailto:','data:')):continue
            target,_,fragment=ref.partition('#');q=(p.parent/target).resolve() if target else p.resolve();links+=1
            if not q.is_relative_to(root.resolve()) or not q.exists():broken.append((str(p.relative_to(root)),ref))
            if fragment and q.suffix=='.md' and q.exists():
                headings=re.findall(r'^#{1,6}\s+(.+)$',q.read_text(),re.M);anchors=[re.sub(r'[^\w\- ]','',h).lower().replace(' ','-') for h in headings]
                if fragment not in anchors:broken.append((str(p.relative_to(root)),ref+' [anchor]'))
    if broken:raise ValueError('broken local links: '+repr(broken))
    return {'text_files_privacy_checked':len([p for p in files if p.suffix!='.png']),'relative_links_checked':links,'broken_links':0}

def main():
    ap=argparse.ArgumentParser()
    for k in ['case','private','output','reference-python','cpu-python']:ap.add_argument('--'+k,required=True)
    a=ap.parse_args();case=Path(a.case).resolve();pr=Path(a.private);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    tree=out/'case014_P2_core_review';tree.mkdir();selected=[]
    # Only P2 evidence/source and exact unmodified CPU dependencies; no prior result bundles.
    for rel in ['followups/p2_core','results/p2_r1','figures/p2_r1']:
        for p in sorted((case/rel).rglob('*')):
            if p.is_symlink():raise ValueError('symlink in allowlist')
            if p.is_file():selected.append(p)
    dependencies=['LICENSE','NOTICE.md','references/source_audit.json','tests/test_core.py','tests/fixtures/matrix_identity.json']
    dependencies += ['scripts/'+n for n in ['codec.py','common.py','decompose.py','fit.py','fixtures.py']]
    dependencies += ['followups/p1b_selection/tests/test_p1b.py']
    dependencies += ['followups/p1b_selection/source/'+n for n in ['p1b_common.py','policies.py','bootstrap.py','run_readonly.py']]
    selected += [case/n for n in dependencies]
    if len(selected)!=len(set(selected)):raise ValueError('duplicate allowlist path')
    allowed_suffix={'.md','.txt','.py','.cjs','.json','.jsonl','.csv','.png','.svg','.log','.html',''}
    for p in selected:
        if not p.is_file() or p.is_symlink() or p.suffix not in allowed_suffix or p.stat().st_size>20*2**20:raise ValueError('unapproved export file '+str(p))
        dst=tree/p.relative_to(case);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    (tree/'START_HERE.ko.md').write_text('# Case014 P2 로컬 검토본\n\n[한국어 결과](followups/p2_core/REPORT.ko.md) · [English](followups/p2_core/REPORT.md) · [간단 결과표](followups/p2_core/demo/index.html) · [CPU/GPU 재현](followups/p2_core/REPRODUCE.md)\n\nP0/P1/P1b는 원형 보호 대상이며 이 ZIP은 새 P2와 필요한 CPU 의존성만 포함합니다. root NOTICE는 과거 P1 범위, 새 P2 범위는 followups/p2_core/NOTICE.md와 보고서에 있습니다. 모델·K·INT4 weights·토큰·X/Z·logits·원문은 제외했습니다.\n')
    freeze=json.loads((tree/'results/p2_r1/source_freeze.json').read_text())
    for rel,h in freeze['files'].items():assert sha(tree/rel)==h,rel
    assert sha(tree/'followups/p2_core/specs/p2.json')==freeze['spec_sha256']
    static=scan(tree)
    allfiles=sorted(p for p in tree.rglob('*') if p.is_file());manifest={str(p.relative_to(tree)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in allfiles}
    save(tree/'REVIEW_MANIFEST.json',{'schema_version':1,'closed_file_set_excluding_this_manifest':True,'files':manifest,'execution_source_sha256':freeze['execution_source_sha256'],
       'allowlist_roots':['followups/p2_core','results/p2_r1','figures/p2_r1'],'exact_CPU_dependency_files':dependencies,'privacy':static,
       'excluded':['model/INT4 weights/K/Gram/X/Z/logits','source document text and token arrays','private identities and absolute local paths','prior P0/P1/P1b result bundles'],
       'verification_scope':'All bundled files and frozen execution source hashes checked; external weights are identified by receipts, not verified from this ZIP.'})
    archive=out/'case014_P2_weak_compensation_review_v1.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(tree.rglob('*')):
            if p.is_file():z.write(p,str(Path(tree.name)/p.relative_to(tree)))
    extracted=out/'extracted_validation';extracted.mkdir();names=set()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for info in z.infolist():
            rel=PurePosixPath(info.filename);mode=info.external_attr>>16
            assert info.filename not in names;names.add(info.filename)
            assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in info.filename and not stat.S_ISLNK(mode)
            target=extracted/info.filename;assert target.resolve().is_relative_to(extracted.resolve());target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
    root=extracted/tree.name;M=json.loads((root/'REVIEW_MANIFEST.json').read_text())
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()};assert actual==set(M['files'])|{'REVIEW_MANIFEST.json'}
    for rel,meta in M['files'].items():assert (root/rel).stat().st_size==meta['bytes'] and sha(root/rel)==meta['sha256']
    extracted_scan=scan(root);validation=out/'validation';validation.mkdir();commands=[];env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def run(label,cmd):
        start=time.monotonic();r=subprocess.run(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
        (validation/(label+'.log')).write_text(r.stdout);private_command={'actual_command':cmd,'cwd':str(root),'exit_code':r.returncode,'wall_seconds':time.monotonic()-start}
        save(pr/'attempts'/('review-'+label+'.private.json'),private_command)
        public=' '.join(cmd).replace(a.reference_python,'${REFERENCE_PYTHON}').replace(a.cpu_python,'${CPU_PYTHON}').replace(str(validation),'${VALIDATION_OUTPUT}')
        commands.append({'step':label,'command':public,'exit_code':r.returncode,'wall_seconds':private_command['wall_seconds'],'log_sha256':sha(validation/(label+'.log'))})
        if r.returncode:raise RuntimeError(label+' failed: '+r.stdout[-3000:])
    for name,path in [('old15','tests'),('p1b10','followups/p1b_selection/tests'),('p2_new9','followups/p2_core/tests')]:run(name,[a.reference_python,'-m','unittest','discover','-s',path,'-v'])
    run('independent_saved_audit',[a.cpu_python,'followups/p2_core/source/audit.py','--run','results/p2_r1','--output',str(validation/'audit_extracted.json')])
    newrun=validation/'raw_only_reanalysis'
    run('copy_raw_only',[a.cpu_python,'followups/p2_core/source/copy_for_reanalysis.py','--run','results/p2_r1','--output',str(newrun)])
    run('regenerate_from_raw',[a.cpu_python,'followups/p2_core/source/analyze.py','--run',str(newrun)])
    run('independent_reanalysis_audit',[a.cpu_python,'followups/p2_core/source/audit.py','--run',str(newrun),'--output',str(validation/'reanalysis_audit.json')])
    for name in ['method_summary.csv','paired_comparisons.csv','paired_document_differences.csv']:assert (newrun/name).read_bytes()==(root/'results/p2_r1'/name).read_bytes(),name
    orig=json.loads((root/'results/p2_r1/summary.json').read_text());new=json.loads((newrun/'summary.json').read_text())
    for k in ['method_summaries','paired_comparisons','resources']:assert orig[k]==new[k],k
    assert json.loads((root/'results/p2_r1/decision.json').read_text())==json.loads((newrun/'decision.json').read_text())
    # Tests/audits wrote to external validation paths only; bundled closed set remains untouched.
    assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==actual
    for rel,meta in M['files'].items():assert sha(root/rel)==meta['sha256']
    result={'status':'PASS','ZIP_name':archive.name,'ZIP_bytes':archive.stat().st_size,'ZIP_sha256':sha(archive),'selected_files_including_manifest':len(actual),'CRC':'PASS','paths_duplicates_symlinks':'PASS','all_selected_file_hashes':'PASS',
       'static_original_export':static,'extracted_links_privacy':extracted_scan,'frozen_execution_file_hashes':len(freeze['files']),'commands':commands,'CPU_unique_test_scope':{'new_P2':9,'existing_regressions':25,'extraction_repeats_are_not_new_unique_tests':True},
       'raw_to_summary_and_decision':'identical paired tables/summary values/decision; no original summary was copied to the reanalysis input','actual_GPU_rerun_during_ZIP_validation':False,'remote_upload':False}
    save(out/'review_verification.json',result);(out/(archive.name+'.sha256')).write_text(sha(archive)+'  '+archive.name+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
