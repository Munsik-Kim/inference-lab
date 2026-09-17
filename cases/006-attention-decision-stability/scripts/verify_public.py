"""Publication identity, selection, safe paths and local-link checks; no network."""
import argparse,json,re,sys
from pathlib import Path
from urllib.parse import unquote
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import load,digest,file_hash,write_new


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=a.case.resolve()
    for phase in ('design','eval_manifest'):
        freeze=load(c/f'configs/{phase}_freeze_v1.json')
        for name,sha in freeze['code_hashes'].items():
            if file_hash(c/name)!=sha:raise ValueError('Frozen code mismatch')
    pool=[r for f in sorted((c/'results/raw/pool').glob('*.json')) for r in load(f)]
    if len(pool)!=192 or any(r['arm']!='B' for r in pool):raise ValueError('Pool is not complete B-only evidence')
    selected=[];counts={}
    for task in ('RETRIEVAL','COMPARISON','CODE'):
        bins=[[] for _ in range(5)]
        for r in pool:
            if r['task']!=task:continue
            v=sorted(r['option_logits']);gap=v[-1]-v[-2]
            i=0 if gap==0 else 1 if gap<.1 else 2 if gap<.5 else 3 if gap<1 else 4
            bins[i].append(r['base_id'])
        for i,ids in enumerate(bins):
            counts[f'{task}/{i}']=len(ids)
            if i:selected.extend(sorted(ids)[:6])
    spec=load(c/'configs/boundary_selection_v1.json')
    if selected!=spec['selected_ids']:raise ValueError('Independent boundary selection differs')
    manifest=load(c/'results/raw/source_manifest.json');checked=0
    for kind in ('paired','pool','secondary','timing'):
        for f in sorted((c/f'results/raw/{kind}').glob('*.json')):
            for r in load(f):
                if kind in ('paired','pool'):
                    stage=f.name.split('-'+r['task'].lower()+'-')[0]
                    name=f"{stage}/cells/{r['item_id']}--{r['arm']}.json"
                    if r['split']=='boundary_pool' and kind=='paired' and r['base_id'] not in selected:
                        raise ValueError('Candidate outside B-only selection')
                elif kind=='secondary':name=f"secondary/cells/{r['item_id']}.json"
                else:
                    first=r['rows'][0];name=f"model-timing/cells/{first['item_id']}--r{first['process']}.json"
                if digest(r)!=manifest['sources'][name]['payload_sha256']:raise ValueError('Export changed scalar payload')
                checked+=1
    if checked!=len(manifest['sources']):raise ValueError('Missing exported payload')
    findings=[];links=0;files=[]
    patterns={'private_home':r'/h[o]me/[A-Za-z0-9_.-]+/','private_mount':r'/m[n]t/',
              'private_windows':r'[A-Za-z]:\\Users\\',
              'credential':r'(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,})',
              'private_key':r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'}
    for f in c.rglob('*'):
        if f.is_symlink():raise ValueError('Symlink in public candidate')
        if not f.is_file():continue
        files.append(f)
        if f.suffix in ('.pt','.pth','.safetensors','.npy','.npz','.so','.bin','.pyc') or any(k in f.parts for k in ('.git','.venv','__pycache__')):
            raise ValueError('Private/binary/cache artifact in candidate')
        if f.suffix not in ('.md','.py','.txt','.json','.html','.log'):continue
        text=f.read_text()
        # The scanner's literal test patterns are not leaked paths/credentials.
        if f.name!='verify_public.py':
            for kind,pattern in patterns.items():
                if re.search(pattern,text):findings.append({'file':str(f.relative_to(c)),'category':kind})
        if f.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
                if '://' in target or target.startswith('#'):continue
                path=unquote(target.split('#')[0].strip('<>'))
                if not (f.parent/path).exists():findings.append({'file':str(f.relative_to(c)),'missing_link':path})
                links+=1
    if findings:raise ValueError(json.dumps(findings))
    result={'status':'PASS','frozen_code_files':len(freeze['code_hashes']),'independent_boundary_selection':len(selected),
            'immutable_payloads_checked':checked,'public_files_scanned':len(files),'local_markdown_links':links,
            'private_path_secret_or_excluded_artifact_findings':0,'scope':'Pattern/static hygiene and file correspondence; not a security certification'}
    write_new(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
