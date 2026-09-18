"""Add completed held-out/timing scalars to an existing partial public case.

No tensors or calibration/development overwrites. Full private records remain intact.
"""
import argparse
import hashlib
import json
from pathlib import Path

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-run',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    a=p.parse_args(); case=a.case; run=a.private_run
    if sha(run/'selection.json') != sha(case/'results/raw/selection.json'):
        raise ValueError('Selection changed')
    files=[]
    for stage in ['heldout','timing0','timing1','timing2']:
        if read(run/stage/'summary.json')['status'] != 'PASS':
            raise ValueError('Incomplete stage')
        files.extend((stage,f) for f in sorted((run/stage).glob('c007-*.json')))
        files.append((stage,run/stage/'summary.json'))
        if stage != 'heldout':files.append((stage,run/stage/'blocks.json'))
    targets=[case/'results/raw'/stage/f.name for stage,f in files]
    record=case/'provenance/completion_export.json'
    if record.exists() or any(p.exists() for p in targets):
        raise FileExistsError('Refusing to overwrite public evidence')
    entries=[]
    for (stage,src),dst in zip(files,targets):
        obj=read(src)
        if isinstance(obj,dict):
            for u in obj.get('local',[]):
                tokens=u['token_metrics']
                if len(tokens)!=32:raise ValueError('Wrong private sample count')
                index=max(range(32),key=lambda i:tokens[i]['relative_error'])
                u['token_metrics']=[dict(tokens[index],sample_index=index,
                                         token_position=index*511//31)]
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('x') as f:
            json.dump(obj,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
            f.write('\n')
        entries.append({'file':str(dst.relative_to(case)),
                        'private_original_sha256':sha(src),'public_sha256':sha(dst)})
    record.write_text(json.dumps({'kind':'COMPLETION_SCALAR_EXPORT',
        'originals_preserved':True,'tensor_files_exported':0,
        'token_policy':'Worst of 32 sampled tokens per method/prompt; all prompt sufficient statistics unchanged',
        'files':entries},indent=2)+'\n')
    print(json.dumps({'exported_files':len(entries),'status':'PASS'}))

if __name__=='__main__':main()
