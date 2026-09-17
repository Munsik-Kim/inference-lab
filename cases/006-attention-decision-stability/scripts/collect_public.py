"""Export scalar evidence only; private tensors and environments are never copied."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import load,write_new,file_hash,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);sources={};counts={}
    stages={'dev-paired':'paired','standard-eval':'paired','boundary-eval':'paired','length-support':'paired',
            'boundary-pool':'pool','model-timing':'timing','secondary':'secondary'}
    for stage,kind in stages.items():
        files=sorted((a.run/stage/'cells').glob('*.json'))
        if not files:raise ValueError('Missing measured stage: '+stage)
        groups={}
        for f in files:
            cell=load(f);row=cell['payload']
            if kind in ('paired','pool'):
                if row.get('evidence_kind')!='gpu_measurement' or row.get('mock'):
                    raise ValueError('Non-measurement record')
                key=f"{stage}-{row['task'].lower()}-{row['arm']}"
            else:key=stage
            groups.setdefault(key,[]).append(row)
            sources[str(f.relative_to(a.run))]={'ledger_sha256':file_hash(f),'payload_sha256':digest(row)}
        for key,rows in groups.items():write_new(a.output/kind/(key+'.json'),rows)
        counts[stage]=len(files)
    write_new(a.output/'source_manifest.json',{'scope':'Scalar payloads from immutable local ledgers; full tensors/vocabulary vectors excluded',
               'counts':counts,'sources':sources})
    print(json.dumps(counts))


if __name__=='__main__':main()
