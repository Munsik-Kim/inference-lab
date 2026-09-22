"""Read-only byte verification of the two existing local checkpoint directories."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    p=argparse.ArgumentParser();p.add_argument('--bf16',type=Path,required=True);p.add_argument('--w4',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'provenance/artifacts.json');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    expected=json.loads(a.manifest.read_text());roots={'BF16':a.bf16,'W4':a.w4};records={}
    for name,record in expected.items():
        part=PurePosixPath(name)
        if part.is_absolute() or '..' in part.parts or len(part.parts)<2 or part.parts[0] not in roots:raise ValueError('unsafe artifact member')
        path=roots[part.parts[0]].joinpath(*part.parts[1:]);h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=record['sha256'] or record['sha256']!=record['expected'] or path.stat().st_size!=record['bytes']:raise ValueError('artifact identity differs: '+name)
        records[name]={'sha256':h.hexdigest(),'bytes':path.stat().st_size}
    with a.output.open('x') as f:f.write(json.dumps({'status':'PASS','files':records,'scope':'existing local checkpoint bytes; no load, quantization or model forward'},indent=2)+'\n')
    print(json.dumps({'status':'PASS','files':len(records)}))


if __name__=='__main__':main()
