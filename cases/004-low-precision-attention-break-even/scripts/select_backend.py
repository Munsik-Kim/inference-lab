"""Inspect an exact matching experimental table entry; never patches dispatch."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def select(table,query):
    unknown=dict(backend='bf16',status='UNKNOWN',reason='No exact validated fingerprint/model/input-family/shape entry')
    for field in ['environment_fingerprint','model_revision','input_family','spec_sha256']:
        if query.get(field)!=table.get(field):return unknown
    matches=[e for e in table['entries'] if e['shape']==query.get('shape')]
    if len(matches)!=1:return unknown
    e=matches[0]
    if e['status']=='LOCAL_CANDIDATE':return dict(backend='sage',status=e['status'],reason_codes=e['reason_codes'])
    return dict(backend='bf16',status=e['status'],reason_codes=e['reason_codes'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--table',type=Path,required=True);p.add_argument('--query',type=Path,required=True);a=p.parse_args()
    print(json.dumps(select(json.loads(a.table.read_text()),json.loads(a.query.read_text())),indent=2))
if __name__=='__main__':main()
