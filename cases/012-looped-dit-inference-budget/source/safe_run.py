"""Current entrypoint: verify and reuse a completed phase before GPU loading.
The frozen generation runner remains source/run.py. This guard prevents a
completed native-parity receipt from being overwritten by a repeated command.
"""
import argparse
from pathlib import Path
from source.contracts import load,sha,digest,verify_freeze,verify_attempt
ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'pilot':3,'trace':3,'dev':27,'verify':6,'smoke':24,'main':192,'repeat':6}

def completed(root,phase,round_id=0):
    root=Path(root);verify_freeze(root)
    if phase=='parity':
        path=root/'results/parity.json'
        if not path.exists():return False
        r=load(path)
        if len(r.get('rows',[]))!=3 or {x['loops'] for x in r['rows']}!={1,2,4}:
            raise ValueError('Malformed existing parity receipt; refuse overwrite')
        if not all(x['pre_pil_bitwise'] and x['official_generate_png_equal'] and x['finite'] for x in r['rows']):
            raise ValueError('Existing parity failure is preserved; explicit diagnosis required')
        return True
    if phase not in EXPECTED:raise ValueError('Unknown phase')
    mf=root/'results'/f'jobs-{phase}-{round_id}.json'
    if not mf.exists():return False
    m=load(mf)
    if m['phase']!=phase or m['round']!=round_id or len(m['jobs'])!=EXPECTED[phase] or m['protocol_sha256']!=sha(root/'configs/protocol.json'):
        raise ValueError('Existing job manifest mismatch')
    for job in m['jobs']:
        parent=root/'results/attempts'/digest(job)[:24];success=False
        for p in sorted(parent.glob('attempt-*')):
            if p.name.endswith('.partial'):continue
            rec=verify_attempt(p)
            if rec['job']!=job:raise ValueError('Existing job identity mismatch')
            if rec['status']=='SUCCESS':success=True
        if not success:return False
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--encoder',required=True)
    p.add_argument('--phase',choices=['pilot','parity','trace','dev','verify','smoke','main','repeat'],required=True)
    p.add_argument('--round',type=int,choices=[0,1,2],default=0);a=p.parse_args()
    if completed(ROOT,a.phase,a.round):
        print('VERIFIED_COMPLETED_PHASE: reused saved records, no GPU load/generation, no result writes')
        return
    from source.run import main as original
    original()
if __name__=='__main__':main()
