"""Time-only, single bounded interpolation/extrapolation. Never reads pixels."""
import argparse, statistics
from pathlib import Path
import numpy as np
from source.contracts import load,dump,sha,digest,verify_attempt
ROOT=Path(__file__).resolve().parents[1]
def rows(phase):
    out=[]
    for p in sorted((ROOT/'results/attempts').glob('*/attempt-*')):
        if not p.name.endswith('.partial'):
            r=verify_attempt(p)
            if r['status']=='SUCCESS' and r['job']['phase']==phase:out.append(r)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['predict','freeze']);a=ap.parse_args()
    dev=rows('dev')
    if len(dev)!=27:raise ValueError('Need all 27 DEV records; no partial selection')
    med={(l,s):statistics.median(r['complete_seconds'] for r in dev if r['job']['setting']['loops']==l and r['job']['setting']['steps']==s) for l in (1,2,4) for s in (25,50,75)}
    target=med[(4,50)]
    dest=ROOT/'configs/latency_prediction.json'
    if a.action=='predict':
        if dest.exists():raise ValueError('Prediction already frozen')
        predictions=[]
        for l in (1,2):
            slope,intercept=np.polyfit([25,50,75],[med[(l,s)] for s in (25,50,75)],1)
            if slope<=0:raise ValueError('Nonpositive DEV time slope')
            s=max(25,min(125,int(np.rint((target-intercept)/slope))))
            predictions.append({'id':f'verify_L{l}','loops':l,'steps':s,'extrapolated':not 25<=s<=75})
        dump(dest,{'schema':'case012-time-prediction-v1','quality_observed_for_selection':False,'target_seconds':target,'predictions':predictions,'dev_records':{r['job_id']:digest(r) for r in dev}})
    else:
        pred=load(dest);ver=rows('verify')
        if len(ver)!=6:raise ValueError('Need all six verification rows')
        settings=[]
        for l,name in [(1,'A_time'),(2,'B_time'),(4,'C_time')]:
            if l==4:s=50;t=target
            else:
                candidates=[(s,med[(l,s)]) for s in (25,50,75)]
                s0=next(x['steps'] for x in pred['predictions'] if x['loops']==l)
                tv=statistics.median(r['complete_seconds'] for r in ver if r['job']['setting']['loops']==l)
                candidates.append((s0,tv));s,t=min(candidates,key=lambda st:(abs(st[1]-target),st[0]))
            settings.append({'id':name,'loops':l,'steps':s,'dev_complete_median_seconds':t,'relative_time_mismatch':(t-target)/target,'within_5_percent':abs(t-target)/target<=.05})
        freeze=ROOT/'configs/main_settings.json'
        if freeze.exists():raise ValueError('MAIN settings already frozen')
        dump(freeze,{'schema':'case012-main-settings-v1','status':'FROZEN_BEFORE_MAIN','selected_by':'complete DEV time only, one estimate/verification per L; closest actually measured setting','target_seconds':target,'settings':settings,'protocol_sha256':sha(ROOT/'configs/protocol.json'),'prediction_sha256':sha(dest),'quality_annotations_used':0})
        print(load(freeze))
if __name__=='__main__':main()
