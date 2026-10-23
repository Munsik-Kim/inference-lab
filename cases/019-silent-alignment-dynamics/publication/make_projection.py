"""Display-only projections from saved grids; no integration or simulation."""
from pathlib import Path
import json,argparse
import numpy as np

def project(root):
    p1=root/'results/p1_20261023-104046';p2=root/'results/p2_20261023-135644'
    rows=json.loads((p1/'input_manifest.json').read_text())['planned_trajectories']
    def curve(run,keys,fields):
        groups={k:[] for k in fields};times=None
        for key in keys:
            with np.load(run/'raw'/(key+'.npz'),allow_pickle=False) as a:
                if times is not None and not np.array_equal(times,a['times']):raise ValueError('DISPLAY_GRID_MISMATCH')
                times=a['times']
                for k in fields:groups[k].append(a[k])
        # Select actual stored times, including zero/end and key reported times.
        idx=sorted(set(list(range(0,len(times),5))+[len(times)-1]+[int(i) for i,t in enumerate(times) if t in [0,.25,.5,1,2,5,10,20,40,80,155,160]]))
        return {'times':times[idx].tolist(),'saved_grid_indices':idx,
                'fields':{k:np.median(v,axis=0)[idx].tolist() for k,v in groups.items()},
                'aggregation':'pointwise median of R8' if len(keys)==8 else 'single disclosed R2 trajectory',
                'R_repetitions':len(keys),'source_keys':keys,'sampling':'selected actual saved grid indices, no new rho/g or generated trajectory'}
    rho={}
    for ri,r in enumerate([.01,.0001,.000001]):rho[str(r)]=curve(p1,[f'g0_rho{ri}_R{i}' for i in range(8)],['learnable_residual','right_distance'])
    gaps={}
    for gi,g in enumerate([.4,.1,.025]):
        gaps[str(g)]={'median':curve(p1,[f'g{gi}_rho2_R{i}' for i in range(8)],['right_distance','right_worst']),
                      'R2':curve(p1,[f'g{gi}_rho2_R2'],['right_distance','right_worst'])}
    arms={}
    summary=json.loads((p2/'derived/summary.json').read_text())
    for gi,g in [(0,.4),(2,.025)]:
        for star in [1,5]:
            key=f'{g}/{star}';arms[key]={}
            for arm in ['CONTINUE','GAUGE','ALIGN','ROTATE']:
                arms[key][arm]=curve(p2,[f'g{gi}_R{i}_t{star}_{arm}' for i in range(8)],['learnable_residual','right_distance','right_worst'])
    effects=[{'g':x['g'],'t_star':x['t_star'],'arm':x['arm'],'tau':x['tau'],
              **{k:x['statistics'][k] for k in ['J_normalized','G_normalized','A_normalized']}} for x in summary['groups']]
    return {'schema':'case019-display-v1','rho':rho,'gap':gaps,'arms':arms,'effects':effects,
            'scope':'R8 pointwise median curves; R2 separately labelled. Event tables first reduce each trajectory then aggregate. Actual saved conditions only.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--recorded-case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('New output only')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(project(a.recorded_case),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
if __name__=='__main__':main()
