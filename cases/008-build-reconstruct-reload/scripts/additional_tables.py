"""Planned same-size cost contrasts and split-separated reconstruction tables."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.modelpack.common import read,write
from tools.modelpack.analysis import timing_summary
from tools.modelpack.numerics import recovery
import numpy as np


def summarize(case,output):
    raw=case/'results/raw/R';cal=read(raw/'calibration_readout.json');dev=read(raw/'development.json')
    eta=read(raw/'selection.json')['eta'];heldout=read(raw/'model_records.json');timings=read(raw/'timing.json')
    result={'eta':eta,'recovery_by_split':{},'same_size_cost':{},'scope':'planned contrasts from unchanged raw records; no new model runs'}
    for name in ['I25','P25','S50']:
        selected=[r for r in cal if r['structure']==name]
        a=[r['local']['uncorrected']['squared_error'] for r in selected];b=[r['local']['repaired']['squared_error'] for r in selected]
        ca={'n_prompts':len(selected),'pooled_recovery':recovery(np.array(a),np.array(b))}
        a=[r['local'][name]['uncorrected']['squared_error'] for r in dev];b=[r['local'][name][str(eta)]['squared_error'] for r in dev]
        dv={'n_prompts':len(dev),'pooled_recovery':recovery(np.array(a),np.array(b))}
        a={r['id']:r for r in heldout if r['arm']==name};b={r['id']:r for r in heldout if r['arm']==name+'-R'}
        if set(a)!=set(b):raise ValueError('Missing heldout pairs')
        ho={'n_prompts':len(a),'pooled_recovery':recovery(np.array([a[i]['local']['squared_error'] for i in sorted(a)]),np.array([b[i]['local']['squared_error'] for i in sorted(a)]))}
        result['recovery_by_split'][name]={'calibration':ca,'development':dv,'heldout':ho}
        cost=timing_summary([r for r in timings if r['arm'] in [name,name+'-R']],name)
        result['same_size_cost'][name]={boundary:values[name+'-R'] for boundary,values in cost.items()}
    write(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.case,a.output)
