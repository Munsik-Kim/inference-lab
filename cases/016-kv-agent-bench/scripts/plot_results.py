#!/usr/bin/env python3
"""Optional CPU plots. All eight cells/all96 episodes are retained."""
import argparse
import collections
import hashlib
import json
import pathlib
import platform
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=pathlib.Path,required=True);args=p.parse_args();case=args.case.resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    summary=json.loads((case/'results/p1_r1/summary.json').read_text());eps=[json.loads(p.read_text())for p in sorted((case/'results/p1_r1/episodes').glob('*.json'))]
    figdir=case/'figures';figdir.mkdir(exist_ok=True);plt.rcParams.update({'svg.hashsalt':'case016-v1','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    groups=summary['groups'];y=np.arange(len(groups));fig,ax=plt.subplots(figsize=(10,6))
    for i,(key,label,color)in enumerate([('verified','Verified workflow','#06685f'),('state','Correct state','#a34b32'),('final','Correct final JSON','#7d8690')]):
        values=[g[key]for g in groups];bars=ax.barh(y+(i-1)*.23,values,height=.22,label=label,color=color);ax.bar_label(bars,padding=3,fontsize=8)
    ax.set_yticks(y,[f'{g["kv"]} | menu{g["menu"]} | cap{g["cap"]}'for g in groups]);ax.invert_yaxis();ax.set_xlim(0,12.8);ax.set_xticks(range(0,13,2));ax.set_xlabel('Observed successes /12 DEV fixtures (higher is better)');ax.set_title('P1: all8 frozen cells; the same12 fixtures recur across settings');ax.legend(loc='lower right');fig.tight_layout();fig.savefig(figdir/'p1_success.svg',metadata={'Date':None});plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));bottom=np.zeros(2)
    for reason,color in [('TOOL_CALL_LIMIT','#a34b32'),('FINAL','#7d8690'),('PARSE_STOP','#d5ab6f')]:
        vals=np.array([sum(e['terminal_reason']==reason for e in eps if e['kv']==kv)for kv in ('B16','F4')]);bars=ax.bar(['B16','F4'],vals,bottom=bottom,label=reason,color=color);ax.bar_label(bars,label_type='center');bottom+=vals
    ax.set_ylim(0,52);ax.set_ylabel('Episodes /48 conditions');ax.set_title('FINAL is a response termination, not a correctness verdict');ax.legend(bbox_to_anchor=(1.02,1),loc='upper left');fig.tight_layout();fig.savefig(figdir/'p1_termination.svg',metadata={'Date':None});plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for kv,marker in [('B16','o'),('F4','x')]:
        for success,color in [(False,'#a34b32'),(True,'#06685f')]:
            rows=[e for e in eps if e['kv']==kv and e['VERIFIED_WORKFLOW_SUCCESS']==success]
            if rows:ax.scatter([e['output_tokens']for e in rows],[e['wall_seconds']for e in rows],marker=marker,c=color,alpha=.8,label=f'{kv} '+('verified'if success else'not verified'))
    ax.set_xlim(left=0);ax.set_ylim(bottom=0);ax.set_xlabel('Actual generated tokens per episode');ax.set_ylabel('Episode wall seconds (load excluded)');ax.set_title('All96 conditions:12 shared DEV fixtures, C1');ax.legend();fig.tight_layout();fig.savefig(figdir/'p1_tokens_cost.svg',metadata={'Date':None});plt.close(fig)
    caches={kv:json.loads((case/f'results/p0_diagnostics_r3/{kv}_final_config_cache_after.json').read_text())for kv in ('B16','F8','F4')};fig,ax=plt.subplots(figsize=(9,4))
    bars=ax.bar(['B16','F8 (P0 only)','F4'],[v['cache_config']['kv_cache_size_tokens']for v in caches.values()],color=['#7d8690','#d5ab6f','#06685f']);ax.bar_label(bars,labels=[f'{v["cache_config"]["kv_cache_size_tokens"]:,}'for v in caches.values()],padding=3);ax.set_ylim(0,120000);ax.set_ylabel('Allocated token capacity');ax.set_title('Same requested4GiB cache pool; this is capacity, not job throughput');fig.tight_layout();fig.savefig(figdir/'kv_capacity.svg',metadata={'Date':None});plt.close(fig)
    receipt={'python_version':platform.python_version(),'environment_label':'EXISTING_CPU_PLOT_ENV_READ_ONLY','matplotlib':matplotlib.__version__,'numpy':np.__version__,'model_loaded':False,'figures':[{'file':str(f.relative_to(case)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}for f in sorted(figdir.glob('*.svg'))],'all_cells_plotted':8,'all_episode_points':len(eps)}
    (case/'provenance/plot_environment.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'figures':len(receipt['figures']),'episodes':len(eps)}))

if __name__=='__main__':main()
