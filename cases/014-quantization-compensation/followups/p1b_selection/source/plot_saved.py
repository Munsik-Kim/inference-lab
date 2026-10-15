"""Four standalone figures from all six targets' saved scalar records. CPU only."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TARGETS=[f'model.layers.{b}.'+('self_attn.' if p=='o_proj' else 'mlp.')+p for b in [4,12,20] for p in ['o_proj','down_proj']]
def label(t):return f"block {t.split('.')[2]} / {t.split('.')[-1]}"

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((run/'alpha_results.csv').open()));s=json.loads((run/'summary.json').read_text());choices={r['target']:r['policy_alpha'] for r in s['choices']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def export(fig,name):
        fig.savefig(out/(name+'.svg'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    colors=['#335c81','#b86b30','#587c52']
    for metric,name,title in [('SSE_vs_alpha0_ratio','01_alpha_sse','Saved-weight local SSE: all six independent interventions'),('NLL_vs_alpha0_doc_mean','02_alpha_nll','Gold next-token NLL: all six independent interventions')]:
        fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
        for t,ax in zip(TARGETS,axes.flat):
            for split,c,ls in zip(['SELECT-ID','FRESH-EVAL-ID','FRESH-SHIFT'],colors,['--','-',':']):
                rr=sorted([r for r in rows if r['target']==t and r['split']==split],key=lambda r:float(r['alpha']))
                ax.plot([float(r['alpha']) for r in rr],[float(r[metric])*(100 if metric.startswith('SSE') else 1) for r in rr],marker='o',linestyle=ls,color=c,label=split+f" n={rr[0]['documents']}")
            er=[r for r in rows if r['target']==t and r['split']=='FRESH-EVAL-ID']
            for pol,marker,col,offset in [('OLD_FIT','s','#666666',0),('SELECT_SSE','^','#a13d63',.008),('SELECT_NLL','D','#634a9d',-.008)]:
                aa=choices[t][pol];r=next(r for r in er if float(r['alpha'])==aa)
                ax.scatter([aa+offset],[float(r[metric])*(100 if metric.startswith('SSE') else 1)],s=90,marker=marker,facecolors='none',edgecolors=col,label=pol,zorder=5)
            ax.axhline(0,color='#999999',lw=.7);ax.set_title(label(t));ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('alpha (stored candidate)')
            ax.set_ylabel('SSE change vs alpha0 (%)' if metric.startswith('SSE') else 'delta NLL vs alpha0 (nats/token)');ax.grid(alpha=.15)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
        fig.suptitle(title+'\nSame old K and artifacts; development SELECT excluded from fresh evaluation')
        export(fig,name)
    raw=[json.loads(l) for f in sorted((run/'cells/FRESH-EVAL-ID').glob('*.jsonl')) for l in f.read_text().splitlines()]
    lookup={(r['target'],r['alpha'],r['document_id_hash']):r for r in raw}
    fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    for t,ax in zip(TARGETS,axes.flat):
        for alpha,col in zip([.25,.5,.75,1],['#4477aa','#228833','#ccbb44','#aa3377']):
            rr=[r for r in raw if r['target']==t and r['alpha']==alpha]
            x=[100*(r['SSE_after']/r['SSE_before']-1) for r in rr]
            y=[(r['NLL_sum']-lookup[t,0,r['document_id_hash']]['NLL_sum'])/255 for r in rr]
            ax.scatter(x,y,s=18,alpha=.65,color=col,label=f'alpha={alpha:g}')
        ax.axhline(0,color='#888888',lw=.7);ax.axvline(0,color='#888888',lw=.7);ax.set_title(label(t)+' / 64 books × 4 nonzero strengths')
        ax.set_xlabel('document local SSE change vs alpha0 (%)');ax.set_ylabel('gold delta NLL (nats/token)');ax.grid(alpha=.15)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=4)
    fig.suptitle('Local fidelity versus suffix gold score on FRESH-EVAL-ID\n1,536 paired comparisons share 64 books; points are not independent documents')
    export(fig,'03_local_sse_vs_nll')
    fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    configs=[('SELECT_SSE','OLD_FIT','SSE'),('SELECT_SSE','FIXED_QUARTER','SSE'),('SELECT_SSE','FIXED_HALF','SSE'),('SELECT_NLL','SELECT_SSE','NLL')]
    for ax,(cp,bp,kind) in zip(axes.flat,configs):
        rr=[next(r for r in s['target_contrasts'] if r['target']==t and r['split']=='FRESH-EVAL-ID' and r['candidate']==cp and r['baseline']==bp) for t in TARGETS]
        m=next(r for r in s['macro_contrasts'] if r['split']=='FRESH-EVAL-ID' and r['candidate']==cp and r['baseline']==bp)
        values=[r['SSE_difference_over_alpha0'] if kind=='SSE' else r['NLL_doc_mean_difference'] for r in rr]+[m['D_macro'] if kind=='SSE' else m['NLL_macro_doc_mean_difference']]
        cis=[r['SSE_pointwise_CI95'] if kind=='SSE' else r['NLL_pointwise_CI95'] for r in rr]+[m['D_macro_CI95'] if kind=='SSE' else m['NLL_macro_CI95']]
        factor=100 if kind=='SSE' else 1
        for j,(v,ci) in enumerate(zip(values,cis)):
            ax.plot(np.array(ci)*factor,[j,j],color='#5b6572',lw=1.7);ax.scatter([v*factor],[j],s=42,color='#335c81',marker='D' if j==6 else 'o')
        ax.set_yticks(range(7),[label(t) for t in TARGETS]+['mean of 6 fixed interventions']);ax.invert_yaxis();ax.axvline(0,color='#999999',lw=.8)
        ax.set_title(cp+' minus '+bp);ax.set_xlabel('percentage points of alpha0 SSE' if kind=='SSE' else 'delta NLL (nats/token)');ax.grid(axis='x',alpha=.15)
    fig.suptitle('Frozen-policy paired effects / FRESH-EVAL-ID: 64 new books\n10,000 shared document resamples; only SSE-vs-OLD_FIT macro is primary')
    export(fig,'04_paired_effects')
    print('Generated 4 SVG and 4 PNG figures from saved rows, no GPU inference.')

if __name__=='__main__':main()
