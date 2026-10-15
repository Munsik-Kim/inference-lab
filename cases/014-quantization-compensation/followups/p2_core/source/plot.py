"""Four standalone P2 figures from saved raw/summary data; CPU only."""
import argparse,json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    s=json.loads((run/'summary.json').read_text());methods=['BF16','RTN','COMP_QUARTER','COMP_HALF'];colors=['#657586','#b57532','#376b8c','#905373'];labels=['BF16','RTN','Quarter','Half']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def export(fig,name):
        fig.savefig(out/(name+'.svg'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
    for split,name,corpus in [('EVAL-ID','01_pg19','PG19:64 new books'),('SHIFT','02_arxiv','ArXiv:16 new content groups')]:
        rr=[next(r for r in s['method_summaries'] if r['method']==m and r['split']==split) for m in methods];fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
        for ax,key,title in zip(axes,['NLL_token_weighted','PPL'],['Gold NLL (nats/token)','PPL = exp(total NLL / tokens)']):
            vals=[r[key] for r in rr];ax.bar(labels,vals,color=colors,width=.65)
            for j,v in enumerate(vals):ax.annotate(f'{v:.5f}' if key.startswith('NLL') else f'{v:.3f}',(j,v),xytext=(0,5),textcoords='offset points',ha='center')
            ax.set_title(title);ax.set_ylim(0,max(vals)*1.18);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        fig.suptitle(corpus+'\nFour actual whole models / fixed CAL32 / stored INT4 → dense BF16')
        export(fig,name)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    with (run/'paired_document_differences.csv').open() as f:document_differences=list(csv.DictReader(f))
    for row,split in enumerate(['EVAL-ID','SHIFT']):
        for col,method in enumerate(['COMP_QUARTER','COMP_HALF']):
            vals=np.array([r['delta_NLL'] for r in document_differences if r['split']==split and r['candidate']==method and r['baseline']=='RTN'],dtype=float)
            ax=axes[row,col];ax.hist(vals,bins=12,color=colors[2+col],alpha=.75,edgecolor='white');ax.axvline(0,color='#555',ls='--',lw=1)
            q=next(r for r in s['paired_comparisons'] if r['split']==split and r['candidate']==method and r['baseline']=='RTN');ax.axvline(q['mean_document_delta_NLL'],color='#222',lw=1.4,label='paired mean')
            ax.set_title(f"{split}: {method} minus RTN / n={len(vals)}");ax.set_xlabel('delta gold NLL (nats/token)');ax.set_ylabel('documents');ax.legend()
        
    fig.suptitle('Document-paired changes; negative values favor compensation\nAll documents shown; SHIFT is not pooled with PG19')
    export(fig,'03_paired_document_nll')
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for method,color,label in zip(methods,colors,labels):
        q=next(r for r in s['method_summaries'] if r['method']==method and r['split']=='EVAL-ID');x=q['projection_core_bytes']/2**20;y=q['PPL'];axes[0].scatter(x,y,color=color,s=65)
        offset=(8,14) if method=='COMP_QUARTER' else (8,-14) if method=='COMP_HALF' else (8,6)
        axes[0].annotate(label,(x,y),xytext=offset,textcoords='offset points',color=color)
    axes[0].set_xlabel('projection core MiB (common weights excluded)');axes[0].set_ylabel('fresh PG19 PPL');axes[0].set_title('Storage accounting and measured quality');axes[0].grid(alpha=.15)
    times=s['resources']['stage_seconds_inclusive'];vals=[times['RTN_build'],times['COMP_QUARTER_build'],times['COMP_HALF_build']];axes[1].bar(['RTN','Quarter','Half'],vals,color=colors[1:])
    for j,v in enumerate(vals):axes[1].annotate(f'{v:.1f}s',(j,v),xytext=(0,5),textcoords='offset points',ha='center')
    axes[1].set_ylim(0,max(vals)*1.2);axes[1].set_ylabel('offline construction seconds');axes[1].set_title('Native CAL capture + CPU fit/pack/diagnostics/I/O')
    fig.suptitle('All196 projections per model; quantized methods have identical core bytes\nBF16 point uses tensor payload; quantized points include file header/scales/padding. No serving speed measurement.')
    export(fig,'04_storage_and_offline_cost');print('Created4 SVG/PNG P2 figures from saved rows.')
if __name__=='__main__':main()
