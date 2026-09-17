"""Publication figures from verified measured pairs; no illustrative score data."""
import argparse,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import load,external_directory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary=load(a.summary);pairs=load(a.pairs);out=external_directory(a.output,a.case)
    primary=[r for r in pairs if r['split']=='standard' and r['length']==4096]
    if not primary:raise ValueError('No measured primary pairs; refuse empty/fake figures')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':160})
    colors={'A_PUBLIC':'C0','V4':'C1'}
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.png'),metadata={'Software':'Case006 measured-data figure generator'});plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for arm in colors:
        rows=sorted([r for r in primary if r['arm']==arm],key=lambda r:r['item_id'])
        ax.scatter([r['base_gap'] for r in rows],[r['delta_choice_nll'] for r in rows],s=18,alpha=.65,label=arm,color=colors[arm])
    ax.axhline(0,color='0.4',lw=1);ax.set_xlabel('BF16 winner gap (native raw-logit units)');ax.set_ylabel('Candidate − BF16 gold choice NLL (nats)')
    ax.set_title('Paired score changes · STANDARD_EVAL · 192 scenarios, L4096');ax.legend();save(fig,'01_score_changes')
    for split in ('standard','boundary_pool'):
        key=split+'/L4096'
        if key not in summary['groups']:continue
        group=summary['groups'][key]
        for arm in colors:
            fig,ax=plt.subplots(figsize=(8,4.7));bins=group['arms'][arm]['bins']
            labels=['EXACT_TIE','[0.0,0.1)','[0.1,0.5)','[0.5,1.0)','[1.0,inf)']
            for ti,task in enumerate(('RETRIEVAL','COMPARISON','CODE')):
                for bi,label in enumerate(labels):
                    b=next(r for r in bins if r['task']==task and r['gap_bin']==label);n=b['flips']['denominator']
                    if not n:continue
                    point=b['flips']['rate'];lo,hi=b['flip_wilson95'];x=bi+(ti-1)*.18
                    ax.errorbar(x,point*100,yerr=[[max(0,point-lo)*100],[max(0,hi-point)*100]],fmt='o',color=f'C{ti}',capsize=2,label=task if bi==next(i for i,l in enumerate(labels) if next(z for z in bins if z['task']==task and z['gap_bin']==l)['n']) else None)
                    ax.annotate(f"{b['flips']['numerator']}/{n}",(x,point*100),xytext=(0,-14 if point>.9 else 7),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xticks(range(len(labels)),['tie','(0, .1)','[.1, .5)','[.5, 1)','[1, ∞)']);ax.set_xlabel('BF16 winner-gap bin');ax.set_ylabel('Choice flips (%)');ax.set_ylim(-2,102)
            ax.set_title(f'{arm} · {split} · {group["independent_scenarios"]} scenarios\nPointwise Wilson intervals; empty bins omitted, not connected');ax.legend();save(fig,f'02_flips_{split}_{arm}')
    fig,ax=plt.subplots(figsize=(8,4.8))
    for arm in colors:
        rows=[r for r in primary if r['arm']==arm]
        ax.scatter([100*r['local_reference_error'] for r in rows],[r['delta_choice_nll'] for r in rows],s=18,alpha=.65,label=arm,color=colors[arm])
    ax.axhline(0,color='0.4',lw=1);ax.set_xlabel('Pooled local output relative error vs FP32 reference (%)');ax.set_ylabel('Paired gold choice NLL change (nats)');ax.set_title('Local error and answer-score change · one point per scenario/arm\nExploratory association; 16 heads are not independent answer trials');ax.legend();save(fig,'03_local_to_score')
    from src.study_analysis import wilson
    for split in ('standard','boundary_pool'):
        fig,ax=plt.subplots(figsize=(8,4.6))
        labels=['EXACT_TIE','[0.0,0.1)','[0.1,0.5)','[0.5,1.0)','[1.0,inf)']
        for ai,arm in enumerate(colors):
            for bi,label in enumerate(labels):
                rows=[r for r in pairs if r['split']==split and r['length']==4096 and r['arm']==arm and r['B_score']['correct'] and r['gap_bin']==label]
                if not rows:continue
                k=sum(r['cell']=='regression' for r in rows);n=len(rows);lo,hi=wilson(k,n);x=bi+(ai-.5)*.18
                ax.errorbar(x,100*k/n,yerr=[[max(0,100*(k/n-lo))],[max(0,100*(hi-k/n))]],fmt='o',color=colors[arm],capsize=3)
                ax.annotate(f'{arm}: {k}/{n}',(x,100*k/n),xytext=(0,10+ai*13),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(5),['tie','(0, .1)','[.1, .5)','[.5, 1)','[1, ∞)']);ax.set_ylim(-2,102)
        ax.set_xlabel('BF16-correct gold-margin bin (equals winner gap for a unique correct winner)');ax.set_ylabel('Regressions / BF16-correct items (%)')
        ax.set_title(f'Gold-grounded regression · {split}\nPointwise Wilson intervals; tasks pooled for display, per-task records retained')
        save(fig,'03_regressions_'+split)
    boundaries=['layer13_attention_before_o_proj_last','layer13_post_attention_residual_last','block_13_output_last','block_14_output_last','block_18_output_last','block_22_output_last','block_27_output_last','final_norm_last','final_logits']
    fig,ax=plt.subplots(figsize=(9,4.8))
    for arm in colors:
        rows=[r for r in primary if r['arm']==arm]
        ys=[np.median([r['hidden_differences'][b]['relative_error'] for r in rows])*100 for b in boundaries]
        ax.plot(range(len(boundaries)),ys,'o-',label=arm,color=colors[arm])
    ax.set_xticks(range(len(boundaries)),['attn 13','residual 13','block 13','block 14','block 18','block 22','block 27','final norm','logits'],rotation=30,ha='right')
    ax.set_ylabel('Median relative difference vs BF16 (%)');ax.set_title('Last-position propagation · STANDARD_EVAL\nEach boundary has its own denominator; not an amplification factor');ax.legend();save(fig,'04_hidden_propagation')
    fig,ax=plt.subplots(figsize=(8,3.8));categories=['both_correct','regression','gain','both_wrong'];vals=[]
    for arm in colors:vals.append([summary['groups']['standard/L4096']['arms'][arm]['outcomes'][c] for c in categories])
    image=ax.imshow(vals,cmap='Blues',aspect='auto');ax.set_xticks(range(4),['Both correct','B correct → wrong','B wrong → correct','Both wrong']);ax.set_yticks([0,1],list(colors))
    for i,row in enumerate(vals):
        for j,v in enumerate(row):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>96 else 'black')
    ax.set_title('Gold-grounded outcomes · 192 paired standard scenarios\nBoth-wrong may include choice disagreement');save(fig,'05_outcome_table')
    if 'A_PUBLIC' in summary.get('model_timing',{}):
        fig,ax=plt.subplots(figsize=(7.5,4.8))
        for arm in colors:
            timing=summary['model_timing'][arm];metric=summary['groups']['standard/L4096']['arms'][arm]['task_balanced']['delta_choice_nll']
            x=timing['paired_ratio_median'];y=metric['mean'];xc=timing['ci95'];yc=metric['ci95']
            ax.errorbar(x,y,xerr=[[max(0,x-xc[0])],[max(0,xc[1]-x)]],yerr=[[max(0,y-yc[0])],[max(0,yc[1]-y)]],fmt='o',capsize=4,label=arm,color=colors[arm])
        ax.axvline(1,color='0.5',lw=1);ax.axhline(0,color='0.5',lw=1)
        ax.set_xlim(0,1.2) # Do not magnify a sub-percent timing change with an automatic tight axis.
        ax.set_xlabel('Measured complete model-prefill speedup vs BF16 (12 scenarios × 3 rounds)');ax.set_ylabel('Gold choice NLL change (192 standard scenarios)');ax.set_title('Model prefill cost and gold score · separate fixed subsets\nPointwise 95% intervals; no combined winner score');ax.legend();save(fig,'06_model_cost_and_score')
    print('Figures generated from actual records only')


if __name__=='__main__':main()
