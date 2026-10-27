"""CPU-only presentation from audited tables; no GPU runner imports."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read,write,identity

def table(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument('--a',type=Path,required=True);p.add_argument('--b',type=Path,required=True);p.add_argument('--c',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'case021-posthoc','figure.dpi':130})
    docs=table(a.a/'cross_precision_document.csv');js=table(a.a/'js_cross_precision.csv');drift=table(a.a/'fixed_reference_drift.csv')
    ev=[r for r in docs if r['original_split']=='EVAL' and r['length']=='512'];jv=[r for r in js if r['original_split']=='EVAL' and r['length']=='512']
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,rows,keys,title in [(axes[0,0],ev,['K00','K01','K10','K11'],'Directed KL: D1 first, P second'),(axes[0,1],jv,['JS00','JS01','JS10','JS11'],'Symmetric JSD')]:
        for i,key in enumerate(keys):
            vals=np.array([float(r[key]) for r in rows])*1000;ax.scatter(np.full(len(vals),i)+np.linspace(-.13,.13,len(vals)),vals,s=16,alpha=.6,color='#3267a8');ax.plot([i-.2,i+.2],[vals.mean()]*2,color='#d64b35',lw=2)
        ax.set(xticks=range(4),xticklabels=['R0 / R0','R0 / R1','R1 / R0','R1 / R1'],ylabel='Divergence (nats × 1000)',title=title);ax.grid(axis='y',alpha=.2)
    ax=axes[1,0]
    for i,key in enumerate(['delta_P_given_D1R0','delta_P_given_D1R1']):
        vals=np.array([float(r[key]) for r in ev])*1000;ax.scatter(np.full(24,i)+np.linspace(-.13,.13,24),vals,s=18,alpha=.65);ax.plot([i-.2,i+.2],[vals.mean()]*2,color='#d64b35',lw=2)
    ax.axhline(0,color='gray',lw=1);ax.set(xticks=[0,1],xticklabels=['Fixed D1_R0','Fixed D1_R1'],ylabel='KL before − after P change (nats × 1000)',title='Positive: P_R1 is closer to that fixed reference')
    ax=axes[1,1]
    for i,cond in enumerate(['D1_R0','D1_R1','P_R0','P_R1']):
        vals=np.array([float(r['KL']) for r in drift if r['original_split']=='EVAL' and r['length']=='512' and r['candidate']==cond])*1000;ax.scatter(np.full(24,i)+np.linspace(-.13,.13,24),vals,s=16,alpha=.6);ax.plot([i-.2,i+.2],[vals.mean()]*2,color='#d64b35',lw=2)
    ax.set(xticks=range(4),xticklabels=['D1 R0','D1 R1','P R0','P R1'],ylabel='KL(D0_R0 || candidate), nats × 1000',title='A different fixed reference: D0_R0')
    fig.suptitle('Post-hoc existing EVAL: 24 documents, 512-token input\nEach dot = mean of 16 selected positions; red line = document mean',fontsize=14)
    save(fig,a.output,'01_cross_precision')
    pairs=table(a.b/'rounding_cancellation_pairs.csv');tensors=table(a.b/'rounding_cancellation_tensors.csv');evb=[r for r in pairs if r['analysis_split']=='original_EVAL_posthoc']
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for layer,marker in [('25','o'),('27','^')]:
        for tensor,color in [('q','#3267a8'),('k','#d27b22')]:
            rows=[r for r in evb if r['layer']==layer and r['tensor']==tensor];x=[float(r['individual_error_change_R1_minus_R0']) for r in rows];y=[float(r['pair_gap_change_R1_minus_R0']) for r in rows]
            axes[0].scatter(x,y,label=f'Layer {layer} {tensor.upper()} (24)',marker=marker,color=color,s=25,alpha=.65)
            axes[1].scatter([float(r['cross_change_R1_minus_R0']) for r in rows],y,marker=marker,color=color,s=25,alpha=.65)
    axes[0].set(xlabel='Change in sum of individual FP64 errors (SSE)',ylabel='Change in D1–P gap (SSE)',title='Left/up: individual error falls, pair gap grows');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Change in signed cross term',ylabel='Change in D1–P gap (SSE)',title='Cross term is one part of the exact identity')
    for ax in axes[:2]:ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='gray',lw=.8)
    ax=axes[2]
    for mode,color in [('R0','#3267a8'),('R1','#d27b22')]:
        rows=[r for r in tensors if r['analysis_split']=='original_EVAL_posthoc' and r['rounding']==mode];offset=-.13 if mode=='R0' else .13
        vals=[sum(float(r[k]) for r in rows) for k in ['d_energy','n_energy','signed_cross','pair_gap_SSE']];ax.bar(np.arange(4)+offset,vals,width=.26,label=mode,color=color)
    ax.axhline(0,color='gray',lw=.8);ax.set(xticks=range(4),xticklabels=['Input d²','Noise n²','2 dot(d,n)','Pair gap²'],ylabel='Summed energy over same coordinates',title='Gap² = d² + n² + 2 dot(d,n)');ax.legend()
    fig.suptitle('Local replay: 96 EVAL document/layer/Q-or-K cells, 16 positions per tensor\nR1 − R0 changes; selected coordinates, not a full-model KL attribution',fontsize=13)
    save(fig,a.output,'02_rounding_cancellation')
    group=table(a.c/'operator_grouped.csv');raw=table(a.c/'operator_metrics.csv')
    kinds=['ONE_HOT_PLUS','ONE_HOT_MINUS','PAIR_WITHIN_GROUP','PAIR_ACROSS_GROUP','GROUP64_SIGNS','ACTUAL_DEV'];labels=['+ one-hot','− one-hot','Within group','Across groups','64 signs','DEV activation'];comparisons=['D1_vs_dense64','P_vs_dense64','D1_vs_exact64','P_vs_exact64','P_vs_D1'];colors=['#2865a8','#88b7db','#d97b20','#d33645','#683b9e']
    fig,axes=plt.subplots(3,1,figsize=(13,11),layout='constrained')
    for ax,layer in zip(axes,[0,25,27]):
        for j,(comp,color) in enumerate(zip(comparisons,colors)):
            x=np.arange(6)+(j-2)*.13;values=[float(next(r for r in group if int(r['layer'])==layer and r['kind']==kind and r['comparison']==comp)['relative_RMSE']) for kind in kinds];ax.plot(x,values,marker='o',ms=5,lw=.7,label=comp.replace('_vs_',' / '),color=color)
            for i,kind in enumerate(kinds):
                points=[float(r['relative_RMSE']) for r in raw if int(r['layer'])==layer and r['kind']==kind and r['comparison']==comp];ax.scatter(np.full(len(points),x[i])+np.linspace(-.025,.025,len(points)),points,s=10,alpha=.5,color=color)
        ax.set(xticks=range(6),xticklabels=labels,ylabel='Relative RMSE (symlog)',title=f'Layer {layer} q_proj | M512 / I1024 / O2048');ax.set_yscale('symlog',linthresh=1e-5);ax.set_ylim(bottom=-2e-6);ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=5,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.29))
    fig.suptitle('Two local FP64 references and the D1–P gap\nLarge markers: pooled SSE/energy; small dots: each fixture. Zero retained.\nFirst block outputs; 3 process blocks verify repeats, not 3 independent models.',fontsize=13)
    save(fig,a.output,'03_projection_oracles')
    write(a.output/'figure_receipt.json',{'figures':3,'PNG_and_SVG':True,'sources':{str(f):identity(f) for folder in [a.a,a.b,a.c] for f in folder.glob('*.csv')},'CPU_only':True,'Torch_imported':'torch' in sys.modules,'point_units':'Documents/tensor cells/fixtures; never counted as independent models','plot_type':'symlog only in projection relative RMSE, zeros retained'})
def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=160);fig.savefig(out/(name+'.svg'),metadata={'Date':None});plt.close(fig)
if __name__=='__main__':main()
