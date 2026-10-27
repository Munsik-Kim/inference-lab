"""CPU figures from recorded numeric CSVs. No model or GPU dependency."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rows(path):return list(csv.DictReader(Path(path).open()))
def val(row,key):return float(row[key])
def grouped_error(records):
    return np.sqrt(sum(val(r,'SSE') for r in records)/sum(val(r,'reference_energy') for r in records))

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    layer=[r for r in rows(a.summary/'layer_metrics.csv') if r['split']=='EVAL' and int(r['length'])==512]
    doc=[r for r in rows(a.summary/'document_metrics.csv') if r['split']=='EVAL' and int(r['length'])==512]
    primary=[r for r in rows(a.summary/'primary.csv') if int(r['length'])==512]
    decision=json.loads((a.summary/'decision.json').read_text());colors=['#247ba0','#df7c24','#5950a3'];pairs=[('D0','D1'),('D1','P'),('D0','P')]
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for j,(ref,cand) in enumerate(pairs):
        subset=[r for r in layer if r['reference']==ref+'_R0' and r['candidate']==cand+'_R0']
        ys=[grouped_error([r for r in subset if int(r['layer'])==i]) for i in range(28)]
        axs[0].plot(range(28),ys,color=colors[j],label=ref+' - '+cand,lw=2)
        ds=[val(r,'relative_RMSE') for r in doc if r['reference']==ref+'_R0' and r['candidate']==cand+'_R0']
        axs[1].scatter(j+np.linspace(-.12,.12,len(ds)),ds,s=22,color=colors[j],alpha=.65)
        axs[1].plot([j-.2,j+.2],[np.median(ds)]*2,color=colors[j],lw=3)
    axs[0].set(xlabel='Block index (0-based)',ylabel='sqrt(sum SSE / sum reference energy)',title='Full-path block outputs, R0; 24 EVAL documents')
    axs[0].legend();axs[1].set(xticks=range(3),xticklabels=['D0-D1','D1-P','D0-P'],ylabel='Selected-logit relative RMSE',title='Each point is one document; bars are medians')
    fig.suptitle('Same INT4 artifact: representation and execution both change results\n512 tokens; each path has its own upstream history (not causal amplification)',fontsize=13)
    save(fig,a.output,'01_representation_execution')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for row in primary:
        axs[0].plot([0,1],[val(row,'E0'),val(row,'E1')],marker='o',ms=3,lw=.8,alpha=.65,color='#247ba0')
    axs[0].set(xticks=[0,1],xticklabels=['R0 native','R1 FP32 intermediate'],ylabel='KL(D1 || P), nats / selected position',title=f'{len(primary)} paired EVAL512 documents, 16 positions each')
    for j,c in enumerate(['D1_R0','P_R0','D0_R1','D1_R1','P_R1']):
        ds=[val(r,'KL') for r in doc if r['reference']=='D0_R0' and r['candidate']==c]
        axs[1].scatter(j+np.linspace(-.12,.12,len(ds)),ds,s=18,alpha=.6,color=colors[j%3]);axs[1].plot([j-.2,j+.2],[np.mean(ds)]*2,lw=3,color=colors[j%3])
    axs[1].set(xticks=range(5),xticklabels=['D1/R0','P/R0','D0/R1','D1/R1','P/R1'],ylabel='KL(D0_R0 || condition), nats',title='Drift from one fixed reference; bars are means')
    effect=decision.get('rounding_effect','NOT_RUN');stat=decision.get('primary512',{})
    fig.suptitle(f'Changing a shared rounding boundary: {effect}\nDelta = E0-E1; mean {stat.get("mean_delta",float("nan")):.6g}; document bootstrap95% [{stat.get("ci_low",float("nan")):.6g}, {stat.get("ci_high",float("nan")):.6g}]',fontsize=12)
    save(fig,a.output,'02_rounding_discrepancy')
    local=rows(a.summary/'local_replay_metrics.csv');selected=decision['selected_layers'];boundaries=['self_attn.q_proj','self_attn.q_norm.output','post_rope_q','attention_output','self_attn.o_proj','mlp','block_output'];labels=['Q projection','Q norm','Q after RoPE','Attention out','O projection','MLP','Block out']
    has_intervention=decision['intervention']['status']=='SELECTED'
    fig,axs=plt.subplots(1,3 if has_intervention else 2,figsize=(16 if has_intervention else 13,5.4),constrained_layout=True)
    for ax,idx in zip(axs[:2],selected):
        for j,(ref,cand) in enumerate(pairs):
            for r,style in [('R0','-'),('R1','--')]:
                sub=[x for x in local if x['split']=='EVAL' and int(x['layer'])==idx and x['reference']==ref and x['candidate']==cand and x['rounding']==r]
                ys=[grouped_error([x for x in sub if x['boundary']==b]) for b in boundaries]
                ax.plot(range(len(boundaries)),ys,style,marker='o',ms=3,color=colors[j],label=f'{ref}-{cand} {r}')
        ax.set(xticks=range(len(boundaries)),xticklabels=labels,ylabel='sqrt(sum SSE / sum reference energy)',title=f'Layer {idx}: identical saved D0_R0 block input')
        ax.tick_params(axis='x',rotation=35,labelsize=9)
    axs[0].legend(ncol=2,fontsize=8)
    if has_intervention:
        intervention=rows(a.summary/'eval_intervention_metrics.csv');change=[val(r,'KL_change') for r in intervention];candidate=decision['intervention']['candidate']
        axs[2].scatter(range(len(change)),change,c=['#df7c24' if x>0 else '#247ba0' for x in change],s=23)
        axs[2].axhline(0,color='#444444',lw=1);axs[2].axhline(np.mean(change),color='#5950a3',lw=1.5,ls='--',label='Document mean')
        axs[2].set(xlabel='EVAL document (fixed order)',ylabel='KL after replacement - baseline KL (nats)',title=f'Layer {candidate["layer"]}: {candidate["boundary"]}\nPositive=worse, negative=less discrepancy');axs[2].legend(fontsize=8)
    fig.suptitle('Same-input local replay and a DEV-selected boundary replacement\n24 EVAL documents; local curves compare distinct tensor spaces, not causal amplification',fontsize=12)
    save(fig,a.output,'03_same_input_local_replay')
    (a.output/'figure_scope.json').write_text(json.dumps({'figures':3,'formats':['PNG','SVG'],'primary_documents':len(primary),'main_length':512,'local_layers':selected,'error_bars':'No inferred confidence bars; points show documents; bootstrap interval is labeled separately','units':'dimensionless relative RMSE and natural-log KL nats'},indent=2)+'\n')

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.svg'));plt.close(fig)

if __name__=='__main__':main()
