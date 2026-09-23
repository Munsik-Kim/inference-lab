"""Regenerate presentation figures from retained v2 scalar records only."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
ARMS=['NATIVE_FP32','UNIFORM_8','UNIFORM_5','LOWRANK_4_8_R2','MIXED_5_6_BUDGET']
COLORS=['#202e39','#45758b','#b27b40','#a34636','#658369']
SHORT=['Native FP32','Uniform 8','Uniform 5','Rank2 4+8','Mixed 5/6']
def read(p):return json.loads(p.read_text())
def save(fig,out,name):
 fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
def curve(s):
 tau=np.array([s['T']+1 if x is None else x for x in s['tau']]);return (tau[:,None]>np.arange(1,s['T']+1)).mean(0)
def style(ax):
 ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.facecolor':'#ffffff','axes.labelcolor':'#202e39','text.color':'#202e39'})
 fresh={(s,k):read(ROOT/'results/fresh'/f'seed{s}'/k/'summary.json') for s in range(3) for k in ARMS}
 fig,axs=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
 for s,ax in enumerate(axs):
  for k,c,l in zip(ARMS,COLORS,SHORT):ax.plot(np.arange(1,2049),curve(fresh[s,k]),label=l,color=c,lw=1.7)
  ax.axhline(.95,color='#888',ls=':',lw=.8);ax.set(xscale='log',xlim=(1,2048),ylim=(0,1.01),title=f'Checkpoint seed {s}',xlabel='Scored group-token position');style(ax)
 axs[0].set_ylabel('Never yet failed, S(t)');axs[-1].legend(fontsize=8)
 fig.suptitle('Fresh 1024 sequences per checkpoint · all prefixes, same inputs');fig.tight_layout();save(fig,a.output,'fresh_survival')
 fig,axs=plt.subplots(2,3,figsize=(13,7),sharey='row')
 for s in range(3):
  for mode,c,ls in zip(['D00','D10','D01','D11'],COLORS,['-','--',':','-.']):
   folder=ROOT/'results/precision'/f'seed{s}'/mode;m=read(folder/'summary.json')
   with np.load(folder/'predictions.npz',allow_pickle=False) as z:acc=(z['predictions'][:,1:]==z['gold']).mean(0)
   axs[0,s].plot(np.arange(1,2049),curve(m),label=mode,color=c,ls=ls,lw=1.4)
   axs[1,s].plot(np.arange(1,2049),acc,color=c,ls=ls,lw=.7,alpha=.7)
  for ax in axs[:,s]:ax.set(xscale='log',xlim=(1,2048),ylim=(0,1.01));style(ax)
  axs[0,s].set_title(f'Checkpoint seed {s}');axs[1,s].set_xlabel('Group-token position')
 axs[0,0].set_ylabel('First-error survival');axs[1,0].set_ylabel('Accuracy at this token');axs[0,2].legend(fontsize=8)
 fig.suptitle('Fixed FP32 coefficient values; 32 diagnostic sequences\nD00 FP32/FP32 · D10 FP64/FP32 · D01 FP32/FP64 · D11 FP64/FP64 (recurrence/readout)')
 fig.tight_layout();save(fig,a.output,'precision_survival_accuracy')
 fig,axs=plt.subplots(2,3,figsize=(13,6.8),sharey='row')
 for s in range(3):
  for k,c,l in zip(ARMS[2:],COLORS[2:],SHORT[2:]):
   r=fresh[s,k];x=r['ledger']['total_bytes']['128']/1024
   for row,metric in enumerate(['empirical_T05_all_tokens','supported_T05_grid']):
    val=r[metric];axs[row,s].scatter(x,0 if val is None else val,color=c,s=65,marker={'UNIFORM_5':'s','LOWRANK_4_8_R2':'o','MIXED_5_6_BUDGET':'^'}[k],label=l);axs[row,s].annotate(l,(x,0 if val is None else val),xytext={'UNIFORM_5':(0,10),'LOWRANK_4_8_R2':(0,16),'MIXED_5_6_BUDGET':(0,-20)}[k],textcoords='offset points',ha='center',fontsize=8,color=c,arrowprops={'arrowstyle':'-','color':c,'lw':.6})
  for ax in axs[:,s]:ax.set(xlim=(274,289),ylim=(0,190));style(ax)
  axs[0,s].set_title(f'Seed {s} · low-bit region');axs[1,s].set_xlabel('Total serialized bytes / 1024, N=128')
 axs[0,0].set_ylabel('Empirical T0.05, all token positions');axs[1,0].set_ylabel('Supported lower horizon, fixed grid')
 fig.suptitle('Fresh budget–horizon comparison · includes shared config, basis and token table\nConfidence family 195; zero denotes no qualifying declared horizon, not zero memory')
 fig.tight_layout();save(fig,a.output,'fresh_budget_horizon')
 # The saved paired analysis follows the one frozen bootstrap specification.
 from source.metrics import paired_summary
 pairs=[(ARMS[3],ARMS[4]),(ARMS[3],ARMS[2]),(ARMS[1],ARMS[0])]
 fig,axs=plt.subplots(1,3,figsize=(12,3.8),sharex=True)
 for j,(a1,b1) in enumerate(pairs):
  for s in range(3):
   r=paired_summary(fresh[s,a1],fresh[s,b1],bootstrap_seed=60101+100*s+j);v=r['RMST0_delta'];lo,hi=r['pointwise_95_CI']
   axs[j].errorbar(v,s,xerr=[[v-lo],[hi-v]],fmt='o',color=COLORS[3] if j<2 else COLORS[1],capsize=4)
  axs[j].axvline(0,color='#777',lw=.8);axs[j].set(yticks=[0,1,2],yticklabels=['Seed0','Seed1','Seed2'],title=f'{SHORT[ARMS.index(a1)]} − {SHORT[ARMS.index(b1)]}',xlabel='Paired RMST0 difference, tokens');style(axs[j])
 fig.suptitle('Fresh mean consecutive-correct length · pointwise 95% paired bootstrap intervals\n5000 sequence-level resamples within each checkpoint; no pooled seed inference')
 fig.tight_layout();save(fig,a.output,'fresh_paired_rmst')
 fig,axs=plt.subplots(3,4,figsize=(15,9))
 for s in range(3):
  folder=ROOT/'results/trace'/f'seed{s}'/'LOWRANK_4_8_R2';m=read(folder/'summary.json')
  with np.load(folder/'time_aggregates.npz',allow_pickle=False) as z:
   data={k:z[k].copy() for k in z.files}
  fields=['state_norm','native_state_error_l2','projection_leakage_mse','gold_margin']
  for j,field in enumerate(fields):
   ax=axs[s,j]
   for pop,ls in [('all','-'),('before_first_failure','--'),('at_or_after_first_failure',':')]:
    vals=data[f'{field}__{pop}__mean'];ax.plot(np.arange(len(vals)),vals,color=COLORS[3],ls=ls,label=pop,lw=1.1)
   if j==0:ax.plot(np.arange(2049),data['native_state_norm__all__mean'],color=COLORS[0],label='native state norm',lw=1)
   if j<3:ax.set_yscale('symlog',linthresh=1e-7)
   finite_taus=[x for x in m['tau'] if x is not None]
   if finite_taus:ax.axvline(np.median(finite_taus),color='#555',lw=.8,alpha=.7)
   ax.set_title(f'Seed{s} · {field}');ax.set_xlabel('Write index (BOS=0)');style(ax)
 axs[0,0].legend(fontsize=7)
 fig.suptitle('Rank2 diagnostic cohort: 32 sequences, scalar-only trajectories\nSolid/all, dashed/pre-failure, dotted/at-or-after: changing conditional subsets. Terminal rows excluded.\nAt t128, pre/post N=31/1, 28/4, 31/1; by t512, N=0/32 in every seed. Vertical line: median first error.')
 fig.tight_layout();save(fig,a.output,'rank2_time_diagnostics')
if __name__=='__main__':main()
