"""Figures from complete stored scalar/event records, no inference."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from interface_sensitivity.common import read,rows,sha,atomic
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();case=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=True)
 summary=read(case/'results/p1_r1/summary.json');methods=['BF16','RTN','D16'];interfaces=['I0_ORIGINAL','I1_ORDER_ONLY','I2_STYLE_RENAME','I3_OPAQUE_RENAME'];plt.rcParams.update({'font.size':10,'svg.hashsalt':'case015-v1'})
 fig,ax=plt.subplots(figsize=(9,3.5));values=np.array([[next(r['success'] for r in summary['success_table'] if r['method']==m and r['interface_id']==i) for i in interfaces] for m in methods]);ax.imshow(values,vmin=0,vmax=8,cmap='Blues');ax.set_xticks(range(4),['Original','Order only','lowerCamelCase','Opaque']);ax.set_yticks(range(3),methods)
 for y in range(3):
  for x in range(4):ax.text(x,y,f'{values[y,x]}/8',ha='center',va='center',color='white' if values[y,x]>=6 else '#17283e',weight='bold')
 ax.set_title('Functional success: existing 8-task DEV (higher is better)');fig.tight_layout();fig.savefig(a.output/'success_matrix.svg',metadata={'Date':None});plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
 for ax,m in zip(axes,methods):
  rr=[r for r in summary['paired_interface_counts'] if r['method']==m];x=np.arange(3)
  ax.bar(x-.18,[-r['baseline_only'] for r in rr],.36,label='Original success -> failure',color='#c35240');ax.bar(x+.18,[r['changed_only'] for r in rr],.36,label='Original failure -> success',color='#207a68');ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,['Order','Style','Opaque']);ax.set_ylim(-8,8);ax.set_title(m);ax.set_ylabel('Task count vs I0 (n=8)')
 axes[0].legend(fontsize=8,loc='lower left');fig.suptitle('Losses and gains kept separate; each changed interface paired to original');fig.tight_layout();fig.savefig(a.output/'paired_losses_gains.svg',metadata={'Date':None});plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,3.8));rr=list(summary['contrasts'].items());y=np.arange(len(rr));points=[r['estimate']*100 for k,r in rr];lo=[(r['estimate']-r['CI95'][0])*100 for k,r in rr];hi=[(r['CI95'][1]-r['estimate'])*100 for k,r in rr];ax.errorbar(points,y,xerr=[lo,hi],fmt='o',capsize=5,color='#21548b');ax.set_yticks(y,[k for k,r in rr]);ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Signed interface-change interaction (percentage points)');ax.set_title('Whole-task paired bootstrap95%, 10,000 draws; existing DEV8');fig.tight_layout();fig.savefig(a.output/'paired_interaction.svg',metadata={'Date':None});plt.close(fig)
 atomic(a.output/'figure_manifest.json',{'source_records_sha256':sha(case/'results/p1_r1/episodes.jsonl'),'summary_sha256':sha(case/'results/p1_r1/summary.json'),'generator_sha256':sha(Path(__file__)),'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(a.output.glob('*.svg'))},'confidence_interval':'only interaction plot; exploratory whole-task resampling, no invented success-count CI'},True)
 print(json.dumps({'figures':3,'source':'raw-derived summary'}))
if __name__=='__main__':main()
