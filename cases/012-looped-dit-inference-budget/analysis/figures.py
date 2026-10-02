"""Plot saved timing only. No quality values are invented for pending images."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from source.contracts import load,dump,sha
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
s=load(ROOT/'analysis/summary.json');ts=s['timing'];labels=[];mid=[];low=[];high=[]
for name,t in ts.items():
 labels.append(f"L{t['loops']} / S{t['steps']}");m=t['main_complete_median_seconds'];mid.append(m);low.append(m-t['main_complete_minmax_seconds'][0]);high.append(t['main_complete_minmax_seconds'][1]-m)
fig,ax=plt.subplots(figsize=(7,4));ax.set_facecolor('#f7f6f2');fig.patch.set_facecolor('#f7f6f2')
ax.errorbar(range(len(labels)),mid,yerr=[low,high],fmt='o',color='#a34b32',capsize=6,label='MAIN median; bars = observed min–max')
ax.set_xticks(range(len(labels)),labels);ax.set_ylabel('Complete request (seconds)');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_title('Same checkpoint: measured generation budgets');ax.legend(fontsize=9);fig.tight_layout()
(ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/complete-request-time.png',dpi=160);fig.savefig(ROOT/'figures/complete-request-time.svg');plt.close(fig)
dump(ROOT/'figures/source.json',{'summary_sha256':sha(ROOT/'analysis/summary.json'),'unit':'seconds','boundary':'warmed complete request','observations_per_setting':64,'error_bars':'observed MAIN min–max, not confidence intervals','quality_curve':'ANNOTATION_PENDING; no values plotted'})
