"""Figures from saved exact/scalar records only. Existing Matplotlib env; no GPU."""
import argparse,json
from pathlib import Path
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'figure.dpi':150,'savefig.dpi':150})
def main(root,out):
 root=Path(root);out=Path(out);out.mkdir(exist_ok=True)
 p=json.loads((root/'p0.json').read_text())['exact'];fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 neg=[r for r in p['fixed_errors'] if r['second_error_sign']==-1];eps=[float(Fraction(r['epsilon'])) for r in neg]
 for k,label in [('BASE_abs_error','BASE'),('COMP_abs_error','COMP')]:ax[0].loglog(eps,[float(Fraction(r[k])) for r in neg],'o-',label=label)
 ax[0].set(xlabel='fixed additive error epsilon',ylabel='absolute output error',title='Fixed additive errors: cancellation removed');ax[0].legend();f=p['finite_codebook'];ax[1].bar(['BASE (1,3,1)','COMP (1,2,2)'],[float(Fraction(f[k])) for k in ['BASE_abs_error','COMP_abs_error']],color=['#317873','#cc6044']);ax[1].set(title='Independent 4-code scalar fixture (2 bit)',ylabel='absolute output error')
 for ext in ['svg','png']:fig.savefig(out/f'fig1_exact.{ext}')
 plt.close(fig)
 summary=root/'derived_summary.json'
 if not summary.exists():
  for name,title in [('fig2_components','Actual INT4 curves'),('fig3_transfer','CAL to EVAL selection transfer'),('fig4_local_nll','Local SSE versus downstream NLL')]:
   fig,ax=plt.subplots(figsize=(10,3));ax.axis('off');ax.text(.5,.6,title,ha='center',fontsize=16);ax.text(.5,.35,'P1 NOT RUN: model storage identity awaiting clarification. No measured curve.',ha='center');fig.savefig(out/(name+'.svg'));fig.savefig(out/(name+'.png'));plt.close(fig)
  return
 rows=json.loads(summary.read_text())['rows'];targets=[(b,p) for b in [4,12,20] for p in ['o_proj','down_proj']]
 fig,axes=plt.subplots(3,2,figsize=(13,11),layout='constrained')
 for ax,(b,p) in zip(axes.flat,targets):
  rs=sorted([r for r in rows if r['block']==b and r['projection']==p and r['split']=='EVAL-ID'],key=lambda r:r['alpha']);scale=rs[0]['SSE_before'];aa=[r['alpha'] for r in rs]
  for k,label in [('SSE_after','actual R SSE'),('fixedB_SSE','fixed-B R SSE'),('T_inherited','inherited term'),('T_cancellation','cancellation term'),('T_requant','requantization term')]:ax.plot(aa,[r[k]/scale for r in rs],'o-',label=label,ms=3)
  ax.set(title=f'block {b} {p} | EVAL 32 documents',xlabel='alpha',ylabel='SSE / alpha0 SSE');ax.axhline(0,color='gray',lw=.6)
 axes[0,0].legend(fontsize=7)
 for ext in ['svg','png']:fig.savefig(out/f'fig2_components.{ext}')
 plt.close(fig)
 fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
 for ax,(b,p) in zip(axes.flat,targets):
  for split,style in [('CAL-FIT','o-'),('EVAL-ID','s--')]:
   rs=sorted([r for r in rows if (r['block'],r['projection'],r['split'])==(b,p,split)],key=lambda r:r['alpha']);order=sorted(rs,key=lambda r:r['SSE_after']);ranks={r['alpha']:i+1 for i,r in enumerate(order)};ax.plot([r['alpha'] for r in rs],[ranks[r['alpha']] for r in rs],style,label=split)
  a=next(r['alpha'] for r in rs if r['selected_CAL']);ax.axvline(a,color='#cc6044',alpha=.5,label='CAL-frozen alpha');ax.set(title=f'block {b} {p}',xlabel='alpha',ylabel='local SSE rank (1 best)',yticks=[1,2,3,4,5])
 axes[0,0].legend(fontsize=8)
 for ext in ['svg','png']:fig.savefig(out/f'fig3_transfer.{ext}')
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
 for b,p in targets:
  rs=[r for r in rows if (r['block'],r['projection'],r['split'])==(b,p,'EVAL-ID')];ax.scatter([r['DeltaSSE_percent'] for r in rs],[r['DeltaNLL_vs_alpha0_doc_mean'] for r in rs],label=f'{b} {p}')
  for r in rs:ax.annotate(str(r['alpha']),(r['DeltaSSE_percent'],r['DeltaNLL_vs_alpha0_doc_mean']),fontsize=7)
 ax.axvline(0,color='gray',lw=.6);ax.axhline(0,color='gray',lw=.6);ax.set(xlabel='local SSE change vs alpha0 (%)',ylabel='NLL change vs alpha0 (nats/token)',title='All six targets × five alpha | same 32 EVAL documents');ax.legend(fontsize=8)
 for ext in ['svg','png']:fig.savefig(out/f'fig4_local_nll.{ext}')
 plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--figures',required=True);a=p.parse_args();main(a.results,a.figures)
