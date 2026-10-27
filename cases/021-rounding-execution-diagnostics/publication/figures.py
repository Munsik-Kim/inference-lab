"""Publication figures from public scalar rows; no private data/GPU."""
import argparse,json,csv,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
f=json.loads((a.case/'summary/facts.json').read_text());plt.rcParams.update({'font.size':11,'svg.hashsalt':'case021-publication','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for x in f['sparse']:axes[0].plot(x['K'],[100*v/(512*2048) for v in x['mismatches']],marker='o',label=f"L{x['layer']} {x['document_id']}")
axes[0].set_xscale('log',base=2);axes[0].set_xticks([1,2,8,64,256,1024],labels=['1','2','8','64','256','1024']);axes[0].set(xlabel='Retained-axis budget K (log spacing)',ylabel='D1/P numeric mismatch (%)',title='One/two terms agree in all four inputs');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
rows=[x for x in csv.DictReader((a.case/'evidence/p0p3/primary.csv').open()) if int(x['length'])==512]
for i,x in enumerate(rows):axes[1].plot([0,1],[float(x['E0']),float(x['E1'])],color='#3274a1',alpha=.35,marker='.',ms=5)
axes[1].plot([0,1],[f['E0'],f['E1']],color='#cd5328',marker='o',lw=3,label='Equal-document mean');axes[1].set(xticks=[0,1],xticklabels=['R0 native','R1 FP32 intermediates'],ylabel='KL(D1 || P), nats',title='Original24 EVAL documents: INCONCLUSIVE');axes[1].legend(fontsize=9)
fig.suptitle('Different execution paths: local inputs and full-model distributions\nLeft: fixed P5 posthoc probes. Right: original16-position document means.',fontsize=13)
for ext in ['png','svg']:fig.savefig(a.output/('01_overview.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else None)
plt.close(fig)
g=next(x for x in f['cross'] if x['split']=='original_EVAL_posthoc' and x['length']==512);fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,prefix in zip(axes,['K','JS']):
 values=[[g[prefix+'00'],g[prefix+'01']],[g[prefix+'10'],g[prefix+'11']]];im=ax.imshow(values,cmap='Blues');ax.set(xticks=[0,1],xticklabels=['P_R0','P_R1'],yticks=[0,1],yticklabels=['D1_R0','D1_R1'],title='Directed KL (nats)' if prefix=='K' else 'Symmetric JSD (nats)')
 for i in range(2):
  for j in range(2):ax.text(j,i,f'{values[i][j]:.7f}',ha='center',va='center',color='black',bbox={'facecolor':'white','alpha':.7,'edgecolor':'none'})
 fig.colorbar(im,ax=ax,fraction=.046)
fig.suptitle('P4 posthoc: fixed-reference comparisons stay within a row\n24 original EVAL documents, length512,16 positions/document',fontsize=12)
for ext in ['png','svg']:fig.savefig(a.output/('02_cross_precision.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else None)
plt.close(fig)
(a.output/'scope.json').write_text(json.dumps({'figures':2,'public_scalar_rows_only':True,'new_GPU':0,'points':'documents or fixed inputs, not output coordinates as independent samples'},indent=2)+'\n')
