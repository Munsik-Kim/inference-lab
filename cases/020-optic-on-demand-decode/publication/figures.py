"""Publication figures from raw-derived tables; block origins retained, CPU only."""
from pathlib import Path
import statistics as st
import numpy as np

def render(data,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'case020-publication'})
 out=Path(out);out.mkdir(parents=True,exist_ok=True);colors={'A':'#334155','B':'#0284c7','C':'#e76f16'};marks=['o','s','^','D','v','P']
 def save(f,name):f.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight',facecolor='white');f.savefig(out/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});svg=out/(name+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n');plt.close(f)
 # Same shape, original method-isolated accounting, all3 blocks represented.
 f,axs=plt.subplots(1,2,figsize=(11,4.5));f.suptitle('Same output, different memory costs | P2 LARGE Z32, M2048 / U512')
 for ax,key,title in [(axs[0],'resident_bytes','Tensors retained between calls'),(axs[1],'total_allocated_peak_bytes','PyTorch allocated peak during calls')]:
  for i,m in enumerate('ABC'):
   rs=[x for x in data['P2_memory'] if x['shape']=='LARGE' and x['M']==2048 and x['U']==512 and x['method']==m];v=[x[key]/2**20 for x in rs];med=st.median(v);ax.bar(i,med,color=colors[m],width=.6,alpha=.8)
   for r in rs:ax.scatter(i+(r['block']-1)*.055,r[key]/2**20,color='black',s=20,marker=marks[r['block']])
   ax.text(i,med+10,f'{med:.4f}' if med<1 else f'{med:.3f}',ha='center')
  ax.set_xticks(range(3),['A: precompute all','B: requested rows','C: distinct rows']);ax.tick_params(axis='x',labelsize=9);ax.set_ylim(0,350 if key.startswith('total') else 300);ax.set_ylabel('MiB (2^20 bytes)');ax.set_title(title);ax.grid(axis='y',alpha=.14)
 f.text(.5,.015,'Each marker = one of3 isolated process blocks; overlapping values retained. Output16MiB. Allocated includes common32MiB baseline; not device-wide VRAM.',ha='center',fontsize=8);f.tight_layout(rect=(0,.06,1,.91));save(f,'01_memory')
 f,axs=plt.subplots(2,3,figsize=(13,8));f.suptitle('Direct synchronized calls | primary H, same process, alternating method order')
 for j,(z,M,U) in enumerate([(32,1,1),(32,2048,512),(644,2048,512)]):
  ax=axs[0,j];rr=[x for x in data['representative_direct_block'] if (x['Z'],x['M'],x['U'])==(z,M,U)];n=len(rr)//3
  for i,m in enumerate('ABC'):
   rs=[x for x in rr if x['method']==m]
   for r in rs:
    x=i+(r['block']-(n-1)/2)*.035;ax.scatter(x,r['median_us'],color=colors[m],marker=marks[r['block']],s=30);ax.plot([x,x],[r['median_us'],r['p95_us']],color=colors[m],alpha=.35,lw=1);ax.scatter(x,r['p95_us'],color=colors[m],marker='_',s=30)
  ax.set_xticks(range(3),list('ABC'));ax.set_ylabel('us');ax.set_ylim(bottom=0);ax.set_title(f'Z{z} | M{M} / U{U} | {n} blocks');ax.grid(axis='y',alpha=.15)
  ax=axs[1,j];rs=sorted([x for x in data['P4_paired_H'] if (x['Z'],x['M'],x['U'],x['comparison'])==(z,M,U,'C/B')],key=lambda x:x['block'])
  for r in rs:ax.scatter(r['block'],r['median_ratio'],marker=marks[r['block']],color=colors['C'],s=40)
  ax.axhline(1,color='#64748b',ls='--');ax.axhline(st.median(x['median_ratio'] for x in rs),color=colors['C'],alpha=.5);ax.set_ylim(bottom=0);ax.set_ylabel('C/B block-median ratio');ax.set_xlabel('Fresh process block');ax.set_xticks(range(n));ax.grid(axis='y',alpha=.15)
 f.text(.5,.01,'Top: filled marker=within-block median; tick=p95 of120 H calls (line is not an uncertainty bar). Bottom: each block ratio; orange line=block median, dashed=equal.',ha='center',fontsize=9);f.tight_layout(rect=(0,.05,1,.93));save(f,'02_direct_calls')
 f,axs=plt.subplots(1,2,figsize=(12,5));f.suptitle('Intermediate width changes storage and the observed C/B call-time ratio')
 for ax,U in zip(axs,[2048,512]):
  for z in [32,128,644]:
   wr=next(x for x in data['width_storage'] if x['Z']==z);mem=wr['module_MiB'];rr=[x for x in data['P4_paired_H'] if x['Z']==z and x['M']==2048 and x['U']==U and x['comparison']=='C/B']
   for r in rr:ax.scatter(mem,r['median_ratio'],color=colors['C'],marker=marks[r['block']],s=36)
   ax.annotate(f'Z{z} | n{len(rr)}',(mem,max(x['median_ratio'] for x in rr)),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
  ax.axhline(1,ls='--',color='#64748b');ax.set_xlim(-.5,12);ax.set_ylim(0,2.35);ax.set_title(f'M2048 / U{U} | A/B/C measured');ax.set_xlabel('Compact module unique storage MiB (saved inventory)');ax.set_ylabel('C/B H block-median ratio');ax.grid(alpha=.15)
 f.text(.5,.005,'The dense BF16 table stays256MiB. Width changes groups, codebooks, storage and work together; these are synthetic shapes, not trained E4B.',ha='center',fontsize=9);f.tight_layout(rect=(0,.05,1,.9));save(f,'03_width_storage')
 f,axs=plt.subplots(1,2,figsize=(12,5));f.suptitle('Preparation plus real repeated calls | isolated Z32, M2048 / U512')
 for ax,key,title in [(axs[0],'session_wall_seconds','Session: read + setup + N calls + cleanup'),(axs[1],'worker_wall_seconds','Whole worker: Python start through exit')]:
  for i,N in enumerate([128,1024,4096]):
   for m in 'ABC':
    rs=[x for x in data['lifecycle_block'] if x['N']==N and x['method']==m];x=i+{'A':-.19,'B':0,'C':.19}[m];vals=[r[key] for r in rs]
    for r in rs:ax.scatter(x+(r['block']-1)*.035,r[key],marker=marks[r['block']],color=colors[m],s=34)
    med=st.median(vals);ax.errorbar(x,med,yerr=[[med-min(vals)],[max(vals)-med]],fmt='_',color=colors[m],capsize=3,label=m if i==0 else None)
  ax.set_xticks(range(3),['128','1024','4096']);ax.set_xlabel('N actual calls; independent workers for each N');ax.set_ylabel('seconds');ax.set_ylim(bottom=0);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.15);ax.legend()
 f.text(.5,.005,'All3 origins per method/N; center=median, range=min–max, not a confidence interval. Cold first call included. No precise crossover interpolated.',ha='center',fontsize=9);f.tight_layout(rect=(0,.05,1,.9));save(f,'04_session_worker')
