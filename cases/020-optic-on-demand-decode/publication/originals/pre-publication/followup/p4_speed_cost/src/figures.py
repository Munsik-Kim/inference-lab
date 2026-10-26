"""Four P4 figures; CPU only. All block points retained; no pooled P2/P4 samples."""
import pathlib,statistics,json
import numpy as np
def render(root,out,data):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'case020-p4','font.family':'DejaVu Sans'})
 root=pathlib.Path(root);out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);colors={'A':'#334155','B':'#0284c7','C':'#ea580c','S':'#7c3aed'};markers=['o','s','^','D','v','P'];rows=data['controlled_latency']
 def filt(z,c,pas,clock,method=None):return [r for r in rows if r['Z']==z and r['condition']==c and r['pass']==pas and r['clock']==clock and (method is None or r['method']==method)]
 def save(fig,name):
  fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight',facecolor='white');fig.savefig(out/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});plt.close(fig)
 fig,axs=plt.subplots(3,3,figsize=(14,11));fig.suptitle('Shared-context core: H host, same-call D, and batched E are separate passes',fontsize=14)
 for i,c in enumerate(['m1','unique2048','repeat2048']):
  methods=list('ABCS' if c=='unique2048' else 'ABC');label={'m1':'M1 / U1','unique2048':'M2048 / U2048','repeat2048':'M2048 / U512'}[c]
  for j,ax in enumerate(axs[i]):
   for mi,m in enumerate(methods):
    if j==1:
     for clock,shift,color in [('host',-.09,'#0369a1'),('event',.09,'#d97706')]:
      rs=filt(32,c,'D',clock,m);vals=[r['median_us'] for r in rs]
      for r in rs:ax.scatter(mi+shift+(r['block']-2.5)*.014,r['median_us'],marker=markers[r['block']],s=28,color=color,alpha=.8)
      ax.plot([mi+shift-.035,mi+shift+.035],[statistics.median(vals)]*2,color=color,lw=2)
    else:
     rs=filt(32,c,'H' if j==0 else 'E','host' if j==0 else 'event',m);key='median_us' if j==0 else 'mean_us'
     for r in rs:ax.scatter(mi+(r['block']-2.5)*.045,r[key],marker=markers[r['block']],s=31,color=colors[m],alpha=.85)
     med=statistics.median(r[key] for r in rs);ax.plot([mi-.13,mi+.13],[med]*2,color=colors[m],lw=2)
   ax.set_xticks(range(len(methods)),methods);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.17);ax.set_ylabel('us');ax.set_title(label+' | '+['H host median','D host / event medians','E event bracket mean /16'][j])
 handles=[plt.Line2D([],[],marker='o',ls='',color='#0369a1',label='D host median'),plt.Line2D([],[],marker='o',ls='',color='#d97706',label='D same-call event median')];axs[0,1].legend(handles=handles,fontsize=7)
 fig.text(.5,.007,'Each marker is one fresh process block (six distinct shapes). Bars: median across blocks. H/D K1; E K16. Shared residency is latency-only, not isolated memory.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.035,1,.95));save(fig,'01_shared_context_latency')
 fig,axs=plt.subplots(1,2,figsize=(12,5));fig.suptitle('Decoder width: paired H median ratios, same original ID bytes',fontsize=14)
 for ax,cond in zip(axs,['unique2048','repeat2048']):
  for comp,color,offset in [('C/B','#ea580c',-.09),('A/B','#334155',.09)]:
   mids=[]
   for i,z in enumerate([32,128,644]):
    rs=[r for r in data['width_comparison'] if r['Z']==z and r['condition']==cond and r['comparison']==comp];mids.append(statistics.median(r['median_ratio'] for r in rs))
    for r in rs:ax.scatter(i+offset+(r['block']-(2.5 if z==32 else 1))*.018,r['median_ratio'],color=color,marker=markers[r['block']],s=37)
   ax.plot(np.arange(3)+offset,mids,color=color,label=comp)
  ax.axhline(1,color='#64748b',ls='--',lw=1,label='equal');ax.axhline(.9,color='#b91c1c',ls=':',lw=1,label='0.90 descriptive cutoff');ax.set_xticks(range(3),['Z32 | n6','Z128 | n3','Z644 | n3']);ax.set_ylabel('Paired ratio (lower is faster)');ax.set_title('M2048 / U'+('2048' if cond=='unique2048' else '512'));ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.17);ax.legend(fontsize=8)
 fig.text(.5,.005,'A/B/C only; S belongs exclusively to Z32/M2048/U2048. Width also changes groups, codebook, resident storage and work. All shapes remain synthetic.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,.91));save(fig,'02_width_comparison')
 fig,axs=plt.subplots(1,2,figsize=(13,5));fig.suptitle('Actual isolated sessions: setup + N real synchronized calls + cleanup',fontsize=14);Ns=[128,1024,4096];life=data['lifecycle'];maxleft=0
 oldcase=root.parents[1]
 for ax,key,title in [(axs[0],'session_wall_seconds','Session wall'),(axs[1],'worker_wall_seconds','Parent spawn-to-exit wall')]:
  for m in 'ABC':
   for ni,N in enumerate(Ns):
    rs=sorted([r for r in life if r['method']==m and r['N']==N],key=lambda r:r['block']);vals=[r[key] for r in rs];med=statistics.median(vals)
    for r in rs:ax.scatter(ni+{'A':-.16,'B':0,'C':.16}[m]+(r['block']-1)*.025,r[key],color=colors[m],marker=markers[r['block']],s=36,zorder=3)
    ax.errorbar(ni+{'A':-.16,'B':0,'C':.16}[m],med,yerr=[[med-min(vals)],[max(vals)-med]],color=colors[m],capsize=4,fmt='_',label=m if ni==0 else None)
  ax.set_xticks(range(3),[str(n) for n in Ns]);ax.set_xlabel('N real calls (fresh worker for each N)');ax.set_ylabel('seconds');ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.18);ax.legend(fontsize=8,loc='upper left')
 # Original A/B linear scenario is unchanged and visually distinct from measured points.
 for m in 'AB':
  curves=[]
  for b in range(3):
   rr=json.loads((oldcase/f'results/20261026-1235/b{b}_LARGE_M2048_repeat75_{m}.json').read_text());curves.append([rr['setup']['wall_seconds']+n*statistics.median(rr['timing']['host_us'])/1e6 for n in Ns])
  vals=np.median(curves,axis=0);axs[0].plot(range(3),vals,color=colors[m],ls='--',lw=1,alpha=.65,label='P2 linear '+m)
 axs[0].legend(fontsize=8,loc='upper left');fig.text(.5,.005,'All3 block origins/ranges per method/N. Dashed: old setup+N*host-median scenarios, not refitted. Actual crossing reported only between tested N values.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,.91));save(fig,'03_actual_lifecycle')
 fig,axs=plt.subplots(2,2,figsize=(13,9));fig.suptitle('32MiB: historical baseline and new isolated memory-stage diagnostics',fontsize=14)
 ax=axs[0,0]
 for i,m in enumerate('ABC'):
  rs=[r for r in data['legacy_memory'] if r['shape']=='LARGE' and r['M']==2048 and r['U']==512 and r['method']==m];r=rs[0];parts=[r['baseline_allocated'],r['module_bytes']+r['ID_bytes'],r['residual_after_baseline']];bottom=0
  for val,color,label in zip(parts,['#475569','#0284c7','#f59e0b'],['Initialized baseline','Module + IDs','Additional residual']):ax.bar(i,val/2**20,bottom=bottom,color=color,label=label if i==0 else None);bottom+=val/2**20
  ax.text(i,bottom+5,f'{bottom:.2f}',ha='center')
 ax.set_xticks(range(3),list('ABC'));ax.set_ylabel('P2 isolated warm allocated MiB');ax.set_title('Original raw: baseline already32MiB');ax.set_ylim(bottom=0);ax.legend(fontsize=8)
 stageorder=['context_initialized_before_library_probe','library_probe_output_live','library_probe_released_baseline','after_artifact_load','before_first_fn','first_output_live','after_output_release','after_warmup20','after_module_ID_removal'];labels=['context','probe live','probe del','load','pre fn','out live','out del','warm20','module del']
 for ax,m in zip([axs[0,1],axs[1,0],axs[1,1]],'ABC'):
  rs={r['stage']:r for r in data['memory_stages'] if r['method']==m};x=np.arange(len(stageorder));ax.plot(x,[rs[k]['allocated_bytes']/2**20 for k in stageorder],marker='o',color=colors[m],label='allocated endpoint');ax.plot(x,[rs[k]['reserved_bytes']/2**20 for k in stageorder],ls='--',color='#64748b',label='reserved endpoint');ax.axhline(32,color='#b91c1c',ls=':',lw=.9);ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_ylim(bottom=0);ax.set_ylabel('MiB');ax.set_title('P4 isolated '+m+' | diagnostic, not timing peak');ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8)
 fig.text(.5,.005,'New stack: setWorkspaceForHandle / getCurrentCUDABlasHandle for active33554432-byte block after module removal. Shared latency-worker memory is excluded here.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.045,1,.95));save(fig,'04_memory_attribution')
