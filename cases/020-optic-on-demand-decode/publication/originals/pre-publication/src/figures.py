"""CPU Matplotlib figures; statistical scope is three process blocks."""
import pathlib,numpy as np
def render(data,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'case020','savefig.facecolor':'white'})
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
 colors={'A':'#334155','B':'#0284c7','C':'#ea580c'};names={'A':'A dense gather','B':'B request decode','C':'C unique decode'};shapes=['SMALL','LARGE'];conds=[(1,1),(128,128),(128,32),(2048,2048),(2048,512)];x=np.arange(5);labels=[f'{m:,} / {u:,}' for m,u in conds]
 def subset(key,shape,method=None,M=None,U=None):return [r for r in data[key] if r['shape']==shape and (method is None or r['method']==method) and (M is None or r['M']==M) and (U is None or r['U']==U)]
 def save(fig,name):
  fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight',metadata={'Software':'Case020 independent CPU postprocess'});fig.savefig(out/(name+'.svg'),bbox_inches='tight',metadata={'Date':None,'Creator':'Case020 independent CPU postprocess'});plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(13,8.5));fig.suptitle('Same compressed representation: resident payload and allocated peak',fontsize=15)
 for j,shape in enumerate(shapes):
  ax=axs[0,j]
  for i,m in enumerate(['A','B','C']):
   rows=subset('memory',shape,m);v=rows[0]['gpu_resident_tensor_bytes']/2**20;ax.bar(i,v,color=colors[m]);ax.text(i,v,f'{v:.3f} MiB',ha='center',va='bottom',fontsize=10)
  ax.set_xticks(range(3),['A','B','C']);ax.set_ylabel('Resident module tensor storage (MiB)');ax.set_title(shape);ax.margins(y=.2)
  ax=axs[1,j]
  for m in ['A','B','C']:
   mid=[]
   for ci,(M,U) in enumerate(conds):
    vals=np.array([r['steady_allocated_peak_bytes']/2**20 for r in subset('memory',shape,m,M,U)]);mid.append(float(np.median(vals)));ax.scatter(np.full(len(vals),ci)+{'A':-.07,'B':0,'C':.07}[m],vals,color=colors[m],s=16,alpha=.65)
   ax.plot(x,mid,color=colors[m],label=names[m])
  ax.set_xticks(x,labels,rotation=20);ax.set_xlabel('Request M / actual unique U');ax.set_ylabel('Timed PyTorch allocated peak (MiB)');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
 fig.text(.5,.005,'90 cell-blocks. Payload excludes common IDs/output; peak includes them. Dots: three process blocks; lines: block median. Setup/cold peaks in memory.csv.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.035,1,.95));save(fig,'01_memory')
 fig,axs=plt.subplots(2,2,figsize=(13,9));fig.suptitle('Warm-cache operator latency: synchronized host wall and event brackets',fontsize=15)
 for j,shape in enumerate(shapes):
  for row in [0,1]:
   ax=axs[row,j]
   for m in ['A','B','C']:
    med=[];p95=[]
    for ci,(M,U) in enumerate(conds):
     rows=subset('latency',shape,m,M,U);key='host_median_us' if row==0 else 'event_bracket_mean_per_call_us';vals=[r[key] for r in rows];med.append(float(np.median(vals)))
     for r in rows:ax.scatter(ci+{'A':-.06,'B':0,'C':.06}[m],r[key],marker=['o','s','^'][r['block']],s=20,color=colors[m],alpha=.65)
     if row==0:p95.append(float(np.median([r['host_p95_us'] for r in rows])))
    ax.plot(x,med,color=colors[m],label=names[m])
    if row==0:ax.plot(x,p95,color=colors[m],ls='--',alpha=.7)
   ax.set_xticks(x,labels,rotation=20);ax.set_xlabel('Request M / actual unique U');ax.grid(axis='y',alpha=.2);ax.set_ylim(bottom=0);ax.legend(fontsize=8)
   ax.set_title(shape+(' | host: solid median, dashed p95' if row==0 else ' | event: mean bracket / K'))
   ax.set_ylabel('Synchronized host wall (us)' if row==0 else 'CUDA stream bracket / call (us)')
 fig.text(.5,.005,'Markers o/s/^: blocks0/1/2; lines: median across blocks. 100 host calls or100 event brackets/block.\nK=32,8,1 for M=1,128,2048; event p95/K is not a single-request tail.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.045,1,.95));save(fig,'02_latency')
 fig,axs=plt.subplots(2,2,figsize=(13,8.5));fig.suptitle('Does deduplication pay? Total C minus B and separate diagnostics',fontsize=15)
 for j,shape in enumerate(shapes):
  ax=axs[0,j];mid=[];lo=[];hi=[];all_values=[]
  for ci,(M,U) in enumerate(conds):
   rs=[r for r in data['paired'] if r['shape']==shape and r['M']==M and r['U']==U];vals=np.array([r['C_minus_B_host_us'] for r in rs]);all_values.extend(vals);mid.append(float(np.median(vals)));lo.append(mid[-1]-vals.min());hi.append(vals.max()-mid[-1])
   for r in rs:ax.scatter(ci,r['C_minus_B_host_us'],marker=['o','s','^'][r['block']],color=colors['C'],s=26,zorder=3)
  ax.errorbar(x,mid,yerr=[lo,hi],color=colors['C'],capsize=4);ax.axhline(0,color='black',lw=.9);ax.set_ylim(min(-20,min(all_values)-40),max(all_values)+60);ax.set_ylabel('C host median minus B (us)\npositive = added cost');ax.set_title(shape);ax.set_xticks(x,labels,rotation=20);ax.set_xlabel('Request M / actual unique U');ax.grid(axis='y',alpha=.2)
  ax=axs[1,j];bottom=np.zeros(5)
  for part,color in [('unique','#fbbf24'),('decode','#38bdf8'),('inverse','#a78bfa')]:
   values=[]
   for M,U in conds:
    rs=[r for r in data['actual'].values() if r['cell']['shape']==shape and r['cell']['M']==M and r['cell']['U']==U and r['cell']['method']=='C'];values.append(np.median([np.median(r['diagnostic_parts_us'][part]) for r in rs]))
   ax.bar(x,values,bottom=bottom,color=color,label=part);bottom+=np.array(values)
  ax.set_xticks(x,labels,rotation=20);ax.set_xlabel('Request M / actual unique U');ax.set_ylabel('Separate synchronized diagnosis (us)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
 fig.text(.5,.005,'Top: main paired C-B; markers o/s/^=blocks0/1/2, bars=observed range. Bottom:10 separate diagnostic passes/block.\nAdded synchronization changes diagnostic cost; sums need not match C.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.045,1,.95));save(fig,'03_unique_cost')
