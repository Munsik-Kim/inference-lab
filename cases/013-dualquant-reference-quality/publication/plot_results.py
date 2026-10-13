"""Three descriptive figures from existing records only; no inference or fitting."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
CASE=Path(__file__).resolve().parents[1]
COLORS=['#d8c9b8','#716b65','#b46d42','#242f35']
METHODS=['BF16','RTN','D16','D4']

def plot(output):
    output=Path(output)
    if output.exists() or output.resolve().is_relative_to(CASE.resolve()):raise ValueError('Choose a new external output')
    spec=importlib.util.spec_from_file_location('case013_plot_derive',CASE/'publication/derive.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);d=m.derive(CASE)
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'case013-recorded-v1','font.size':11})
    figures=[]
    def finish(fig,name):
        fig.tight_layout();fig.savefig(output/name,metadata={'Date':None});plt.close(fig);figures.append(name)
    def bars(ax,values,denominator=None):
        ax.bar(METHODS,values,color=COLORS,edgecolor='#242f35',linewidth=.6)
        ax.set_ylim(0,(denominator+2) if denominator else max(values)*1.24)
        ax.spines[['right','top']].set_visible(False)
        for i,v in enumerate(values):ax.text(i,v+.02*ax.get_ylim()[1],f'{v}/{denominator}' if denominator else f'{v:.3f}',ha='center',fontsize=10)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,key in zip(axes,['06b','4b']):
        values=[next(x['ppl'] for x in d['quality'] if x['model']==key and x['method']==method) for method in METHODS]
        bars(ax,values);ax.set_title('Qwen3-0.6B' if key=='06b' else 'Qwen3-4B-Instruct-2507');ax.set_ylabel('Subset PPL (lower is better)')
    fig.suptitle('Recorded reference quality: 65,504 scored tokens / condition',y=1.03)
    finish(fig,'ppl.svg')
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    panels=[('8K exact DEV',12,[next(x['successes'] for x in d['long'] if x['length']==8192 and x['method']==method) for method in METHODS]),
            ('16K exact DEV',12,[next(x['successes'] for x in d['long'] if x['length']==16384 and x['method']==method) for method in METHODS])]
    for cohort,mode,label in [('base','REPLAY','Base replay DEV'),('base','FREE','Base free DEV'),('extension16k','REPLAY','16K replay follow-up DEV'),('extension16k','FREE','16K free follow-up DEV')]:
        rows=[next(x for x in d['agents'] if x['cohort']==cohort and x['mode']==mode and x['method']==method) for method in METHODS]
        panels.append((label,rows[0]['n'],[x['successes'] for x in rows]))
    for ax,(label,n,values) in zip(axes.flat,panels):bars(ax,values,n);ax.set_title(label);ax.set_ylabel('Completed tasks (higher is better)');ax.set_yticks(range(n+1))
    fig.suptitle('DEV counts; length variants reuse the same task IDs',y=1.01)
    finish(fig,'dev-success.svg')
    packed=json.loads((CASE/'records/diagnostics/packed_precision_aligned_03.json').read_text())
    vals=[packed['gemm']['relative_rmse'],packed['one_block']['relative_rmse'],packed['short_full_decode']['forward_relative_rmse']]
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    for ax,value,label in zip(axes,vals,['Projection GEMM','One block output','32-token full-model logits']):
        ax.bar([0],[value],width=.5,color='#b46d42',edgecolor='#242f35');ax.set_xticks([]);ax.set_ylim(0,.034)
        ax.text(0,value+.001,f'{value:.6f}',ha='center');ax.set_title(label);ax.set_ylabel('Relative L2 error (linear scale)');ax.spines[['right','top']].set_visible(False)
    axes[2].axhline(packed['thresholds_before_probe']['full_forward_relative_rmse'],color='#242f35',linestyle='--',label='Full-model limit 0.02')
    axes[2].legend(loc='upper left',fontsize=9)
    fig.suptitle('Different output tensors / denominators; not a layer-error trajectory',y=1.03)
    finish(fig,'packed-gates.svg')
    (output/'figure_manifest.json').write_text(json.dumps({'schema':'case013-figures-v1','source_identity':d['source_identity'],
        'plot_environment':{'Python':sys.version.split()[0],'Matplotlib':matplotlib.__version__,'NumPy':np.__version__},
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files':{name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in figures},
        'new_gpu_runs':0,'intervals':None,'reason':'Descriptive frozen DEV/reference records, not formal inference'},indent=2)+'\n')
    print(json.dumps({'figures':figures,'new_gpu_runs':0}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args();plot(a.output)
