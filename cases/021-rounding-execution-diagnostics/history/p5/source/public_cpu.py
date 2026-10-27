"""Portable public-only audit: summaries, figures and exact synthetic primitives."""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from arithmetic_v2 import candidates,exact_candidates,decode,steps,rank,round_bits,rational_bits
from fractions import Fraction
def read(p):return json.loads(Path(p).read_text())
def table(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def digest(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--quick',action='store_true');p.add_argument('--building',action='store_true',help=argparse.SUPPRESS);a=p.parse_args();r=a.root.resolve();out=a.output.resolve()
    if out.exists() or out.is_relative_to(r):raise ValueError('NEW_EXTERNAL_OUTPUT_REQUIRED')
    out.mkdir(parents=True,exist_ok=False);checked=0
    if not a.building:
        manifest=read(r/'PUBLIC_MANIFEST.json')
        actual={str(f.relative_to(r)) for f in r.rglob('*') if f.is_file() and f.name!='PUBLIC_MANIFEST.json'}
        if actual!=set(manifest['files']):raise ValueError('MANIFEST_COVERAGE')
        for name,row in manifest['files'].items():
            raw=r/name;f=raw.resolve()
            if not f.is_relative_to(r) or raw.is_symlink() or digest(f)!=row:raise ValueError('MANIFEST_MISMATCH:'+name)
            checked+=1
    for x,expected in [(1,(3.03125,3.03125)),(.125,(.37890625,.37890625)),(1.5,(4.5625,4.53125))]:
        v=candidates(x,3,1.0078125);assert tuple(float(decode(b)) for b in v)==expected;assert tuple(int(b) for b in v)==exact_candidates(x,3,1.0078125)
    assert int(steps(np.array(0x8000,np.uint16),np.array(0,np.uint16)))==0
    for q in [Fraction(1,2**134),Fraction(3,2**134),Fraction(1)+Fraction(1,256)+Fraction(1,2**30)]:assert int(round_bits(float(q)))==rational_bits(q)
    facts=read(r/'facts.json');B=read(r/'summary/B_decision.json');components=table(r/'summary/local_gap_components.csv');single=table(r/'summary/single_term_metrics.csv');fits=table(r/'summary/candidate_fit.csv');groups=[]
    for split in ['original_DEV_posthoc','original_EVAL_posthoc']:
        for cls in ['DN_NONPOS_DC_POS','DN_POS_DC_POS','DN_POS_DC_NONPOS','NEAR_TOLERANCE_OR_OTHER','NOT_INCREASE_OR_NEAR_TOLERANCE']:
            own=[x for x in components if x['split']==split and x['classification']==cls];groups.append({'split':split,'classification':cls,'cells':len(own)})
    for x in components:
        dg,dn,dc=map(float,[x['delta_G'],x['delta_N'],x['delta_C']]);scale=max(abs(float(x[k])) for k in ['R0_G','R1_G','R0_N','R1_N','R0_C','R1_C']);assert abs(dg-dn-dc)<=1e-10*scale if scale else dg-dn-dc==0
        tau=max(1e-12,1e-10*scale);assert tau==float(x['sign_tolerance'])
        expected=('NOT_INCREASE_OR_NEAR_TOLERANCE' if dg<=tau else 'DN_NONPOS_DC_POS' if dn<=-tau and dc>tau else 'DN_POS_DC_POS' if dn>tau and dc>tau else 'DN_POS_DC_NONPOS' if dn>tau and dc<=-tau else 'NEAR_TOLERANCE_OR_OTHER')
        assert x['classification']==expected
    total={k:sum(int(x[k]) for x in fits if x['scope']=='ALL_VALID' and x['method']=='P') for k in ['coordinates','discriminating_coordinates','A_only','B_only','both','neither','undefined']};assert total==B['candidate_fit_totals_nonoverlapping_all_valid_P']
    assert sum(int(x['numeric_mismatch']) for x in single if x['comparison']=='D1_vs_A')==B['D1_A_numeric_mismatch']
    assert total==facts['B']['candidate_fit_totals_nonoverlapping_all_valid_P']
    assert len(components)==facts['A']['local_tensor_pairs']
    ulp=table(r/'summary/projection_ulp_documents.csv')
    assert sum(int(x['numeric_mismatch']) for x in ulp)==facts['A']['ULP_numeric_mismatches']
    sparse=r/'summary/sparse_metrics.csv'
    if sparse.exists():
        rows=[x for x in table(sparse) if x['comparison']=='P_vs_D1']
        for item in facts['C']['first_observation']:
            own=[x for x in rows if int(x['layer'])==item['layer'] and x['document_id']==item['document_id']]
            assert [int(x['K']) for x in own]==[1,2,8,64,256,1024]
            assert next((int(x['K']) for x in own if int(x['numeric_mismatch'])),None)==item['first_observed_K']
            assert [int(x['numeric_mismatch']) for x in own]==[x['numeric_mismatch'] for x in item['conditions']]
    receipt={'scope':'Public summaries and small synthetic exact oracle only; private raw/GPU replay not performed','manifest_files':checked,'synthetic_checks':'example, nearest-even/subnormal/double-round and numeric zero','group_counts':groups,'candidate_fit_recomputed':total,'Torch_imported':'torch' in sys.modules,'GPU_calls':0,'figures':0}
    if not a.quick:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'case021-p5'})
        def save(fig,name):fig.savefig(out/(name+'.png'),dpi=160);fig.savefig(out/(name+'.svg'),metadata={'Date':None});plt.close(fig);receipt['figures']+=1
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');labels=['Noise falls,\ncross rises','Both rise','Noise rises,\ncross falls'];classes=['DN_NONPOS_DC_POS','DN_POS_DC_POS','DN_POS_DC_NONPOS']
        for split,offset,color in [('original_DEV_posthoc',-.17,'#3274a1'),('original_EVAL_posthoc',.17,'#da8731')]:
            values=[next(x['cells'] for x in groups if x['split']==split and x['classification']==cl) for cl in classes];bars=axes[0].bar(np.arange(3)+offset,values,.34,color=color,label=split);axes[0].bar_label(bars)
        axes[0].set(xticks=range(3),xticklabels=labels,ylabel='Tensor cells with increased gap',title='Increases: DEV14 / EVAL34');axes[0].legend(fontsize=8)
        ev=[x for x in components if x['split']=='original_EVAL_posthoc'];xx=np.array([float(x['delta_N']) for x in ev]);yy=np.array([float(x['delta_C']) for x in ev]);increase=np.array([float(x['delta_G'])>float(x['sign_tolerance']) for x in ev]);axes[1].scatter(xx,yy,c=np.where(increase,'#d64b35','#3274a1'),s=22,alpha=.7);lo,hi=min(xx.min(),-yy.max()),max(xx.max(),-yy.min());axes[1].plot([lo,hi],[-lo,-hi],'--',color='gray',label='delta G = 0');axes[1].axhline(0,lw=.7,color='gray');axes[1].axvline(0,lw=.7,color='gray');axes[1].set(xlabel='delta N (noise energy)',ylabel='delta C (signed cross term)',title='All96 EVAL cells; red = gap increase');axes[1].legend()
        fig.suptitle('Local gap change = delta N + delta C\nExisting selected coordinates, not a full-model causal contribution',fontsize=13);save(fig,'01_gap_components')
        fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');names=['POWER1','POWER_EIGHTH','FIXED_NONPOWER','ACTUAL_ONE','CHALLENGE'];short=['±1','±1/8','Fixed\nnon-power','Actual\ncoefficient','Challenge'];colors=['#3274a1','#d64b35','#b5cbd9','#874aa1']
        for ax,layer in zip(axes,[0,25]):
            selected=[x for x in fits if x['layer']==str(layer) and x['scope']=='ALL_VALID' and x['method']=='P'];bottom=np.zeros(len(selected))
            for key,color in zip(['A_only','B_only','both','neither'],colors):
                vals=np.array([100*int(next(x for x in selected if x['fixture']==name)[key])/int(next(x for x in selected if x['fixture']==name)['coordinates']) for name in names[:len(selected)]]);ax.bar(range(len(selected)),vals,bottom=bottom,color=color,label=key);bottom+=vals
            ax.set(xticks=range(len(selected)),xticklabels=short[:len(selected)],ylabel='Output coordinates (%)',ylim=(0,105),title=f'Layer {layer}: P fits arithmetic candidates');ax.legend(ncol=2,fontsize=8)
        fig.suptitle('Single nonzero coefficient per row, M512\nCandidate A: BF16 weight first; B: final rounding only. Both is non-discriminating.',fontsize=13);save(fig,'02_single_term_candidates')
        sparse=r/'summary/sparse_metrics.csv'
        if sparse.exists():
            rows=[x for x in table(sparse) if x['comparison']=='P_vs_D1'];fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
            for layer,doc in dict.fromkeys((x['layer'],x['document_id']) for x in rows):
                own=[x for x in rows if x['layer']==layer and x['document_id']==doc];K=[int(x['K']) for x in own];lab='L'+layer+' '+doc
                for ax,key,factor in [(axes[0],'mismatch_fraction',100),(axes[1],'max_steps',1),(axes[2],'norm64',1)]:ax.plot(K,[factor*float(x[key]) for x in own],marker='o',ms=4,label=lab);ax.set_xscale('log',base=2);ax.set_xticks(K,labels=list(map(str,K)));ax.grid(alpha=.2)
            axes[0].set(ylabel='D1/P mismatched coordinates (%)',title='Numeric mismatch');axes[1].set_yscale('symlog',linthresh=1);axes[1].set(ylabel='Maximum BF16 representable steps (symlog)',title='Zero retained; not rounding-operation count');axes[2].set(ylabel='Input Frobenius norm',title='Norm changes with the retained values')
            for ax in axes:ax.set_xlabel('K mask budget per row')
            for ax in axes:ax.set_ylim(bottom=0)
            axes[0].legend(fontsize=7);fig.suptitle('Fixed nested actual-value replay: two projections × two DEV documents\nPoints are conditions, not independent workloads;3 process repeats checked separately.',fontsize=13);save(fig,'03_sparse_replay')
    assert 'torch' not in sys.modules;write(out/'verification.json',receipt);print({'status':'COMPLETE','figures':receipt['figures'],'output':str(out)})
if __name__=='__main__':main()
