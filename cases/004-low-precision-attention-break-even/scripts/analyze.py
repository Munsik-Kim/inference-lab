"""Deterministic CPU reanalysis of actual GPU records; no illustrative speed data."""
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import CASE,verify_spec,ensure_real,write_json,sha
from src.metrics import paired_timing_bootstrap,document_bootstrap,numerical_screen,decision


def analyze(case=CASE):
    spec,spec_sha=verify_spec(case);manifest=json.loads((case/'inputs/manifest.json').read_text())
    docs=sorted(d['document_id'] for d in manifest['documents'] if d['split']=='confirmation')
    raw=[];sources={};process=[]
    for p in sorted((case/'results/confirmation').glob('round-*.json')):
        x=json.loads(p.read_text());assert x['evidence_kind']=='gpu_measurement' and not x['mock']
        assert x['spec_sha256']==spec_sha and x['environment_fingerprint']==spec['environment_fingerprint']
        assert x['started_at_utc']>spec['frozen_at_utc'];process.append(x);sources[str(p.relative_to(case))]=sha(p)
        for m in x['measurements']:
            ensure_real(m);assert m['stage']=='confirmation' and m['spec_sha256']==spec_sha
            for e in m.get('errors_by_head',[]):
                if e['invalid_rows']:
                    assert e['relative_output_error'] is None
                    continue
                e2=math.fsum(t*t for t in e['row_error_norms']);r2=math.fsum(t*t for t in e['row_reference_norms'])
                assert math.isclose(e2,e['error_squared_sum'],rel_tol=1e-12,abs_tol=1e-12)
                assert math.isclose(r2,e['reference_squared_sum'],rel_tol=1e-12,abs_tol=1e-12)
                if e['relative_output_error'] is not None:assert math.isclose(math.sqrt(e2/r2),e['relative_output_error'],rel_tol=1e-12,abs_tol=1e-12)
            raw.append(m)
    index={(m['shape']['id'],m['round'],m['document_id'],m['backend']):m for m in raw};assert len(index)==len(raw)
    entries=[]
    for si,shape in enumerate(spec['shapes']):
        sid=shape['id'];base=spec['baseline_selection'].get(sid)
        ids=docs if shape['family']=='qwen' else ['synthetic-'+str(410000+si)]
        errors=[];backend_summaries={};complete=True;comparisons=[]
        for round_id in range(spec['timing']['process_rounds']):
            aa=[];bb=[]
            for did in ids:
                pair=[index.get((sid,round_id,did,name)) for name in [base,'sage']]
                if not all(p and p['status']=='OK' for p in pair):complete=False;continue
                a,b=pair;assert a['input_sha256']==b['input_sha256'];assert a['inputs_unchanged'] and b['inputs_unchanged']
                assert len(a['wall_ms'])==len(b['wall_ms'])==spec['timing']['counts']['blocks']
                aa.append(a['wall_ms']);bb.append(b['wall_ms'])
            if len(aa)==len(ids):comparisons.append((aa,bb))
        speed=paired_timing_bootstrap(comparisons,seed=470004+si,repetitions=spec['statistics']['timing_bootstrap_replicates']) if complete and len(comparisons)==spec['timing']['process_rounds'] else None
        for name in ['default','flash','sage','kernel_only']:
            ms=[m for m in raw if m['shape']['id']==sid and m['backend']==name and m['status']=='OK']
            if not ms:continue
            wall=[v for m in ms for v in m['wall_ms']];event=[v for m in ms for v in m['event_ms']]
            units=[e for m in ms if m['round']==0 for e in m.get('errors_by_head',[])]
            numerical=numerical_screen(units)
            grouped={did:[e['relative_output_error'] for m in ms if m['round']==0 and m['document_id']==did for e in m.get('errors_by_head',[]) if e['relative_output_error'] is not None] for did in ids}
            interval=document_bootstrap(grouped,repetitions=spec['statistics']['document_bootstrap_replicates']) if shape['family']=='qwen' and all(grouped.values()) and numerical['status'] not in ['NUMERICAL_REVIEW','NOT_RUN'] else None
            peaks=[m['peak_allocated_bytes'] for m in ms if 'peak_allocated_bytes' in m];reserved=[m['peak_reserved_bytes'] for m in ms if 'peak_reserved_bytes' in m]
            by_round={str(ri):float(np.median([v for m in ms if m['round']==ri for v in m['wall_ms']])) for ri in sorted({m['round'] for m in ms})}
            backend_summaries[name]=dict(blocks=len(wall),wall_block_mean_ms=dict(median=float(np.median(wall)),p95=float(np.quantile(wall,.95))),event_block_mean_ms=dict(median=float(np.median(event)),p95=float(np.quantile(event,.95))),round_median_wall_ms=by_round,peak_allocated_bytes=max(peaks,default=None),peak_reserved_bytes=max(reserved,default=None),numerical=numerical,document_cluster_uncertainty=interval)
        candidate=backend_summaries.get('sage',{}).get('numerical')
        expected_units=len(ids)*shape['hq']
        quality_complete=bool(candidate and candidate.get('units')==expected_units)
        if not quality_complete and shape['family']=='qwen':candidate={'status':'NOT_RUN'}
        interface=bool(base) and spec['gate0_status']=='PASS'
        status,reasons=decision(speed,candidate,shape['family']=='qwen',interface)
        if not complete:reasons.append('INCOMPLETE_PROCESS_OR_INPUT_COVERAGE')
        process_failures=[err for process_record in process for err in process_record['errors'] if err.get('shape') and err['shape']['id']==sid]
        failures=[m for m in raw if m['shape']['id']==sid and m['status']!='OK']
        if any(m['status']=='OOM' for m in failures+process_failures):status='OOM';reasons.append('OOM')
        elif failures:status='UNSUPPORTED';reasons.append('BACKEND_FAILURE')
        elif process_failures:status='NOT_RUN';reasons.append('PROCESS_FAILURE')
        # Primary fidelity uses round 0 only. Later rounds check repeatability.
        repeatable=True
        for name in ['default','flash','sage']:
            for did in ids:
                first=index.get((sid,0,did,name))
                if not first:repeatable=False;continue
                for ri in range(1,spec['timing']['process_rounds']):
                    later=index.get((sid,ri,did,name))
                    if not later or later.get('errors_by_head')!=first.get('errors_by_head'):repeatable=False
        entries.append(dict(shape=shape,baseline=base,status=status,reason_codes=reasons,speed=speed,numerical=candidate,backend_summaries=backend_summaries,primary_unit_count=expected_units,confirmation_document_count=len(ids),same_input_repeated_numerical_records_identical=repeatable,evidence=list(sources)))
    gpu=[]
    for x in process:
        gpu.append(dict(round=x['round'],process_start=x['process_start'],sampled_whole_device_peak_bytes=x['gpu_samples']['peak_bytes'],samples=len(x['gpu_samples']['samples']),status=x['status']))
    return dict(protocol_version=spec['protocol_version'],spec_sha256=spec_sha,environment_fingerprint=spec['environment_fingerprint'],hardware_software=spec['environment'],model_revision=spec['model']['revision'],input_family='case004-self-authored-v1-layer13',primary_cost='complete public BF16-input/BF16-output adapter; synchronized wall-clock block mean',speedup_definition='median of paired BF16/Sage block-mean cost ratios; secondary ratio_of_median_costs retained',uncertainty='pointwise 95% percentile intervals; outer process round, inner paired blocks within each retained document; no multiple-comparison coverage claim',quality_independent_documents=16,primary_quality_round=0,source_hashes=sources,entries=entries,gpu_memory=gpu,all_confirmation_rounds_completed=len(process)==spec['timing']['process_rounds'] and all(p['status']=='COMPLETED' for p in process))


def figures(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    fig,ax=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for i,causal in enumerate([True,False]):
        for j,dim in enumerate([64,128]):
            entries=[e for e in result['entries'] if e['shape']['family']=='synthetic' and e['shape']['causal']==causal and e['shape']['dim']==dim and e['speed']]
            a=ax[i,j]
            if entries:
                x=[e['shape']['length'] for e in entries];y=[e['speed']['speedup'] for e in entries];lo=[e['speed']['ci95'][0] for e in entries];hi=[e['speed']['ci95'][1] for e in entries]
                a.errorbar(x,y,yerr=[np.maximum(0,np.array(y)-lo),np.maximum(0,np.array(hi)-y)],fmt='o',capsize=3,label='complete operator, pointwise 95% CI')
            a.axhline(1,color='grey');a.axhline(1.1,color='grey',linestyle=':');a.set_xscale('log',base=2);a.set_title(f"Synthetic only: D={dim}, causal={causal}");a.set_xlabel('Length');a.set_ylabel('BF16 / Sage paired cost');a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'figures/complete_speedup.png',dpi=150);plt.close(fig)
    real=[e for e in result['entries'] if e['shape']['family']=='qwen'];fig,ax=plt.subplots(1,2,figsize=(10,4))
    for a,metric,limit in zip(ax,['median','p95'],[.01,.03]):
        for name,color in [('baseline','#346b9e'),('sage','#b75d2a')]:
            x=[];y=[]
            for e in real:
                n=e['backend_summaries'].get(e['baseline'] if name=='baseline' else name,{}).get('numerical',{})
                if n.get(metric) is not None:x.append(e['shape']['length']);y.append(100*n[metric])
            if x:a.plot(x,y,'o-',color=color,label=name)
        a.axhline(100*limit,color='grey',linestyle='--',label='fixed local screen');a.set_title(f'Real Qwen, layer 13: {metric}');a.set_xlabel('Length');a.set_ylabel('Sampled-query output error (%)');a.legend()
    fig.suptitle('16 confirmation documents; 16 heads per document; 32 query positions per head');fig.tight_layout();fig.savefig(out/'figures/qwen_output_error.png',dpi=150);plt.close(fig)
    rows=[]
    for e in result['entries']:
        s=e['shape'];rows.append([s['family'],f"{s['hq']}/{s['hkv']}",s['dim'],s['length'],str(s['causal']),f"{e['speed']['speedup']:.2f}x" if e['speed'] else 'not run',e['status']])
    fig,ax=plt.subplots(figsize=(11,10));ax.axis('off');t=ax.table(cellText=rows,colLabels=['Input','Hq/Hkv','D','Length','Causal','Complete speedup','Decision'],loc='center',cellLoc='center',colWidths=[.12,.1,.07,.1,.09,.18,.27]);t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.6);fig.tight_layout();fig.savefig(out/'figures/selection_table.png',dpi=150);plt.close(fig)
    applicable=[e for e in result['entries'] if 'kernel_only' in e['backend_summaries'] and 'sage' in e['backend_summaries']]
    if applicable:
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        for a,family in zip(ax,['synthetic','qwen']):
            es=[e for e in applicable if e['shape']['family']==family and e['shape']['causal'] and e['shape']['dim']==128]
            for name in ['baseline','sage','kernel_only']:
                x=[e['shape']['length'] for e in es];y=[e['backend_summaries'][e['baseline'] if name=='baseline' else name]['wall_block_mean_ms']['median'] for e in es]
                if x:a.plot(x,y,'o-',label=name)
            a.set_xscale('log',base=2);a.set_yscale('log');a.set_title(f'{family}, causal, D=128');a.set_ylabel('Median block-mean wall time (ms)');a.set_xlabel('Length');a.legend()
        fig.suptitle('Kernel-only is auxiliary and prequantized; differences are not isolated quantization time');fig.tight_layout();fig.savefig(out/'figures/kernel_vs_complete.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'results').mkdir(exist_ok=True);(a.output_dir/'figures').mkdir(exist_ok=True)
    result=analyze();write_json(a.output_dir/'results/selection_table.json',result)
    with (a.output_dir/'results/selection_table.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['shape_id','baseline','status','speedup','ci_low','ci_high','sage_median_error','sage_p95_error'])
        for e in result['entries']:
            speed=e['speed'] or {};num=e['numerical'] or {};w.writerow([e['shape']['id'],e['baseline'],e['status'],speed.get('speedup'),*(speed.get('ci95',[None,None])),num.get('median'),num.get('p95')])
    figures(result,a.output_dir);print(json.dumps({'all_rounds_completed':result['all_confirmation_rounds_completed'],'statuses':{status:sum(e['status']==status for e in result['entries']) for status in sorted({e['status'] for e in result['entries']})}}))
if __name__=='__main__':main()
