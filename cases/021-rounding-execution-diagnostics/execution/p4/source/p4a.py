"""Post-hoc 2x2 KL/JS from retained full-vocabulary logits, no Torch."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from common import read,write,identity,csvwrite
from numerics import bfdecode,distribution_pair,independent_pair,bootstrap

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.process_time();loc=read(a.root/'references/locations.json');case=Path(loc['original_case']);private=Path(loc['original_private'])
    manifest=read(case/'inputs/manifest.json');original_docs=list(csv.DictReader((case/'results/summary/document_metrics.csv').open()))
    previous={(r['split'],r['document_id'],int(r['length']),r['reference'],r['candidate']):float(r['KL']) for r in original_docs}
    meta_docs={r['document_id']:r for r in manifest['documents']}
    pairs={'00':('D1_R0','P_R0'),'01':('D1_R0','P_R1'),'10':('D1_R1','P_R0'),'11':('D1_R1','P_R1')}
    conditions=['D0_R0','D1_R0','D1_R1','P_R0','P_R1'];positions_rows=[];document_rows=[];js_rows=[];drift=[];sources={};maxdiff=0.;min_kl=float('inf');min_js=float('inf')
    for split in ['DEV','EVAL']:
        for doc in manifest['document_order'][split]:
            ids=np.load(case/'inputs'/(doc+'.npy'),allow_pickle=False)
            for length in [128,512]:
                positions=manifest['selected_positions'][str(length)];expected=[(j+1)*length//17 for j in range(16)]
                if positions!=expected:raise ValueError('POSITION_OFFSET')
                arrays={}
                for condition in conditions:
                    base=private/'main'/split/condition;name=f'{doc}_n{length}_r0';meta=read(base/(name+'.json'));file=base/(name+'_logits.npy')
                    if meta['positions']!=positions or meta['next_token_ids']!=ids[np.array(positions)+1].tolist() or meta['full_logits_shape']!=[1,length,151936]:raise ValueError('INPUT_POSITION_CONTRACT')
                    if meta['logits']['encoding']!='BF16_BITS_UINT16' or meta['logits']['dtype']!='torch.bfloat16' or identity(file)!={k:meta['logits'][k] for k in ['bytes','sha256']}:raise ValueError('RAW_IDENTITY_OR_DTYPE')
                    bits=np.load(file,allow_pickle=False)
                    if bits.shape!=(16,151936):raise ValueError('FULL_VOCAB_GEOMETRY')
                    arrays[condition]=bfdecode(bits).astype(np.float64);sources[str(file)]=identity(file)
                values={};js={}
                for key,(ref,cand) in pairs.items():
                    k,j=distribution_pair(arrays[ref],arrays[cand]);ind=independent_pair(arrays[ref],arrays[cand])
                    if not np.allclose(np.column_stack([k,j]),ind,rtol=1e-9,atol=1e-12):raise ValueError('INDEPENDENT_KL_JS')
                    maxdiff=max(maxdiff,float(np.max(np.abs(np.column_stack([k,j])-ind))))
                    min_kl=min(min_kl,float(k.min()));min_js=min(min_js,float(j.min()));values[key]=k;js[key]=j
                if min_kl < -1e-12 or min_js < -1e-12:raise ValueError('MEANINGFUL_NEGATIVE_DIVERGENCE')
                base={'original_split':split,'analysis_split':'original_'+split+'_posthoc','document_id':doc,'category':meta_docs[doc]['category'],'length':length,'positions':16,'vocabulary':151936}
                for j,pos in enumerate(positions):
                    row={**base,'position':pos,'next_token_index':pos+1}
                    for key in pairs:row['K'+key]=float(values[key][j]);row['JS'+key]=float(js[key][j])
                    row.update(delta_P_given_D1R0=row['K00']-row['K01'],delta_P_given_D1R1=row['K10']-row['K11'],reference_changed_K00_minus_K10=row['K00']-row['K10'])
                    positions_rows.append(row)
                row={**base,**{'K'+key:float(values[key].mean()) for key in pairs}}
                row.update(delta_P_given_D1R0=row['K00']-row['K01'],delta_P_given_D1R1=row['K10']-row['K11'],reference_changed_K00_minus_K10=row['K00']-row['K10'],diagonal_delta=row['K00']-row['K11'])
                document_rows.append(row);js_rows.append({**base,**{'JS'+key:float(js[key].mean()) for key in pairs}})
                for key in ['00','11']:
                    ref,cand=pairs[key]
                    if not np.isclose(row['K'+key],previous[(split,doc,length,ref,cand)],rtol=1e-9,atol=1e-12):raise ValueError('LEGACY_KL_REPLAY')
                for candidate in ['D1_R0','D1_R1','P_R0','P_R1']:
                    k,j=distribution_pair(arrays['D0_R0'],arrays[candidate]);ind=independent_pair(arrays['D0_R0'],arrays[candidate])
                    if not np.allclose(k,ind[:,0],rtol=1e-9,atol=1e-12):raise ValueError('DRIFT_INDEPENDENT')
                    if not np.isclose(k.mean(),previous[(split,doc,length,'D0_R0',candidate)],rtol=1e-9,atol=1e-12):raise ValueError('LEGACY_FIXED_DRIFT')
                    drift.append({**base,'reference':'D0_R0','candidate':candidate,'KL':float(k.mean()),'JS':float(j.mean()),'previous_metric_reused_as_comparison_only':True})
                del arrays
    facts=[]
    for split in ['DEV','EVAL']:
        for n in [128,512]:
            rows=[r for r in document_rows if r['original_split']==split and r['length']==n]
            item={'split':'original_'+split+'_posthoc','length':n,'documents':len(rows),'cells':{k:float(np.mean([r['K'+k] for r in rows])) for k in pairs},'JS_cells':{k:float(np.mean([r['JS'+k] for r in js_rows if r['original_split']==split and r['length']==n])) for k in pairs}}
            for name in ['delta_P_given_D1R0','delta_P_given_D1R1','reference_changed_K00_minus_K10','diagonal_delta']:
                v=[r[name] for r in rows];item[name]=bootstrap(v);item[name]['positive_documents']=sum(x>0 for x in v);item[name]['negative_documents']=sum(x<0 for x in v)
            facts.append(item)
    # Reconstruct the original Monte Carlo interval in its original document order.
    old=list(csv.DictReader((case/'results/summary/primary.csv').open()));mapped={r['document_id']:r for r in document_rows if r['original_split']=='EVAL' and r['length']==512}
    original_order=[r['document_id'] for r in old if int(r['length'])==512]
    old_replay=bootstrap([mapped[d]['diagonal_delta'] for d in original_order],seed=61027023);decision=read(case/'results/summary/decision.json')
    expected=decision['primary512']
    if not np.allclose([old_replay['mean'],old_replay['ci_low'],old_replay['ci_high']],[expected['mean_delta'],expected['ci_low'],expected['ci_high']],rtol=1e-9,atol=1e-12):raise ValueError('ORIGINAL_PRIMARY_INTERVAL')
    for name,rows in [('cross_precision_document.csv',document_rows),('cross_precision_positions.csv',positions_rows),('fixed_reference_drift.csv',drift),('js_cross_precision.csv',js_rows)]:csvwrite(a.output/name,rows)
    categories=[]
    for split in ['DEV','EVAL']:
        for n in [128,512]:
            for category in ['explanation','key_value','code','json']:
                selected=[r for r in document_rows if r['original_split']==split and r['length']==n and r['category']==category]
                categories.append({'split':split,'length':n,'category':category,'documents':len(selected),**{k:float(np.mean([r[k] for r in selected])) for k in ['K00','K01','K10','K11','delta_P_given_D1R0','delta_P_given_D1R1']}})
    csvwrite(a.output/'category_metrics.csv',categories)
    write(a.output/'p4a_facts.json',{'scope':'POSTHOC_FOLLOWUP','groups':facts,'original_primary_verdict':decision['rounding_effect'],'first_argument_warning':'K00-K10 changes the KL reference; only within-row P comparisons have one fixed D1 reference','original_interval_replay':old_replay})
    write(a.output/'p4a_verification.json',{'status':'COMPLETE','raw_files':len(sources),'document_length_rows':len(document_rows),'position_rows':len(positions_rows),'cross_pair_positions':len(positions_rows)*4,'fixed_drift_document_rows':len(drift),'max_KL_JS_vector_fsum_difference':maxdiff,'minimum_KL':min_kl,'minimum_JS':min_js,'clipping':False,'atol':1e-12,'rtol':1e-9,'sources':sources,'cpu_process_seconds':time.process_time()-started,'torch_imported':False})
    print(json.dumps({'status':'COMPLETE','groups':facts,'max_audit_diff':maxdiff}))

if __name__=='__main__':main()
