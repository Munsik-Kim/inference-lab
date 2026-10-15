"""Portable raw-to-display projection. stdlib only; never imports model runners."""
import argparse,hashlib,json,math
from collections import defaultdict
from pathlib import Path

METHODS=('BF16','RTN','COMP_QUARTER','COMP_HALF');REV='c1899de289a04d12100db370d81485cdf75e47ca'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def close(x,y):
    if abs(x-y)>1e-10*max(1,abs(x),abs(y)):raise ValueError('recorded aggregate mismatch')
def unique(rows,keys):
    result={}
    for r in rows:
        k=tuple(r[x] for x in keys)
        if k in result:raise ValueError('duplicate row identity')
        result[k]=r
    return result
def p2_metrics(rows,documents):
    by=unique(rows,('method','split','document_id_hash'));docs={r['document_id_hash']:r for r in documents if r['split'] in ['EVAL-ID','SHIFT']};out=[]
    if len(by)!=4*len(docs):raise ValueError('missing whole-model arm/document')
    for split in ['EVAL-ID','SHIFT']:
        ids=sorted(k for k,v in docs.items() if v['split']==split)
        for method in METHODS:
            rr=[by[method,split,k] for k in ids]
            for r in rr:
                if r['token_hash']!=docs[r['document_id_hash']]['token_hash'] or r['status']!='COMPLETE' or r['valid_token_count']!=255 or r['KL_position_count']!=16:raise ValueError('input, validity or denominator mismatch')
            n=sum(r['valid_token_count'] for r in rr);loss=math.fsum(r['NLL_sum'] for r in rr);base=[by['RTN',split,k] for k in ids];delta=[r['NLL_sum']/255-b['NLL_sum']/255 for r,b in zip(rr,base)]
            v=loss/n;baseline=math.fsum(r['NLL_sum'] for r in base)/n
            out.append({'method':method,'split':split,'document_count':len(ids),'valid_token_count':n,'NLL_sum':loss,'NLL':v,'PPL':math.exp(v),'mean_KL':math.fsum(r['sampled_full_vocab_KL_sum'] for r in rr)/(16*len(ids)),
              'flip_rate':sum(r['flip_count'] for r in rr)/n,'NLL_minus_RTN':math.fsum(delta)/len(delta),'PPL_relative_change':math.expm1(v-baseline),'lower_documents':sum(x<0 for x in delta),'higher_documents':sum(x>0 for x in delta),'tie_documents':sum(x==0 for x in delta),'core_bytes':rr[0]['projection_core_bytes']})
    return out,by

def derive(case):
    case=Path(case);sources={}
    def get(rel):sources[rel]=sha(case/rel);return load(case/rel)
    def getrows(pattern):
        result=[]
        for p in sorted(case.glob(pattern)):sources[str(p.relative_to(case))]=sha(p);result+=lines(p)
        return result
    p0=get('results/local_r1/p0_full.json');p1=getrows('results/p1_r2/cells/*.jsonl');p1by=unique(p1,('block','projection','split','alpha','document_id_hash'));fit=get('results/p1_r2/fit_frozen.json')
    if len(p1)!=2880:raise ValueError('P1 row count')
    target_rows=[]
    for selected in sorted(fit['selected'],key=lambda r:r['target']):
        t=selected['target'];b=int(t.split('.')[2]);proj=t.split('.')[-1];a=selected['selected_alpha'];pool=lambda split,alpha:[r for r in p1 if r['block']==b and r['projection']==proj and r['split']==split and r['alpha']==alpha]
        cal={x:math.fsum(r['SSE_after'] for r in pool('CAL-FIT',x)) for x in [0,.25,.5,.75,1]}
        if min(cal,key=lambda x:(cal[x],x))!=a:raise ValueError('P1 frozen CAL selector')
        baseline=math.fsum(r['SSE_after'] for r in pool('EVAL-ID',0));chosen=pool('EVAL-ID',a);quarter=pool('EVAL-ID',.25)
        target_rows.append({'target':t,'selected_alpha':a,'EVAL_documents':32,'selected_SSE_change':math.fsum(r['SSE_after'] for r in chosen)/baseline-1,'quarter_SSE_change':math.fsum(r['SSE_after'] for r in quarter)/baseline-1,
             **{k:math.fsum(r[k] for r in chosen) for k in ['T_inherited','T_cancellation','T_requant']},'baseline_SSE':baseline})
    sign=get('results/p1_r2/arithmetic_sign_robustness.json')['rows'];p1b=getrows('results/p1b_r2/cells/*/*.jsonl');frozen=get('results/p1b_r2/selection_frozen.json');bsum=get('results/p1b_r2/summary.json');bby=unique(p1b,('target','alpha','split','document_id_hash'))
    if len(p1b)!=3360:raise ValueError('P1b rows')
    choices={r['target']:r['policy_alpha'] for r in frozen['choices']};macros=[]
    for rec in bsum['macro_contrasts']:
        split=rec['split'];ratios=[];nlls=[]
        for t,choice in sorted(choices.items()):
            ids=sorted(d for (tt,a,s,d) in bby if tt==t and a==0 and s==split);ca=choice[rec['candidate']];ba=choice[rec['baseline']]
            base=math.fsum(bby[t,0,split,d]['SSE_after'] for d in ids)
            ratios.append(math.fsum(bby[t,ca,split,d]['SSE_after']-bby[t,ba,split,d]['SSE_after'] for d in ids)/base)
            nlls.append(math.fsum((bby[t,ca,split,d]['NLL_sum']-bby[t,ba,split,d]['NLL_sum'])/255 for d in ids)/len(ids))
        D=math.fsum(ratios)/6;N=math.fsum(nlls)/6;close(D,rec['D_macro']);close(N,rec['NLL_macro_doc_mean_difference']);macros.append({**rec,'D_macro':D,'NLL_macro_doc_mean_difference':N})
    r2=getrows('results/p2_r1/metrics/*.jsonl');im=get('results/p2_r1/input_manifest.json');metrics,by=p2_metrics(r2,im['documents']);rs=get('results/p2_r1/summary.json');art=getrows('results/p2_r1/artifacts.jsonl');diag=getrows('results/p2_r1/CAL_diagnostics.jsonl')
    if len(diag)!=392 or len(art)!=588:raise ValueError('P2 construction completeness')
    for m in metrics:
        q=next(x for x in rs['method_summaries'] if x['method']==m['method'] and x['split']==m['split'])
        for k,j in [('NLL','NLL_token_weighted'),('PPL','PPL'),('mean_KL','mean_full_vocab_KL'),('flip_rate','argmax_flip_rate')]:close(m[k],q[j])
    contrasts=[]
    for q in rs['paired_comparisons']:
        ids=sorted(d for mm,s,d in by if mm==q['candidate'] and s==q['split']);delta=math.fsum((by[q['candidate'],q['split'],d]['NLL_sum']-by[q['baseline'],q['split'],d]['NLL_sum'])/255 for d in ids)/len(ids);close(delta,q['mean_document_delta_NLL']);contrasts.append({**q,'mean_document_delta_NLL':delta})
    storage=[]
    for method in METHODS[1:]:
        rr=[r for r in art if r['method']==method];unique(rr,('name',))
        if len(rr)!=196:raise ValueError('projection count')
        fields={k:sum(r[k] for r in rr) for k in ['artifact_bytes','header_bytes','scale_bytes','code_bytes']}
        if fields['artifact_bytes']!=sum(fields[k] for k in ['header_bytes','scale_bytes','code_bytes']):raise ValueError('byte accounting')
        storage.append({'method':method,'files':len(rr),**fields})
    if len({x['artifact_bytes'] for x in storage})!=1:raise ValueError('same-budget claim violated')
    posthoc=[]
    for split in ['EVAL-ID','SHIFT']:
        ids=sorted(d for m,s,d in by if m=='RTN' and s==split);v=math.fsum((by['COMP_HALF',split,d]['NLL_sum']-by['COMP_QUARTER',split,d]['NLL_sum'])/255 for d in ids)/len(ids)
        posthoc.append({'comparison':'HALF minus QUARTER','split':split,'delta_NLL':v,'CI95':None,'CI_reason':'Independent reviewer code/seed unavailable locally; raw point estimate only','status':'POSTHOC_PUBLICATION_ANALYSIS'})
    primary=next(x for x in contrasts if x['primary']);get('references/source_audit.json');source_identity=get('results/p2_r1/source_freeze.json')
    observations=[]
    for m in metrics:
        rel=f"results/p2_r1/metrics/{m['method']}_{m['split']}.jsonl"
        for metric,unit,denominator in [('PPL','dimensionless','exp(total NLL / valid next tokens)'),('NLL','nats/token','total valid next tokens'),('PPL_relative_change','relative to RTN','RTN PPL on identical documents')]:
            observations.append({'stage':'P2','run_id':'p2_r1','model_revision':REV,'evidence_kind':'CALCULATED_FROM_RECORDED_SCALARS','corpus_split':m['split'],'document_unit':'book excerpt' if m['split']=='EVAL-ID' else 'content-group excerpt','document_count':m['document_count'],'tokens_per_document':256,'valid_targets_per_document':255,'candidate':m['method'],'baseline':'RTN' if metric=='PPL_relative_change' else None,'alpha':{'BF16':None,'RTN':0,'COMP_QUARTER':.25,'COMP_HALF':.5}[m['method']],'metric':metric,'unit':unit,'denominator':denominator,'value':m[metric],'CI':None,'primary_or_secondary_or_posthoc':'secondary absolute/relative summary','execution_boundary':'INT4 storage -> dense BF16 quality reference','source_file':rel,'source_fields':['NLL_sum','valid_token_count'],'source_sha256':sources[rel]})
    observations.append({'stage':'P2','run_id':'p2_r1','model_revision':REV,'evidence_kind':'RECORDED_PAIRED_DOCUMENT_BOOTSTRAP_RECOMPUTED_IN_PUBLICATION_CHECK','corpus_split':'EVAL-ID','document_unit':'book excerpt','document_count':64,'tokens_per_document':256,'candidate':'COMP_QUARTER','baseline':'RTN','alpha':.25,'metric':'mean_document_delta_NLL','unit':'nats/token','denominator':'64 paired document means of255 next tokens','value':primary['mean_document_delta_NLL'],'CI':primary['CI95'],'CI_type':'paired percentile bootstrap10000 seed6101523, conditional on model/CAL32','primary_or_secondary_or_posthoc':'registered primary','execution_boundary':'dense BF16 whole-model reference','source_file':'results/p2_r1/summary.json','source_fields':['paired_comparisons[primary]'],'source_sha256':sources['results/p2_r1/summary.json']})
    def observation(stage,run_id,split,unit,n,candidate,baseline,alpha,metric,value,denominator,source_file,fields,ci=None,kind='diagnostic'):
        return {'stage':stage,'run_id':run_id,'model_revision':None if stage=='P0' else REV,'evidence_kind':'CALCULATED_FROM_RECORDED_SCALARS','corpus_split':split,'document_unit':unit,'document_count':n,'tokens_per_document':256 if stage!='P0' else None,'candidate':candidate,'baseline':baseline,'alpha':alpha,'metric':metric,'unit':'relative to alpha0 SSE' if 'SSE_change' in metric else 'nats/token' if 'NLL' in metric else 'squared error','denominator':denominator,'value':value,'CI':ci,'CI_type':None if ci is None else 'original paired document bootstrap','primary_or_secondary_or_posthoc':kind,'execution_boundary':'CPU synthetic fixture' if stage=='P0' else 'dense BF16 reference; local independent intervention','source_file':source_file,'source_fields':fields,'source_sha256':sources[source_file]}
    get('results/p1_r2/derived_summary.json')
    for x in target_rows:
        for field,alpha,kind in [('selected_SSE_change',x['selected_alpha'],'frozen CAL selection'),('quarter_SSE_change',.25,'posthoc weak-strength observation')]:
            observations.append(observation('P1','p1_r2','EVAL-ID','book excerpt reused across6 fixed targets',32,x['target'],'same target alpha0',alpha,field,x[field],'sum alpha0 local projection SSE on32 excerpts','results/p1_r2/derived_summary.json',['rows: target/split/alpha','SSE_after','SSE_before'],kind=kind))
    for x in macros:
        if x['candidate']=='SELECT_SSE' and x['baseline'] in ['OLD_FIT','BASE_ZERO','FIXED_QUARTER']:
            for field,ci_field in [('D_macro','D_macro_CI95'),('NLL_macro_doc_mean_difference','NLL_macro_CI95')]:
                v=observation('P1b','p1b_r2',x['split'],'document excerpt reused across6 fixed interventions',x['documents'],'SELECT_SSE',x['baseline'],None,field,x[field],'equal mean of6 fixed target differences; SSE divided by each target alpha0 SSE' if field=='D_macro' else 'mean per-document255-token NLL difference, then mean over6 fixed targets','results/p1b_r2/summary.json',['macro_contrasts: split/candidate/baseline',field],x[ci_field],'registered primary' if x['primary'] and field=='D_macro' else 'secondary pointwise')
                v['unit']='relative to baseline alpha0 SSE' if field=='D_macro' else 'nats/token';observations.append(v)
    finite=p0['exact']['finite_codebook']
    for method in ['BASE','COMP']:
        v=observation('P0','local_r1','synthetic finite4-code','not a document',None,method,'teacher',None,'absolute_error',float(__import__('fractions').Fraction(finite[method+'_abs_error'])),'absolute scalar output error;2bit4-code fixture, not an INT4 LLM','results/local_r1/p0_full.json',['exact.finite_codebook.'+method+'_abs_error']);v['unit']='absolute scalar error';observations.append(v)
    for method in METHODS[1:]:
        z=next(x for x in storage if x['method']==method);v=observation('P2','p2_r1','storage ledger','not a document',None,method,'same-format RTN',{'RTN':0,'COMP_QUARTER':.25,'COMP_HALF':.5}[method],'projection_core_bytes',z['artifact_bytes'],'196 projection files;header+FP32scale+codes including padding','results/p2_r1/artifacts.jsonl',['artifact_bytes','header_bytes','scale_bytes','code_bytes']);v['unit']='bytes';v['execution_boundary']='serialized projection core, not checkpoint/VRAM';v['tokens_per_document']=None;observations.append(v)
    return {'schema_version':1,'publication_version':'v1','case':'014','model_id':'Qwen/Qwen3-0.6B','model_revision':REV,'new_GPU_inference_during_publication':0,'execution_source_sha256':source_identity['execution_source_sha256'],
      'p0':{'exact':p0['exact'],'dither':[ {k:v for k,v in r.items() if k!='seed_SSE'} for r in p0['dither']['rows']],'dither_contract':'infinite lattice subtractive dither; lambda0/no ridge/clipping;2000 stored seeds per cell, MC not rerun'},
      'p1':{'raw_rows':len(p1),'targets':target_rows,'selected_EVAL_SSE_increased':sum(x['selected_SSE_change']>0 for x in target_rows),'quarter_EVAL_SSE_decreased':sum(x['quarter_SSE_change']<0 for x in target_rows),'unit':'six independent hybrid interventions;32 EVAL documents reused across targets',
        'arithmetic':{'comparisons':len(sign),'measured_direction_agrees':sum(r['actual_and_algebra_sign_agree'] is True for r in sign),'bound_certified':sum(r['algebra_sign_certified_against_arithmetic_bound'] for r in sign)}},
      'p1b':{'raw_rows':len(p1b),'choices':choices,'macro_contrasts':macros,'arithmetic':bsum['arithmetic_sign_summary'],'unit':'six independent interventions;SELECT32/FRESH-EVAL64/SHIFT16;7 policy rows can alias the same artifact'},
      'p2':{'raw_rows':len(r2),'methods':metrics,'contrasts':contrasts,'primary':primary,'storage':storage,'resources':rs['resources'],'posthoc':posthoc,'scope':'four actual whole models; original CAL32; fresh PG19 64 books and ArXiv16 content groups; no packed inference'},'observations':observations,'source_identity':sources}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=derive(a.case);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('Derived P0/P1/P1b/P2 display from saved raw scalar records; no model execution.')
