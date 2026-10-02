"""Recalculate saved records without loading models. Missing labels stay null."""
import argparse,sys,statistics,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from source.contracts import *

def all_records(root):
    rows=[]
    for p in sorted((root/'results/attempts').glob('*/attempt-*')):
        if not p.name.endswith('.partial'):
            r=verify_attempt(p);r['relative_image_path']=str((p/'image.png').relative_to(root)) if (p/'image.png').exists() else None
            rows.append(r)
    # Multiple failed attempts remain visible. Completed job IDs may appear only once.
    unique_records([r for r in rows if r['status']=='SUCCESS']);return rows

def quality(rows,ann,items,rubric_hash):
    coverage=annotation_validate(ann,items,rubric_hash)
    if not coverage['complete']:return {'status':'ANNOTATION_PENDING','primary':None,'coverage':coverage}
    lookup={r['image_id']:r for r in ann['rows']};scores={};uncertain=0
    for r in rows:
        if r['status']!='SUCCESS':scores[r['job_id']]={'pass':0,'fraction':0.,'optimistic_pass':0,'values':{}};continue
        values=lookup[r['job_id']]['values'];v=list(values.values());uncertain+=v.count('uncertain')
        scores[r['job_id']]={'pass':int(all(x=='satisfied' for x in v)),'fraction':sum(x=='satisfied' for x in v)/len(v),'optimistic_pass':int(all(x!='not_satisfied' for x in v)),'values':values}
    ps=sorted({r['job']['prompt_id'] for r in rows});mapping={(r['job']['prompt_id'],r['job']['seed_label'],r['job']['setting']['id']):r for r in rows}
    diffs=[];pairs={'both_pass':0,'A_only':0,'C_only':0,'neither':0};gains={};losses={}
    for p in ps:
        ds=[]
        for seed in [72301,72302,72303,72304]:
            a=mapping[(p,seed,'A_time')];c=mapping[(p,seed,'C_time')];sa=scores[a['job_id']];sc=scores[c['job_id']]
            x,y=sa['pass'],sc['pass'];ds.append(y-x)
            pairs['both_pass' if x and y else 'A_only' if x else 'C_only' if y else 'neither']+=1
            for k in set(sa['values'])|set(sc['values']):
                va=sa['values'].get(k)=='satisfied';vc=sc['values'].get(k)=='satisfied';key=p+'/'+k
                gains[key]=gains.get(key,0)+int(vc and not va);losses[key]=losses.get(key,0)+int(va and not vc)
        diffs.append(sum(ds)/4)
    rng=np.random.default_rng(72501);d=np.asarray(diffs);boots=d[rng.integers(0,len(d),(5000,len(d)))].mean(axis=1)
    aggregate={}
    for setting in ['A_time','B_time','C_time']:
        ss=[scores[r['job_id']] for r in rows if r['job']['setting']['id']==setting]
        aggregate[setting]={'images':len(ss),'all_constraint_pass_rate':sum(x['pass'] for x in ss)/len(ss),'mean_constraint_fraction':sum(x['fraction'] for x in ss)/len(ss),'all_uncertain_as_pass_rate':sum(x['optimistic_pass'] for x in ss)/len(ss)}
    categories={}
    for cat in sorted({i['category'] for i in items}):
        category_ids={i['image_id'] for i in items if i['category']==cat};categories[cat]={}
        for s in ['A_time','B_time','C_time']:
            rr=[r for r in rows if r['job_id'] in category_ids and r['job']['setting']['id']==s]
            categories[cat][s]={'n':len(rr),'all_constraint_pass_rate':sum(scores[r['job_id']]['pass'] for r in rr)/len(rr)}
    return {'status':'ANNOTATED','evaluator':ann['evaluator'],'coverage':coverage,'primary':{'direction':'C_time minus A_time, higher better','prompt_clusters':len(ps),'estimate':float(d.mean()),'interval':np.quantile(boots,[.025,.975],method='linear').tolist(),'level':.95,'method':'paired prompt-cluster bootstrap, seeds and settings kept together','seed':72501,'repetitions':5000,'template_family_limit':'four related families; no population guarantee'},'settings':aggregate,'categories':categories,'paired':pairs,'constraint_gains':gains,'constraint_losses':losses,'uncertain_constraint_count':uncertain,'per_image':scores}

def build(root,annotation=None):
    verify_freeze(root);rows=all_records(root);ps={x['prompt_id']:x for x in load(root/'configs/prompts.json')['prompts']}
    main=[r for r in rows if r['job']['phase']=='main'];smoke=[r for r in rows if r['job']['phase']=='smoke']
    # Same input noise for all arms. Never aggregate repeated timings as independent quality images.
    for split in [main,smoke]:
        pairs={}
        for r in split:
            if r['status']=='SUCCESS':pairs.setdefault((r['job']['prompt_id'],r['job']['seed_label']),set()).add(r['initial_noise_sha256'])
        if any(len(x)!=1 for x in pairs.values()):raise ValueError('Noise hash differs across L/S')
    items=[]
    for r in main+smoke:
        if r['status']!='SUCCESS':continue
        p=ps[r['job']['prompt_id']]
        items.append({'image_id':r['job_id'],'image_sha256':r['image_sha256'],'path':r['relative_image_path'],'split':r['job']['phase'].upper(),'category':p['category'],'english':p['english'],'korean_display':p['korean_display'],'constraints':p['constraints']})
    dump(root/'analysis/annotation_items.json',{'schema':'case012-blind-items-v1','rubric_sha256':sha(root/'configs/rubric.json'),'items':items})
    main_items=[i for i in items if i['split']=='MAIN']
    if annotation:
        ann=load(annotation)
        full_coverage=annotation_validate(ann,items,sha(root/'configs/rubric.json'))
        main_ids={i['image_id'] for i in main_items}
        main_ann=dict(ann,rows=[r for r in ann['rows'] if r['image_id'] in main_ids])
        q=quality(main,main_ann,main_items,sha(root/'configs/rubric.json'))
        q['all_cohort_annotation_coverage']=full_coverage
    else:q={'status':'ANNOTATION_PENDING','primary':None,'coverage':{'annotated':0,'expected':len(main_items)},'reason':'No evaluator annotations supplied. Image validity is not constraint correctness.'}
    timing={}
    for setting in ['A_time','B_time','C_time']:
        rr=[r for r in main if r['status']=='SUCCESS' and r['job']['setting']['id']==setting]
        repeats=[r for r in rows if r['status']=='SUCCESS' and r['job']['phase']=='repeat' and r['job']['setting']['id']==setting]
        if rr:
            values=[r['complete_seconds'] for r in rr]
            timing[setting]={'images':len(rr),'loops':rr[0]['job']['setting']['loops'],'steps':rr[0]['job']['setting']['steps'],
                'main_complete_median_seconds':statistics.median(values),'main_complete_minmax_seconds':[min(values),max(values)],'main_sampler_gpu_median_ms':statistics.median(r['sampler_gpu_ms'] for r in rr),
                'main_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rr),'main_peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in rr),
                'repeat_process_medians':[{'round':n,'n':sum(r['job']['round']==n for r in repeats),'median_seconds':statistics.median(r['complete_seconds'] for r in repeats if r['job']['round']==n)} for n in range(3) if any(r['job']['round']==n for r in repeats)]}
    summary={'schema':'case012-summary-v1','status':q['status'],'generation':{'main_expected':192,'main_records':len(main),'main_success':sum(r['status']=='SUCCESS' for r in main),'smoke_expected':24,'smoke_records':len(smoke),'smoke_success':sum(r['status']=='SUCCESS' for r in smoke),'failures':sum(r['status']=='FAILED' for r in rows),'independent_main_prompt_seed_pairs':64,'prompt_clusters':16,'all_recorded_attempts':len(rows)},'timing':timing,'quality':q,'budget':load(root/'results/budget.json'),'main_settings':load(root/'configs/main_settings.json') if (root/'configs/main_settings.json').exists() else None,
      'source_hashes':{'protocol':sha(root/'configs/protocol.json'),'prompts':sha(root/'configs/prompts.json'),'rubric':sha(root/'configs/rubric.json'),'models':sha(root/'provenance/models.json') if (root/'provenance/models.json').exists() else None},'new_training':0,'remote_writes':0}
    dump(root/'analysis/summary.json',summary)
    dump(root/'analysis/record_index.json',{'rows':rows})
    return summary
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--annotations',type=Path);a=ap.parse_args()
    s=build(a.root,a.annotations);print(json.dumps({k:s[k] for k in ['status','generation','timing']},ensure_ascii=False,indent=2))
