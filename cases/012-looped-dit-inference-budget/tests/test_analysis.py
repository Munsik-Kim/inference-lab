from pathlib import Path
import copy
import pytest
from analysis.analyze import quality
from source.contracts import digest

def fixture():
    rows=[];items=[];annotations=[]
    for p in range(16):
        for seed in [72301,72302,72303,72304]:
            for s in ['A_time','B_time','C_time']:
                id=f'synthetic-{p}-{seed}-{s}'
                rows.append({'job_id':id,'status':'SUCCESS','job':{'prompt_id':f'p{p}','seed_label':seed,'setting':{'id':s}}})
                items.append({'image_id':id,'image_sha256':'h','constraints':[{'id':'c1'},{'id':'c2'}],'category':'fixture'})
                # C gains one constraint for everyone; all-constraint outcome is uncertain/fail on half of A.
                annotations.append({'image_id':id,'image_sha256':'h','values':{'c1':'satisfied','c2':'satisfied' if s=='C_time' or p%2==0 else 'uncertain'}})
    ann={'schema':'case012-annotations-v1','rubric_sha256':'r','evaluator':{'type':'human','id':'synthetic_test_not_real_quality'},'rows':annotations}
    return rows,items,ann

def test_quality_prompt_cluster_gain_and_denominator():
    rows,items,ann=fixture();q=quality(rows,ann,items,'r')
    assert q['primary']['estimate']==.5 and q['primary']['prompt_clusters']==16
    assert q['settings']['A_time']['images']==64 and q['settings']['C_time']['all_constraint_pass_rate']==1
    assert q['paired']=={'both_pass':32,'A_only':0,'C_only':32,'neither':0}
    assert sum(q['constraint_gains'].values())==32 and sum(q['constraint_losses'].values())==0
    assert q['settings']['A_time']['all_uncertain_as_pass_rate']==1

def test_partial_annotation_no_score():
    r,i,a=fixture();a['rows']=a['rows'][:-1];q=quality(r,a,i,'r')
    assert q['primary'] is None and q['status']=='ANNOTATION_PENDING'

def test_annotation_permutation_invariant():
    r,i,a=fixture();q=quality(r,a,i,'r');a['rows'].reverse()
    assert quality(r,a,i,'r')==q

def test_pair_direction_loss():
    r,i,a=fixture()
    for x in a['rows']:
        if x['image_id'].endswith('C_time'):x['values']['c1']='not_satisfied'
    q=quality(r,a,i,'r');assert q['primary']['estimate']==-.5 and q['paired']['A_only']==32

def test_failure_image_counts_as_failure():
    r,i,a=fixture();r[-1]['status']='FAILED'
    q=quality(r,a,i,'r');assert q['settings']['C_time']['all_constraint_pass_rate']==63/64
