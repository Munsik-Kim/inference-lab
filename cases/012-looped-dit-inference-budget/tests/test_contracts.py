import json, math
from pathlib import Path
import pytest
from source.contracts import *
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('setting',[{'loops':0,'steps':50},{'loops':8,'steps':50},{'loops':True,'steps':50},{'loops':2,'steps':0},{'loops':2,'steps':126},{'loops':2,'steps':1.5}])
def test_invalid_settings(setting):
    with pytest.raises(ValueError): validate_setting(setting)

@pytest.mark.parametrize('l,s,total',[(1,94,3196),(2,73,3212),(4,50,3200)])
def test_proxy_cfg_counts(l,s,total):
    c=counts({'loops':l,'steps':s}); assert c['joint_blocks']==total and c['model']==2*s
    assert c['text_preamble']==4*s and c['decode']==2*s

def test_frozen_original_inputs(): assert verify_freeze(ROOT)['status']=='FROZEN_BEFORE_GENERATION'

def test_modified_freeze_rejected(tmp_path):
    (tmp_path/'configs').mkdir();dump(tmp_path/'configs/input_freeze.json',{'files':{'configs/x':'wrong'}});(tmp_path/'configs/x').write_text('x')
    with pytest.raises(ValueError): verify_freeze(tmp_path)

@pytest.mark.parametrize('literal',['NaN','Infinity','-Infinity'])
def test_nonfinite_json_rejected(tmp_path,literal):
    p=tmp_path/'x.json';p.write_text('{"x":'+literal+'}')
    with pytest.raises(ValueError):load(p)

def test_duplicate_record():
    with pytest.raises(ValueError):unique_records([{'job_id':'a','status':'FAILED'}]*2)

def test_atomic_failure_retained(tmp_path):
    p=tmp_path/'attempt-1.partial';p.mkdir();(p/'started.json').write_text('{}')
    r={'status':'FAILED','error':'synthetic_test'};commit_attempt(p,tmp_path/'attempt-1',r)
    assert not p.exists() and verify_attempt(tmp_path/'attempt-1')==r

def test_atomic_refuses_overwrite(tmp_path):
    p=tmp_path/'partial';p.mkdir();f=tmp_path/'final';f.mkdir()
    with pytest.raises(ValueError):commit_attempt(p,f,{})

def test_modified_attempt_detected(tmp_path):
    p=tmp_path/'partial';p.mkdir();commit_attempt(p,tmp_path/'final',{'status':'FAILED'})
    (tmp_path/'final/record.json').write_text('{}')
    with pytest.raises(ValueError):verify_attempt(tmp_path/'final')

ITEM=[{'image_id':'opaque','image_sha256':'hash','constraints':[{'id':'c1'},{'id':'c2'}]}]
def ann():return {'schema':'case012-annotations-v1','rubric_sha256':'r','evaluator':{'type':'human','id':'fixture'},'rows':[{'image_id':'opaque','image_sha256':'hash','values':{'c1':'satisfied','c2':'uncertain'}}]}

def test_annotations_complete():assert annotation_validate(ann(),ITEM,'r')['complete']
def test_annotations_pending():
    x=ann();x['rows']=[];assert not annotation_validate(x,ITEM,'r')['complete']
@pytest.mark.parametrize('kind',['hash','duplicate','constraint','value','evaluator','rubric'])
def test_bad_annotation(kind):
    x=ann()
    if kind=='hash':x['rows'][0]['image_sha256']='bad'
    if kind=='duplicate':x['rows']*=2
    if kind=='constraint':del x['rows'][0]['values']['c1']
    if kind=='value':x['rows'][0]['values']['c1']='looks_good'
    if kind=='evaluator':x['evaluator']['type']='independent_human_assumed'
    if kind=='rubric':x['rubric_sha256']='wrong'
    with pytest.raises(ValueError):annotation_validate(x,ITEM,'r')
