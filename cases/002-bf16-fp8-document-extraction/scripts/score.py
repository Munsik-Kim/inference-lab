"""Strict scoring. Schema failures count as wrong for every primary field."""
import datetime,json,re,unicodedata
FIELDS=('owner','task','due_date','amount_krw')

class DuplicateKey(ValueError): pass

def object_pairs(pairs):
    result={}
    for k,v in pairs:
        if k in result: raise DuplicateKey(k)
        result[k]=v
    return result

def normalize(x):
    return unicodedata.normalize('NFC',x).strip() if isinstance(x,str) else x

def score(raw,gold):
    result={'json_valid':False,'schema_valid':False,'error':None,'parsed':None,'fields':{f:False for f in FIELDS},'document_correct':False}
    try: obj=json.loads(raw,object_pairs_hook=object_pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except DuplicateKey: result['error']='duplicate_key';return result
    except (ValueError,TypeError): result['error']='json_parse';return result
    result['json_valid']=True
    if not isinstance(obj,dict): result['error']='not_object';return result
    missing=set(FIELDS)-obj.keys();extra=obj.keys()-set(FIELDS)
    if missing or extra:
        result['error']='missing_and_extra_keys' if missing and extra else 'missing_keys' if missing else 'extra_keys';return result
    obj={k:normalize(v) for k,v in obj.items()};result['parsed']=obj
    for f in FIELDS:
        v=obj[f]
        if v is None: continue
        if f=='amount_krw':
            if type(v) is not int:result['error']='amount_type';return result
        elif not isinstance(v,str):result['error']=f+'_type';return result
        elif f=='due_date':
            try:
                if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',v):raise ValueError(v)
                datetime.date.fromisoformat(v)
            except ValueError:result['error']='date_format';return result
    result['schema_valid']=True
    result['fields']={f:obj[f]==normalize(gold[f]) for f in FIELDS}
    result['document_correct']=all(result['fields'].values())
    return result

def self_test():
    gold={'owner':'김민수','task':'서버 점검','due_date':'2026-10-15','amount_krw':250000}
    good=json.dumps(gold,ensure_ascii=False)
    assert score(good,gold)['document_correct']
    assert score(json.dumps({k:(' '+v+' ' if isinstance(v,str) else v) for k,v in reversed(list(gold.items()))}),gold)['document_correct']
    bad={**gold,'owner':'이서연'};assert not score(json.dumps(bad),gold)['document_correct']
    nullgold={**gold,'amount_krw':None}
    assert score(json.dumps(nullgold),nullgold)['document_correct']
    assert not score(json.dumps({**gold,'amount_krw':0}),nullgold)['fields']['amount_krw']
    assert score(json.dumps({**gold,'amount_krw':0}),{**gold,'amount_krw':0})['document_correct']
    assert score(json.dumps({**gold,'amount_krw':True}),gold)['error']=='amount_type'
    for v in ['', 'null']:
        assert not score(json.dumps({**nullgold,'owner':v}),{**nullgold,'owner':None})['fields']['owner']
    assert score(json.dumps({**gold,'due_date':'2026-2-3'}),gold)['error']=='date_format'
    assert score(json.dumps({**gold,'due_date':'2026-02-30'}),gold)['error']=='date_format'
    assert score(json.dumps({k:v for k,v in gold.items() if k!='owner'}),gold)['error']=='missing_keys'
    assert score(json.dumps({**gold,'extra':1}),gold)['error']=='extra_keys'
    assert score('{"owner":"x",'+good[1:],gold)['error']=='duplicate_key'
    assert score(good[:-1],gold)['error']=='json_parse'
    assert score('```json\n'+good+'\n```',gold)['error']=='json_parse'
    assert score('[]',gold)['error']=='not_object'
    assert score(json.dumps({**gold,'amount_krw':250000.0}),gold)['error']=='amount_type'
    assert score(json.dumps({**gold,'owner':unicodedata.normalize('NFD',gold['owner'])}),gold)['document_correct']
    return {'fixed_checks':19,'passed':True}

if __name__=='__main__':print(json.dumps(self_test()))
