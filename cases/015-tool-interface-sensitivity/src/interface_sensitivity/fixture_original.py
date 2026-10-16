"""Exact function excerpts from Case013 paper_model_runner.py (Apache-2.0).
See provenance/reuse.json for original source identity. No GPU imports.
"""
import json, math
from pathlib import Path
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.partial')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)

class OriginalFixture:
    def tool(self,name,args,fixture,config_path):
            params={'read_document':[],'read_record':[],'query_table':[],'calculate':['a','b'],'apply_fixture_patch':['enabled','limit'],'validate_fixture_test':[]}
            if name not in params or set(args)!=set(params[name]):raise ValueError('unknown tool/incorrect argument keys')
            if name=='read_document':return fixture['document']
            if name=='read_record':return fixture['record']
            if name=='query_table':return fixture['updates']
            if name=='calculate':
                if any(type(args[k]) not in (int,float) or not math.isfinite(args[k]) for k in ['a','b']):raise ValueError('invalid numerical argument')
                return {'product':args['a']*args['b']}
            if name=='apply_fixture_patch':
                if type(args['enabled'])!=bool or type(args['limit'])!=int:raise ValueError('incorrect patch argument types')
                atomic(config_path,args);return {'applied':True,'config':args}
            return {'valid':json.loads(config_path.read_text())==fixture['expected_config']}
