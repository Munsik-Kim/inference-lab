"""New synthetic tasks; source, oracle, schema and input freeze before smoke."""
import copy
import hashlib
import random
from pathlib import Path
from .common import ROOT,P1,METHODS,APIS,SCORER,atomic,read,require,digest,sha,token_hash,utc
from .registry import Registry,MAPS

TEMPLATES=[
    'Settle order {order_id}. First look up its quantity and unit price. After receiving that result, use the multiplication tool to compute the total in integer minor units. After receiving the product, save it as the order total. Validate the saved order. After receiving successful validation, return only JSON with order_id, total_minor, and validated. Do not change any other order.',
    'For order {order_id}, retrieve the order record and wait for the result. Use the returned quantity and unit price as the two integer multiplication arguments, then wait for the product. Save that product to the same order, and validate the saved record. Once you have the validation result, finish with JSON containing order_id, total_minor, and validated. Preserve all other orders and immutable fields.',
    'Complete the settlement of {order_id} using the provided tools: read the order; receive its data; multiply quantity by unit price in minor units; receive the product; write that total; validate the stored record; receive validation; return JSON with exactly order_id, total_minor, and validated. Other orders and all non-total fields must remain unchanged.',
    'Order to process: {order_id}. Obtain its quantity and unit price from a tool result before requesting their multiplication. Obtain the multiplication result before requesting a total update. Validate after the update executes, and observe the validation result before returning the final JSON (order_id, total_minor, validated). Work only on this order and preserve every other field.',
]
SYSTEM='Use the declared local order tools to complete the requested workflow. Amounts are integer minor units. Return the requested final JSON. Do not execute code.'


def generate_tasks(seed=6101621):
    tasks=[]
    for k in range(17):
        kind='smoke' if k==16 else 'primary';rng=random.Random(int(hashlib.sha256(f'{seed}|{kind}|{k}|values'.encode()).hexdigest(),16))
        ids=['ORD-'+hashlib.sha256(f'{seed}|{kind}|{k}|id|{i}'.encode()).hexdigest()[:12].upper() for i in range(2)]
        orders={oid:{'order_id':oid,'quantity':rng.randint(2,40),'unit_price_minor':rng.randint(100,20000),'total_minor':None if rng.randrange(2)==0 else -1} for oid in ids}
        template=k//4 if k<16 else 0;scenario={'scenario_id':f'A1-{k:02d}' if k<16 else 'A1-SMOKE','template_id':f'T{template+1}',
            'target_order_id':ids[0],'control_order_id':ids[1],'orders':orders,
            'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':TEMPLATES[template].format(order_id=ids[0])}],
            'role':'new_synthetic_same_workflow; template-dependent','fixture_seed':seed,'split':kind}
        scenario['canonical_task_hash']=digest(scenario)
        tasks.append(scenario)
    require(len({oid for s in tasks for oid in s['orders']})==34,'order ID collision')
    return tasks


def register(root,private):
    root=Path(root);private=Path(private);require(not (root/'specs/protocol.json').exists(),'preserve prior registration')
    tasks=generate_tasks();atomic(root/'data/scenarios.json',tasks,True)
    atomic(root/'data/api_mappings.json',MAPS,True)
    atomic(root/'data/schema.json',{api:Registry(api).tools() for api in APIS},True)
    from transformers import AutoTokenizer
    runtime=read(private/'runtime.json');tok=AutoTokenizer.from_pretrained(runtime['model']['path'],local_files_only=True,trust_remote_code=False)
    manifest=[];plan=[];ids_private={};oracle=[]
    for task in tasks:
        r=task['orders'][task['target_order_id']];total=r['quantity']*r['unit_price_minor']
        oracle.append({'scenario_id':task['scenario_id'],'order_id':task['target_order_id'],'quantity':r['quantity'],'unit_price_minor':r['unit_price_minor'],'total_minor':total,
            'gold':{'order_id':task['target_order_id'],'total_minor':total,'validated':True},'oracle_sha256':digest([r['quantity'],r['unit_price_minor'],total]),'evidence_kind':'synthetic_contract_definition'})
        for api in APIS:
            tools=Registry(api).tools();ids=tok.apply_chat_template(task['messages'],tools=tools,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
            ids_private[task['scenario_id']+'/'+api]=ids
            info={'scenario_id':task['scenario_id'],'api_version':api,'template_id':task['template_id'],'canonical_task_hash':task['canonical_task_hash'],
                'surface_prompt_hash':digest({'messages':task['messages'],'tools':tools}),'input_token_sha256':token_hash(ids),'input_tokens':len(ids),
                'name_tokenization':[{'name':t['function']['name'],'standalone_ids':tok.encode(t['function']['name'],add_special_tokens=False)} for t in tools],
                'payload_hash':digest(task['orders']),'gold_hash':digest(oracle[-1]['gold']),'actual_values_not_in_initial_prompt':True,'split':task['split']}
            manifest.append(info)
            if task['split']=='primary':
                for method in METHODS:plan.append({'episode_id':method+'__'+api+'__'+task['scenario_id'],'method':method,**{k:info[k] for k in ['scenario_id','template_id','api_version','canonical_task_hash','surface_prompt_hash','input_token_sha256']}})
    generation=copy.deepcopy(read(P1/'specs/protocol.json')['generation'])
    protocol={'version':'case015-a1-v1','scorer_version':SCORER,'created_utc':utc(),'design':'post-P1 new synthetic application; not P2 diagnostics or cross-business confirmation',
        'methods':list(METHODS),'apis':list(APIS),'scenario_count':16,'template_count':4,'scenario_per_template':4,'expected_primary_episodes':96,'smoke_episodes':2,
        'fixture_seed':6101621,'execution_order_seed':6101622,'analysis_seed':6101623,'bootstrap_replicates':10000,'bootstrap_unit':'whole template group, exploratory; no deployment/equivalence claim',
        'model_load_order':list(METHODS),'within_method_order':'same seed6101622 shuffle of32scenario/APIcells, no performance-dependent change',
        'generation':generation,'worker_wall_budget_seconds':7200,
        'primary_endpoint':'VERIFIED_WORKFLOW_SUCCESS: actual ordered chain, read and multiply results observed in later generations, unchanged other orders/immutable fields throughout, current validation observed, strict final JSON',
        'observation_contract':{'multiply':'target read result available before generation','write':'correct multiply on observed target read available before generation','write_then_validate_same_response':'allowed if execution order and current validation match','final':'matching current-state validation available before final generation'},
        'schema_policy':'exact keys; int excludes bool; schema-valid wrong values executed without oracle correction','error_feedback':'same generic unknown tool or invalid arguments across all versions/methods',
        'parser':'P1 first valid JSON object within tool markers/final; no grammar repair; malformed marker attempts consume tool budget',
        'final_schema':{'order_id':'strict string, target','total_minor':'strict integer, oracle product','validated':'strict bool true'},
        'mitigations':{'ENUM':False,'enhanced_feedback':False,'aliases':False},'P1_GPU_rerun':False,'new_quantization':False,'training':False,'remote_publication_authorized':True,
        'input_manifest_sha256':None,'planned_cells_sha256':None,'private_runtime_sha256':sha(private/'runtime.json')}
    atomic(root/'data/oracle.json',oracle,True);atomic(root/'results/a1_r1/planned_cells.json',plan,True)
    atomic(root/'provenance/input_manifest.json',{'inputs':{str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted((root/'data').glob('*.json'))},'rendered_inputs':manifest},True)
    atomic(private/'initial_token_ids.json',ids_private,True)
    protocol['input_manifest_sha256']=sha(root/'provenance/input_manifest.json');protocol['planned_cells_sha256']=sha(root/'results/a1_r1/planned_cells.json')
    atomic(root/'specs/protocol.json',protocol,True)
    return {'status':'REGISTERED','scenarios':16,'smoke_scenarios':1,'planned_episodes':96,'model_forwards':0}


def freeze(root):
    root=Path(root);files={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for d in ['src','tests','data','specs'] for p in sorted((root/d).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    reused={str(p.relative_to(P1)):{'sha256':sha(p)} for p in [P1/'src/interface_sensitivity/common.py',P1/'src/interface_sensitivity/model_adapter.py',P1/'src/interface_sensitivity/runner.py',*sorted((P1/'src/interface_sensitivity/upstream_case013').glob('*.py'))]}
    value={'created_utc':utc(),'files':files,'reused_P1_files':reused,'execution_source_sha256':digest({'A1':files,'reused_P1':reused}),
        'P1_source_sha256':read(P1/'provenance/source_freeze.json')['execution_source_sha256'],'git_research_base':'e647b474bdc9fa558d851bd9c998b664b0e33638'}
    atomic(root/'provenance/source_freeze.json',value,True);return value['execution_source_sha256']


def validate(root):
    root=Path(root);p=read(root/'specs/protocol.json');require(p['expected_primary_episodes']==96,'protocol matrix changed')
    require(sha(root/'provenance/input_manifest.json')==p['input_manifest_sha256'],'input manifest changed')
    require(sha(root/'results/a1_r1/planned_cells.json')==p['planned_cells_sha256'],'planned cells changed')
    for name,e in read(root/'provenance/input_manifest.json')['inputs'].items():require(sha(root/name)==e['sha256'],'fixture/schema/oracle changed')
    if (root/'provenance/source_freeze.json').exists():
        f=read(root/'provenance/source_freeze.json')
        for name,e in f['files'].items():require(sha(root/name)==e['sha256'],'A1 frozen source changed: '+name)
        for name,e in f['reused_P1_files'].items():require(sha(P1/name)==e['sha256'],'P1 source changed')
    return {'status':'PASS','primary_episodes':96,'scenarios':16,'apis':2,'model_forwards':0}
