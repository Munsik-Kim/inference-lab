"""Register inputs and identities before model forward; local files only."""
import random
from pathlib import Path
from .common import CASE, METHODS, INTERFACES, HARNESS, atomic, digest, inputs, read, require, sha, token_hash, utc
from .transform import make_interface
from .equivalence import oracle_check


def register(case, private):
    case=Path(case);private=Path(private)
    require(not (case/'specs/protocol.json').exists(),'registration already exists; preserve frozen version')
    runtime=read(private/'runtime.json')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(runtime['model']['path'],local_files_only=True,trust_remote_code=False)
    tasks,messages,tools=inputs(case);historical={r['id']:r for r in read(case/'data/agent_token_manifest.json')}
    interface_records=[]; manifest=[]; private_ids={};plan=[]
    for task in tasks:
        for interface in INTERFACES:
            spec=make_interface(task,messages[task['id']],tools,interface)
            ids=tokenizer.apply_chat_template(spec['messages'],tools=spec['tools'],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
            if interface==INTERFACES[0]:
                require(token_hash(ids)==historical[task['id']]['initial_token_sha256'],'historical I0 token mismatch')
            item={k:spec[k] for k in ('task_id','interface_id','canonical_task_hash','rendered_prompt_hash','mapping','permutation','changes')}
            item['mapping_sha256']=digest(item['mapping']);interface_records.append(item)
            names=[t['function']['name'] for t in spec['tools']]
            manifest.append({'task_id':task['id'],'family_id':task['family'],'interface_id':interface,
                'canonical_task_hash':spec['canonical_task_hash'],'rendered_prompt_hash':spec['rendered_prompt_hash'],
                'input_token_sha256':token_hash(ids),'input_tokens':len(ids),'historical_I0_matched':interface==INTERFACES[0],
                'names_tokenization':[{'name':n,'standalone_token_ids':tokenizer.encode(n,add_special_tokens=False)} for n in names],
                'name_tokenization_scope':'standalone auxiliary; actual model input hash from full apply_chat_template',
                'requested_context_target':task['initial_context_target'],'task_text_character_start':len(spec['messages'][-1]['content'].split('\n\nTASK: ')[0])+len('\n\nTASK: ')})
            private_ids[task['id']+'/'+interface]=ids
            for method in METHODS:
                plan.append({'episode_id':method+'__'+interface+'__'+task['id'],'method':method,'task_id':task['id'],
                    'family_id':task['family'],'interface_id':interface,'canonical_task_hash':spec['canonical_task_hash'],
                    'rendered_prompt_hash':spec['rendered_prompt_hash'],'input_token_sha256':token_hash(ids)})
    generation={'greedy':True,'enable_thinking':False,'max_new_tokens':192,'max_model_calls':8,'max_tool_calls':6,
        'deadline_seconds':900,'context_limit':24576,'batch_size':1,'seed':610132,'KV':'BF16 within request ON; across-turn prefix cache OFF',
        'logits_to_keep':1,'stop_policy':'unchanged saved model generation_config/eos_token_id','automatic_truncate':False}
    artifacts=read(case/'provenance/artifact_identity.json')
    protocol={'version':'case015-v1','created_utc':utc(),'harness_version':HARNESS,'role':'EXPLORATORY_PREVIOUSLY_OBSERVED_8TASK_DEV',
        'methods':list(METHODS),'interfaces':list(INTERFACES),'expected_primary_episodes':96,'generation':generation,
        'interface_seed':6101601,'execution_order_seed':6101602,'bootstrap_seed':6101603,'bootstrap_replicates':10000,
        'worker_wall_budget_seconds':7200,'model_load_order':['BF16']+random.Random(6101602).sample(['RTN','D16'],2),
        'within_method':'all 8 I0 first in original task order; remaining 24 cells shuffled with seed6101602 per method',
        'primary':'mean_task[(mean_I1_I2_I3 D16-I0 D16)-(mean_I1_I2_I3 BF16-I0 BF16)]',
        'functional_success':'final gold + required ordered original actions + actual final state; fixture.validate true after correct patch',
        'legacy_success':'pinned Case013 gold subset equality; fixture_patch additionally actual config and valid validation event',
        'required_actions':{'document':['document.read'],'combine':['record.read','math.multiply(quantity,unit_price), either operand order'],
                            'updates':['updates.read'],'fixture_patch':['fixture.patch(expected_config)','fixture.validate(valid=true on final state)']},
        'semantic_allowed_sets':'required family actions, with record before multiply and correct patch before validation; sidecar only, recoverable errors do not force final failure',
        'strict_argument_schema':'original keys/types; correct required action values checked independently without hints to model',
        'mitigations':{'ENUM':False,'enhanced_feedback':False,'aliases':False},
        'smoke':{'method':'BF16','interface':'I0_ORIGINAL','tasks':['AGENT-6','AGENT-7'],'counted_in_primary':False},
        'P2':'NOT_RUN_SCOPE','P3':'NOT_RUN_SCOPE','D4':'NOT_RUN_SCOPE','training':'NOT_RUN_SCOPE','publication':'NOT_AUTHORIZED',
        'code_changes_after_freeze':'preserve failed attempts, amendment and new source identity; no tolerance/score tuning',
        'input_manifest_sha256':None,'planned_cells_sha256':None}
    atomic(case/'specs/interfaces.json',interface_records,True)
    atomic(case/'specs/tasks_manifest.json',[{'id':t['id'],'family':t['family'],'task_sha256':digest(t),'context_target':t['initial_context_target']} for t in tasks],True)
    atomic(case/'specs/methods.json',{'model_id':runtime['model']['id'],'revision':runtime['model']['revision'],
        'execution':'dense BF16 restored reference, D16 FP32 X*beta_inverse->BF16, not folded/double transported',
        'methods':{m:{'projection_count':0 if m=='BF16' else 252,'artifact_manifest_sha256':artifacts[m]['manifest_sha256'] if m!='BF16' else None,
                      'artifact_id':runtime['model']['revision'] if m=='BF16' else artifacts[m]['manifest_sha256']} for m in METHODS}},True)
    atomic(case/'provenance/input_manifest.json',{'inputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((case/'data').iterdir()) if p.is_file()},
        'rendered_inputs':manifest,'fixtures':'owned synthetic Case013 fixtures, no new tasks/lengths'},True)
    atomic(case/'results/p1_r1/planned_cells.json',plan,True)
    atomic(private/'initial_token_ids.json',private_ids,True)
    protocol['input_manifest_sha256']=sha(case/'provenance/input_manifest.json')
    protocol['planned_cells_sha256']=sha(case/'results/p1_r1/planned_cells.json')
    atomic(case/'specs/protocol.json',protocol,True)
    p0=oracle_check(case);atomic(case/'results/p1_r1/p0_oracle.json',p0,True)
    return {'status':'REGISTERED','planned':len(plan),'oracle_episodes':p0['oracle_episodes'],'historical_I0_hash_matches':8,'model_forwards':0}


def freeze(case):
    case=Path(case)
    files={str(p.relative_to(case)):{'sha256':sha(p),'bytes':p.stat().st_size} for folder in ('src','tests','data','specs') for p in sorted((case/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    freeze={'created_utc':utc(),'files':files,'execution_source_sha256':digest({k:v['sha256'] for k,v in files.items()}),
            'git_base':'e647b474bdc9fa558d851bd9c998b664b0e33638','source_hash_is_not_git_commit':True}
    atomic(case/'provenance/source_freeze.json',freeze,True)
    return freeze['execution_source_sha256']


def validate(case):
    case=Path(case);p=read(case/'specs/protocol.json');tasks,messages,tools=inputs(case)
    require(p['expected_primary_episodes']==8*3*4==96,'planned count mismatch')
    require(sha(case/'provenance/input_manifest.json')==p['input_manifest_sha256'],'input manifest changed')
    require(sha(case/'results/p1_r1/planned_cells.json')==p['planned_cells_sha256'],'plan changed')
    for name,e in read(case/'provenance/input_manifest.json')['inputs'].items():
        require(sha(case/'data'/name)==e['sha256'],'raw fixture identity mismatch')
    expected={(r['task_id'],r['interface_id']):r for r in read(case/'specs/interfaces.json')}
    for task in tasks:
        for interface in INTERFACES:
            s=make_interface(task,messages[task['id']],tools,interface)
            r=expected[(task['id'],interface)]
            for key in ('mapping','permutation','changes','canonical_task_hash','rendered_prompt_hash'):
                require(s[key]==r[key],'registered interface changed: '+key)
    f=case/'provenance/source_freeze.json'
    if f.exists():
        for name,e in read(f)['files'].items():
            require(sha(case/name)==e['sha256'],'execution source/spec changed: '+name)
    return {'status':'PASS','registered_interfaces':32,'planned_primary_episodes':96,'mitigations_OFF':True}
