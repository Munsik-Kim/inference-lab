"""Fresh synthetic scenarios, exact oracles and prompt-only answer boundaries."""
from __future__ import annotations
import hashlib
import random
from pathlib import Path
from .common import CASE, digest, read, require, write

TASKS = ('retrieval', 'comparison', 'code')
SPLITS = ('smoke', 'calibration', 'development', 'heldout')


def scenario(track: str, split: str, task: str, index: int) -> dict:
    require(track in ('Q','R') and split in SPLITS and task in TASKS and index >= 0, 'Unknown scenario')
    ident = f'c008-{track.lower()}-{split}-{task}-{index:03d}'
    rng = random.Random(int(hashlib.sha256(ident.encode()).hexdigest(),16))
    if task == 'code':
        # Different split/track namespaces force disjoint complete program facts.
        start=1000+(0 if track=='Q' else 50000)+SPLITS.index(split)*1000+index
        step=rng.randrange(2,8); count=2+index%4; modulus=19+index%11
        program=f'total = {start}\nfor i in range({count}):\n    total = total + {step} * i\nanswer = total % {modulus}'
        value=start
        for i in range(count): value+=step*i
        answer=value%modulus
        require(answer==(start+step*count*(count-1)//2)%modulus,'Independent oracle mismatch')
        facts={'start':start,'step':step,'count':count,'modulus':modulus}
        core='Python integer program. range excludes its end.\n'+program
        question='What is the final integer value of answer?'
        options=[str(answer),str(answer+1),str(answer+3),str(answer+7)]
    else:
        facts=[{'id':f'{ident}-record-{j}','value':v} for j,v in enumerate(rng.sample(range(10,990),6))]
        # Short visible names, while the complete manifest carries unique scenario IDs.
        names=['Amber','Birch','Cedar','Delta','Ember','Flint']
        rng.shuffle(names)
        for f,name in zip(facts,names):f['name']=name
        first,second=rng.sample(facts,2)
        core='Inventory report. Every name occurs once; amounts use the same units.\n'+'\n'.join(f"{f['name']}: {f['value']} units." for f in facts)
        if task=='retrieval':
            answer=str(first['value']);question=f"What amount is listed for {first['name']}?"
            options=[answer]+[str(f['value']) for f in facts if f!=first][:3]
        else:
            answer=max((first,second),key=lambda f:f['value'])['name']
            question=f"Between {first['name']} and {second['name']}, which has the larger amount?"
            options=[answer]+[f['name'] for f in facts if f['name']!=answer][:3]
    answer=str(answer);rng.shuffle(options); gold=index%4;old=options.index(answer)
    options[old],options[gold]=options[gold],options[old]
    require(len(set(options))==4 and options[gold]==answer,'Gold/foil failure')
    return {'id':ident,'track':track,'split':split,'task':task,'facts':facts,'facts_hash':digest({'task':task,'facts':facts}),
            'core':core,'question':question,'options':options,'gold':gold,'gold_value':answer}


def render(row: dict, tokenizer, maximum: int) -> dict:
    content=('Return only one answer letter: A, B, C, or D.\n'+row['core']+'\nQuestion: '+row['question']+
             '\nOptions:\n'+'\n'.join(f'{a}. {b}' for a,b in zip('ABCD',row['options']))+'\n')
    text=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,
                                       add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer.encode(text,add_special_tokens=False)
    require(32 <= len(ids) <= maximum, 'Essential prompt outside frozen length cap')
    labels=[]
    for label in 'ABCD':
        joined=tokenizer.encode(text+label,add_special_tokens=False)
        require(joined[:-1]==ids and len(joined)==len(ids)+1,'BLOCKED_ANSWER_INTERFACE')
        labels.append(joined[-1])
    require(len(set(labels))==4,'Duplicate label tokens')
    return {**row,'text':text,'text_hash':digest(text),'token_ids':ids,'token_hash':digest(ids),
            'label_ids':labels,'length':len(ids),'future_answer_tokens_present':False}


def prepare(track: str, snapshot: Path) -> dict:
    from transformers import AutoTokenizer
    proto=read(CASE/'configs/protocol_draft.json');cfg=proto[track]
    require(snapshot.name==cfg['revision'],'Unexpected source revision')
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    root=CASE/'inputs'/track
    require(not root.exists(),'Inputs already exist')
    rows=[]
    for split,count in cfg['counts'].items():
        for task_index,task in enumerate(TASKS):
            n=count//3+(task_index<count%3)
            for index in range(n):rows.append(render(scenario(track,split,task,index),tok,cfg['max_length']))
    require(len({r['id'] for r in rows})==len(rows),'Duplicate ID')
    require(len({r['facts_hash'] for r in rows})==len(rows),'Base scenario overlap')
    require(len({r['token_hash'] for r in rows})==len(rows),'Exact token overlap')
    for split in cfg['counts']:write(root/(split+'.json'),[r for r in rows if r['split']==split])
    timing=[]
    for task in TASKS:
        timing += [r['id'] for r in sorted([r for r in rows if r['split']=='heldout' and r['task']==task],key=lambda r:digest(r['id']))[:4]]
    manifest={'track':track,'counts':cfg['counts'],'tokenizer_template_hash':digest(tok.chat_template),
              'all_input_hash':digest(rows),'timing_ids':timing,'model_revision':snapshot.name,
              'length_min':min(r['length'] for r in rows),'length_max':max(r['length'] for r in rows),
              'total_calibration_tokens':sum(r['length'] for r in rows if r['split']=='calibration'),
              'rows':{r['id']:{'token_hash':r['token_hash'],'facts_hash':r['facts_hash'],'split':r['split'],'task':r['task']} for r in rows}}
    write(root/'manifest.json',manifest)
    return manifest
