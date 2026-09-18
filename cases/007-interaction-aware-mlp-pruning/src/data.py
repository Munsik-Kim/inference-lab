"""New fact-first synthetic scenarios; deterministic irrelevant filler only."""
import random
from .core import TASKS,COUNTS,SEED,digest
from .oracle import interpret

def scenario(split,task,index):
 if split not in (*COUNTS,'SMOKE') or task not in TASKS:raise ValueError('Unknown split/task')
 sid=f'c007-{split.lower()}-{task.lower()}-{index:03d}';rng=random.Random(sid+str(SEED));serial=list((*COUNTS,'SMOKE')).index(split)*1000+index
 if task=='CODE':
  start=10000+serial;step=rng.randrange(2,10);count=2+index%5;mod=17+index%13
  program=f'total = {start}\nfor i in range({count}):\n    total = total + {step} * i\nanswer = total % {mod}'
  answer=(start+step*count*(count-1)//2)%mod
  if interpret(program)!=answer:raise ValueError('Oracles disagree')
  facts={'start':start,'step':step,'count':count,'mod':mod};core='Integer Python subset (range excludes its end):\n'+program;question='What is the final integer value of answer?';foils=[answer+1,answer+3,answer+7]
 else:
  vals=rng.sample(range(100,990),8);facts=[{'id':f'unit-{serial}-{j}','value':v} for j,v in enumerate(vals)];rng.shuffle(facts)
  core='Each identifier occurs exactly once:\n'+'\n'.join(f"{f['id']} = {f['value']} units." for f in facts)
  a,b=rng.sample(facts,2)
  if task=='RETRIEVAL':answer=a['value'];question=f"How many units belong to {a['id']}?";foils=[f['value'] for f in facts if f!=a][:3]
  else:answer=a['value']-b['value'];question=f"What is the value of {a['id']} minus the value of {b['id']}?";foils=[answer+1,answer+3,answer+7]
 options=list(map(str,[answer]+foils));rng.shuffle(options);target=index%4;at=options.index(str(answer));options[target],options[at]=options[at],options[target]
 if len(set(options))!=4 or options[target]!=str(answer):raise ValueError('Bad gold/foils')
 return {'id':sid,'split':split,'task':task,'facts':facts,'facts_hash':digest({'task':task,'facts':facts}),'core':core,'question':question,'options':options,'gold':target,'gold_value':str(answer),'language':'English; restricted Python for CODE'}
def render(row,tok,length=512):
 def make(extra):
  content='Answer with exactly one letter: A, B, C, or D. Do not explain.\n'+row['core']+'\n<irrelevant_notes>'+(' note'*extra)+'</irrelevant_notes>\nQuestion: '+row['question']+'\nOptions:\n'+'\n'.join(f'{l}. {v}' for l,v in zip('ABCD',row['options']))+'\n'
  text=tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  return text,tok.encode(text,add_special_tokens=False)
 text,ids=make(0)
 if len(ids)>length:raise ValueError('Essential evidence exceeds budget; never truncate')
 extra=length-len(ids)
 for adjust in range(-3,4):
  if extra+adjust<0:continue
  text,ids=make(extra+adjust)
  if len(ids)==length:break
 if len(ids)!=length:raise ValueError('Exact filler contract failed')
 labels=[]
 for l in 'ABCD':
  joint=tok.encode(text+l,add_special_tokens=False)
  if joint[:-1]!=ids:raise ValueError('Unstable one-token answer interface')
  labels.append(joint[-1])
 if len(set(labels))!=4:raise ValueError('Label token collision')
 return {**row,'text':text,'text_hash':digest(text),'token_ids':ids,'token_hash':digest(ids),'label_ids':labels,'length':len(ids),'future_answer_tokens_present':False}
