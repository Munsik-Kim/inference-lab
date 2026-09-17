"""Deterministic post-DESIGN_FREEZE inputs; no model access or output filtering."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.tasks import TASKS,scenario,build_prompt
from src.storage import load,write_new,file_hash,digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--design-freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    frozen=load(a.design_freeze)
    if frozen['kind']!='DESIGN_FREEZE':raise ValueError('Genuine design freeze required')
    if a.output.exists():raise FileExistsError(a.output)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True,trust_remote_code=False)
    a.output.mkdir(parents=True);files=[];selected={};allgold=[];allinputs=[]
    for split in ('standard','boundary_pool'):
        for task in TASKS:
            latent=[scenario(split,task,i) for i in range(64)]
            prompts=[build_prompt(r,tok,4096) for r in latent]
            name=f'{split}-{task.lower()}.json';write_new(a.output/name,prompts);files.append(name)
            allgold.extend(latent);allinputs.extend(prompts)
            if split=='standard':
                ordered=sorted(latent,key=lambda r:hashlib.sha256(r['base_id'].encode()).hexdigest())
                selected[task]={'length_ids':[r['base_id'] for r in ordered[:16]],'trajectory_ids':[r['base_id'] for r in ordered[:8]],'timing_ids':[r['base_id'] for r in ordered[:4]]}
                length=[build_prompt(r,tok,n) for r in ordered[:16] for n in (512,2048)]
                secondary=[build_prompt(r,tok,4096,secondary=True) for r in ordered[:8]]
                for label,rows in [('length',length),('secondary',secondary)]:
                    name=f'{label}-{task.lower()}.json';write_new(a.output/name,rows);files.append(name);allinputs.extend(rows)
    if len({r['base_id'] for r in allgold})!=len(allgold) or len({r['base_facts_hash'] for r in allgold})!=len(allgold):
        raise ValueError('Overlapping evaluation base scenarios')
    write_new(a.output/'gold.json',allgold)
    write_new(a.output/'subsets.json',selected)
    write_new(a.output/'input_inventory.json',{'design_freeze_hash':file_hash(a.design_freeze),'files':{f:file_hash(a.output/f) for f in files},
          'input_hashes':{r['item_id']:r['token_hash'] for r in allinputs},'independent_scenarios':384,'prompts_including_dependent_subsets':len(allinputs),
          'tokenizer_template_hash':hashlib.sha256(tok.chat_template.encode()).hexdigest(),'selection':'SHA256(base_id) order within each task; no model output used'})
    print(json.dumps({'scenarios':384,'prompts':len(allinputs),'status':'PREPARED_WITHOUT_MODEL_SCORING'}))

if __name__=='__main__':main()
