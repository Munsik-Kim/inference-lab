"""Fixed DEV-only cyclic label audit. Does not inspect any candidate output."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.tasks import TASKS,scenario,permute,build_prompt
from src.storage import write_new
from src.model_runtime import Runtime,array
from src.metrics import score_full
from scripts.run_case006 import gpu_gate

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    gate=gpu_gate()
    if gate['status']!='RESOURCE_GATE_PASSED':write_new(a.output,gate);return 20
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True,trust_remote_code=False)
    rt=Runtime(a.snapshot);rows=[]
    for task in TASKS:
        base=scenario('dev',task,0)
        for shift in range(4):
            item=permute(base,shift);prompt=build_prompt(item,tok,4096);output=rt.forward(prompt['token_ids'],'B',False)
            score=score_full(array(output.logits[0,-1]),prompt['label_token_ids'],'ABCD'.index(item['gold']))
            rows.append({'base_id':base['base_id'],'task':task,'cyclic_shift':shift,'token_hash':prompt['token_hash'],'gold':item['gold'],'score':score})
    write_new(a.output,{'evidence_set':'DEVELOPMENT','arm':'B','independent_scenarios':3,'dependent_variants':12,'rows':rows,'candidate_outputs_used':False})
    print('Completed three DEV scenarios x four cyclic labels; not twelve independent cases')
    return 0

if __name__=='__main__':raise SystemExit(main())
