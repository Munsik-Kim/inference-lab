"""Adapt public official-task scalars into the installed CLI contract.

Each filter/metric gets separate files. Different views are not extra scenarios.
No benchmark prose, generated rationale, tokenizer or GPU dependency is needed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    artifacts=json.loads((a.case/'provenance/artifacts.json').read_text())
    identities={arm:hashlib.sha256(json.dumps({k:v['sha256'] for k,v in artifacts.items() if k.startswith(arm+'/')},sort_keys=True).encode()).hexdigest() for arm in ['BF16','W4']}
    groups={}
    for line in (a.case/'results/raw/quality_scalars.jsonl').read_text().splitlines():
        x=json.loads(line)
        if x['task']=='wikitext':continue  # PPL has its official summed denominator.
        benchmark='mmlu' if x['task'].startswith('mmlu_') else x['task']
        metric='exact_match' if benchmark=='gsm8k' else 'acc'
        base={'schema_version':1,'sample_id':x['id'],'task':x['task'],
          'task_version':'lm-eval-d6de81643928/'+x['filter']+'/'+metric,
          'prompt_hash':x['prompt_hash'],'gold_definition':'official-'+metric,
          'model_id':'Qwen3-4B-Instruct-2507','artifact_id':identities[x['arm']],
          'output_type':'extracted_answer' if benchmark=='gsm8k' else 'choice',
          'answer':x['answer_hash'] if benchmark=='gsm8k' else str(x['prediction']),
          'gold':[x['gold_hash']] if benchmark=='gsm8k' else [str(x['gold'])],
          'correct':x['correct'],'evidence_kind':'measurement','process_exit_code':x['process_exit_code']}
        if 'scores' in x:
            m=max(x['scores']);v=[math.exp(z-m) for z in x['scores']];total=math.fsum(v)
            base['distribution']={'kind':'choice','labels':[str(i) for i in range(len(v))],
              'probabilities':[z/total for z in v],'tokenizer_id':'Qwen-cdbee75f17c01a7cc42f958dc650907174af0554',
              'prefix_hash':x['prompt_hash'],'dtype':'FP64-softmax-of-continuation-loglikelihoods','normalization':'probabilities_sum_to_one'}
        key=f'{benchmark}-{x["filter"]}-{metric}-{x["arm"]}'
        groups.setdefault(key,[]).append(base)
        if 'correct_norm' in x:
            norm={k:v for k,v in base.items() if k!='distribution'}
            norm.update(task_version='lm-eval-d6de81643928/'+x['filter']+'/acc_norm',gold_definition='official-acc_norm',answer=str(x['prediction_norm']),correct=x['correct_norm'])
            groups.setdefault(f'{benchmark}-{x["filter"]}-acc_norm-{x["arm"]}',[]).append(norm)
    a.output.mkdir(parents=True,exist_ok=False)
    for name,rows in groups.items():(a.output/(name+'.jsonl')).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    (a.output/'scope.json').write_text(json.dumps({'files':{k:{'records':len(v),'process_exit_codes':sorted({r['process_exit_code'] for r in v})} for k,v in groups.items()},'execution_scope':'CLI recalculates stored metrics; nonzero process exits remain failures even where full calculations were retained. See source quality summary.','GSM_answers':'SHA256 of official filtered strings; correctness retained from official parser. Different filters are repeated views.','WikiText':'excluded from choice CLI; official scalar audit checks PPL'},indent=2)+'\n')


if __name__=='__main__':main()
