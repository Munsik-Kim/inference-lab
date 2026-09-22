"""Export existing public scalar records; not a new model evaluation."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.case/'results/raw/Q/model_records.json';rows=json.loads(source.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    counts={}
    for arm in ('Q-BF16','Q-W4'):
        output=[]
        for r in rows:
            if r['arm']!=arm:continue
            s=r['score'];labels=list('ABCD')
            output.append(dict(schema_version=1,sample_id=r['id'],task='case008-'+r['task'],task_version='frozen-20260919',prompt_hash=r['token_hash'],gold=[labels[s['gold']]],gold_definition='case008 independent synthetic oracle',model_id='Qwen/Qwen3-4B-Instruct-2507',artifact_id=arm,output_type='choice',answer=labels[s['prediction']],correct=s['correct'],evidence_kind='historical',distribution=dict(kind='choice',labels=labels,probabilities=s['choice_probabilities'],tokenizer_id='Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554',prefix_hash=r['token_hash'],dtype='float64_analysis_of_native_logits',normalization='probabilities_sum_to_one')))
        (a.output/(arm+'.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in output));counts[arm]=len(output)
    (a.output/'source.json').write_text(json.dumps({'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'relative_source':'results/raw/Q/model_records.json','counts':counts,'evidence_kind':'historical','scope':'four-choice scalar reanalysis; does not reconstruct full-vocabulary KL'},indent=2)+'\n')

if __name__=='__main__':main()
