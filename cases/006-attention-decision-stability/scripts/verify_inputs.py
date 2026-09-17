"""Audit retained exact inputs, separate gold and frozen manifests without a model."""
import argparse,json,math,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import load,digest,file_hash,write_new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--evaluation-inputs',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,help='Optional cached tokenizer only; never loads weights')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    gold=load(a.case/'inputs/gold.json')+load(a.evaluation_inputs/'gold.json')
    if len(gold)!=486 or len({x['base_id'] for x in gold})!=486 or len({x['base_facts_hash'] for x in gold})!=486:
        raise ValueError('Base-scenario overlap/count failure')
    index={x['base_id']:x for x in gold}
    for r in gold:
        if r['task']=='CODE':
            f=r['facts'];answer=str((f['initial']+f['step']*sum(range(f['count'])))%f['modulus'])
        else:
            entities=re.findall(r'unit-[0-9a-f]+-[0-9]+',r['question'])
            facts={f['entity']:f for f in r['facts']}
            if r['task']=='RETRIEVAL':answer=str(facts[entities[0]]['value'])
            else:answer=max(entities,key=lambda e:facts[e]['value'])
        if answer!=r['gold_value'] or r['options']['ABCD'.index(r['gold'])]!=answer or len(set(r['options']))!=4:
            raise ValueError('Independent fact/gold mismatch')
    inv=load(a.evaluation_inputs/'input_inventory.json');inputs=load(a.case/'inputs/prompts.json')
    for name,sha in inv['files'].items():
        if file_hash(a.evaluation_inputs/name)!=sha:raise ValueError('Frozen input file mismatch')
        inputs.extend(load(a.evaluation_inputs/name))
    if len(inputs)!=606 or len({r['item_id'] for r in inputs})!=606:raise ValueError('Input identity/count mismatch')
    tokenizer=None
    if a.snapshot:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True,trust_remote_code=False)
    frozen=load(a.case/'configs/eval_manifest_freeze_v1.json')
    for r in inputs:
        g=index[r['base_id']]
        if r['token_hash']!=digest(r['token_ids']) or len(r['token_ids'])!=r['length']:raise ValueError('Input token identity')
        if 'gold' in r or r['future_answer_tokens_present']:raise ValueError('Answer metadata in request')
        if not all(t in r['prompt'] for t in (g['core'],g['question'])):raise ValueError('Required fact/question truncated')
        for label,option in zip('ABCD',g['options']):
            if f'{label}. {option}' not in r['prompt']:raise ValueError('Option truncated')
        if r['split'] in ('standard','boundary_pool') and frozen['input_hashes'].get(r['item_id'])!=r['token_hash']:
            raise ValueError('Evaluation manifest mismatch')
        if tokenizer:
            if tokenizer.encode(r['prompt'],add_special_tokens=False)!=r['token_ids']:raise ValueError('Tokenizer replay mismatch')
            if not r.get('secondary'):
                for label,token in zip('ABCD',[32,33,34,35]):
                    if tokenizer.encode(r['prompt']+label,add_special_tokens=False)!=r['token_ids']+[token]:
                        raise ValueError('Unstable one-token answer interface')
    for name,sha in frozen['code_hashes'].items():
        if file_hash(a.case/name)!=sha:raise ValueError('Frozen measurement code changed')
    out={'status':'PASS','independent_base_scenarios':486,'exact_prompts_including_dependent_subsets':606,
         'independent_fact_oracles':'PASS','split_overlap':'NONE','required_fact_and_option_retention':'PASS',
         'frozen_code_files':len(frozen['code_hashes']),'tokenizer_replay':'PASS' if tokenizer else 'NOT_RUN',
         'scope':'CPU facts/tokenization/hash audit; no model inference'}
    write_new(a.output,out);print(json.dumps(out))


if __name__=='__main__':main()
