"""Post-hoc descriptive calculations from retained public scalars; no new inference."""
import argparse,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path

def analyze(case):
    names=['results/raw/quality_scalars.jsonl','results/derived/quality_summary.json','results/derived/serving_summary.json']
    sources={n:hashlib.sha256((case/n).read_bytes()).hexdigest() for n in names}
    records=[json.loads(x) for x in (case/names[0]).read_text().splitlines()]
    groups=defaultdict(dict)
    for r in records:
        k=(r['task'],r['filter'],r['id']);assert r['arm'] not in groups[k];groups[k][r['arm']]=r
    pairs=[]
    for k,d in sorted(groups.items()):
        assert set(d)=={'BF16','W4'},k
        b,c=d['BF16'],d['W4'];assert b['prompt_hash']==c['prompt_hash'] and b['doc_hash']==c['doc_hash']
        pairs.append((b,c))
    m=[(b,c) for b,c in pairs if b['task'].startswith('mmlu_')]
    bysubject=defaultdict(list)
    for b,c in m:bysubject[b['task']].append((b,c))
    subjects={s:{'n':len(p),'baseline_correct':sum(b['correct'] for b,c in p),'candidate_correct':sum(c['correct'] for b,c in p),'delta_accuracy':sum(c['correct']-b['correct'] for b,c in p)/len(p)} for s,p in sorted(bysubject.items())}
    labels={str(g):{'n':sum(b['gold']==g for b,c in m),'baseline_correct':sum(b['gold']==g and b['correct'] for b,c in m),'candidate_correct':sum(b['gold']==g and c['correct'] for b,c in m)} for g in range(4)}
    mm={'n_items':len(m),'n_subjects':len(subjects),'subjects_declining':sum(x['delta_accuracy']<0 for x in subjects.values()),'baseline_correct':sum(b['correct'] for b,c in m),'candidate_correct':sum(c['correct'] for b,c in m),'lost':sum(b['correct'] and not c['correct'] for b,c in m),'gained':sum(not b['correct'] and c['correct'] for b,c in m),'D_choice_counts':{a:sum(x['prediction']==3 for p in m for x in [p[0 if a=='BF16' else 1]]) for a in ['BF16','W4']},'gold_label_counts':labels,'subjects':subjects,'interpretation':'Choice-position shift is descriptive, not an identified quantization/tokenizer/runtime cause. Original item-weighted and subject-cluster intervals unchanged.'}
    wp=[(b,c) for b,c in pairs if b['task']=='wikitext'];wa={}
    for i,a in enumerate(['BF16','W4']):
        ll=math.fsum(p[i]['loglikelihood'] for p in wp);words=sum(p[i]['words'] for p in wp);bytes_=sum(p[i]['bytes'] for p in wp)
        wa[a]={'summed_loglikelihood':ll,'words':words,'bytes':bytes_,'word_PPL':math.exp(-ll/words),'byte_PPL':math.exp(-ll/bytes_)}
    wiki={'n_documents':len(wp),'lower_candidate_loglikelihood':sum(c['loglikelihood']<b['loglikelihood'] for b,c in wp),'arms':wa,'word_PPL_relative_increase':wa['W4']['word_PPL']/wa['BF16']['word_PPL']-1,'denominator_source':'Retained public fields; excluded text not reparsed.'}
    q=json.loads((case/names[1]).read_text());gsm=next(t for t in q['tasks'] if t['task']=='gsm8k');filters={}
    for a in ['BF16','W4']:
        v={}
        for r in records:
            if r['task']=='gsm8k' and r['arm']==a:v.setdefault(r['id'],{})[r['filter']]=r['correct']
        assert all(set(x)=={'strict-match','flexible-extract'} for x in v.values())
        budget=gsm['generation_budget'][a]
        filters[a]={'n':len(v),'strict_wrong_flexible_correct':sum(not x['strict-match'] and x['flexible-extract'] for x in v.values()),'strict_correct_flexible_wrong':sum(x['strict-match'] and not x['flexible-extract'] for x in v.values()),'mean_generated_tokens':budget['total_generated_tokens']/budget['n'],'token_cap_count':budget['at_1024_token_cap'],'generation_length_source':'Original public summary totals, not item-level rationales'}
    s=json.loads((case/names[2]).read_text());cells=s['cells'];curves={}
    for a in ['BF16','W4']:
        curves[a]={}
        for l,c in [(128,1),(1024,16),(1024,32)]:
            rows=[x for x in cells if x['execution']=='graph' and x['arm']==a and x['input_tokens']==l and x['client_concurrency']==c];assert len(rows)==3
            curves[a][f'L{l}-C{c}']={k:statistics.mean(x[k] for x in rows) for k in ['ttft_p50_ms','tpot_p50_ms','output_tokens_per_second']}
    long=[x for x in cells if x['execution']=='graph' and x['input_tokens']==1024 and x['client_concurrency']==32]
    return {'evidence_kind':'POST_HOC_SAME_RECORDED_SCALARS','sources':sources,'mmlu':mm,'wikitext':wiki,'gsm8k':{'filter_views':filters,'flexible_transition':gsm['groups']['gsm8k/flexible-extract'],'interpretation':'Two views of identical generations; length ratios are not GSM runtime speedups.'},'serving':{'rounds':3,'means_of_round_metrics':curves,'long_input_C32_cells':len(long),'preemptions':[x['preemptions'] for x in long],'total_preemptions':sum(x['preemptions'] for x in long),'original_paired_contrasts':s['contrasts'],'interpretation':'Graph/compile/fusion bundle. Round min/max bars and paired bootstrap intervals remain distinct. Client concurrency is not GPU batch.'}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f:json.dump(analyze(a.case),f,indent=2,allow_nan=False);f.write('\n')
if __name__=='__main__':main()
