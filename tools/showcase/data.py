"""Extract a bounded display projection of preserved records; no model execution."""
from __future__ import annotations
import math
import re
from collections import Counter
from pathlib import Path
from common import C6, C7, SUP, checked_read, require, source_manifest

VALIDITY = ('full_attention_outputs_finite', 'full_block_outputs_finite',
            'full_final_hidden_finite', 'full_logits_finite', 'inputs_unchanged',
            'same_layer13_qkv', 'routing_valid', 'dtype_layout_valid',
            'answer_interface_valid', 'native_B_validated', 'effective_backend_verified')

def index(rows: list, keys: tuple) -> dict:
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(key not in result, 'Duplicate record key: '+str(key))
        result[key] = row
    return result

def valid6(row: dict) -> None:
    require(all(row.get('validity', {}).get(k) is True for k in VALIDITY), 'Incomplete full-output validity')

def score(s: list, gold: int, nll: float, brier: float, margin: float,
          mass: float, prediction: int) -> dict:
    require(len(s) == 4 and all(math.isfinite(x) for x in s), 'Invalid option logits')
    require(type(gold) is int and 0 <= gold < 4, 'Invalid gold label')
    top = [i for i, v in enumerate(s) if v == max(s)]
    require(prediction == top[0], 'Original first-index argmax mismatch')
    normalizer = max(s) + math.log(sum(math.exp(x-max(s)) for x in s))
    q = [math.exp(x-normalizer) for x in s]
    require(math.isclose(nll, normalizer-s[gold], abs_tol=1e-12), 'NLL mismatch')
    require(math.isclose(brier, sum((v-(i == gold))**2 for i,v in enumerate(q)), abs_tol=1e-12), 'Brier mismatch')
    require(math.isclose(margin, s[gold]-max(v for i,v in enumerate(s) if i != gold), abs_tol=1e-12), 'Margin mismatch')
    require(math.isfinite(mass) and 0 < mass <= 1+1e-12, 'Invalid label mass')
    return dict(logits=s, q=q, gold=gold, prediction=prediction, correct=prediction == gold,
                top=top, nll=nll, brier=brier, margin=margin, label_mass=mass,
                gap=sorted(s, reverse=True)[0]-sorted(s, reverse=True)[1])

def original6(s):
    return score(s['option_logits'], s['gold_index'], s['choice_nll'], s['choice_brier'],
                 s['gold_margin'], s['label_mass'], s['prediction'])

def shadow6(s):
    return score(s['s'], s['gold'], s['nll'], s['brier'], s['margin'], s['label_mass'], s['prediction'])

def original7(s):
    return score(s['option_logits'], s['gold'], s['nll'], s['brier'], s['margin'], s['label_mass'], s['prediction'])

def transition(b: dict, c: dict) -> dict:
    require(b['gold'] == c['gold'], 'Gold mismatch across paired arms')
    cell = ('both_correct' if c['correct'] else 'regression') if b['correct'] else ('gain' if c['correct'] else 'both_wrong')
    flip = b['prediction'] != c['prediction']
    return dict(cell=cell, flip=flip, wrong_to_wrong=flip and cell == 'both_wrong',
                delta_nll=c['nll']-b['nll'], delta_margin=c['margin']-b['margin'])

def excerpt(prompt: str) -> str:
    # Retain instructions, program/facts, question and options; omit only marked filler.
    return re.sub(r'<irrelevant_notes>.*?</irrelevant_notes>', '[irrelevant filler omitted]', prompt, flags=re.S)

def source_url(revision: str, path: str) -> str:
    return f'https://github.com/Munsik-Kim/inference-lab/blob/{revision}/{path}'

def load(root: Path) -> dict:
    manifest = source_manifest(root)
    rev = manifest['evidence_revision']
    get = lambda path: checked_read(root, path, manifest)
    records7 = []
    prompts7 = index(get(C7+'/inputs/HELD_OUT.json'), ('id',))
    local = index(get(C7+'/results/derived/pairs.json'), ('id','budget'))
    summary7 = get(C7+'/results/derived/summary.json')
    selection = get(C7+'/results/raw/selection.json')['selections']
    require(len(prompts7) == 192 and len(local) == 384, 'Case007 coverage')
    for (item_id,), prompt in sorted(prompts7.items()):
        raw = {}
        for method in ('B', 'INDEPENDENT_4', 'PAIRWISE_4', 'INDEPENDENT_8', 'PAIRWISE_8'):
            path = C7+f'/results/raw/heldout/{item_id}--{method}.json'
            r = get(path)
            require(r['evidence_kind'] == 'gpu_measurement' and r['valid'] is True
                    and r['full_blocks_checked'] == 28 and r['full_final_norm_checked'] is True
                    and r['full_logits_finite'] is True, 'Case007 invalid/fixture record')
            require(r['id'] == item_id and r['method'] == method and r['task'] == prompt['task']
                    and r['token_hash'] == prompt['token_hash'], 'Case007 mismatched join')
            raw[method] = r
        b = original7(raw['B']['score'])
        for budget in (4,8):
            pair = local[(item_id, budget)]
            for method in ('INDEPENDENT', 'PAIRWISE'):
                arm = f'{method}_{budget}'
                r = raw[arm]
                c = original7(r['score'])
                t = transition(b,c)
                require(all(t[k] == r['transition'][k] for k in ('cell','flip','wrong_to_wrong')), 'Transition mismatch')
                records7.append(dict(id=item_id, set='HELD_OUT', task=prompt['task'], length=512,
                    budget=str(budget), arm=method, readout='H_NATIVE', B=b, candidate=c, **t,
                    local_error=pair[method.lower()+'_error'], local_definition='MLP output relative Frobenius error, 32 fixed input positions',
                    full_kl=r['full_vocab_kl_B_to_candidate'], selected_groups=selection[arm]['removed'],
                    source=source_url(rev,C7+f'/results/raw/heldout/{item_id}--{arm}.json'),
                    source_path=C7+f'/results/raw/heldout/{item_id}--{arm}.json', token_hash=prompt['token_hash'],
                    validity='RECORDED_FULL_OUTPUT_VALID', evidence_kind='ORIGINAL_GPU_MEASUREMENT'))
    # Verify displayed transitions against retained aggregate, separately per method/budget.
    for arm in selection:
        method,budget = arm.rsplit('_',1)
        rows = [r for r in records7 if r['arm'] == method and r['budget'] == budget]
        expected = summary7['model'][arm]
        require(sum(r['flip'] for r in rows) == expected['choice_flips']
                and sum(r['candidate']['correct'] for r in rows) == expected['correct'], 'Case007 aggregate mismatch')
    prompts6 = index(get(SUP+'/inputs/core.json'), ('item_id',))
    original = index([r for r in get(C6+'/results/study/pairs.json')
                      if r['length'] == 4096 and r['split'] in ('standard','boundary_pool')], ('item_id','arm'))
    pairs = get(SUP+'/results/derived/pairs.json')
    pairmap = index(pairs, ('item_id','arm','readout'))
    require(len(prompts6) == 238 and len(original) == 476 and len(pairmap) == 952, 'Case006 coverage')
    native = index(get(SUP+'/results/raw/native.json'), ('item_id','arm'))
    shadow = index(get(SUP+'/results/raw/shadow.json'), ('item_id','arm'))
    require(len(native) == len(shadow) == 714, 'Readout raw coverage')
    for k,r in native.items():
        valid6(r)
        sr = shadow[k]
        valid6(sr)
        require(all(sr['controls'].get(v) is True for v in ('full_native_finite','full_shadow_finite','full_hidden_finite','C1_cast_only_exact','flags_restored')), 'Readout control failure')
        require(r['token_hash'] == sr['token_hash'] == prompts6[(r['item_id'],)]['token_hash'], 'Readout token mismatch')
    records6 = []
    for (item_id,arm,view), p in sorted(pairmap.items()):
        old = original[(item_id,arm)]
        valid6(old)
        prompt = prompts6[(item_id,)]
        require(old['token_hash'] == prompt['token_hash'], 'Original token mismatch')
        if view == 'H_NATIVE':
            b,c = original6(old['B_score']), original6(old['candidate_score'])
            require(b == shadow6(p['B']) and c == shadow6(p['candidate']), 'Native supplement changed scores')
        else:
            require(view == 'H_FP32', 'Unknown readout')
            b,c = shadow6(p['B']), shadow6(p['candidate'])
            require(b['logits'] == shadow[(item_id,'B')]['option_logits']
                    and c['logits'] == shadow[(item_id,arm)]['option_logits'], 'Shadow raw mismatch')
        t = transition(b,c)
        require(all(t[k] == p[k] for k in ('cell','flip','wrong_to_wrong')), 'Case006 transition mismatch')
        records6.append(dict(id=item_id, set=p['split'], task=p['task'], length=4096,
            budget='NA', arm=arm, readout=view, B=b, candidate=c, **t,
            original_B_top=original6(old['B_score'])['top'],
            local_error=old['native_pair_local']['relative_error'],
            local_reference_error=old['local_reference_error'],
            local_definition='attention candidate versus native BF16, pooled 32 queries; unchanged by shadow head',
            full_kl=old['full_vocab_kl'] if view == 'H_NATIVE' else shadow[(item_id,arm)]['full_vocab_kl_B_to_candidate'],
            source=source_url(rev,C6+'/results/study/pairs.json' if view == 'H_NATIVE' else SUP+'/results/derived/pairs.json'),
            token_hash=prompt['token_hash'], validity='RECORDED_FULL_OUTPUT_VALID',
            evidence_kind='ORIGINAL_GPU_MEASUREMENT' if view == 'H_NATIVE' else 'POST_HOC_SAME_INPUT_READOUT_DIAGNOSTIC'))
    summary6 = get(C6+'/results/study/summary.json')
    supplement = get(SUP+'/results/derived/summary.json')
    for split in ('standard','boundary_pool'):
        for arm in ('A_PUBLIC','V4'):
            rows = [r for r in records6 if r['set']==split and r['arm']==arm and r['readout']=='H_NATIVE']
            require(sum(r['flip'] for r in rows) == summary6['groups'][split+'/L4096']['arms'][arm]['outcomes']['flips']['numerator'], 'Case006 summary mismatch')
    def prompt_projection6(p):
        return {'text':excerpt(p['prompt']), 'source':source_url(rev,SUP+'/inputs/core.json'),
                'prompt_hash':p['prompt_hash'], 'token_hash':p['token_hash']}
    def prompt_projection7(p):
        return {'text':p['core']+'\n'+p['question']+'\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(p['options'])),
                'source':source_url(rev,C7+'/inputs/HELD_OUT.json'), 'prompt_hash':p['text_hash'], 'token_hash':p['token_hash']}
    # Per-record field provenance: reconstruction and full KL do not share a source in every view.
    for r in records7:
        main=r['source_path'];local_path=C7+'/results/derived/pairs.json'
        r['field_sources']={k:{'url':source_url(rev,path),'sha256':manifest['files'][path]} for k,path in
                            [('scores',main),('local_error',local_path),('full_kl',main)]}
    for r in records6:
        main=C6+'/results/study/pairs.json' if r['readout']=='H_NATIVE' else SUP+'/results/derived/pairs.json'
        kl_path=C6+'/results/study/pairs.json' if r['readout']=='H_NATIVE' else SUP+'/results/raw/shadow.json'
        r['field_sources']={k:{'url':source_url(rev,path),'sha256':manifest['files'][path]} for k,path in
                            [('scores',main),('local_error',C6+'/results/study/pairs.json'),('full_kl',kl_path)]}
    return {'case007':dict(case='007', revision=rev, records=records7,
                          prompts={k[0]:prompt_projection7(v) for k,v in prompts7.items()},
                          summary=summary7, selections=selection, n_independent=192),
            'case006':dict(case='006', revision=rev, records=records6,
                          prompts={k[0]:prompt_projection6(v) for k,v in prompts6.items()},
                          summary=summary6, supplement_summary=supplement,
                          n_independent={'standard':192,'boundary_pool':46})}
