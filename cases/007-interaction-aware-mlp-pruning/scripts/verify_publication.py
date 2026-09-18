"""CPU-only publication checks anchored to the reviewed v2 inventory.

Historical scientific files and current package coverage are checked separately.
This validates retained scalars and editorial tables, not new GPU measurements.
"""
import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path, PurePosixPath

ARCHIVE_SHA = '811c392b7efd36c4ed186a5766a396b7ce70adcb68363cde4342f9da6fbcfa53'
ORIGINAL_SUMS_SHA = '979f2f234c6e6ba249817b3321ee09320b5511bb57c679b2c735b99bb9aaaff7'
EDITABLE = {'README.md', 'README.ko.md', 'ANALYSIS.md', 'PORTFOLIO.md', 'REPRODUCTION.md'}
ADDED = {'provenance/publication_snapshot.json', 'PUBLICATION_SHA256SUMS',
         'scripts/verify_publication.py', 'scripts/package_publication.py',
         'publication_tests/test_publication.py'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def unique_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'Duplicate JSON key')
        out[key] = value
    return out

def read(path):
    return json.loads(path.read_text(), object_pairs_hook=unique_object)

def safe_name(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and str(p) == name
            and not any(x in name for x in ['\\', ':', '\n', '\r', '\x00'])
            and not any(x in p.parts for x in ['.', '..']), 'Unsafe path: '+name)
    return name

def manifest(text):
    out = {}
    for line in text.splitlines():
        require('  ' in line, 'Malformed checksum line')
        digest, name = line.split('  ', 1)
        safe_name(name)
        require(bool(re.fullmatch('[0-9a-f]{64}', digest)), 'Malformed checksum')
        require(name not in out, 'Duplicate checksum member')
        out[name] = digest
    return out

def file_map(case):
    out = {}
    for path in case.rglob('*'):
        require(not path.is_symlink(), 'Symlink in public case')
        if path.is_file():
            name = safe_name(path.relative_to(case).as_posix())
            out[name] = sha(path)
    return out

def original_inventory(case):
    require(sha(case/'SHA256SUMS') == ORIGINAL_SUMS_SHA,
            'Historical manifest differs from reviewed v2')
    out = manifest((case/'SHA256SUMS').read_text())
    require('SHA256SUMS' not in out, 'Historical self-reference')
    out['SHA256SUMS'] = ORIGINAL_SUMS_SHA
    require(len(out) == 1282, 'Wrong reviewed inventory')
    return out

def verify_maps(original, actual, edits, added, current=None):
    for name in set(original) | set(actual) | set(edits) | set(added):
        safe_name(name)
    require(set(edits) <= EDITABLE, 'Unapproved editorial path')
    require(len(added) == len(set(added)) and set(added) == ADDED,
            'Added-file allowlist mismatch')
    require(set(actual) == set(original) | set(added), 'Missing/unapproved public files')
    for name, digest in original.items():
        if name in edits:
            e = edits[name]
            require(e.get('reviewed_sha256') == digest and
                    e.get('published_sha256') == actual[name] and
                    bool(e.get('reason', '').strip()), 'Unrecorded editorial change: '+name)
        else:
            require(actual[name] == digest, 'Protected original changed: '+name)
    if current is not None:
        require('PUBLICATION_SHA256SUMS' not in current, 'Current manifest self-reference')
        require(set(current) == set(actual)-{'PUBLICATION_SHA256SUMS'},
                'Current manifest coverage mismatch')
        require(all(actual[n] == h for n,h in current.items()), 'Current checksum mismatch')

def close(a, b):
    require(math.isfinite(a) and math.isfinite(b) and
            math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11), 'Scalar mismatch')

def nll(score):
    values = score['option_logits']; m = max(values)
    return m + math.log(math.fsum(math.exp(v-m) for v in values))-values[score['gold']]

def editorial_values(case):
    """Recompute random means and score transitions from original retained records."""
    raw = case/'results/raw'; spec = read(case/'configs/protocol.json')
    summary = read(case/'results/derived/summary.json')
    selections = read(raw/'selection.json')['selections']
    inputs = read(case/'inputs/HELD_OUT.json')
    ids = {r['id']:r for r in inputs}
    require(len(inputs) == len(ids) == 192, 'Held-out input coverage')
    methods = {m:s['removed'] for m,s in selections.items()}
    for budget in [4,8]:
        sets = spec['random_sets'][str(budget)]
        require(len(sets) == len({tuple(s) for s in sets}) == 20, 'Random set coverage')
        for i, removed in enumerate(sets):
            require(len(removed) == budget, 'Random deletion budget')
            methods[f'RANDOM_{budget}_{i:02d}'] = removed
    means = {m:[] for m in methods}; records = {}
    for f in sorted((raw/'heldout').glob('c007-*.json')):
        r = read(f); key = (r['id'],r['method'])
        require(key not in records, 'Duplicate model cell')
        require(r['id'] in ids and r['token_hash'] == ids[r['id']]['token_hash'], 'Input mismatch')
        require(r.get('evidence_kind') == 'gpu_measurement' and r.get('valid') is True
                and r.get('full_blocks_checked') == 28 and r.get('full_final_norm_checked') is True
                and r.get('full_logits_finite') is True, 'Missing full-output validity')
        records[key] = r
        if r['method'] == 'B':
            local = {u['method']:u for u in r['local']}
            require(len(local) == len(r['local']) == 44 and set(local) == set(methods),
                    'Local comparison coverage')
            for m,u in local.items():
                require(u['removed'] == methods[m], 'Frozen group/index mismatch')
                require(u['r2'] > 0 and u['n'] > 0 and math.sqrt(u['r2']/u['n']) > 1e-6,
                        'Undefined reference norm')
                value = math.sqrt(u['e2']/u['r2']); close(value,u['relative_error'])
                means[m].append(value)
    require(set(records) == {(i,m) for i in ids for m in ['B',*selections]},
            'Model cell coverage')
    for m,values in means.items():
        require(len(values) == 192, 'Local prompt coverage')
        close(statistics.mean(values),summary['local']['HELD_OUT'][m]['relative_error']['mean'])
    random = {}
    for b in [4,8]:
        rv = [statistics.mean(means[f'RANDOM_{b}_{i:02d}']) for i in range(20)]
        im = statistics.mean(means[f'INDEPENDENT_{b}']); pm = statistics.mean(means[f'PAIRWISE_{b}'])
        counts = [sum(v < im for v in rv),sum(v < pm for v in rv)]
        random[str(b)] = {'min':min(rv),'median':statistics.median(rv),'max':max(rv),
                          'independent':im,'pairwise':pm,'beating_selected':counts}
    quality = {}; bc = sum(records[i,'B']['score']['correct'] for i in ids)
    for m in ['INDEPENDENT_4','PAIRWISE_4']:
        out = dict(local_mean=statistics.mean(means[m]), flips=0, regressions=0,
                   gains=0, wrong_to_wrong=0, correct=0, kl=[], nll_delta=[], retrieval_flips=0)
        for i in ids:
            b=records[i,'B'];r=records[i,m];bs=b['score'];cs=r['score']
            require(b['mlp_input_hash'] == r['mlp_input_hash'] and bs['gold'] == cs['gold'],
                    'Pair boundary/gold mismatch')
            bp=max(range(4),key=lambda j:bs['option_logits'][j])
            cp=max(range(4),key=lambda j:cs['option_logits'][j])
            bok=bp==bs['gold'];cok=cp==cs['gold'];flip=bp!=cp
            out['flips']+=flip;out['regressions']+=bok and not cok
            out['gains']+=not bok and cok;out['wrong_to_wrong']+=not bok and not cok and flip
            out['correct']+=cok
            out['retrieval_flips']+=flip and r['task']=='RETRIEVAL'
            out['kl'].append(r['full_vocab_kl_B_to_candidate'])
            out['nll_delta'].append(nll(cs)-nll(bs))
        out['kl']=statistics.mean(out['kl']);out['nll_delta']=statistics.mean(out['nll_delta'])
        target=summary['model'][m]
        for actual,expected in [(out['kl'],target['mean_full_vocab_KL']),
                                (out['nll_delta'],target['delta_nll']['mean']),
                                (out['correct'],target['correct']),
                                (out['flips'],target['choice_flips']),
                                (out['regressions'],target['transitions']['regression']),
                                (out['gains'],target['transitions']['gain']),
                                (out['wrong_to_wrong'],target['wrong_to_wrong'])]:close(actual,expected)
        quality[m]=out
    return {'random':random,'quality':quality,'n':192,'B_correct':bc,'B_wrong':192-bc,
            'full_KL_scope':'Aggregate retained full-vector measurements only; not reconstructed from options'}

def table(values, kind, lang):
    ko = lang == 'ko'
    if kind == 'random':
        labels = (['평균 미관측 국소 오차','고정 random 20개 중 최솟값','집합별 평균의 중앙값','고정 random 20개 중 최댓값',
                   'INDEPENDENT','PAIRWISE','선택법보다 오차가 낮은 random 집합'] if ko else
                  ['Mean held-out local error','Minimum among 20 frozen random sets','Median of the 20 set means',
                   'Maximum among 20 frozen random sets','INDEPENDENT','PAIRWISE','Random sets beating either selection'])
        lines=[f'| {labels[0]} | 25% deletion | 50% deletion |' if not ko else f'| {labels[0]} | 25% 삭제 | 50% 삭제 |','|---|---|---|']
        for label,key in zip(labels[1:6],['min','median','max','independent','pairwise']):
            lines.append('| '+label+' | '+' | '.join(f"{100*values['random'][b][key]:.4f}%" for b in ['4','8'])+' |')
        counts=[values['random'][b]['beating_selected'] for b in ['4','8']]
        require(all(x[0] == x[1] for x in counts),'Editorial combined count ambiguous')
        lines.append('| '+labels[6]+' | '+' | '.join(f'{x[0]}/20' for x in counts)+' |')
    else:
        require(kind == 'quality','Unknown table kind')
        labels=(['지표 · held-out 192개','평균 국소 상대 Frobenius 오차','평균 전체 어휘 KL(B ∥ 후보), nats',
                 'B 대비 선택 변경','B 정답 → 후보 오답 (회귀)','B 오답 → 후보 정답 (개선)',
                 '오답 → 다른 오답','정답 수','평균 Δ정답 선택지 NLL(후보−B), nats'] if ko else
                ['Metric · 192 held-out prompts','Mean local relative Frobenius error','Mean full-vocabulary KL(B ∥ candidate), nats',
                 'Choice flips versus B','B correct → candidate wrong (regression)','B wrong → candidate correct (gain)',
                 'Wrong → different wrong','Correct choices','Mean Δgold-choice NLL (candidate−B), nats'])
        lines=[f'| {labels[0]} | INDEPENDENT_4 | PAIRWISE_4 |','|---|---|---|']
        keys=['local_mean','kl','flips','regressions','gains','wrong_to_wrong','correct','nll_delta']
        for label,key in zip(labels[1:],keys):
            cells=[]
            for m in ['INDEPENDENT_4','PAIRWISE_4']:
                v=values['quality'][m][key]
                if key=='local_mean':cell=f'{100*v:.4f}%'
                elif key=='kl':cell=f'{v:.6f}'
                elif key=='nll_delta':cell=f'{v:+.6f}'
                else:
                    denominator=values['B_correct'] if key=='regressions' else values['B_wrong'] if key=='gains' else values['n']
                    cell=f'{v}/{denominator}'
                cells.append(cell)
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
    return '\n'.join(lines)

def validate_tables(text, values, lang):
    for kind in ['random','quality']:
        start=f'<!-- publication-table: {kind} -->';end=f'<!-- /publication-table: {kind} -->'
        require(text.count(start)==text.count(end)==1,'Missing/duplicate editorial table')
        actual=text.split(start,1)[1].split(end,1)[0].strip()
        require(actual==table(values,kind,lang),'Editorial mean/sign/denominator/table mismatch: '+kind)

def verify(case):
    original=original_inventory(case);snapshot=read(case/'provenance/publication_snapshot.json')
    require(snapshot['original_archive']['sha256']==ARCHIVE_SHA and snapshot['original_files']==original,
            'Original snapshot identity mismatch')
    actual=file_map(case);current=manifest((case/'PUBLICATION_SHA256SUMS').read_text())
    verify_maps(original,actual,snapshot['edited_documents'],snapshot['publication_added_files'],current)
    values=editorial_values(case)
    for name,lang in [('README.md','en'),('README.ko.md','ko')]:validate_tables((case/name).read_text(),values,lang)
    summary=read(case/'results/derived/summary.json')
    require(summary['status']=='COMPLETED_NO_CLEAR_TRANSFER' and summary['deployment']=='NOT_ASSESSED',
            'Scientific/deployment status mismatch')
    return {'status':'PASS','reviewed_files':len(original),'current_files':len(actual),
            'protected_original_files':len(original)-len(snapshot['edited_documents']),
            'edited_documents':sorted(snapshot['edited_documents']),
            'random_sets_per_budget':20,'heldout_prompts':192,'model_cells':960,
            'editorial_values':values,'new_gpu_runs':0,
            'scope':'Historical-byte preservation, package coverage and retained scalar/editorial consistency; not GPU replication'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists() and not a.output.resolve().is_relative_to(a.case.resolve()),'New external output required')
    result=verify(a.case);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='editorial_values'}))

if __name__=='__main__':main()
