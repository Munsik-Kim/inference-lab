"""Read-only Case008 summary projection; no model, fixture or inference execution."""
from html import escape
from pathlib import Path
from common import read, checked_read, require
from layout import C8_PATH, C8_REVISION, anchor, report8_url


def load_case008(root: Path) -> dict:
    manifest = read(root/'presentation/case008_sources.json')
    require(manifest['evidence_revision'] == C8_REVISION, 'Case008 revision mismatch')
    get = lambda name: checked_read(root, C8_PATH+'/'+name, manifest)
    summary = get('results/derived/summary.json')
    models = get('provenance/models.json')
    local = get('results/derived/local_tables.json')['splits']['heldout']
    tracks = {}
    for name in ('Q', 'R'):
        rows = get(f'inputs/{name}/heldout.json')
        require(len(rows) == len({r['id'] for r in rows}) == 192, 'Wrong scenario coverage')
        require(all(r['track'] == name and r['split'] == 'heldout' and r['length'] == len(r['token_ids']) for r in rows), 'Wrong input scope')
        tracks[name] = {'model':models[name]['model'], 'model_revision':models[name]['revision'],
                        'n_scenarios':len(rows), 'token_range':[min(r['length'] for r in rows),max(r['length'] for r in rows)]}
    q = tracks['Q']; q['arms'] = summary['tracks']['Q']['arms']
    q['weight_bytes'] = {'Q-BF16':sum(v['bytes'] for k,v in models['Q']['files'].items() if k.endswith('.safetensors')),
                         'Q-W4':get('results/raw/Q/build.json')['safetensors_bytes']}
    q['file_reduction_fraction'] = 1-q['weight_bytes']['Q-W4']/q['weight_bytes']['Q-BF16']
    q['allocator_peak_bytes'] = {}
    for arm in ('Q-BF16','Q-W4'):
        runtime = get(f'results/raw/Q/{arm}-runtime.json')['after']
        require(len(runtime) == 1 and runtime[0]['diagnostics']['invalid'] == 0, 'Unexpected runtime validity/scope')
        q['allocator_peak_bytes'][arm] = runtime[0]['max_allocated']
        require(q['arms'][arm]['n'] == 192, 'Wrong Q denominator')
    q['timing'] = summary['timing']['Q']
    r = tracks['R']; r['structures'] = {}
    for name in ('I25','P25','S50'):
        before = get(f'results/raw/R/artifacts/{name}.json')['weights']
        after = get(f'results/raw/R/artifacts/{name}-R.json')['weights']
        require(set(before) == set(after), 'Repair changed tensor inventory')
        require(all(all(before[k][field] == after[k][field] for field in ('shape','numel','bytes','dtype')) for k in before), 'Repair changed structure')
        changed = [k for k in before if before[k]['sha256'] != after[k]['sha256']]
        require(changed == ['model.layers.13.mlp.down_proj.weight'], 'Unexpected repaired weights')
        repair = summary['tracks']['R']['repair'][name]
        require(repair['n'] == local[name]['n_prompts'] == local[name+'-R']['n_prompts'] == 192, 'Wrong R denominator')
        require(local[name]['unit'] == local[name+'-R']['unit'] == 'fraction of native teacher Frobenius norm', 'Wrong local metric unit')
        r['structures'][name] = {'width':before['model.layers.13.mlp.gate_proj.weight']['shape'][0],
            'changed_weight':changed[0], 'same_structure':True,
            'mean_relative_before':local[name]['mean_relative'], 'mean_relative_after':local[name+'-R']['mean_relative'],
            'recovery':repair,
            'correct_before':summary['tracks']['R']['arms'][name]['correct'],
            'correct_after':summary['tracks']['R']['arms'][name+'-R']['correct']}
    return {'schema':1, 'evidence_kind':'RECORDED_STUDY_SUMMARY', 'deployment':summary['deployment'],
            'evidence_revision':C8_REVISION, 'sources':manifest['files'], 'tracks':tracks,
            'units':{'weight_bytes':'safetensors file bytes','allocator_peak_bytes':'Torch allocator peak, cache/initialization/execution included',
                'mean_relative':'mean prompt relative Frobenius norm error at 32 fixed positions',
                'recovery':'1 - pooled repaired error energy / pooled uncorrected error energy',
                'timing':'12 fixed inputs, 3 processes, 5 paired blocks of 3 calls; pointwise 95% CI'},
            'scope':'Separate models and tracks; never a pooled 384-scenario quality rate. No tensors or prompt tokens in this display.'}


def render_case008(root: Path, data: dict, lang: str) -> str:
    t = read(root/f'presentation/case008/{lang}.json')
    text = lambda key: escape(t[key])
    q,r = data['tracks']['Q'],data['tracks']['R']
    pct = lambda value: f'{100*value:.2f}%'
    h = '<p class="eyebrow">CASE 008 · RECORDED RESULTS</p><h1>'+text('title')+'</h1><p class="lede">'+text('intro')+'</p>'
    h += '<nav class="actions" aria-label="'+text('tracks')+'">'+anchor('#track-q',t['qtitle'])+anchor('#track-r',t['rtitle'])+anchor('#try-it',t['try'])+'</nav><div class="actions evaluation-jumps">'+anchor('#q-evaluation',t['qEvaluation'])+anchor('#r-evaluation',t['rEvaluation'])+'</div>'
    for name,d in [('Q',q),('R',r)]:
        isq=name=='Q'; key='q' if isq else 'r'
        h += '<section class="panel track-panel" id="track-'+key+'"><p class="eyebrow">TRACK '+name+'</p><h2>'+text(key+'title')+'</h2>'
        h += '<p class="scope">'+escape(d['model'])+' · '+str(d['n_scenarios'])+' '+text('scenarios')+' · '+str(d['token_range'][0])+'–'+str(d['token_range'][1])+' '+text('tokens')+' · RTX 5080</p>'
        h += '<p>'+text(key+'method')+'</p><ol class="artifact-flow">'+''.join('<li>'+escape(s)+'</li>' for s in t[key+'flow'])+'</ol>'
        if isq:
            h += '<div class="result-box"><p class="result-label">'+text('fileReduction')+'</p><p class="result-number" id="q-file-reduction">'+pct(q['file_reduction_fraction'])+'</p><p>'+text('qdefinition')+'</p></div>'
            h += '<div class="scroll"><table><thead><tr><th>'+text('metric')+'</th><th>Q-BF16</th><th>Q-W4</th></tr></thead><tbody>'
            for label,values in [(t['files'],[f'{q["weight_bytes"][a]:,}' for a in ('Q-BF16','Q-W4')]),(t['memory'],[f'{q["allocator_peak_bytes"][a]/2**30:.4f}' for a in ('Q-BF16','Q-W4')]),(t['correct'],[f'{q["arms"][a]["correct"]}/{q["arms"][a]["n"]}' for a in ('Q-BF16','Q-W4')])]:
                h += '<tr><th>'+escape(label)+'</th>'+''.join('<td>'+escape(v)+'</td>' for v in values)+'</tr>'
            h += '</tbody></table></div><p class="scope">'+text('memoryScope')+'</p>'
            w=q['arms']['Q-W4'];tim=q['timing']['fixed_8_token_request']['Q-W4']
            h += '<h3 id="q-evaluation">'+text('scores')+'</h3><p>'+text('qscore')+'</p><dl class="score-list">'
            for key,label in [('delta_full_gold_nll',t['fullNll']),('delta_choice_nll',t['choiceNll'])]:
                v=w[key];h += '<div><dt>'+escape(label)+'</dt><dd>'+f'{v["mean"]:+.6f} [{v["ci95"][0]:+.6f}, {v["ci95"][1]:+.6f}] nats'+'</dd></div>'
            h += '</dl><p class="scope">'+text('nllScope')+'</p><p id="q-timing">'+text('qTiming')+' '+f'{tim["speedup"]:.3f}× [{tim["ci95"][0]:.3f}, {tim["ci95"][1]:.3f}]'+'</p><p class="scope">'+text('timeScope')+'</p>'
        else:
            h += '<p>'+text('rnames')+'</p><div class="repair-grid">'
            for name,v in r['structures'].items():
                rec=v['recovery'];h+='<article class="repair-card"><h3>'+name+' → '+name+'-R</h3><p>'+str(v['width'])+' '+text('channels')+'</p><p class="result-label">'+text('relativeError')+'</p><p class="repair-values">'+pct(v['mean_relative_before'])+' → <strong>'+pct(v['mean_relative_after'])+'</strong></p>'
                h+='<p>'+text('energy')+' <strong>'+pct(rec['pooled_recovery'])+'</strong><br><span class="small">95% CI ['+f'{100*rec["ci95"][0]:.2f}, {100*rec["ci95"][1]:.2f}'+']%</span></p><p>'+text('improved')+f' {rec["improved_prompts"]}/{rec["n"]}'+'</p><p>'+text('correct')+f': {v["correct_before"]}/192 → {v["correct_after"]}/192'+'</p></article>'
            h += '</div><p class="scope">'+text('rScope')+'</p><h3 id="r-evaluation">'+text('rEvaluation')+'</h3><p>'+text('rConclusion')+'</p>'
        h += '<div class="actions">'+anchor(report8_url(lang)+'#'+('results' if isq else 'methods'),t['report'])+'</div></section>'
    h += '<section id="try-it" class="panel"><h2>'+text('try')+'</h2><p>'+text('fixture')+'</p><p>'+text('routes')+'</p><div class="actions">'+anchor('https://github.com/Munsik-Kim/inference-lab/blob/main/docs/'+lang+'/GETTING_STARTED.md#case008-paths',t['fourRoutes'])+anchor('../data/case008.json',t['download'])+'</div></section>'
    h += '<details><summary>'+text('sources')+'</summary><p>COMPLETED · '+escape(data['deployment'])+'</p><ul>'
    for path in data['sources']:
        h += '<li>'+anchor(f'https://github.com/Munsik-Kim/inference-lab/blob/{C8_REVISION}/{path}',path)+'</li>'
    return h+'</ul></details>'
