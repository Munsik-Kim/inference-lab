"""Build plots, tables and a network-free measured-data explorer in a NEW directory."""
from pathlib import Path
import argparse,json,base64,sys,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from readout_analysis import load,save,score

OPENING=('All 5 A_PUBLIC and 8 V4 native choice changes in the 192-scenario standard set involved an exact top-score tie in B or the candidate. '
'The same final hidden states read through a full-vocabulary FP32 output projection produced 3 changes for each candidate, with no exact four-option top ties. '
'This shadow readout also introduced new changes: it does not erase the native outcomes or establish improved model quality. '
'All original results remain unchanged; this is a post-hoc, same-input diagnostic.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.case.resolve()):p.error('NEW external output required')
    a.output.mkdir(parents=True);figdir=a.output/'figures';figdir.mkdir();s=load(a.analysis/'summary.json');pairs=load(a.analysis/'pairs.json');native=load(a.case/'results/raw/native.json');shadow=load(a.case/'results/raw/shadow.json')
    rows=native+shadow;inputs=load(a.case/'inputs/core.json');gold=load(a.case/'inputs/gold.json');g=s['groups']
    lines=['# Result tables','', 'POST-HOC SAME-INPUT READOUT DIAGNOSTIC. Original native results remain primary.','', '| Set | Readout | Arm | B correct | Candidate correct | Flips | Regression | Gain | Wrong-to-wrong | B / candidate top ties |','|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for split in ['standard','boundary_pool']:
        for view in ['H_NATIVE','H_FP32']:
            for arm in ['A_PUBLIC','V4']:
                x=g[f'{split}/{view}/{arm}/ALL'];n=x['n'];lines.append(f"| {split} | {view} | {arm} | {x['B_description']['correct']}/{n} | {x['candidate_description']['correct']}/{n} | {x['flips']} | {x['regression']} | {x['gain']} | {x['wrong_to_wrong_changes']} | {x['B_top_ties']} / {x['candidate_top_ties']} |")
    lines+=['','## Native tie-state partitions','','Each cell reports scenario / flip / regression / gain / wrong-to-wrong counts. The four cells exhaust each paired comparison.','', '| Set / arm | Unique → unique | Tied → unique | Unique → tied | Both tied |','|---|---|---|---|---|']
    for split in ['standard','boundary_pool']:
        for arm in ['A_PUBLIC','V4']:
            x=g[f'{split}/H_NATIVE/{arm}/ALL'];v=[]
            for st in ['unique/unique','tied/unique','unique/tied','tied/tied']:
                t=x['tie_strata'][st];v.append(' / '.join(str(t[f]) for f in ['n','flips','regression','gain','wrong_to_wrong_changes']))
            lines.append('| '+split+' / '+arm+' | '+' | '.join(v)+' |')
    lines+=['','## Task-specific counts and scores','','| Set / task | Readout / candidate | n | B correct | Candidate correct | Flips / regressions / gains | Mean Δ gold NLL (nats) | Mean Δ Brier | Mean Δ gold margin |','|---|---|---:|---:|---:|---|---:|---:|---:|']
    for split in ['standard','boundary_pool']:
        for task in ['RETRIEVAL','COMPARISON','CODE']:
            for view in ['H_NATIVE','H_FP32']:
                for arm in ['A_PUBLIC','V4']:
                    x=g[f'{split}/{view}/{arm}/{task}'];lines.append(f"| {split} / {task} | {view} / {arm} | {x['n']} | {x['B_description']['correct']} | {x['candidate_description']['correct']} | {x['flips']} / {x['regression']} / {x['gain']} | {x['delta_nll']:+.8f} | {x['delta_brier']:+.8f} | {x['delta_margin']:+.8f} |")
    lines+=['','## STANDARD task-balanced score changes','','5000 paired within-task scenario-cluster bootstrap draws; post-hoc pointwise descriptive intervals.','', '| Readout | Arm | Δ gold NLL [95% interval] | Δ Brier [95% interval] |','|---|---|---|---|']
    for view in ['H_NATIVE','H_FP32']:
        for arm in ['A_PUBLIC','V4']:
            ci=g[f'standard/{view}/{arm}/ALL']['posthoc_intervals'];vals=[]
            for f in ['delta_nll','delta_brier']:
                v=ci[f];vals.append(f"{v['task_balanced_mean']:+.8f} [{v['ci95'][0]:+.8f}, {v['ci95'][1]:+.8f}]")
            lines.append('| '+view+' | '+arm+' | '+' | '.join(vals)+' |')
    (a.output/'TABLES.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(7.5,4));x=np.arange(2)
    for j,view in enumerate(['H_NATIVE','H_FP32']):
        vals=[g[f'standard/{view}/{arm}/ALL']['flips'] for arm in ['A_PUBLIC','V4']];ax.bar(x+(j-.5)*.32,vals,.32,label=view)
    ax.set(xticks=x,xticklabels=['A_PUBLIC','V4'],ylabel='Choice changes versus B / 192 scenarios',ylim=(0,10),title='Same inputs; separate native and FP32 readouts');ax.legend();fig.tight_layout();fig.savefig(figdir/'01_readout_flips.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,arm in zip(axes,['A_PUBLIC','V4']):
        sub=[p for p in pairs if p['split']=='standard' and p['arm']==arm and p['readout']=='H_NATIVE'];sh={p['base_id']:p for p in pairs if p['split']=='standard' and p['arm']==arm and p['readout']=='H_FP32'}
        for task in ['RETRIEVAL','COMPARISON','CODE']:
            pp=[p for p in sub if p['task']==task];ax.scatter([p['delta_nll'] for p in pp],[sh[p['base_id']]['delta_nll'] for p in pp],s=15,alpha=.65,label=task)
        ax.axhline(0,color='grey',lw=.6);ax.axvline(0,color='grey',lw=.6);ax.set(title=arm,xlabel='Native paired gold-NLL change (nats)',ylabel='FP32 paired gold-NLL change (nats)')
    axes[1].legend(fontsize=8);fig.suptitle('All 192 STANDARD scenarios; post-hoc readout sensitivity');fig.tight_layout();fig.savefig(figdir/'02_score_readout.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    ix={(r['base_id'],r['arm']):r for r in native}
    for ax,arm in zip(axes,['A_PUBLIC','V4']):
        pp=[p for p in pairs if p['split']=='standard' and p['arm']==arm and p['readout']=='H_NATIVE']
        for task in ['RETRIEVAL','COMPARISON','CODE']:
            q=[p for p in pp if p['task']==task];ax.scatter([100*ix[(p['base_id'],arm)]['native_pair_local']['relative_error'] for p in q],[abs(p['delta_margin']) for p in q],s=15,alpha=.65,label=task)
        ax.set(title=arm,xlabel='Pooled candidate-vs-B local error (%)',ylabel='Absolute gold-margin change (logit units)')
    axes[1].legend(fontsize=8);fig.suptitle('Existing scalar evidence; 32 sampled queries × 16 heads per item');fig.tight_layout();fig.savefig(figdir/'03_local_margin.png',dpi=160);plt.close(fig)
    # CSV preserves named denominators, rather than comparing normalized percentages as amplification.
    with (a.output/'hidden_boundary_summary.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['split','task','arm','boundary','scenarios','median_absolute_rms','median_B_reference_rms','median_relative_error'])
        for split in ['standard','boundary_pool']:
            for task in ['RETRIEVAL','COMPARISON','CODE']:
                for arm in ['A_PUBLIC','V4']:
                    rr=[r for r in native if r['split']==split and r['task']==task and r['arm']==arm]
                    for boundary in sorted(rr[0]['hidden_differences']):
                        v=[r['hidden_differences'][boundary] for r in rr];w.writerow([split,task,arm,boundary,len(v),np.median([x['absolute_rms'] for x in v]),np.median([x['reference_rms'] for x in v]),np.median([x['relative_error'] for x in v])])
    itemrows=[]
    for i in inputs:
        rr=[r for r in rows if r['item_id']==i['item_id']];itemrows.append({'item_id':i['item_id'],'base_id':i['base_id'],'split':i['split'],'task':i['task'],'token_hash':i['token_hash'],'prompt':i['prompt'],'gold':next(x['gold'] for x in gold if x['base_id']==i['base_id']),'records':[{**{k:r[k] for k in ['arm','readout','option_logits','full_lse','full_argmax','gold_index','label_ids','validity']},'score':score(r)} for r in rr]})
    payload=json.dumps({'summary':s,'items':itemrows},ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Case 006 — Readout and ties</title>
<style>body{max-width:1100px;margin:24px auto;padding:0 16px;font:16px system-ui;background:#fafafa;color:#172331}header{position:sticky;top:0;background:#172331;color:white;padding:12px;z-index:1}select,input,button{font:inherit;padding:7px;margin:4px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #b8c2cb;padding:7px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto;background:white;padding:12px}img{max-width:100%}.scroll{overflow:auto}label{display:inline-block}#empty{color:#8c3600}</style>
<header>POST-HOC SAME-INPUT READOUT DIAGNOSTIC</header><h1>Native ties and a shadow FP32 output head</h1><p>OPENING</p><p>192 standard + 46 originally selected stress scenarios. Views are repeated outcomes, not additional samples. The native 192-item result stays primary. No deployment assessment.</p>
<img alt="Measured readout flip counts" src="IMAGE"><h2>Explore a fixed scenario</h2>
<label>Set <select id="set"><option value="standard">STANDARD (192)</option><option value="boundary_pool">Selected stress (46)</option></select></label>
<label>Task <select id="task"><option value="ALL">All tasks</option><option>RETRIEVAL</option><option>COMPARISON</option><option>CODE</option></select></label>
<label>Scores <select id="view"><option>H_NATIVE</option><option>H_FP32</option></select></label>
<label>Candidate <select id="arm"><option>A_PUBLIC</option><option>V4</option></select></label>
<label>Outcome <select id="outcome"><option value="ALL">All outcomes</option><option value="flip">Choice change</option><option value="regression">Regression</option><option value="gain">Gain</option></select></label>
<label>Item ID <input id="search" placeholder="Search item ID"></label><button id="download">Download visible source records</button>
<p id="counts"></p><p id="empty"></p><select id="items" aria-label="Matching scenario"></select><div id="detail"></div><h2>Limits</h2><p>The candidate-to-B contrast always uses the same readout. Removing exact top ties does not prove better answers. Round-trip option scores agree, but full-vocabulary round trips differ. FP32 changes both output storage and GEMM arithmetic. A matching later token is not state restoration; original JSON generation remained 0/24 in all arms. Original prefill cost changes were about 0.4% / 0.14%; no new timing was measured here.</p>
<script type="application/json" id="data">PAYLOAD</script><script>
'use strict';const data=JSON.parse(document.getElementById('data').textContent);const el=id=>document.getElementById(id);let shown=[];
function record(i,a,v){return i.records.find(r=>r.arm===a&&r.readout===v)}
function outcome(i){const b=record(i,'B',el('view').value).score,c=record(i,el('arm').value,el('view').value).score;return {flip:b.prediction!==c.prediction,regression:b.correct&&!c.correct,gain:!b.correct&&c.correct}}
function showDetail(){const i=shown.find(i=>i.item_id===el('items').value);const box=el('detail');box.replaceChildren();if(!i)return;const h=document.createElement('h3');h.textContent=i.item_id+' · gold '+i.gold;box.append(h);const meta=document.createElement('p');const bt=record(i,'B','H_NATIVE').score.top;meta.textContent='Original native B top set: '+bt.map(x=>'ABCD'[x]).join(', ')+'; token SHA256 '+i.token_hash;box.append(meta);
const scroll=document.createElement('div');scroll.className='scroll';const table=document.createElement('table');const hr=document.createElement('tr');for(const name of ['Arm / view','A','B','C','D','Current top set','Decision / correct','Label mass','NLL / margin']){const th=document.createElement('th');th.textContent=name;hr.append(th)}table.append(hr);
for(const arm of ['B','A_PUBLIC','V4']){const r=record(i,arm,el('view').value),s=r.score;const tr=document.createElement('tr');const values=[arm+' / '+r.readout,...r.option_logits.map(String),s.top.map(x=>'ABCD'[x]).join(','),'ABCD'[s.prediction]+' / '+s.correct,s.label_mass.toPrecision(9),s.nll.toPrecision(9)+' / '+s.margin.toPrecision(9)];for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}table.append(tr)}scroll.append(table);box.append(scroll);const prompt=document.createElement('pre');prompt.textContent=i.prompt;box.append(prompt)}
function render(){shown=data.items.filter(i=>i.split===el('set').value&&(el('task').value==='ALL'||i.task===el('task').value)&&i.item_id.includes(el('search').value)&&(el('outcome').value==='ALL'||outcome(i)[el('outcome').value]));el('counts').textContent=shown.length+' existing scenarios in current filter; views are not new samples.';el('empty').textContent=shown.length?'':'None observed / no matching item. No substitute example.';el('items').replaceChildren();for(const i of shown){const o=document.createElement('option');o.value=i.item_id;o.textContent=i.item_id;el('items').append(o)}showDetail()}
for(const id of ['set','task','view','arm','outcome','search'])el(id).addEventListener('input',render);el('items').addEventListener('change',showDetail);el('download').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(shown,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='case006-readout-visible-records.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});render();
</script></html>'''
    html=html.replace('OPENING',OPENING).replace('IMAGE','data:image/png;base64,'+base64.b64encode((figdir/'01_readout_flips.png').read_bytes()).decode()).replace('PAYLOAD',payload)
    (a.output/'demo').mkdir();(a.output/'demo/index.html').write_text(html)
    save(a.output/'material_manifest.json',{'figures':3,'items':len(itemrows),'data_kind':'measured','readouts':['H_NATIVE','H_FP32'],'opening':OPENING})
if __name__=='__main__':main()
