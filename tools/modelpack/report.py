"""Generate figures and an offline scalar explorer from audited measurements."""
from pathlib import Path
import html,json
import numpy as np
from .common import read,write,new_external,sha


def generate(case: Path,output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=new_external(output);summary=read(case/'results/derived/summary.json')
    (out/'figures').mkdir();(out/'demo').mkdir();records=[]
    inputs={}
    for track in ['Q','R']:
        path=case/f'results/raw/{track}/model_records.json'
        if path.exists():records += [{**r,'track':track} for r in read(path)]
        for row in read(case/f'inputs/{track}/heldout.json'):inputs[row['id']]=row
    q=summary['tracks']['Q']
    if 'arms' in q:
        row=q['arms']['Q-W4'];x=row['delta_full_gold_nll'];fig,ax=plt.subplots(figsize=(7.2,2.6))
        ax.errorbar([x['mean']],[0],xerr=[[x['mean']-x['ci95'][0]],[x['ci95'][1]-x['mean']]],fmt='o',capsize=5)
        ax.axvline(0,color='grey',linewidth=1);ax.set_yticks([0],['Q-W4 minus Q-BF16']);ax.set_xlabel('Full-vocabulary gold-token NLL change (nats; positive worse)')
        ax.set_title('Q: 192 held-out scenarios; paired pointwise 95% interval');fig.tight_layout();fig.savefig(out/'figures/q_scores.png',dpi=160);plt.close(fig)
    r=summary['tracks']['R']
    if 'repair' in r:
        fig,ax=plt.subplots(figsize=(7.2,3.2))
        for i,(name,v) in enumerate(r['repair'].items()):
            point=v['pooled_recovery']*100;lo,hi=[x*100 for x in v['ci95']]
            ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',capsize=5,label=name)
        ax.axvline(0,color='grey',linewidth=1);ax.set_yticks(range(3),list(r['repair']));ax.set_xlabel('Removed error energy recovered (%)')
        ax.set_title('R: native BF16 held-out reconstruction, 192 scenarios');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(out/'figures/r_recovery.png',dpi=160);plt.close(fig)
    payload=json.dumps({'summary':summary,'records':records,'inputs':inputs},ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Case 008 — stored evidence</title>
<style>body{font:17px system-ui,sans-serif;max-width:1080px;margin:3rem auto;padding:0 1rem;color:#172f36;background:#f8f9f7}h1{font-size:2rem}label{display:inline-block;margin:1rem 1rem 1rem 0}select,input,button{font:inherit;padding:.4rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;border:1px solid #d4dddd;padding:1rem}a{color:#07586e}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #cb7e39}.muted{color:#405860}</style>
<h1>Case 008 · Build, reconstruct and reload</h1><p>Explore recorded model scores. Q and R use different models and runtimes; choose a track first. No model runs in this page.</p>
<label>Track <select id="track"><option>Q</option><option>R</option></select></label><label>Task <select id="task"><option value="">All</option><option>retrieval</option><option>comparison</option><option>code</option></select></label><label>Input ID <input id="find" type="search"></label><label>Arm <select id="arm"></select></label><p id="count" aria-live="polite"></p><select id="item" aria-label="Scenario"></select><button id="download">Download these records</button>
<h2>Prompt and gold</h2><pre id="prompt"></pre><h2>Selected arm and baseline</h2><p class="muted">Gold NLL evaluates the exact answer; positive candidate−baseline is worse. Local recovery measures error energy, not accuracy. Native ties retain deterministic A/B/C/D ordering.</p><pre id="detail"></pre><h2>Track aggregate</h2><pre id="aggregate"></pre><p><a href="../README.md">README in the review bundle</a> · <a href="../results/derived/summary.json">Source summary JSON</a></p>
<script type="application/json" id="data">PAYLOAD</script><script>
'use strict';const d=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let visible=[];
function refresh(changed=false){const t=$('track').value,names=[...new Set(d.records.filter(r=>r.track===t).map(r=>r.arm))],old=$('arm').value;
$('arm').replaceChildren(...names.map(n=>new Option(n,n)));if(names.includes(old))$('arm').value=old;
visible=d.records.filter(r=>r.track===t&&r.arm===$('arm').value&&(!$('task').value||r.task===$('task').value)&&r.id.includes($('find').value));
const prior=$('item').value;$('item').replaceChildren(...visible.map(r=>new Option(r.id,r.id)));if(visible.some(r=>r.id===prior))$('item').value=prior;
$('count').textContent=visible.length+' recorded scenarios in this filter';$('aggregate').textContent=JSON.stringify(d.summary.tracks[t],null,2);show();}
function show(){const r=visible.find(r=>r.id===$('item').value);if(!r){$('prompt').textContent='No matching recorded item.';$('detail').textContent='';return;}
const input=d.inputs[r.id],base=d.records.find(x=>x.id===r.id&&x.arm===(r.track==='Q'?'Q-BF16':'R-B'));
$('prompt').textContent=input.text+'\\nGold (independent oracle): '+String.fromCharCode(65+input.gold)+' — '+input.gold_value;
$('detail').textContent=JSON.stringify({baseline:base,candidate:r},null,2);location.hash=new URLSearchParams({track:r.track,arm:r.arm,id:r.id,task:$('task').value}).toString();}
for(const id of ['track','task','arm','find'])$(id).addEventListener(id==='find'?'input':'change',()=>refresh());$('item').addEventListener('change',show);
$('download').addEventListener('click',()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(visible,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='case008-records.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});
const h=new URLSearchParams(location.hash.slice(1));if(['Q','R'].includes(h.get('track')))$('track').value=h.get('track');if(h.has('task'))$('task').value=h.get('task');refresh();if(h.has('arm'))$('arm').value=h.get('arm');refresh();if(h.has('id'))$('item').value=h.get('id');show();
</script></html>'''.replace('PAYLOAD',payload)
    (out/'demo/index.html').write_text(page)
    write(out/'generation.json',{'summary_sha256':sha(case/'results/derived/summary.json'),'records':len(records),
        'source':'actual retained measurements only','browser_validation':'NOT_RUN','real_iPad':'NOT_TESTED'})
    return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();generate(a.case,a.output)
