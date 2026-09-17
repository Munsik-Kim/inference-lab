"""Offline measured-data explorer. No server, fetch, CDN, eval or HTML injection."""
import argparse,base64,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import load,external_directory
from scripts.build_demo import embedded_json

HTML='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Case 006 · Decision Stability Audit</title><style>
:root{color-scheme:light;--ink:#152e3d;--muted:#536571;--line:#cdd9df;--accent:#13697b}*{box-sizing:border-box}body{margin:0;background:#f3f6f7;color:var(--ink);font:16px/1.55 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:28px 18px}h1{font-size:clamp(1.9rem,4vw,2.8rem);line-height:1.12}h2{font-size:1.35rem}h3{font-size:1.1rem}.eyebrow{color:var(--accent);letter-spacing:.12em;font-size:.8rem;text-transform:uppercase}.scope{border-left:5px solid var(--accent);padding:15px 20px;background:#e7f1f4}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}section,article{background:white;border:1px solid var(--line);border-radius:8px;padding:18px;margin:16px 0}article{margin:0}.number{font-size:1.8rem;color:var(--accent)}small,.muted{color:var(--muted)}p{max-width:90ch}.filters{display:flex;gap:10px;flex-wrap:wrap;align-items:end}label{font-size:.85rem}select,input,button{font:inherit;max-width:100%;padding:8px;border:1px solid var(--line);border-radius:4px;background:white;color:var(--ink)}button{background:var(--accent);color:white;cursor:pointer}table{border-collapse:collapse;width:100%;font-size:.94rem}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.table-scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:430px;overflow:auto;background:#f5f7f8;padding:14px;font:13px/1.55 ui-monospace,monospace}summary{cursor:pointer;font-weight:600}.good{background:#e9f5ed}.badge{display:inline-block;border:1px solid var(--line);padding:3px 8px;border-radius:4px;font-size:.82rem}img{max-width:100%;height:auto}.figure{margin:14px 0}.metadata{overflow-wrap:anywhere}a{color:var(--accent)}
</style></head><body><main><div class="eyebrow">Inference Lab · Case 006 · Measured offline evidence</div>
<h1>Decision stability beyond aggregate accuracy</h1>
<p>Controlled Qwen3-0.6B prompt-prefill interventions on RTX 5080.</p>
<div class="scope"><strong id="headline"></strong><p>Only layer 13's prompt-prefill attention changes. Weights, all other layers and every decode step remain BF16. Forced-choice accuracy and probabilistic gold scores answer different questions. Deployment verdict: NOT_ASSESSED.</p></div>
<section><h2>Start here · 30 seconds</h2><ol><li>Read the standard-set score and correctness tables below. A flip can be a gain, a regression or a different wrong answer.</li><li>Use the item explorer to inspect the exact question, gold, option logits/probabilities and full-label mass.</li><li>Compare standard data with the separately selected BF16 boundary stress set. Its flip rate is not an ordinary-workload estimate.</li></ol><p class="muted">All numbers come from measured records. Bootstrap intervals are pointwise and preserve base-scenario pairing. No independent third-party GPU reproduction is claimed.</p></section>
<div class="cards" id="cards"></div>
<section><h2>Gold-grounded outcomes · standard L4096</h2><div class="table-scroll"><table><thead><tr><th>Arm</th><th>Accuracy</th><th>Both correct</th><th>Regression</th><th>Gain</th><th>Both wrong</th><th>All flips</th></tr></thead><tbody id="outcomes"></tbody></table></div><p id="baseline"></p><p class="muted">Conditional choice scores renormalize A–D. Full-vocabulary label mass and NLL are retained to expose failures of answer format.</p></section>
<section><h2>Item-level evidence</h2><div class="filters">
<label>Set<br><select id="set"><option value="standard">STANDARD_EVAL / length support</option><option value="boundary_pool">BF16-conditioned stress</option><option value="dev">DEVELOPMENT</option></select></label>
<label>Task<br><select id="task"><option value="">All</option><option>RETRIEVAL</option><option>COMPARISON</option><option>CODE</option></select></label>
<label>Length<br><select id="length"><option>4096</option><option>2048</option><option>512</option></select></label>
<label>Arm<br><select id="arm"><option>A_PUBLIC</option><option>V4</option></select></label>
<label>Outcome<br><select id="outcome"><option value="">All</option><option value="regression">Regression</option><option value="gain">Gain</option><option value="both_correct">Both correct</option><option value="both_wrong">Both wrong</option><option value="flip">Any flip</option></select></label>
<label>ID search<br><input id="search" placeholder="e.g. code-005"></label>
</div><p id="counts" class="muted"></p><label>Item<br><select id="item"></select></label>
<h3 id="title"></h3><p id="judgment"></p><div class="table-scroll"><table><thead><tr><th>Label</th><th>Gold</th><th>B logit</th><th>B conditional probability</th><th>A_PUBLIC logit</th><th>A_PUBLIC probability</th><th>V4 logit</th><th>V4 probability</th></tr></thead><tbody id="choices"></tbody></table></div>
<p id="scores"></p><p id="mass"></p><p id="local"></p><p id="validity" class="metadata"></p>
<details><summary>Exact prompt and separate gold</summary><pre id="prompt"></pre><pre id="gold"></pre></details>
<details><summary>Raw scalar evidence and hidden-boundary norms</summary><pre id="raw"></pre></details><p><button id="download">Export this measured item</button></p>
<p class="muted">R = option-logit perturbation oscillation / BF16 winner gap requires both outputs. It is a post-hoc check, not an online predictor or a speed-saving router. Undefined ties stay undefined.</p></section>
<section><h2>Measured figures</h2><div id="figures"></div></section>
<section><h2>Model-level cost and secondary generation</h2><pre id="cost"></pre><pre id="generation"></pre><h3>Preselected generation examples</h3><p>Own prompts and own caches; examples follow the frozen subset, not largest divergence.</p><select id="generation_item"></select><pre id="generation_example"></pre><p class="muted">Prefill timing uses 12 fixed scenarios, three process rounds and paired five-call blocks. p95 is a block-mean statistic, not service p95. Greedy generation has its own prompt and own cache per arm; different output lengths are not pure compute speedups. Token re-alignment never proves cache/state recovery.</p></section>
<section><h2>Limits</h2><p>These are narrow self-written English retrieval/comparison tasks and restricted Python arithmetic, with deterministic irrelevant filler. Head/query/length repeats are not independent scenarios. One small model, one layer and one GPU do not establish downstream deployment suitability, broad generalization or non-inferiority. No quality or latency tolerance has been specified for deployment.</p><p>Qwen, Transformers and SageAttention provide the upstream implementations. Codex assisted this evaluation harness, local execution, analysis and documentation. All sources and sufficient statistics are in the package; full logits/activations are not bundled.</p></section>
<script id="payload" type="application/json">__DATA__</script><script>
'use strict';const data=JSON.parse(document.getElementById('payload').textContent);const $=id=>document.getElementById(id);const fmt=v=>v===null||v===undefined?'undefined':Number(v).toPrecision(5);const pct=v=>v===null||v===undefined?'undefined':fmt(100*v)+'%';
const primary=data.summary.groups['standard/L4096'];$('headline').textContent=primary.independent_scenarios+' standard scenarios · B, A_PUBLIC and V4 measured · no production verdict';
for(const arm of ['A_PUBLIC','V4']){const s=primary.arms[arm],m=s.task_balanced.delta_choice_nll;const card=document.createElement('article');const title=document.createElement('strong');title.textContent=arm+' vs BF16';const num=document.createElement('div');num.className='number';num.textContent=fmt(m.mean)+' nats';const desc=document.createElement('p');desc.textContent='Task-balanced gold choice NLL change (positive is worse). 95% paired interval ['+fmt(m.ci95[0])+', '+fmt(m.ci95[1])+'].';card.appendChild(title);card.appendChild(num);card.appendChild(desc);$('cards').appendChild(card);
 const tr=document.createElement('tr');for(const v of [arm,pct(s.candidate_accuracy),s.outcomes.both_correct,s.outcomes.regression,s.outcomes.gain,s.outcomes.both_wrong,s.outcomes.flips.numerator+'/'+s.outcomes.flips.denominator]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}$('outcomes').appendChild(tr);}
$('baseline').textContent='BF16 forced-choice accuracy: '+pct(primary.arms.A_PUBLIC.B_accuracy)+'. Exact per-task counts and all wrong-to-wrong flips are retained in the source summary.';
const inputs=new Map(data.inputs.map(r=>[r.item_id,r]));const gold=new Map(data.gold.map(r=>[r.base_id,r]));let visible=[];
function row(){return visible.find(r=>r.item_id===$('item').value);}
function show(){const r=row();$('choices').replaceChildren();$('download').disabled=!r;if(!r){for(const id of ['title','judgment','scores','mass','local','validity','prompt','gold','raw'])$(id).textContent='';return;}
 const b=r.B_score,c=r.candidate_score;$('title').textContent=r.item_id+' · '+r.arm;$('judgment').textContent='Gold '+String.fromCharCode(65+b.gold_index)+' | BF16 '+String.fromCharCode(65+b.prediction)+' | candidate '+String.fromCharCode(65+c.prediction)+' | '+r.cell+(r.flip?' · decision flip':' · same choice');
 const other={};for(const arm of ['A_PUBLIC','V4'])other[arm]=data.pairs.find(x=>x.item_id===r.item_id&&x.arm===arm)?.candidate_score;for(let i=0;i<4;i++){const tr=document.createElement('tr');if(i===b.gold_index)tr.className='good';for(const v of [String.fromCharCode(65+i),i===b.gold_index?'yes':'',fmt(b.option_logits[i]),fmt(b.q[i]),fmt(other.A_PUBLIC?.option_logits[i]),fmt(other.A_PUBLIC?.q[i]),fmt(other.V4?.option_logits[i]),fmt(other.V4?.q[i])]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}$('choices').appendChild(tr);}
 $('scores').textContent='Gold choice NLL B / candidate: '+fmt(b.choice_nll)+' / '+fmt(c.choice_nll)+' nats. Signed ΔNLL '+fmt(r.delta_choice_nll)+'; signed Δgold margin '+fmt(r.delta_gold_margin)+'.';
 $('mass').textContent='Full-vocabulary A–D mass B / candidate: '+fmt(b.label_mass)+' / '+fmt(c.label_mass)+'. Full gold NLL: '+fmt(b.full_gold_nll)+' / '+fmt(c.full_gold_nll)+'. Full argmax allowed: '+b.full_argmax_allowed+' / '+c.full_argmax_allowed+'.';
 $('local').textContent='Pooled local error vs FP32 '+pct(r.local_reference_error)+'; last-query error '+pct(r.local_last_query_error)+'; full-vocabulary KL '+fmt(r.full_vocab_kl)+'; gap '+fmt(r.base_gap)+'; R '+fmt(r.R)+'.';
 $('validity').textContent='Validity '+(Object.values(r.validity).every(v=>v===true)?'PASS':'BLOCKED')+' · token SHA256 '+r.token_hash;
 $('prompt').textContent=inputs.get(r.item_id)?.prompt||'Input missing — inspect package manifest';$('gold').textContent=JSON.stringify(gold.get(r.base_id),null,2);$('raw').textContent=JSON.stringify(r,null,2);
}
function filter(){const q=$('search').value.toLowerCase();visible=data.pairs.filter(r=>r.split===$('set').value&&r.length===Number($('length').value)&&r.arm===$('arm').value&&(!$('task').value||r.task===$('task').value)&&(!$('outcome').value||($('outcome').value==='flip'?r.flip:r.cell===$('outcome').value))&&r.item_id.toLowerCase().includes(q));$('item').replaceChildren();for(const r of visible){const o=document.createElement('option');o.value=r.item_id;o.textContent=r.item_id;$('item').appendChild(o);}$('counts').textContent=visible.length+' paired items match these filters. Standard and selected stress denominators are separate.';show();}
for(const id of ['set','task','length','arm','outcome','search'])$(id).addEventListener('input',filter);$('item').addEventListener('change',show);
$('download').addEventListener('click',()=>{const r=row();if(!r)return;const blob=new Blob([JSON.stringify({evidence:r,input:inputs.get(r.item_id),gold:gold.get(r.base_id)},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=r.item_id+'-'+r.arm+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
for(const f of data.figures){const box=document.createElement('details');box.className='figure';const title=document.createElement('summary');title.textContent=f.title;const img=document.createElement('img');img.src=f.data;img.alt=f.title;box.appendChild(title);box.appendChild(img);$('figures').appendChild(box);}
$('cost').textContent=JSON.stringify(data.summary.model_timing,null,2);$('generation').textContent=JSON.stringify(data.summary.secondary,null,2);for(const r of data.secondary){const o=document.createElement('option');o.value=r.item_id;o.textContent=r.item_id;$('generation_item').appendChild(o);}function generationShow(){const r=data.secondary.find(x=>x.item_id===$('generation_item').value);$('generation_example').textContent=r?JSON.stringify(r,null,2):'NOT_RUN';}$('generation_item').addEventListener('change',generationShow);generationShow();filter();if(visible.some(r=>r.item_id===data.default_id)){$('item').value=data.default_id;show();}
</script></main></body></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--gold',nargs='+',type=Path,required=True);p.add_argument('--figures',type=Path,required=True);p.add_argument('--subsets',type=Path,required=True)
    p.add_argument('--secondary',nargs='*',type=Path);p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary=load(a.summary);pairs=load(a.pairs)
    if not summary['groups'].get('standard/L4096'):raise ValueError('No measured standard data')
    if any(any(v is not True for v in r['validity'].values()) for r in pairs):raise ValueError('Invalid record cannot become a presentation success')
    ids={r['item_id'] for r in pairs};inputs=[];gold=[]
    for path in a.inputs:inputs.extend({k:v for k,v in r.items() if k!='token_ids'} for r in load(path) if r['item_id'] in ids)
    for path in a.gold:gold.extend(load(path))
    if len({r['item_id'] for r in inputs})!=len(ids):raise ValueError('Missing/duplicate input evidence')
    figures=[{'title':p.stem.replace('_',' '),'data':'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()} for p in sorted(a.figures.glob('*.png'))]
    default=load(a.subsets)['RETRIEVAL']['timing_ids'][0]+'-L4096'
    secondary=[]
    for path in a.secondary or []:secondary.extend(load(path))
    payload={'secondary':secondary,'summary':summary,'pairs':pairs,'inputs':inputs,'gold':gold,'figures':figures,'default_id':default,
             'input_token_ids_note':'Exact IDs are in the packaged input files; this embedded viewer keeps prompt text and token hashes.'}
    out=external_directory(a.output,a.case);(out/'index.html').write_text(HTML.replace('__DATA__',embedded_json(payload)),encoding='utf-8')
    print(json.dumps({'paired_items':len(pairs),'inputs':len(inputs),'figures':len(figures),'bytes':(out/'index.html').stat().st_size}))


if __name__=='__main__':main()
