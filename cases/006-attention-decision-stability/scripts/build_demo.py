"""Build an offline input/evidence explorer from retained data, never mock scores."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.analysis import summarize
from src.storage import external_directory, load


def embedded_json(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')


HTML='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Case 006 · Decision Stability Audit</title>
<style>
:root{color-scheme:light;--ink:#172b3a;--accent:#166678;--muted:#51626e;--line:#ccd8dc}
*{box-sizing:border-box}body{margin:0;background:#f4f7f8;color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{max-width:1060px;margin:auto;padding:30px 20px}h1{font-size:clamp(1.8rem,4vw,2.6rem);line-height:1.15;margin-bottom:12px}
h2{font-size:1.25rem}p{max-width:80ch}.eyebrow{font-size:.8rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
.status{border-left:5px solid #be790f;background:#fff3dc;padding:16px 20px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
section,article{background:white;border:1px solid var(--line);border-radius:7px;padding:20px;margin:16px 0}article{margin:0}
.big{font-size:2rem;line-height:1.2;color:var(--accent)}.muted,small{color:var(--muted)}
label{display:block;font-size:.9rem}input,select,button{font:inherit;padding:9px;border:1px solid var(--line);border-radius:4px;background:white;color:var(--ink);max-width:100%}
button{cursor:pointer;background:var(--accent);color:white}.filters{display:flex;flex-wrap:wrap;gap:12px;align-items:end}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7f8;padding:16px;font:14px/1.55 ui-monospace,monospace;max-height:500px;overflow:auto}
table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px;vertical-align:top}
code{overflow-wrap:anywhere}.tabs{margin-top:15px}summary{cursor:pointer;font-weight:600}a{color:var(--accent)}
</style></head><body><main>
<div class="eyebrow">Inference Lab / Case 006 / Offline evidence explorer</div>
<h1>Decision stability beyond accuracy</h1>
<p>Controlled Qwen prefill interventions, paired answer scores and reproducible evidence.</p>
<div class="status"><strong id="state"></strong><p id="state-note"></p></div>
<div class="grid"><article><div class="big" id="smoke-count"></div>prepared smoke scenarios</article><article><div class="big" id="dev-count"></div>prepared DEV scenarios</article><article><div class="big" id="record-count"></div>GPU measurement records</article></div>
<section><h2>30-second walkthrough</h2><ol><li>Read the execution status. This delivery stopped before model loading.</li><li>Select a prepared task and inspect the exact prompt, separate gold and input hashes.</li><li>Check the arm table and download the source records. Missing results remain missing.</li></ol>
<p>No network connection, hosted model or API is used. This is a CPU-tested preparation artifact; it does not answer whether an attention intervention changes Qwen decisions.</p></section>
<section><h2>GPU evidence by arm</h2><table><thead><tr><th>Arm</th><th>Status</th><th>Gold score / decisions / cost</th></tr></thead><tbody id="arms"></tbody></table>
<p class="muted">No accuracy, NLL, margin, flip, hidden-state or timing chart can be drawn before measurements exist. A missing result is not a zero effect.</p></section>
<section><h2>Prepared input explorer</h2><div class="filters">
<label>Task<br><select id="task"><option value="">All tasks</option><option>RETRIEVAL</option><option>COMPARISON</option><option>CODE</option></select></label>
<label>Split<br><select id="split"><option value="">All prepared splits</option><option>smoke</option><option>dev</option></select></label>
<label>Item ID search<br><input id="search" placeholder="e.g. dev-code-023"></label>
<label>Item<br><select id="item"></select></label></div><p id="item-count" class="muted"></p>
<h3 id="item-title"></h3><p id="facts"></p><p id="hash" class="muted"></p>
<details open><summary>Exact model prompt (no gold continuation)</summary><pre id="prompt"></pre></details>
<details><summary>Separate gold and fact provenance (not sent to the model)</summary><pre id="gold"></pre></details>
<p id="scores" class="muted"></p><button id="download">Download this item's source JSON</button>
</section>
<section><h2>What is implemented and what is pending</h2><p>Fact-first generation, an integer AST oracle, four-option metrics, paired statistics, validity gates, immutable records and this offline viewer have CPU tests. A scoped attention adapter and a guarded smoke entry point are present but unvalidated on GPU in Case 006.</p>
<p>DEV model scoring, design and evaluation freezes, complete hidden-boundary instrumentation, primary evaluation, cache-safe generation and model-level timing remain pending. The full study runner is not complete. Deployment verdict: NOT_ASSESSED.</p>
<p>Data: fresh self-written synthetic English tasks and a restricted Python subset. Shared templates and repeated neutral filler limit external validity. Prepared inputs are not independent observed outcomes.</p></section>
<footer class="muted">Upstream Qwen and SageAttention are reused. Codex assisted code and documentation. No third-party GPU reproduction or human review is claimed.</footer>
<script id="payload" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('payload').textContent);
const $=id=>document.getElementById(id);
$('state').textContent=data.summary.status.evidence_completeness+' · GPU_BUSY';
$('state-note').textContent='A pre-existing GPU compute process was present at the resource gate. No Case 006 model forward started. CPU preparation and input checks are complete; GPU semantics and the study are not.';
$('smoke-count').textContent=data.summary.prepared_inputs.smoke;
$('dev-count').textContent=data.summary.prepared_inputs.dev;
$('record-count').textContent=data.summary.measured_record_count;
for(const [arm,r] of Object.entries(data.summary.arms)){
 const tr=document.createElement('tr');for(const value of [arm,r.status,r.metrics===null?'NOT_ESTIMABLE':'See retained source records']){const td=document.createElement('td');td.textContent=value;tr.appendChild(td);} $('arms').appendChild(tr);
}
const gold=new Map(data.gold.map(x=>[x.base_id,x]));let visible=[];
function selected(){return visible.find(x=>x.item_id===$('item').value);}
function show(){const r=selected();if(!r){for(const id of ['item-title','facts','hash','prompt','gold','scores'])$(id).textContent='';$('download').disabled=true;return;}
 $('download').disabled=false;$('item-title').textContent=r.item_id;
 $('facts').textContent=r.task+' · '+r.split+' · '+r.length+' exact input tokens · labels '+r.label_token_ids.join(', ');
 $('hash').textContent='Token SHA256: '+r.token_hash;$('prompt').textContent=r.prompt;$('gold').textContent=JSON.stringify(gold.get(r.base_id),null,2);
 $('scores').textContent='B / A_PUBLIC / V4 probabilities, gold NLL, label mass, decisions, local error and cost: NOT_RUN.';
}
function filter(){const query=$('search').value.toLowerCase();visible=data.inputs.filter(r=>(!$('task').value||r.task===$('task').value)&&(!$('split').value||r.split===$('split').value)&&r.item_id.toLowerCase().includes(query));
 $('item').replaceChildren();for(const r of visible){const o=document.createElement('option');o.value=r.item_id;o.textContent=r.item_id;$('item').appendChild(o);}$('item-count').textContent=visible.length+' prepared inputs match. No evaluation items are present.';show();}
for(const id of ['task','split','search'])$(id).addEventListener('input',filter);$('item').addEventListener('change',show);
$('download').addEventListener('click',()=>{const r=selected();if(!r)return;const blob=new Blob([JSON.stringify({input:r,gold:gold.get(r.base_id),measurements:[]},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=r.item_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
filter();
</script></main></body></html>'''


def build(case,output):
    summary,_=summarize(case)
    if summary['measured_record_count']:
        raise ValueError('This blocked-state viewer requires extension before displaying GPU results; never hide existing measurements')
    data={'summary':summary,'inputs':load(case/'inputs/prompts.json'),'gold':load(case/'inputs/gold.json')}
    output.write_text(HTML.replace('__DATA__',embedded_json(data)),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=external_directory(a.output,a.case);build(a.case,out/'index.html');print('Offline preparation explorer generated; no measured scores.')


if __name__=='__main__':main()
