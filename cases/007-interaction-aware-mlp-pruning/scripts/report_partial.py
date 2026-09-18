"""Generate an explicitly resource-blocked partial report from measured scalars."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import load

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=load(a.summary);c=a.case;o=a.output
 if s['status']!='BLOCKED_RESOURCE' or o.exists():raise ValueError('Expected blocked evidence and new output directory')
 o.mkdir(parents=True);(o/'figures').mkdir();(o/'demo').mkdir();methods=['INDEPENDENT_4','PAIRWISE_4','INDEPENDENT_8','PAIRWISE_8']
 table=['| Budget / method | Removed groups | DEV median / p95 local error |','|---|---|---|']
 for m in methods:
  v=s['splits']['DEVELOPMENT']['methods'][m]['relative_error'];table.append(f"| {s['selection'][m]['budget']}/16 {m.split('_')[0]} | {s['selection'][m]['removed']} | {100*v['median']:.3f}% / {100*v['p95']:.3f}% |")
 table='\n'.join(table);d=s['splits']['DEVELOPMENT']['paired_differences']['4'];delta=100*d['mean_pairwise_minus_independent'];lo,hi=[100*x for x in d['ci95']]
 en=f'''# Interaction-Aware Structured MLP Pruning

English | [한국어](README.ko.md)

This case implements signed interaction-based selection and physically smaller dense Qwen3-0.6B MLPs. **BLOCKED_RESOURCE:** calibration and development completed, but an external GPU job blocked the start of held-out evaluation. At 25% deletion, pairwise selection reduced development mean local error by {-delta:.3f} percentage points versus independent selection; at 50%, both selected the same groups. **Held-out transfer, pruned-model quality and latency are NOT_RUN**, so the scientific question remains unanswered.

## Measured development evidence

{table}

The table uses 48 development prompts, one fixed layer (zero-based 13), length 512 and 32 fixed positions per prompt. Relative error compares actual sliced BF16 outputs with sampled native BF16 MLP outputs. It is not task accuracy. The mean paired 25% difference (PAIRWISE minus INDEPENDENT) was {delta:+.4f} percentage points, 95% descriptive interval [{lo:+.4f}, {hi:+.4f}]. This development observation does not replace the frozen held-out test.

![Recorded calibration and development errors](figures/01_partial_errors.png)

## Implementation and validity

The 3072 intermediate channels are partitioned into 16 contiguous groups of 192. Ninety-six calibration prompts estimate a signed 16×16 Gram matrix. Exact enumeration evaluates 1,820 four-group sets and 12,870 eight-group sets. Twenty deterministic random sets per budget remain in the full [scalar summary](results/partial_summary.json).

The adapters physically slice gate/up rows and down columns. Native width 3072 and 9,437,184 parameters shrink to width 2304 / 7,077,888 parameters or width 1536 / 4,718,592 parameters. These are chosen-layer reductions, not whole-model savings. Same-set 50% selectors are aliases of the same structural design.

Six GPU smoke prompts verified decomposition and sliced/masked mapping. All calibration and development sampled inputs then passed the fixed equivalence check; its largest relative error was 0.050947%. Native repeats matched bitwise. The tests do not establish integrated pruned-model quality, which was not evaluated. See [validation](provenance/development_validation.json), [actual structural sizes](provenance/structural_sizes.json) and [independent scalar audit](provenance/independent_partial_audit.json).

## What ran and what remains

- Ran: 6 smoke, 96 calibration and 48 development prompts; 156 native Qwen prompt forwards; selected/random local MLP comparisons and CPU audits.
- Frozen but not run: 192 held-out prompts and their B + four selected model configurations; three planned timing processes.
- No new layer/model, recovery training, oracle selection, generation, kernel build or installation.

Deployment is **NOT_ASSESSED**. A new private resume directory preserves completed evidence; the resource guard must pass before any further GPU work. [Reproduction and resume](REPRODUCTION.md) describes exact commands. [Methods](METHODS.md), [analysis](ANALYSIS.md) and [limitations](LIMITATIONS.md) separate planned stages from recorded evidence.

## Explore and attribute

Open [demo/index.html](demo/index.html) locally for measured calibration/development records. It shows missing held-out, task-transition and timing evidence explicitly. GitHub HTML source preview is not a running demo. Figures and scalar JSON need no GPU to read.

Qwen/Transformers supply the model implementation. HOPE motivates interaction-aware deletion in a separate MoE context; this is neither its reproduction nor a new algorithm claim. The contribution is the controlled harness, structural surgery and auditable evidence. OpenAI Codex assisted the work. [NOTICE](NOTICE.md), [portfolio card](PORTFOLIO.md), [run state](RUN_STATE.json).
'''
 (o/'README.md').write_text(en)
 ko=f'''# 상호작용을 고려한 구조화 MLP 압축 실험

[English](README.md) | 한국어

상호작용을 포함한 삭제 집합 선택과 실제로 크기가 줄어든 Qwen3-0.6B MLP를 구현했습니다. 현재 상태는 **BLOCKED_RESOURCE**입니다. 보정·개발 실험은 끝났지만 외부 GPU 작업이 감지되어 미관측 평가를 시작하지 않았습니다. 25% 삭제에서 상호작용 방식의 개발 평균 국소 오차가 독립 방식보다 {-delta:.3f} 퍼센트포인트 낮았고, 50%에서는 같은 그룹을 선택했습니다. **미관측 전이·압축 모델의 과제 성능·지연시간은 미실행**이므로 주 연구 질문의 결론은 아직 없습니다.

## 실제 측정한 개발 결과

{table.replace('Budget / method','삭제 예산 / 방법').replace('Removed groups','삭제 그룹').replace('DEV median / p95 local error','개발 국소 오차 중앙값 / p95')}

개발 프롬프트 48개, 0부터 센 layer 13 한 곳, 길이 512, 문서당 고정 위치 32개를 사용했습니다. 표는 실제 축소 MLP의 BF16 출력을 원래 BF16 출력과 비교한 국소 오차이며 정답률 손실이 아닙니다. 25%의 평균 쌍별 차이(상호작용−독립)는 {delta:+.4f}%p, 기술적 95% 구간은 [{lo:+.4f}, {hi:+.4f}]%p입니다. 개발 결과를 고정된 미관측 평가의 대체 근거로 삼지 않습니다.

![보정·개발의 실제 국소 오차. 원래 그림 라벨은 영어입니다.](figures/01_partial_errors.png)

## 구현과 확인 범위

중간 채널 3072개를 연속된 192개씩 16그룹으로 나누고, 보정 프롬프트 96개로 부호를 유지한 상호작용 행렬을 추정했습니다. 4그룹·8그룹 삭제 집합 1,820개·12,870개를 전수 탐색했습니다. 예산별 고정 무작위 집합 20개도 같은 입력으로 국소 비교했습니다.

Gate/up 행과 down 열을 실제로 잘라, 해당 MLP의 중간 폭을 3072에서 2304 또는 1536으로 줄였습니다. 파라미터는 9,437,184개에서 7,077,888개 또는 4,718,592개입니다. 모델 전체가 25%·50% 줄었다는 뜻이 아닙니다. 50%의 두 선택은 같은 구조입니다.

GPU smoke 6개와 보정·개발 입력의 구조 축소/마스킹 비교가 고정 기준을 통과했습니다. 개발 검사에서 가장 큰 상대 차이는 0.050947%였습니다. 원래 모델의 반복 출력도 일치했습니다. 다만 축소 MLP를 통합한 모델의 품질은 아직 검증하지 않았습니다.

## 실행과 미실행

실행한 것은 smoke 6개, 보정 96개, 개발 48개와 원래 모델 forward 156회, 국소 MLP 비교, CPU 검산입니다. 미관측 192개 입력은 고정됐지만 모델 실행은 0회이며 속도 측정도 하지 않았습니다. 다른 GPU 작업을 종료하거나 선점하지 않았습니다. 배포 적합성은 **NOT_ASSESSED**입니다.

[방법 — 영어](METHODS.md), [분석 — 영어](ANALYSIS.md), [재현·재개 명령 — 영어](REPRODUCTION.md), [스칼라 근거](results/partial_summary.json), [독립 계산 검사](provenance/independent_partial_audit.json)를 확인할 수 있습니다. [오프라인 탐색기](demo/index.html)는 저장된 보정·개발 자료만 보여주며 UI는 영어입니다. 실제 iPad는 미검증입니다.

Qwen·Transformers의 모델과 연산을 사용했습니다. HOPE는 MoE 전문가 삭제의 상호작용을 고려한다는 발상 출처이며 직접 재현이 아닙니다. 프로젝트 기여는 통제된 비교, 실제 구조 축소, 검증과 근거 전달입니다. Codex가 구현·실행·분석·문서를 지원했습니다. [출처 — 영어](NOTICE.md), [한·영 포트폴리오](PORTFOLIO.md).
'''
 (o/'README.ko.md').write_text(ko)
 analysis='# Partial analysis — no held-out conclusion\n\n'+table+'\n\nThe resource block occurred before any held-out model forward. Do not classify this as positive, negative or no-clear held-out transfer.\n\n## Calibration and development\n\n'
 for split,v in s['splits'].items():
  for b,dd in v['paired_differences'].items():analysis+=f"- {split}, {b}/16: paired mean difference {dd['mean_pairwise_minus_independent']:.8f}, interval {dd['ci95']}, lower pairwise error on {dd['pairwise_better_prompts']}/{dd['n']} prompts. Units are relative error fractions. Calibration is in-sample; development is not the held-out set.\n"
 analysis+='\n## Same-objective comparison\n\nIndependent diagonal minima and pairwise quadratic minima are different objective functions. Evaluate both selected sets using the common quadratic objective:\n\n'
 for m,v in s['selection'].items():analysis+=f"- {m}: J_ind={v['J_ind_common']:.6f}; J_pair={v['J_pair_common']:.6f}.\n"
 analysis+='\nThe 50% selectors choose identical groups, so their zero observed difference is an alias result. It does not establish general equivalence.\n\n## Random reference distribution\n\nEach random set uses the same prompts. Set counts are not independent prompt counts.\n\n'
 for split,dd in s['splits'].items():
  for b in [4,8]:
   vals=[dd['methods'][f'RANDOM_{b}_{i:02d}']['relative_error']['mean'] for i in range(20)];analysis+=f"- {split}, {b}/16: random mean-error range [{min(vals):.6f}, {max(vals):.6f}], median {np.median(vals):.6f}. Sets below independent mean: {sum(x<dd['methods'][f'INDEPENDENT_{b}']['relative_error']['mean'] for x in vals)}/20; below pairwise: {sum(x<dd['methods'][f'PAIRWISE_{b}']['relative_error']['mean'] for x in vals)}/20.\n"
 analysis+=f"\nCalibration Q has {s['calibration_Q']['negative_pairs']} negative and {s['calibration_Q']['positive_pairs']} positive off-diagonal unordered pairs. The presymmetry residual is {s['calibration_Q']['pre_symmetry_max_abs']}; maximum diagonal discrepancy is {s['calibration_Q']['diagonal_max_abs']:.3e}. Eigenvalues in the summary are descriptive. No held-out-Q comparison or oracle search was run.\n\n## Limits of interpretation\n\nThe local loss is about one MLP boundary. It does not imply answer-quality changes. A smaller dense module does not guarantee lower latency; no timing was recorded. The random comparison covers 20 fixed sets rather than the entire candidate space. Public prompt sums and worst-token scalars permit calculation checks, not reconstruction of the excluded activations.\n"
 (o/'ANALYSIS.md').write_text(analysis)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10})
 fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4)
 for split,offset in [('CALIBRATION',-.16),('DEVELOPMENT',.16)]:
  vals=[100*s['splits'][split]['methods'][m]['relative_error']['median'] for m in methods];ax.bar(x+offset,vals,width=.3,label=f'{split} ({s["splits"][split]["prompts"]} prompts)')
 ax.set_xticks(x,['Independent 25%','Pairwise 25%','Independent 50%','Pairwise 50%']);ax.set_ylabel('Median prompt local error (%)');ax.set_title('Recorded local reconstruction · HELD_OUT NOT_RUN');ax.legend();fig.tight_layout();fig.savefig(o/'figures/01_partial_errors.png',dpi=150,metadata={'Software':'Case007'});plt.close(fig)
 q=np.array(s['calibration_Q']['Q']);fig,ax=plt.subplots(figsize=(6,5));v=max(abs(q.min()),abs(q.max()));im=ax.imshow(q,cmap='coolwarm',vmin=-v,vmax=v);ax.set_title('Calibration Q · signed terms');ax.set_xlabel('Group');ax.set_ylabel('Group');fig.colorbar(im,ax=ax,label='Mean output-vector inner product');fig.tight_layout();fig.savefig(o/'figures/02_interactions.png',dpi=150,metadata={'Software':'Case007'});plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4))
 for b,xoff in [(4,0),(8,1)]:
  dd=s['splits']['DEVELOPMENT']['methods'];vals=[100*dd[f'RANDOM_{b}_{i:02d}']['relative_error']['mean'] for i in range(20)];ax.scatter(np.linspace(xoff-.15,xoff+.15,20),vals,label=f'{b}/16 random sets (20)')
  for method,marker in [('INDEPENDENT','s'),('PAIRWISE','x')]:ax.scatter([xoff],[100*dd[f'{method}_{b}']['relative_error']['mean']],marker=marker,s=80,label=f'{b}/16 {method}')
 ax.set_xticks([0,1],['25%','50%']);ax.set_ylabel('Mean prompt local error (%)');ax.set_title('Development reference distribution · same 48 prompts');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(o/'figures/03_random.png',dpi=150,metadata={'Software':'Case007'});plt.close(fig)
 prompts={r['id']:r['text'] for split in ['CALIBRATION','DEVELOPMENT'] for r in load(c/f'inputs/{split}.json')};rows=[]
 for f in sorted((c/'results/raw/development').glob('c007-*.json')):
  r=load(f)
  for u in r['local']:rows.append({'id':r['id'],'split':r['split'],'task':r['task'],'method':u['method'],'budget':u['budget'],'removed':u['removed'],'relative_error':u['relative_error'],'absolute_rms':u['absolute_rms'],'cosine':u['cosine'],'worst_token':u['token_metrics'][0],'task_transition':'NOT_RUN','cost':'NOT_RUN','source':'../results/raw/development/'+f.name})
 data=json.dumps({'prompts':prompts,'rows':rows},ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
 page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Case007 partial evidence</title><style>body{font:16px system-ui;max-width:1000px;margin:auto;padding:20px;line-height:1.5}select,input,button{padding:8px;margin:4px;max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:12px}img{max-width:100%}</style><h1>Case007: measured partial evidence</h1><p><strong>BLOCKED_RESOURCE · HELD_OUT / pruned-model quality / timing NOT_RUN</strong></p><p>96 calibration and 48 development prompts. Repeated groups, methods and tokens are not new independent samples. Deployment NOT_ASSESSED.</p><label>Split <select id="split"><option value="">All</option><option>CALIBRATION</option><option>DEVELOPMENT</option><option>HELD_OUT</option></select></label><label>Task <select id="task"><option value="">All</option><option>RETRIEVAL</option><option>COMPARISON</option><option>CODE</option></select></label><label>Method <select id="method"><option value="">All</option></select></label><input id="query" aria-label="Prompt ID" placeholder="Prompt ID"><p id="count"></p><select id="items" aria-label="Matching record"></select><button id="download">Download selected record</button><pre id="record"></pre><details><summary>Exact prompt</summary><pre id="prompt"></pre></details><img src="../figures/01_partial_errors.png" alt="Measured calibration and development local errors"><script type="application/json" id="data">'''+data+'''</script><script>const d=JSON.parse(document.getElementById('data').textContent),split=document.getElementById('split'),task=document.getElementById('task'),method=document.getElementById('method'),query=document.getElementById('query'),items=document.getElementById('items');let visible=[];for(const m of [...new Set(d.rows.map(r=>r.method))]){let o=document.createElement('option');o.textContent=m;method.append(o)}function show(){let r=visible[items.selectedIndex];document.getElementById('record').textContent=r?JSON.stringify(r,null,2):'No measured records; HELD_OUT is NOT_RUN.';document.getElementById('prompt').textContent=r?d.prompts[r.id]:'';}function filter(){visible=d.rows.filter(r=>(!split.value||r.split===split.value)&&(!task.value||r.task===task.value)&&(!method.value||r.method===method.value)&&r.id.includes(query.value));items.replaceChildren(...visible.map(r=>{let o=document.createElement('option');o.textContent=r.id+' / '+r.method;return o}));document.getElementById('count').textContent=visible.length+' method records (not independent prompts)';show()}for(const e of [split,task,method,query])e.addEventListener('input',filter);items.addEventListener('change',show);document.getElementById('download').onclick=()=>{let r=visible[items.selectedIndex];if(!r)return;let u=URL.createObjectURL(new Blob([JSON.stringify({...r,prompt:d.prompts[r.id]},null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=u;a.download=r.id+'-'+r.method+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};filter();</script></html>'''
 (o/'demo/index.html').write_text(page);print('Generated partial-only reports, figures and explorer')
if __name__=='__main__':main()
