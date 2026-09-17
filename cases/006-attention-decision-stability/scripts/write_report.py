"""Regenerate this completed study's narrative tables into a new directory."""
from pathlib import Path
import json,argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import external_directory
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--summary',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
case=external_directory(args.output,Path(__file__).resolve().parents[1])
s=json.loads(args.summary.read_text())
p=s['groups']['standard/L4096'];stress=s['groups']['boundary_pool/L4096'];tim=s['model_timing'];sec=s['secondary']
def interval(m):return f"{m['mean']:+.5f} [{m['ci95'][0]:+.5f}, {m['ci95'][1]:+.5f}]"
def correct(a):return p['arms'][a]['outcomes']['both_correct']+p['arms'][a]['outcomes']['gain']
rows='| Arm | Correct / 192 | Gold choice NLL change, nats [95% CI] | Regressions | Gains | All choice flips |\n|---|---:|---|---:|---:|---:|\n| B | 104 / 192 | Reference | — | — | — |\n'
for a,r in p['arms'].items():
 o=r['outcomes'];rows+=f"| {a} | {correct(a)} / 192 | {interval(r['task_balanced']['delta_choice_nll'])} | {o['regression']} / 104 B-correct | {o['gain']} / 88 B-wrong | {o['flips']['numerator']} / 192 |\n"
opening=(f"On an RTX 5080, replacing only Qwen3-0.6B layer 13's prompt-prefill attention changed {p['arms']['A_PUBLIC']['outcomes']['flips']['numerator']} of 192 standard-set choices with A_PUBLIC and {p['arms']['V4']['outcomes']['flips']['numerator']} with V4. These included {p['arms']['A_PUBLIC']['outcomes']['gain']} and {p['arms']['V4']['outcomes']['gain']} gains against independently computed gold, with no standard-set regressions; a separately selected boundary stress set did contain regressions. The task-balanced mean gold-NLL changes had intervals spanning zero. This case retains the score changes, decision transitions and actual model-prefill costs rather than treating similar aggregate accuracy as equivalent behavior.")
timingrows='| Arm | Complete model-prefill speedup [95% CI] | Wall median / block-mean p95 (ms) |\n|---|---|---|\n'
for a in ('B','A_PUBLIC','V4'):
 l=tim['latencies'][a]['wall_seconds'];t=tim.get(a)
 speed='1.000x (control)' if not t else f"{t['paired_ratio_median']:.4f}x [{t['ci95'][0]:.4f}, {t['ci95'][1]:.4f}]"
 timingrows+=f"| {a} | {speed} | {1000*l['median']:.3f} / {1000*l['p95']:.3f} |\n"
readme=f'''# Decision Stability Audit for Low-Precision Attention

Controlled Qwen prefill interventions, paired answer scores and reproducible evidence.

{opening}

**Measured scope:** Qwen3-0.6B BF16, zero-based layer 13, B1 / Hq16 / Hkv8 / D128, causal L4096. All other layers and every decode step remain BF16. **Status: COMPLETED_CONTROLLED_STUDY; deployment NOT_ASSESSED.** This is a narrow synthetic forced-choice study, not evidence of general model-quality preservation.

## Primary result

{rows}
Positive NLL change is worse. Intervals are paired, task-stratified, scenario-cluster bootstrap intervals (5000 draws). A flip can be a gain, regression or different wrong answer. A_PUBLIC/V4 had 3/4 wrong-to-wrong flips. Zero regressions here does not establish zero risk: the BF16-conditioned stress set had {stress['arms']['A_PUBLIC']['outcomes']['regression']}/{stress['arms']['V4']['outcomes']['regression']} regressions among {stress['independent_scenarios']} selected scenarios.

![Paired gold score changes](figures/01_score_changes.png)

## Question and implementation

When does a small attention perturbation change answer scores or choices even if aggregate accuracy looks similar? Three fact-first task families have independent exact gold: key retrieval, value comparison and bounded Python arithmetic. Native one-token option logits are scored without supplying a gold or future continuation. A scoped adapter changes only the chosen prompt-prefill operator, restores the original path after exceptions, and records full-output validity and actual kernel routes.

A_PUBLIC uses the pinned SageAttention INT8-QK / FP8-PV bundle; V4 uses INT8-QK / FP16-PV. These are bundles, not an isolated accumulator experiment or a monotonic precision scale. The model weights and output head remain BF16.

## Read the evidence

- [Offline explorer](demo/index.html): download/open the HTML in a browser. It embeds the measured data and figures, uses no network, and filters exact items by task, set, length, arm and outcome. GitHub's source view is not a running demo.
- **30 seconds:** read the standard outcome table; inspect one item’s gold, four probabilities and allowed-label mass; switch to the separately conditioned stress set. Do not interpret its flip rate as a workload rate.
- [Analysis](ANALYSIS.md), [methods](METHODS.md), [raw evidence index](results/README.md), [summary](results/study/summary.json), [paired item records](results/study/pairs.json).
- [English/Korean portfolio card](PORTFOLIO.md), [Korean explanation](README.ko.md), [limitations](LIMITATIONS.md).

## Model-level cost

{timingrows}
These are new measurements of complete prompt forward with a last-position LM head: 12 fixed scenarios, three processes, five paired blocks of five calls. They are not Case005's operator-only speedups or server TTFT. Block-mean p95 is not service p95. All arms retain the same BF16 weights.

## Limits that affect interpretation

BF16 answered retrieval 59/64, comparison 30/64 and code 15/64 correctly. Weak code utility and shared templates limit conclusions. Inputs use repeated irrelevant filler for exact length; they are not customer documents. Only one model/layer/device and two fixed interventions were measured. Local numerical fidelity is not task accuracy, and non-significant mean differences do not prove equivalence.

The separate 24-scenario JSON-generation diagnostic had **0/24 strict schema-valid outputs in every arm**. Markdown fences, unsupported output format and truncated responses remain in the evidence. It supports bounded token-path observations, not a successful structured-output task benchmark. No output repair or prompt retuning was applied.

## CPU reproduction

Use the tested CPU dependencies in `requirements-cpu.txt`; no GPU, model download or network is needed to reanalyze retained scalar evidence. Run from this case directory, with new output locations outside it:

```bash
python -B -m unittest discover -s tests -v
python -B scripts/analyze_study.py --records results/raw/paired/*.json --timing results/raw/timing/*.json --secondary results/raw/secondary/*.json --output /tmp/case006-analysis-new
python -B scripts/audit_study.py --records results/raw/paired/*.json --summary /tmp/case006-analysis-new/summary.json --output /tmp/case006-audit-new.json
python -B scripts/audit_timing.py --timing results/raw/timing/model-timing.json --summary /tmp/case006-analysis-new/summary.json --output /tmp/case006-timing-audit-new.json
python -B scripts/plot_study.py --summary /tmp/case006-analysis-new/summary.json --pairs /tmp/case006-analysis-new/pairs.json --output /tmp/case006-figures-new
```

[REPRODUCTION.md](REPRODUCTION.md) includes demo, package and pinned-local GPU commands. CPU consistency checks cannot independently reproduce omitted full tensors or kernel execution. The frozen design and evaluation manifests remain unchanged; reporting files were completed afterward.

## Attribution

Qwen, Transformers and SageAttention provide the model and kernels. This case contributes controlled intervention, gold-grounded paired evaluation, explicit validity handling, CPU auditing and an offline evidence explorer. Codex assisted implementation, local execution, analysis and documentation. [NOTICE](NOTICE.md) records pinned sources and reuse from Cases004/005. Case005's STOP_DEV_SCREEN remains unchanged. No third-party GPU reproduction or human review is claimed.
'''
(case/'README.md').write_text(readme)
analysis=f'''# Analysis

## What the controlled study found

{opening}

{rows}

The mean paired NLL effects are small compared with their intervals. V4's four gains are observations from this fixed input family, not a broadly validated improvement. No arm is selected as a winner or approved for deployment. The secondary generation floor is an important negative result: all three arms failed the exact JSON schema on all 24 selected scenarios.

## Standard, selected stress and dependent lengths

| Set / length | Independent base scenarios in this table | A_PUBLIC regressions / gains / flips | V4 regressions / gains / flips |
|---|---:|---|---|
'''
for key,g in s['groups'].items():
 vals=[]
 for a in ('A_PUBLIC','V4'):
  o=g['arms'][a]['outcomes'];vals.append(f"{o['regression']} / {o['gain']} / {o['flips']['numerator']}")
 analysis+=f"| {key} | {g['independent_scenarios']} | {vals[0]} | {vals[1]} |\n"
analysis+='''
The 48 base scenarios at each shorter length are reused standard IDs, not 96 new independent samples. The 46 stress scenarios were selected from a separate 192-item BF16-only pool before candidate access. The pool had 12 exact ties, no nonzero gap below 0.1, and sparse retrieval bins. Descriptively, the smallest positive observed gap was 0.125; native BF16 logits limit score resolution, and FP64 analysis cannot recover discarded bits. No bins were refilled. Stress regressions show why the zero standard-set count is not a safety guarantee.

## Task utility and format mass

| Task | BF16 correct | A_PUBLIC correct | V4 correct | A_PUBLIC NLL change [95% CI] | V4 NLL change [95% CI] |
|---|---:|---:|---:|---|---|
'''
for task in ('RETRIEVAL','COMPARISON','CODE'):
 a=p['arms']['A_PUBLIC']['per_task'][task];v=p['arms']['V4']['per_task'][task]
 def nll(t):return f"{t['mean_delta_nll']:+.5f} [{t['delta_nll_ci95'][0]:+.5f}, {t['delta_nll_ci95'][1]:+.5f}]"
 analysis+=f"| {task} | {a['B_correct']}/64 | {a['candidate_correct']}/64 | {v['candidate_correct']}/64 | {nll(a)} | {nll(v)} |\n"
for arm,r in p['arms'].items():
 analysis+=f"\n{arm}: mean paired Brier change {interval(r['task_balanced']['delta_brier'])}; gold-margin change {interval(r['task_balanced']['delta_gold_margin'])}. Mean allowed-label mass B/candidate: {r['B_label_mass']['mean']:.8f}/{r['candidate_label_mass']['mean']:.8f}; full-vocabulary argmax was an allowed label for {r['B_full_argmax_allowed']}/{r['candidate_full_argmax_allowed']} of 192. Conditional choice normalization is therefore visible rather than silently replacing format compliance.\n"
analysis+='''
## Local error and score propagation

| Arm | Median / p95 pooled local error versus FP32 | Mean full-vocabulary KL(B || candidate) |
|---|---|---:|
'''
for arm,r in p['arms'].items():
 e=r['local_reference_error'];analysis+=f"| {arm} | {100*e['median']:.3f}% / {100*e['p95']:.3f}% | {r['full_vocab_kl']['mean']:.7f} |\n"
analysis+='''
These item-level errors pool 32 sampled queries across 16 heads. They are not Case005's distribution of document×head units, and no 1%/3% gate is imported. The reference uses common BF16 inputs with FP32 arithmetic; its output is not injected into the model. Native-BF16 differences, last-query differences and hidden-boundary absolute/reference RMS are retained separately. Different boundary denominators prevent reading a percentage ratio as an absolute amplification factor.

The local-error/scoring scatter and gap/flip bins are exploratory associations. R requires candidate as well as B logits; it cannot predict risk before execution or save inference cost. Exact BF16 ties remain flagged. No predictive classifier or AUC claim is made.

## Model cost and generation

'''+timingrows
for a in ('B','A_PUBLIC','V4'):
 r=sec['arms'][a];analysis+=f"\n{a}: strict generation schema {r['schema_valid']}/24; scored answer/evidence {r['answer_correct']}/{r['evidence_correct']}; median output length {r['length']['median']:.1f}; median warmed request {r['request_seconds']['median']:.4f} s. Diagnostic and uninstrumented greedy tokens matched on {r['timed_tokens_repeat_matches']}/24.\n"
for a in ('A_PUBLIC','V4'):
 r=sec['arms'][a];analysis+=f"\n{a}: common-B-prefix argmax disagreement in {r['common_prefix_disagreements']}/24; free-running divergence in {r['free_running_divergences']}/24; eight-token re-alignment in {r['eight_token_realignments']}/24. A later token match would not establish state/cache recovery.\n"
analysis+='''
Generation prompts differ from one-token scoring prompts. Invalid formatting is not proof that every underlying factual choice was wrong; no post-hoc repair converts it into a successful task. Responses at the 64-token cap are censored. Different lengths make elapsed-generation comparisons unequal-work observations. No isolated decode-time estimate was collected.

## Validity and evidence limits

All completed primary arm records passed full-output validity checks. The original overstrict LM-head shape diagnostic and its resolution remain in provenance. Frozen measurement code was not changed after evaluation began. The independent checker recalculates scores, norm errors, correctness transitions and 5000-draw intervals from public scalar records; a separate checker repeats timing calculations. Neither is an independent third-party GPU reproduction. Full-vocabulary KL and omitted hidden vectors have a narrower public audit boundary.

The initial task family, one model and one replaced layer bound this conclusion. BF16's low comparison/code accuracy and the failed JSON-generation interface prevent a broad utility claim. No non-inferiority margin, deployment tolerance or production verdict was specified. The appropriate next decision is whether to design a separate, more useful task/interface validation study—not to promote either bundle from these results.
'''
(case/'ANALYSIS.md').write_text(analysis)
(case/'opening.txt').write_text(opening+'\n')
