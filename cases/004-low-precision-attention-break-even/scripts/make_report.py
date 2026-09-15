"""Post-measurement presentation only; never changes the frozen experiment."""
import argparse
import json
import sys
import shutil
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import CASE, verify_spec, write_json


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |',
                      '| ' + ' | '.join(['---']*len(header)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def present_figures(summary, analysis_dir, output_dir):
    """Publication styling only. Keep frozen numerical analysis unchanged."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=output_dir/'figures';figures.mkdir(parents=True,exist_ok=True)
    for name in ['qwen_output_error.png','selection_table.png','kernel_vs_complete.png']:
        shutil.copy2(analysis_dir/'figures'/name,figures/name)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
    for row,causal in enumerate([True,False]):
        for col,dim in enumerate([64,128]):
            ax=axes[row,col]
            es=[e for e in summary['entries'] if e['shape']['family']=='synthetic'
                and e['shape']['causal']==causal and e['shape']['dim']==dim]
            x=[e['shape']['length'] for e in es]
            y=np.array([e['speed']['speedup'] for e in es])
            lo=np.array([e['speed']['ci95'][0] for e in es])
            hi=np.array([e['speed']['ci95'][1] for e in es])
            ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o',capsize=3)
            ax.axhline(1,color='grey',label='equal complete cost')
            ax.axhline(1.1,color='grey',linestyle=':',label='1.10 timing threshold')
            ax.set_xscale('log',base=2);ax.set_xlim(440,19000)
            ax.set_xticks(x,[str(n) for n in x],rotation=25)
            ax.set_ylim(0,3.2);ax.grid(alpha=.2)
            ax.set_title(f'D={dim}, causal={causal}')
            ax.set_xlabel('Length');ax.set_ylabel('Median paired BF16 / Sage cost')
    axes[0,0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Synthetic H8/H8 only: complete operator, pointwise 95% intervals')
    fig.tight_layout();fig.savefig(figures/'complete_speedup.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--analysis-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    spec,spec_sha=verify_spec()
    summary=json.loads((a.analysis_dir/'results/selection_table.json').read_text())
    assert summary['spec_sha256']==spec_sha and summary['all_confirmation_rounds_completed']
    raw=[json.loads(p.read_text()) for p in sorted((CASE/'results/confirmation').glob('round-*.json'))]
    first=raw[0]['measurements']
    real=[e for e in summary['entries'] if e['shape']['family']=='qwen']
    synthetic=[e for e in summary['entries'] if e['shape']['family']=='synthetic']
    main_rows=[];cost_rows=[];error_rows=[];details=[];worst=[]
    for e in real:
        n=e['shape']['length'];bs=e['backend_summaries'];b=bs[e['baseline']];s=bs['sage'];sp=e['speed'];ns=s['numerical']
        main_rows.append([n,f"{sp['speedup']:.3f}× [{sp['ci95'][0]:.3f}, {sp['ci95'][1]:.3f}]",
                          f"{100*b['numerical']['median']:.3f}%",f"{100*ns['median']:.3f}%",f"{100*ns['p95']:.3f}%",e['status']])
        for name in [e['baseline'],'sage','kernel_only']:
            z=bs[name];w=z['wall_block_mean_ms'];d=z['event_block_mean_ms']
            cost_rows.append([n,name,f"{w['median']:.4f}",f"{w['p95']:.4f}",f"{d['median']:.4f}",f"{d['p95']:.4f}",
                              'not measured' if z['peak_allocated_bytes'] is None else f"{z['peak_allocated_bytes']/2**20:.2f}"])
        for name in [e['baseline'],'sage']:
            records=[m for m in first if m['shape']['id']==e['shape']['id'] and m['backend']==name]
            units=[u for m in records for u in m['errors_by_head']]
            z=dict(length=n,backend=name,units=len(units),documents=len(records),
                   median_relative_error=float(np.median([u['relative_output_error'] for u in units])),
                   p95_relative_error=float(np.quantile([u['relative_output_error'] for u in units],.95)),
                   median_absolute_rms_error=float(np.median([u['absolute_rms_error'] for u in units])),
                   p95_absolute_rms_error=float(np.quantile([u['absolute_rms_error'] for u in units],.95)),
                   median_cosine=float(np.median([u['cosine'] for u in units])),
                   invalid_rows=sum(u['invalid_rows'] for u in units),near_zero_units=sum(u['near_zero'] for u in units),
                   document_cluster_uncertainty=bs[name]['document_cluster_uncertainty'])
            details.append(z)
            error_rows.append([n,name,z['units'],f"{100*z['median_relative_error']:.3f}%",f"{100*z['p95_relative_error']:.3f}%",f"{z['median_absolute_rms_error']:.6f}",f"{z['median_cosine']:.6f}"])
        candidates=[(u['relative_output_error'],m,u) for m in first
                    if m['shape']['id']==e['shape']['id'] and m['backend']=='sage' for u in m['errors_by_head']]
        _,m,u=max(candidates,key=lambda t:t[0]);base=next(x for x in first if x['shape']['id']==e['shape']['id'] and x['document_id']==m['document_id'] and x['backend']==e['baseline'])
        bu=next(t for t in base['errors_by_head'] if t['head']==u['head'])
        worst.append(dict(length=n,document_id=m['document_id'],head=u['head'],sage_relative_error=u['relative_output_error'],
                          baseline_relative_error=bu['relative_output_error'],absolute_rms_error=u['absolute_rms_error'],cosine=u['cosine'],
                          evidence='results/confirmation/round-0.json',positions=m['positions']))
    main_table=table(['Length','Complete speedup [95% CI]','BF16 median error','Sage median error','Sage p95 error','Decision'],main_rows)
    synth_rows=[];crossovers=[]
    for e in synthetic:
        sh=e['shape'];sp=e['speed'];s=e['backend_summaries'];base=s[e['baseline']]['wall_block_mean_ms']['median'];low=s['sage']['wall_block_mean_ms']['median']
        synth_rows.append([sh['length'],sh['dim'],sh['causal'],e['baseline'],f'{base:.4f}',f'{low:.4f}',f"{sp['speedup']:.3f}",f"[{sp['ci95'][0]:.3f}, {sp['ci95'][1]:.3f}]",e['status']])
    for dim in [64,128]:
        for causal in [True,False]:
            points=[e for e in synthetic if e['shape']['dim']==dim and e['shape']['causal']==causal]
            wins=[e['shape']['length'] for e in points if e['speed']['speedup']>1]
            gates=[e['shape']['length'] for e in points if e['speed']['speedup']>=1.1 and e['speed']['ci95'][0]>1]
            crossovers.append(dict(dim=dim,causal=causal,measured_speedup_above_one= wins,measured_timing_gate_pass=gates,
                                   first_observed_latency_win=min(wins,default=None),fidelity_eligible_lengths=[]))
    counts=dict(Counter(e['status'] for e in summary['entries']))
    samples=[s for r in raw for s in r['gpu_samples']['samples']]
    telemetry={k:dict(min=min(s[k] for s in samples),max=max(s[k] for s in samples)) for k in ['temperature_c','graphics_clock_mhz','gpu_utilization','device_used_bytes']}
    peak_alloc=max(m.get('peak_allocated_bytes',0) for r in raw for m in r['measurements'])
    peak_reserved=max(m.get('peak_reserved_bytes',0) for r in raw for m in r['measurements'])
    invalid=sum(u['invalid_rows'] for r in raw for m in r['measurements'] for u in m.get('errors_by_head',[]))
    near=sum(u['near_zero'] for r in raw for m in r['measurements'] for u in m.get('errors_by_head',[]))
    captured=json.loads((CASE/'provenance/confirmation_capture.json').read_text())
    report=dict(reporting_scope='Post-measurement derived presentation; frozen protocol and primary rules unchanged',
                spec_sha256=spec_sha,shape_status_counts=counts,process_rounds=len(raw),
                backend_input_records=sum(len(r['measurements']) for r in raw),invalid_rows_across_repeated_checks=invalid,
                near_zero_units_across_repeated_checks=near,real_fidelity=details,worst_real_units=worst,
                synthetic_crossovers=crossovers,telemetry=telemetry,operator_peak_allocated_bytes=peak_alloc,
                operator_peak_reserved_bytes=peak_reserved,capture_peak_allocated_bytes=captured['torch_peak_allocated_bytes'])
    write_json(a.output_dir/'results/report_details.json',report)
    candidate_count=counts.get('LOCAL_CANDIDATE',0)
    lead=(f"The measured SageAttention path was faster on some shapes, but none of the three real-Qwen conditions passed the fixed local numerical screen. "
          f"Sage's median sampled-query output error was {min(e['numerical']['median'] for e in real)*100:.2f}–{max(e['numerical']['median'] for e in real)*100:.2f}%, above the 1% limit. "
          f"The selection table therefore contains {candidate_count} local candidates. This is a completed operator comparison, not a model-quality or deployment verdict.")
    readme=f'''# Low-Precision Attention Break-Even on RTX 5080

Can official SageAttention INT8-QK/FP8-PV attention beat the fastest development-verified BF16 SDPA adapter after including all preprocessing, while meeting a fixed local output-error screen on real Qwen inputs?

{lead}

{main_table}

BF16 means the shape's frozen default/forced-Flash choice, not math fallback. Speed is the median paired complete-operator wall-cost ratio over five process rounds. Error uses a common FP32 reference and round 0 only: 16 new synthetic documents, 16 heads each, 32 sampled queries per head. The screen is median ≤1%, p95 ≤3%, with no invalid or unresolved near-zero units. These are local engineering limits, not downstream task-quality tolerances.

## What ran

- RTX 5080 SM120, WSL2 Ubuntu 24.04.4, driver 610.47; Python 3.12.14, Torch 2.13.0+cu130, Triton 3.7.1, Transformers 5.17.0, SageAttention 2.2.0.
- All 24 synthetic shape conditions and all three real-Qwen GQA geometries ran in all five confirmation processes. There were no OOM or unsupported shape results. Synthetic results remain `SYNTHETIC_ONLY`; they do not share Qwen's H16/H8 head geometry.
- Full layer-13 Q/K/V from pinned Qwen3-0.6B; all 48 confirmation captures matched native BF16 replay bitwise. Full activations and weights are not distributed.
- Actual fused BF16 and INT8/FP8 kernel execution was profiled outside timing. Matching static GPU instructions were inspected; they are not dynamic instruction counters.
- Complete cost includes smoothing, quantization, layout/padding, attention, output conversion and wrapper overhead. Kernel-only figures are explicitly auxiliary.

## Experimental design

The experiment separates two questions: whether the full operator is cheaper at a given shape, and whether its local attention output remains within a fixed numerical tolerance. A fast synthetic result alone cannot answer the second question for Qwen.

| Component | Fixed design | Purpose |
| --- | --- | --- |
| Synthetic sweep | Batch 1, Hq=Hkv=8; D=64/128; lengths 512, 1024, 2048, 4096, 8192, 16384; causal and noncausal: 24 conditions | Map measured cost and support across shapes, without treating random tensors as model-quality evidence. |
| Real Qwen inputs | Qwen3-0.6B, layer 13, batch 1, Hq=16/Hkv=8, D=128; causal lengths 512, 2048, 4096 | Measure complete-operator cost on real head geometry and local output error against the same reference. |
| Development | Eight existing synthetic documents; separate Gaussian seeds | Check capture and backend semantics and choose the BF16 path before confirmation. |
| Confirmation | Sixteen new synthetic documents: five English, five Korean, six code | Evaluate the fixed choices on document IDs and prefixes not used for development. |
| Common contract | GPU-resident BF16 Q/K/V → BF16 O, same scale, causal mask and GQA mapping, dropout=0 | Include necessary conversions and preprocessing without changing the operator interface. |
| Repetition | Five fresh processes; 20 warmups and 20 blocks of 10 calls per backend/input in each process | Retain timing variability and balanced backend order, rather than select a favorable run. |

The model revision, exact input tokens, sampling positions, source hashes, thresholds and baseline choices are in the [frozen protocol](configs/experiment_spec.json). The [input manifest](inputs/manifest.json) and [model manifest](provenance/model.json) make the inputs recapturable without distributing weights or full activations. This is a local protocol fixed before confirmation, not an external preregistration.

Both BF16 choices were verified fused SDPA executions: default dispatch and explicitly forced Flash. For each shape, development timing selected one and confirmation retained that choice. The reference is therefore the faster of these two tested adapters in development, not a claim to have found the fastest BF16 implementation across all libraries. No math fallback was used for the speed denominator.

Real-Qwen timing uses **every query position** in the captured tensor. Only error calculation samples 32 fixed queries per head, each against every causal-valid key. The capture is post-QK-normalization and post-RoPE; GQA heads are not expanded. Model capture and operator timing run in separate processes.

At each length, 16 documents × 16 heads give 256 document/head units. They are repeated observations within 16 document clusters. Lengths are prefixes of the same documents; the five timing processes do not increase the independent document count. The inputs share a narrow self-authored template family, so unique IDs do not establish broad domain coverage.

## Validation and evidence

| Check | Recorded outcome | What it establishes |
| --- | --- | --- |
| CPU harness tests | 28 passed; imports of Torch, SageAttention and Transformers blocked | Small FP64 references, mask/GQA examples, near-zero handling, timing boundaries, paired resampling, threshold boundaries, hash checks and rejection of mock timing records. |
| Explicit GPU Gate 0 | 12 checks passed | Actual BF16/Sage execution, output shape/dtype, explicit scale, GQA mapping, causal behavior and unchanged inputs on small development fixtures. |
| Capture versus native BF16 | All 48 confirmation captures bitwise equal to BF16 adapter replay | The captured full attention inputs reproduce the native interface under the tested configuration. |
| Low-precision execution | Profiler specialization matched compiled INT8 QK and FP8 PV instruction-bearing symbols | The candidate runs low-precision matrix instructions; it is not FP8 storage converted back to BF16 matmul. Static instruction occurrences are not dynamic call counts. |
| Complete versus isolated call | Kernel-only output matched the public wrapper bitwise for each measured input | The auxiliary timing corresponds to the same kernel result, while excluding preparation costs explicitly. |
| Confirmation execution | Five processes completed; 1,440 backend/input records, no OOM or unsupported condition | Coverage of all 24 synthetic and three Qwen geometries. This record count is not an independent sample count. |
| Scalar recomputation | 14,400 repeated unit errors and 594 aggregate values matched | Recorded row norms, relative/RMS errors and latency/error summaries are internally consistent under a separate scalar calculation. |
| Reanalysis and integrity | Tables/figures regenerated; protocol and input hashes verified | Public outputs can be traced to the included records without rerunning GPU measurements. |

The [GPU probe](provenance/backend_probe.json), [instruction evidence](provenance/instruction_evidence.json), [artifact audit](provenance/artifact_audit.json) and [execution log](provenance/validation.log) support these checks. They were performed with Codex assistance; they are not a claim of independent human or third-party GPU reproduction. The numerical audit recomputes metrics from recorded row norms. Independently recalculating attention outputs requires recapture using the supplied inputs and model revision.

The small Gate 0 tolerance was a coarse semantic check, separate from the 1% median/3% p95 confirmation screen. Passing that check, producing finite output, or obtaining high cosine similarity did not override a failed confirmation screen. The development error was already above 1%; the confirmation threshold was kept unchanged.

## What the result implies

1. **Kernel-only speed can give the wrong adoption signal.** At Qwen length 512, the prequantized internal call had a 0.0202 ms wall median, versus 0.0383 ms for BF16. The complete Sage call took 0.0996 ms. Excluding preparation would turn an observed full-call slowdown into an apparent advantage. These medians are descriptive; the primary speedup uses paired block ratios.
2. **Longer inputs can amortize the complete-call cost.** Qwen lengths 2048 and 4096 reached 1.465× and 2.100× paired median speedup, including preprocessing. This identifies measured points, not a universal sequence-length threshold. Several smaller synthetic conditions had wide intervals, so a single smooth crossover rule is not supported.
3. **Latency and local fidelity lead to different answers.** All three Qwen lengths exceeded both error limits. BF16-reference median errors were about 0.17%, versus 2.30–2.87% for Sage. Under this experiment's combined rule, the eligible real-Qwen region is empty even where timing improves.
4. **The rejection is local and conditional.** It does not establish a downstream task-quality loss, reject other SageAttention precision settings, or generalize to other layers, models or GPUs. A separate test would be needed to decide whether these output differences matter for a specific task.
5. **Causal masking does not guarantee prefix-invariant approximation.** A separate development stress test changed future V ranges and changed earlier Sage outputs through sequence-wide quantization statistics. It was not pooled into ordinary-input fidelity; no decode or prefix-invariance guarantee follows from this prefill experiment.

The practical outcome is to keep BF16 for this exact real-Qwen configuration. A useful follow-up would compare a separately fixed precision setting against the same complete-cost and fidelity criteria. No such follow-up, new kernel or automatic model routing was implemented here.

![Complete operator speedup on synthetic shapes](figures/complete_speedup.png)
![Real Qwen sampled-query output error](figures/qwen_output_error.png)

Read [ANALYSIS](ANALYSIS.md) for all 24 speed conditions, kernel-only costs, error tails and observed crossovers; [METHODS](METHODS.md) for boundaries and statistics; and [selection_table.csv](results/selection_table.csv) or [JSON](results/selection_table.json) for exact decisions. The [shape table](figures/selection_table.png) is a compact visual index. Raw [round 0](results/confirmation/round-0.json), [1](results/confirmation/round-1.json), [2](results/confirmation/round-2.json), [3](results/confirmation/round-3.json), [4](results/confirmation/round-4.json) retain all block timings, numerical records and device samples.

## Reproduce the recorded analysis

From this case directory, use NumPy 2.3.5 and Matplotlib 3.10.8 for analysis. CPU tests import neither Torch nor SageAttention:

```bash
python -m unittest discover -s tests -v
sha256sum -c SHA256SUMS
python scripts/analyze.py --output-dir /tmp/case004-analysis
python scripts/make_report.py --analysis-dir /tmp/case004-analysis --output-dir /tmp/case004-report
```

Use a fresh output directory for derived files. Analysis verifies the frozen source/input hashes and recomputes unit errors from retained row norms before aggregating.

## Replay GPU measurements

[INSTALL](provenance/INSTALL.md) describes the isolated environment and both build attempts, including the first link failure. Set `CASE004_PYTHON`, `CASE004_WORK` and `QWEN_SNAPSHOT` to your own compatible interpreter, private working directory and existing pinned model snapshot. No weights download is implicit. Preserve the distributed results; use a disposable copy of the case with its `results/confirmation/round-*.json` and `provenance/confirmation_capture.json` moved aside before replay. Those confirmation files are not members of `frozen_files`.

```bash
"$CASE004_PYTHON" scripts/verify_artifacts.py --snapshot "$QWEN_SNAPSHOT" --model-only
"$CASE004_PYTHON" scripts/probe_backend.py --output "$CASE004_WORK/replay-gate0.json"
"$CASE004_PYTHON" scripts/capture_qwen.py --split confirmation \\
  --snapshot "$QWEN_SNAPSHOT" --work-dir "$CASE004_WORK"
for round in 0 1 2 3 4; do
  "$CASE004_PYTHON" scripts/benchmark.py --stage confirmation --round "$round" \\
    --work-dir "$CASE004_WORK" || break
done
```

The preserved model hashes are in [model.json](provenance/model.json). The exact binary/environment fingerprint is intentionally checked: a different installation needs a separately versioned development run and local protocol, not a silent claim to match this run. To create a new protocol, use a fresh experiment copy, generate/check inputs with `prepare_inputs.py`, capture development traces, run `benchmark.py --stage dev --round 0`, and then `freeze_protocol.py`; never replace published confirmation records.

The [selector](scripts/select_backend.py) only looks up an exact environment/model/input-family/spec/shape record and otherwise returns BF16/UNKNOWN. It does not modify model dispatch. [NOTICE](NOTICE.md) credits SageAttention's authors and Codex assistance. This case contains independent measurement code and self-authored synthetic inputs; it does not claim a new attention algorithm, preserved model quality or production readiness.
'''
    (a.output_dir/'README.md').write_text(readme)
    failure_table=table(['Length','Document','Head','Sage error','BF16 error','Absolute RMS','Cosine'],
                        [[w['length'],w['document_id'],w['head'],f"{100*w['sage_relative_error']:.3f}%",f"{100*w['baseline_relative_error']:.3f}%",f"{w['absolute_rms_error']:.6f}",f"{w['cosine']:.6f}"] for w in worst])
    crossover_table=table(['D','Causal','First measured latency win','Lengths passing timing gate','Real fidelity eligibility'],
                          [[x['dim'],x['causal'],x['first_observed_latency_win'] or 'not observed',', '.join(map(str,x['measured_timing_gate_pass'])) or 'none','unverified synthetic geometry'] for x in crossovers])
    analysis=f'''# Analysis

{lead}

## Complete cost and the observed crossover

The following synthetic measurements include the whole public adapter. Intervals are pointwise paired process/block bootstrap intervals. Missing lengths are not interpolated. Both causal curves and head dimensions are separate; a speed result on one is not a result on the others.

{table(['Length','D','Causal','Frozen BF16','BF16 ms','Sage ms','Paired speedup','95% CI','Status'],synth_rows)}

The ms columns are medians of block-mean costs. Their ratio can differ from the reported median of paired ratios.

{crossover_table}

“First” refers only to observed points, not an exact continuous break-even length or a monotonic guarantee. No synthetic shape earns real-model eligibility: H8/H8 is different from the captured Qwen H16/H8 geometry. In the actual Qwen matrix, timing advantages and numerical rejection coexist:

{main_table}

## Prequantized kernel versus complete call

{table(['Qwen length','Backend','Wall median ms','Wall p95 ms','Event median ms','Event p95 ms','Allocator peak MiB'],cost_rows)}

`kernel_only` has prequantized operands and a preallocated output. Every measured input passed bitwise equality with the public wrapper before its auxiliary timing. It is measured after complete-call blocks, not interleaved in the primary comparison. Its p95 is also a block-mean distribution. The difference between kernel-only and complete cost includes more than quantization and is not a stage-level timing decomposition.

![Complete versus kernel-only cost](figures/kernel_vs_complete.png)

Wall and device-event numbers are separate measurements. Host-side scheduling, wrapper dispatch and synchronization can affect wall cost. Process-round median variation is retained in the JSON; no favorable process was selected. The five processes run on one device and desktop environment, not five independent machines.

## Fidelity against the common reference

{table(['Length','Backend','Dependent units','Median error','p95 error','Median absolute RMS','Median cosine'],error_rows)}

Each length has 16 independent document clusters and 256 repeated document/head units. Round 0 defines fidelity; the same inputs in later rounds do not add independent documents. The [detailed JSON](results/report_details.json) includes document-cluster intervals and absolute errors. No invalid rows ({invalid}) or near-zero units ({near}) were observed across repeated numerical checks. Passing finite-value checks does not imply that the stricter 1%/3% screen passed.

The largest sampled-output unit error at each length was:

{failure_table}

These examples use all valid keys for each of the 32 fixed query positions. Their high output cosine does not cancel the Frobenius error. Head 12 appears in these worst units, but this experiment does not isolate which of Q/K quantization, P/V quantization or accumulation caused its error. Native-to-adapter BF16 equality and the much smaller BF16-reference errors support a capture/semantic check; they do not prove the approximation is acceptable downstream.

A separate development stress input changed only future V ranges by a factor of 1000. This altered Sage prefix outputs while leaving BF16 unchanged. Range-preserving future permutations did not change the prefix. The [Gate 0 record](provenance/backend_probe.json) preserves this nonlocal effect of global quantization statistics. It is excluded from ordinary-input error summaries and rules out a prefix-invariance claim.

## Memory and execution conditions

Maximum observed operator allocator peak was {peak_alloc/2**20:.2f} MiB allocated and {peak_reserved/2**20:.2f} MiB reserved. Confirmation model capture separately peaked at {captured['torch_peak_allocated_bytes']/2**30:.3f} GiB allocated. These scopes must not be compared as model-weight savings. Complete-call peak includes live input tensors and other allocator state; extension allocations may not all appear in PyTorch counters.

The 100 ms whole-device sampled peak was {telemetry['device_used_bytes']['max']/2**30:.3f} GiB, including WSL/Windows background use. Temperature ranged {telemetry['temperature_c']['min']}–{telemetry['temperature_c']['max']} °C and observed graphics clock {telemetry['graphics_clock_mhz']['min']}–{telemetry['graphics_clock_mhz']['max']} MHz across the samples. No clock or thermal setting was forced. Sampling does not establish an exact peak or isolate unrelated device activity. Original per-round startup and telemetry values remain in the raw files.

## What the result supports

The chosen official SM120 path executes real INT8-QK/FP8-PV attention, and its complete cost can beat verified fused BF16 on measured shapes. Including preprocessing changes the comparison substantially relative to a prequantized internal kernel. At the three captured real-Qwen shapes, the fixed local numerical screen rejects the path. The combined eligible region is therefore empty under this protocol; keep BF16 for the queried real-model entries.

This result is limited to one RTX 5080/software fingerprint, one Qwen3-0.6B layer, selected query rows and a narrow self-authored prompt family. It does not establish downstream quality, model-wide failure, decode behavior, video quality, another accumulator option's behavior, or an optimal attention backend across installations. No new quantization algorithm or automatic production router was built. Thresholds were not relaxed after observing the results.

The next technical question is whether a separately specified precision configuration can reduce the observed output error without losing the measured complete-cost advantage. That comparison was not run here. There is no basis in this case to enable the rejected configuration automatically.
'''
    (a.output_dir/'ANALYSIS.md').write_text(analysis)
    present_figures(summary,a.analysis_dir,a.output_dir)
    print(json.dumps({'statuses':counts,'report_details':str(a.output_dir/'results/report_details.json')}))


if __name__=='__main__':
    main()
