"""Render this recorded case narrative; not a conclusion template for new runs."""
import json
from prepare import CASE
from analyze import LABELS


def main():
    r=json.loads((CASE/'results/aggregate.json').read_text())
    spec=json.loads((CASE/'configs/experiment_spec.json').read_text())
    cal=json.loads((CASE/'results/calibration.json').read_text())
    details=json.loads((CASE/'results/representative_details.json').read_text())
    index={(x['length'],x['method']):x for x in r['length']}
    methods=['nearest_ceil','nearest_efq_scale','efq_mmlu','efq_mean','efq_balance','efq_lut','efq_calibrated']
    table='| Method | 512 tokens | 2048 tokens | 4096 tokens |\n|---|---:|---:|---:|\n'
    for m in methods:
        cells=[f"{100*index[n,m]['median']:.2f}% / {100*index[n,m]['p95']:.2f}%" for n in [512,2048,4096]]
        table+='| '+LABELS[m]+' | '+' | '.join(cells)+' |\n'
    trans={(x['split'],x['method']):x['relative_frobenius'] for x in r['calibration_transfer']}
    ratios={n:{metric:next(c['ratio'] for c in r['screen']['checks'] if c['length']==n and c['scope']==metric) for metric in ['median','p95']} for n in [512,2048,4096]}
    ratio_table='| Length | Median ratio | p95 ratio |\n|---|---:|---:|\n'+''.join(f"| {n} | {ratios[n]['median']:.3f} | {ratios[n]['p95']:.3f} |\n" for n in ratios)
    ci_table='| Length | Mean document error difference | Paired 95% interval |\n|---|---:|---:|\n'
    for x in r['paired_document_bootstrap']:
        ci_table+=f"| {x['length']} | {x['mean_document_error_difference']*100:+.3f} pp | [{x['ci95'][0]*100:+.3f}, {x['ci95'][1]*100:+.3f}] pp |\n"
    max_group=max(c['ratio'] for c in r['screen']['checks'] if c['scope']=='layer_length_median')
    online_worst=max(x['worst'] for x in r['length'] if x['method']=='online_fp32')
    mean=details[0]; poor=details[2]
    worsts={m:max(index[n,m]['worst'] for n in [512,2048,4096]) for m in methods}
    specsha=(CASE/'configs/experiment_spec.sha256').read_text().split()[0]
    readme=f'''# Independent numerical audit of EFQ-Softmax on Qwen attention

**EFQ-Mean was a poor default on these traces.** Its median attention-output error was 11.49–14.49%, compared with 6.51–8.09% for the development-selected setting. The selected setting passed the prespecified screen against nearest rounding with the same scale. A different, headroom-based nearest scale still had lower median and p95 error at every length. These are attention-output errors, not model accuracy scores.

This case independently implements [EFQ-Softmax v1](https://arxiv.org/html/2609.09721v1) and checks it on actual post-QK-normalization/post-RoPE attention inputs from the official BF16 Qwen3-0.6B model. The inputs are self-authored synthetic English, Korean and code documents. No packed FP4 computation, A5 benchmark, model-quality test or new kernel is involved.

## Recorded comparison

Values below are **median / p95 relative output Frobenius error** over 192 head units per length: 16 held-out documents × 3 layers × 4 heads. Each unit contains 16 prespecified query rows, using all valid keys. Prefix lengths overlap within a document. The independent document count is 16, not the number of heads or query rows.

{table}
Attention tile size is 128 and microscaling block size is 32. “Nearest, EFQ scale” isolates the code mapping from the scale choice and is the prespecified screen baseline. “Nearest, headroom scale” is an independent MXFP4 numerical reference; its equivalence to the paper's MXFP4 implementation was not established. The latter result shows why a favorable same-scale comparison should not be read as an overall accuracy win for EFQ.

![Length error](results/length_error.png)

FP32 online attention agreed with the FP32 dense reference to a worst unit error of {online_worst:.3e}. All real and synthetic evaluation methods had zero valid-row denominator/output/probability numerical failures. No real unit crossed the near-zero reference-norm threshold. This distinguishes approximation error from an implementation crash or a zero-denominator failure.

## Published parameters and development calibration

A 49-candidate search on eight development documents selected **tau={cal['selected']['tau']:.2f}, h={cal['selected']['h']:.2f}**. This is exactly the published EFQ-MMLU point, so those two rows are identical. It is not a new parameterization invented by this project. The search used length 2048, layer 13 and heads 0/10, with no evaluation input or refinement.

On that matched subset definition, the concatenated-output objective changed from **{100*trans['dev','efq_calibrated']:.3f}% development to {100*trans['eval','efq_calibrated']:.3f}% evaluation**. There was no large transfer collapse in this sample. The corresponding same-scale nearest errors were {100*trans['dev','nearest_efq_scale']:.3f}% and {100*trans['eval','nearest_efq_scale']:.3f}%. This pooled objective differs from the all-head median table above.

EFQ-Mean and EFQ-Balance produced higher median errors than the selected/MMLU point at all three lengths. Balance was originally selected for the paper's vision-language experiments. EFQ-LUT improved typical error over Mean, but its worst unit error reached {100*worsts['efq_lut']:.2f}%; finer intermediate indices did not remove the tail. All three published points and the LUT remain reported; none was selected after seeing evaluation outcomes.

## Engineering screen and limits on that decision

The calibrated/same-scale-nearest ratios were:

{ratio_table}
All median ratios were <=1.10, all p95 ratios <=1.25, and the largest layer×length median ratio was {max_group:.3f}, below 2.0. There were no valid-row numerical failures or floor-based ratio holds. **The predefined screen passed.** This supports considering a limited kernel-feasibility follow-up. No such kernel work was started here, and the result does not prove speedup, model-quality preservation or universal robustness.

The paired document bootstrap below concerns the **mean per-document error difference**, calibrated minus same-scale nearest, rather than the median ratio. It resamples 16 documents, preserving all their heads and prefixes. Intervals describe this small shared-generator sample.

{ci_table}
The headroom-scale reference remains a relevant competing design. In addition, individual units can be much worse under calibrated EFQ even though aggregate groups pass: one unit reached {poor['unit']['relative_error']/poor['unit']['same_scale_nearest']:.2f}× the same-scale nearest error ({100*poor['unit']['relative_error']:.2f}% versus {100*poor['unit']['same_scale_nearest']:.2f}%). The screen's grouped thresholds do not protect every head or input.

## Where error grew

The clearest EFQ-Mean example was `{mean['unit']['document_id']}`, length {mean['unit']['length']}, layer {mean['unit']['layer']}, head {mean['unit']['head']}. Its unit error was **{100*mean['unit']['relative_error']:.2f}%**, versus {100*mean['unit']['same_scale_nearest']:.2f}% for nearest rounding with the same scale. At query {mean['largest_row_query_position']}, the secondary key's normalized probability rose from 0.06970 in the reference to 0.10880 under EFQ-Mean; the nearest control gave 0.07553. EFQ assigned code 7 where nearest rounding assigned code 6. Removed reference mass was only 0.356%, and that row had no finite row-max jump. Zero pruning or historical rescaling alone therefore does not explain this example: the nonzero code mapping also redistributes weight.

The largest calibrated unit error was {100*details[1]['unit']['relative_error']:.2f}%, in `eval-15`, length 512, layer 13, head 5. A separate unit with the largest calibrated/nearest ratio showed a secondary probability of 0.07046 becoming 0.13801, with code 2 instead of nearest code 1. Detailed original key probabilities, codes and row diagnostics are in [representative cases](results/representative_cases.md) and [machine-readable details](results/representative_details.json).

![Representative case](results/representative_failure.png)

For calibrated EFQ, row error had exploratory Spearman correlations of +0.559 with removed reference mass and +0.511 with attention entropy, and -0.405 with the largest-logit gap. The correlation with maximum row-max jump was -0.013. These pooled rows are dependent; the associations are not causal explanations or new theoretical results.

![Exploratory score associations](results/error_associations.png)

Layer 0's median error was 1.27–1.31× the same-scale nearest baseline, while layer 27's was 0.76–0.80×. The layer breakdown matters even when the global median looks similar. [Joint layer/head/length summaries](results/layer_head_length.json) retain all groups.

![Layer error](results/layer_error.png)

## Stress and block-size sensitivity

Nine synthetic families were measured separately with distinct development/evaluation seeds. Gaussian scores with standard deviation 2 produced the largest EFQ-Mean synthetic unit error, 28.10% at length 4096. Near-uniform inputs, sharp peaks, heavy tails, causal masks and large late row-max changes were also retained, including unsuccessful approximation cases. The absence of NaN/Inf in these stress tests is not evidence of model quality.

With the same fixed calibration, block 16 gave calibrated median errors of 7.11%, 6.33% and 6.16%; block 64 gave 9.85%, 7.21% and 6.95%. These are sensitivity results. The main screen remains the preselected block 32; the experiment was not relabeled around the best block size after evaluation.

## Reproduce and inspect

- [Methods, equations and exact metric definitions](METHODS.md)
- [Fresh-output reproduction commands](provenance/REPRODUCE.md)
- [Frozen experiment specification](configs/experiment_spec.json), SHA256 `{specsha}`
- [Model revision, file hashes and BF16 tensor verification](provenance/model.json)
- [Environment and versions](provenance/environment.json)
- [Input texts and exact tokenizer IDs](inputs/manifest.json)
- [Aggregate tables, screen and document bootstrap](results/aggregate.json)
- [Per-unit results](results/eval_units.jsonl), [compressed row measurements](results/eval_rows.csv.gz)
- [Numerical tests](provenance/numerical_tests.json), [native trace check](provenance/trace_validation.json)
- [Static artifact checks](provenance/artifact_validation.json); run `python scripts/verify_artifacts.py`

The runs used an RTX 5080 16GB, WSL2 Ubuntu 24.04.4, driver 610.47, Python 3.12.14, PyTorch 2.13.0+cu130 and Transformers 5.17.0. The exact model revision is `c1899de289a04d12100db370d81485cdf75e47ca`. The full attention inputs are private working files; the supplied text, tokens, scripts, hashes and small development fixture support regeneration without distributing model weights or large activations.

Twelve numerical tests passed, including an independent scalar FP64 oracle, GPU/CPU checks, code boundaries, masking, shared operands, online rescaling and scale underflow. The initial FP32 JS diagnostic had tiny negative roundoff values during development; its correction and preserved prior evidence are [documented](provenance/development_corrections.json). All attention-output error records were unchanged by that diagnostic correction. Evaluation began only after the freeze.

These findings are limited to one small model, selected layers/heads/query rows and 24 synthetic documents sharing generation rules. The paper's original block layout, finite-exponent policy and baseline implementation were not independently recovered. This case audits an explicit independent implementation of v1, not the authors' complete experiments. No human annotation review, independent third-party replication, model-quality gain or hardware acceleration is claimed.

The EFQ method and published parameters are Han et al.'s work; Qwen and the inference libraries belong to their respective authors. OpenAI Codex assisted with the inputs, implementation, analysis and documentation. See [NOTICE](NOTICE.md).
'''
    (CASE/'README.md').write_text(readme)
    cases='# Representative approximation errors\n\nThese cases were selected from the completed fixed evaluation by the rules in `analyze.py`. The inspection reuses stored Q/K/V and the frozen arithmetic; it does not run the model again or change parameters.\n\n'
    for d in details:
        u=d['unit'];cases+=f"## {u['method']}: {u['document_id']}, length {u['length']}, layer {u['layer']}, head {u['head']}\n\n"
        cases+=f"Selection: {d['selection']}. Unit error {u['relative_error']:.8f}; same-scale nearest {u['same_scale_nearest']:.8f}. Largest row error at query {d['largest_row_query_position']}: {d['row_diagnostics']['relative_error']:.8f}. See [input document](../inputs/{u['document_id']}.txt).\n\n"
        cases+='| Key | Reference probability | Method probability | Nearest same-scale | Method code | Nearest code | Scale exponent |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for k in d['top16_reference_keys']:
            cases+=f"| {k['key_position']} | {k['reference_probability']:.8f} | {k['method_effective_probability']:.8f} | {k['nearest_same_scale_probability']:.8f} | {k['method_code']} | {k['nearest_code']} | {k['scale_exponent']:.0f} |\n"
        cases+=f"\nRemoved reference mass: {d['row_diagnostics']['removed_reference_mass']:.8f}; score spread: {d['row_diagnostics']['score_spread']:.5f}; entropy: {d['row_diagnostics']['entropy_nats']:.5f}; maximum finite row-max increase: {d['row_diagnostics']['max_row_max_jump']:.5f}. Reference output cancellation ratio: {d['reference_output_cancellation_ratio']:.5f}. These are descriptive diagnostics, not additional pass/fail criteria.\n\n"
    (CASE/'results/representative_cases.md').write_text(cases.rstrip()+'\n')
    print('Wrote English case README and representative cases.')


if __name__=='__main__':main()
