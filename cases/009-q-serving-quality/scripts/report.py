"""Render measured serving curves and separate official quality tables."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def fmt(x,d=3):
    return f'{x:.{d}f}'


def tables(s,q):
    groups=defaultdict(list)
    for c in s['cells']:
        if c['status']=='COMPLETE':groups[(c['execution'],c['input_tokens'],c['client_concurrency'],c['arm'])].append(c)
    lines=['| Mode / input / concurrency | Arm | TTFT p50 / p95 ms | TPOT p50 / p95 ms/token | Request p50 / p95 ms | Output tokens/s | Requests / failures |',
           '|---|---|---:|---:|---:|---:|---:|']
    for (mode,l,c,arm),rs in sorted(groups.items()):
        mean=lambda k:statistics.fmean(r[k] for r in rs)
        lines.append(f'| {mode} / {l} / {c} | {arm} | {fmt(mean("ttft_p50_ms"))} / {fmt(mean("ttft_p95_ms"))} | {fmt(mean("tpot_p50_ms"))} / {fmt(mean("tpot_p95_ms"))} | {fmt(mean("latency_p50_ms"))} / {fmt(mean("latency_p95_ms"))} | {fmt(mean("output_tokens_per_second"))} | {sum(r["n_requests"] for r in rs)} / {sum(r["failures"] for r in rs)} |')
    ratios=['| Mode / input / concurrency | TPOT time ratio BF16/W4 [95% interval] | Throughput W4/BF16 [95% interval] |','|---|---:|---:|']
    for r in s['contrasts']:
        label=f'{r["execution"]} / {r["input_tokens"]} / {r["client_concurrency"]}'
        if 'tpot_ratio' not in r:ratios.append(f'| {label} | INCOMPLETE | INCOMPLETE |');continue
        form=lambda v:f'{v["mean_ratio"]:.3f} [{v["pointwise_95_interval"][0]:.3f}, {v["pointwise_95_interval"][1]:.3f}]'
        ratios.append(f'| {label} | {form(r["tpot_ratio"])} | {form(r["throughput_ratio"])} |')
    quality=['| Official task / metric | BF16 | W4 | Paired W4−BF16 [pointwise 95% interval] |','|---|---:|---:|---:|']
    transitions=['| Task / metric | n | Lost correct | New correct | Both wrong | Changed wrong | All answer disagreements |','|---|---:|---:|---:|---:|---:|---:|']
    for t in q['tasks']:
        if 'groups' not in t:quality.append(f'| {t["task"]} | {t["arms"]["BF16"]} | {t["arms"]["W4"]} | unavailable |');continue
        if t['status']!='COMPLETE':
            quality.append(f'| {t["task"]} **process status** | {t["process_status"]["BF16"]} | {t["process_status"]["W4"]} | retained computation below; not clean execution |')
        if t['task']=='wikitext':
            r=next(iter(t['groups'].values()));a=r['arms'];lo,hi=r['word_NLL_delta_pointwise_95_interval']
            quality.append(f'| WikiText-2 word perplexity | {a["BF16"]["word_perplexity"]:.4f} | {a["W4"]["word_perplexity"]:.4f} | word NLL {r["word_NLL_delta_nats"]:+.5f} [{lo:+.5f}, {hi:+.5f}] nats |')
            quality.append(f'| WikiText-2 byte perplexity | {a["BF16"]["byte_perplexity"]:.4f} | {a["W4"]["byte_perplexity"]:.4f} | separate original UTF-8 denominator |')
            continue
        if t['task']=='mmlu':
            r=t['overall_paired'];metrics=[('MMLU acc (57 subjects)',r,r['accuracy_delta'],r['pointwise_95_interval'])]
        else:
            metrics=[]
            for key,r in t['groups'].items():
                metrics.append((key+' '+('exact_match' if t['task']=='gsm8k' else 'acc'),r,r['accuracy_delta_candidate_minus_baseline'],r['accuracy_delta_pointwise_95_interval']))
                if 'acc_norm' in r:
                    n=r['acc_norm'];metrics.append((key+' acc_norm',n,n['accuracy_delta_candidate_minus_baseline'],n['accuracy_delta_pointwise_95_interval']))
        for name,r,d,ci in metrics:
            quality.append(f'| {name} | {r["baseline_correct"]}/{r["n"]} ({100*r["baseline_correct"]/r["n"]:.2f}%) | {r["candidate_correct"]}/{r["n"]} ({100*r["candidate_correct"]/r["n"]:.2f}%) | {100*d:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] pp |')
            transitions.append(f'| {name} | {r["n"]} | {r["correct_to_wrong"]} | {r["wrong_to_correct"]} | {r["both_wrong"]} | {r["wrong_to_different_wrong"]} | {r["all_answer_disagreement"]} |')
    memory=['| Server process | Whole-device sampled peak MiB | Sampled maximum running / waiting | Graph capture log |','|---|---:|---:|---|']
    for p in s['processes']:memory.append(f'| {p["process_id"]} | {p["sampled_device_peak_used_MiB"]} | {p["sampled_running_max"]} / {p["sampled_waiting_max"]} | {p["capture_logged"]} |')
    return '\n'.join(lines),'\n'.join(ratios),'\n'.join(quality),'\n'.join(transitions),'\n'.join(memory)


def plots(s,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    for length in (128,1024):
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,metric,label in zip(axes,['tpot_p50_ms','output_tokens_per_second'],['TPOT p50 (ms / generated token)','Output tokens / second']):
            for arm,color in [('BF16','#263747'),('W4','#b66e43')]:
                xs=[];ys=[];los=[];his=[]
                for c in [1,4,16,32]:
                    vals=[r[metric] for r in s['cells'] if r['status']=='COMPLETE' and (r['execution'],r['arm'],r['input_tokens'],r['client_concurrency'])==('graph',arm,length,c)]
                    if len(vals)!=3:continue
                    m=statistics.fmean(vals);xs.append(c);ys.append(m);los.append(m-min(vals));his.append(max(vals)-m)
                ax.errorbar(xs,ys,yerr=[los,his],label=arm,color=color,marker='o',capsize=4)
            ax.set_xticks([1,4,16,32]);ax.set_xlabel('Client max concurrency');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend()
        fig.suptitle(f'Qwen3-4B · RTX 5080 · vLLM 0.29 · graph · {length} input / 256 output\nMean of three server rounds; whiskers show round min–max (not CI)',fontsize=10)
        fig.savefig(out/f'serving-L{length}.png',dpi=160);fig.savefig(out/f'serving-L{length}.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--serving',type=Path,required=True);p.add_argument('--quality',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    s=json.loads(a.serving.read_text());q=json.loads(a.quality.read_text());a.output.mkdir(parents=True,exist_ok=False)
    ts=tables(s,q)
    for name,data in zip(['serving_table','paired_ratios','quality_table','transitions','memory'],ts):(a.output/(name+'.md')).write_text(data+'\n')
    plots(s,a.output/'figures')
    print(json.dumps({'status':'RENDERED','complete_serving_cells':s['complete_cells'],'quality_tasks':{t['task']:t['status'] for t in q['tasks']}}))


if __name__=='__main__':main()
