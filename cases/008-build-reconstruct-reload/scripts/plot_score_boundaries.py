"""Display both pre-specified Q score boundaries; no metric or gate changes."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(summary: Path,output: Path):
    if output.exists():raise ValueError('New output file required')
    row=json.loads(summary.read_text())['tracks']['Q']['arms']['Q-W4']
    fig,ax=plt.subplots(figsize=(8.2,3.3))
    for i,(key,label) in enumerate([('delta_full_gold_nll','Full vocabulary (primary)'),('delta_choice_nll','Four choices (secondary)')]):
        v=row[key];ax.errorbar(v['mean'],i,xerr=[[v['mean']-v['ci95'][0]],[v['ci95'][1]-v['mean']]],fmt='o',capsize=5,label=label)
    ax.axvline(0,color='grey',linewidth=1);ax.set_yticks([0,1],['Full vocabulary\n(primary)','Four choices\n(secondary)'])
    ax.set_xlabel('Gold NLL: W4 minus BF16 (nats; positive worse)')
    ax.set_title('Q: opposite score changes on the same 192 scenarios')
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(output,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();draw(a.summary,a.output)
