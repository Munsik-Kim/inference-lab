"""Case015 A1: CPU contract/replay tools; explicit offline GPU boundary."""
import argparse
import json
from pathlib import Path
from .common import ROOT,atomic,read,rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('validate',help='Validate protocol, mappings, inputs and frozen code without a model.')
    for name,help_ in [('oracle-check','Execute equivalent synthetic oracle traces for both APIs.'),('audit','Replay stored measured responses and independently recompute endpoints.'),('analyze','Recompute paired regression table from measured rows.')]:
        a=sub.add_parser(name,help=help_);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('render',help='Build bilingual static measured episode explorer.');a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('register',help='Register new fixed synthetic inputs with existing offline tokenizer, no model forward.');a.add_argument('--private',type=Path,required=True)
    sub.add_parser('freeze',help='Freeze source/spec/input identities before smoke.')
    a=sub.add_parser('run',help='Run 96 registered FREE episodes plus two separate smoke episodes; no fitting or encoding.');a.add_argument('--private',type=Path,required=True)
    a=sub.add_parser('worker',help=argparse.SUPPRESS);a.add_argument('--private',type=Path,required=True);a.add_argument('--method',choices=['BF16','D16','RTN'],required=True);a.add_argument('--budget',type=float,required=True)
    a=p.parse_args()
    if a.command=='validate':
        from .prepare import validate
        value=validate(a.root)
    elif a.command=='oracle-check':
        from .equivalence import oracle_check
        value=oracle_check(a.root);atomic(a.output,value,True)
    elif a.command=='audit':
        from .audit import audit
        value=audit(a.root);atomic(a.output,value,True)
    elif a.command=='analyze':
        from .analysis import summarize
        value=summarize(rows(a.root/'results/a1_r1/episodes.jsonl'),read(a.root/'results/a1_r1/planned_cells.json'));atomic(a.output,value,True)
    elif a.command=='render':
        from .render import render
        value=render(a.root,a.output)
    elif a.command=='register':
        from .prepare import register
        value=register(a.root,a.private)
    elif a.command=='freeze':
        from .prepare import freeze
        value={'execution_source_sha256':freeze(a.root)}
    elif a.command=='run':
        from .gpu_worker import launch
        value=launch(a.root,a.private)
    else:
        from .gpu_worker import worker
        value=worker(a.root,a.private,a.method,a.budget)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
