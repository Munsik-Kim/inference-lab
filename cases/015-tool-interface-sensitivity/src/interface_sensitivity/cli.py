"""Case015 local entrypoints; CPU commands do not import Torch or a model."""
import argparse
from pathlib import Path
from .common import CASE, atomic, read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,default=CASE)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('validate',help='Check registered schema, mapping and frozen file identities (CPU).')
    o=sub.add_parser('oracle-check',help='Execute 32 CPU fixture oracle episodes, compare every step.');o.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('analyze',help='Recompute task-paired metrics from stored episode records (CPU).');a.add_argument('--records',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('audit',help='Replay raw responses into isolated fixtures and independently recompute success (CPU).');a.add_argument('--records',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('render',help='Build static bilingual episode explorer from saved records (CPU).');a.add_argument('--records',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('run',help='Run the frozen GPU matrix from read-only local weights; offline, no fitting.');a.add_argument('--private',type=Path,required=True)
    a=sub.add_parser('worker',help=argparse.SUPPRESS);a.add_argument('--private',type=Path,required=True);a.add_argument('--method',choices=('BF16','RTN','D16'),required=True);a.add_argument('--budget',type=float,required=True)
    args=p.parse_args()
    if args.command=='validate':
        from .prepare import validate
        value=validate(args.case)
    elif args.command=='oracle-check':
        from .equivalence import oracle_check
        value=oracle_check(args.case);atomic(args.output,value,True)
        value={k:v for k,v in value.items() if k!='records'}
    elif args.command=='analyze':
        from .analysis import summarize
        from .common import rows
        value=summarize(rows(args.records),read(args.case/'results/p1_r1/planned_cells.json'))
        atomic(args.output,value,True);value={'status':value['status'],'records':value['completed_records'],'primary':value['contrasts']['D16-minus-BF16']}
    elif args.command=='audit':
        from .audit import audit
        value=audit(args.case,args.records);atomic(args.output,value,True)
    elif args.command=='render':
        from .render import render
        value=render(args.case,args.records,args.output)
    elif args.command=='run':
        from .gpu_worker import launch
        value=launch(args.case,args.private)
    else:
        from .gpu_worker import worker
        value=worker(args.case,args.private,args.method,args.budget)
    import json
    print(json.dumps(value,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    main()
