"""Bounded local checkpoint build, reload and comparison commands."""
from pathlib import Path
import argparse
from .common import write,read,sha,require,inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('inspect',help='Fingerprint an existing local source; no download')
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--revision',required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('inspect-artifact',help='Strict standalone dense tensor/structure check')
    p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('reload-test',help='Fresh process dense prompt and decode')
    p.add_argument('--artifact',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('build-q',help='Use the isolated pinned compressor environment')
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',action='store_true')
    p=commands.add_parser('evaluate',help='Frozen native final-token scores; GPU required')
    p.add_argument('--track',choices=['Q','R'],required=True);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--arm');p.add_argument('--baseline',type=Path);p.add_argument('--split',choices=['smoke','development','heldout'],default='heldout')
    p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('benchmark',help='Three explicitly numbered fixed workload rounds')
    p.add_argument('--track',choices=['Q','R'],required=True);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--round',type=int,choices=[1,2,3],required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='inspect':
        require(args.snapshot.name==args.revision,'Source revision mismatch')
        # HF snapshots may contain file symlinks; source hashes dereference them.
        result={'revision':args.revision,'config':read(args.snapshot/'config.json'),
                'files':{p.name:sha(p) for p in sorted(args.snapshot.iterdir()) if p.is_file()},'status':'PASS'}
    elif args.command=='inspect-artifact':
        if read(args.artifact/'artifact.json')['artifact_type']=='quantized_standard':
            from .qartifact import inspect
            result=inspect(args.artifact)
        else:
            from .artifact import inspect_artifact
            result=inspect_artifact(args.artifact)
    elif args.command=='reload-test':
        from .reload import reload_test
        result=reload_test(args.artifact,args.input,args.device)
    elif args.command=='build-q':
        from .quantized import build
        build(args.snapshot,args.output,args.fixture);return
    elif args.command=='evaluate':
        if args.track=='Q':
            from .q_runtime import run
            run(args.artifact,args.output,args.split)
        else:
            from .r_study import evaluate
            require(args.arm is not None and args.split!='development','R requires arm and smoke/heldout')
            evaluate(args.artifact,args.arm,args.output,args.baseline,args.split)
        return
    elif args.command=='benchmark':
        if args.track=='Q':
            from .q_runtime import run
            run(args.artifact,args.output,'heldout',timing_round=args.round)
        else:
            from .timing import r_round
            r_round(args.artifact,args.output,args.round)
        return
    write(args.output,result);print(result['status'])


if __name__=='__main__':main()
