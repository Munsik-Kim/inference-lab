"""Standalone P2 CLI; P0/P1 entry points and source bytes are unchanged."""
import time
LAUNCH_TIME=time.monotonic()
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description="Case019 P2 CPU graph audit and direction interventions")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("run");p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--budget-ledger",type=Path,required=True);p.add_argument("--evidence",type=Path,required=True)
    p=sub.add_parser("verify");p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[3]
    if args.command=="run":
        from .runner import run
        result=run(root,args.config,args.output,args.budget_ledger,args.evidence,LAUNCH_TIME)
        print(result["execution_status"],result["implementation_status"])
    else:
        from .verify import verify
        result=verify(root,args.run,args.output)
        print(result["status"],"stored primary",result["primary_count"],"ODE reruns",result["ODE_reruns"])


if __name__=="__main__":main()
