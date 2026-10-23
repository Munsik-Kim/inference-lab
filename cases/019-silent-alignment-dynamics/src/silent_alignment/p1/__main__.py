"""Separate entry point; P0's root CLI is unchanged."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description="Case019 P1 CPU boundary-pair and initialization-size comparison")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("run")
    p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--budget-ledger",type=Path,required=True)
    p=sub.add_parser("verify")
    p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[3]
    if args.command=="run":
        from .runner import run
        status=run(root,args.config,args.output,args.budget_ledger)
        print(status)
        if status["execution_status"]!="COMPLETE" or status["implementation_status"]!="PASS":raise SystemExit(2)
    else:
        from .verify import verify
        r=verify(root,args.run,args.output);print(r["status"],r["trajectories"],"stored trajectories; ODE reruns=0")


if __name__=="__main__":main()
