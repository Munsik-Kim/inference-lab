import argparse
from pathlib import Path
from .io import write_json


def main():
    parser = argparse.ArgumentParser(description="Case019 P0 CPU finite-dimensional silent alignment")
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("p0", help="17 primary trajectories, fixed numerical references and figures")
    pilot.add_argument("--config", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify", help="saved raw -> summary/figures; no ODE solve")
    check.add_argument("--run", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == "p0":
        from .engine import run_pilot
        status = run_pilot(root, args.config, args.output)
        print(status)
        if status["implementation_validation"] != "PASS":
            raise SystemExit(2)
    else:
        from .verify import verify_run
        receipt = verify_run(root, args.run, args.output)
        print(receipt["status"], "17 raw trajectories, 136 checkpoints, no ODE rerun")


if __name__ == "__main__":
    main()
