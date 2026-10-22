"""CPU paths stay stdlib-only; P1 imports its GPU adapter only when selected."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Case018 P0 CPU synthetic contracts and independent stored-trace audit")
    sub = parser.add_subparsers(dest="command", required=True)
    p0 = sub.add_parser("p0", help="Run CPU contracts and 18 synthetic controls; no model")
    p0.add_argument("--output", required=True, help="Must be a new directory")
    audit = sub.add_parser("audit", help="Independently reconstruct stored CPU raw responses")
    audit.add_argument("--run", required=True)
    audit.add_argument("--output", required=True, help="Must be a new JSON path")
    p1 = sub.add_parser("p1", help="Explicitly authorized existing-runtime P1 technical gates and DEV24 only")
    p1.add_argument("--output", required=True)
    p1.add_argument("--runtime", required=True)
    p1.add_argument("--preparation", required=True)
    p1.add_argument("--evidence", required=True)
    args = parser.parse_args()
    if args.command == "p0":
        from .p0 import execute
        result = execute(args.output)
    elif args.command == "audit":
        from .audit import audit_run
        result = audit_run(args.run, args.output)
    else:
        from .p1 import execute
        result = execute(args.output, args.runtime, args.preparation, args.evidence)
        result = {**result, "status": result["execution_status"]}
    print(json.dumps({k: result[k] for k in ("status", "P0_gate", "controls", "diagnostics", "episodes", "new_model_requests") if k in result}, ensure_ascii=False))
    return 0 if result["status"] in ("PASS", "P1_CPU_READY", "COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
