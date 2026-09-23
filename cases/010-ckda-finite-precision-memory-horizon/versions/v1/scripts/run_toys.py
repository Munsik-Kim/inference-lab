"""Run the fully frozen Case010 Phase A toy sweep on CPU."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codec.toy import run_protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="N=8,T=64 validation; never primary evidence")
    parser.add_argument("--include-dev", action="store_true", help="also report the frozen optional DEV ranking split")
    args = parser.parse_args(argv)
    def progress(key, result):
        if result["status"] == "ok":
            print(json.dumps({"arm": key, "status": "ok", "runtime_seconds": result["runtime_seconds"],
                              "first_failures": result["summary"]["observed_failures"]}), flush=True)
        else:
            print(json.dumps({"arm": key, "status": "error", "error": result["error"]}), flush=True)
    manifest = run_protocol(args.output, smoke=args.smoke, include_dev=args.include_dev, progress=progress)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return int(bool(manifest["failed_arms"]))


if __name__ == "__main__":
    raise SystemExit(main())
