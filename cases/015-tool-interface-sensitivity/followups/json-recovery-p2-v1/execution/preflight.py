"""Existing local-file and token/schema checks in a CPU-guarded process."""
import argparse
from pathlib import Path
import sys

P2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P2 / 'src'))
from json_recovery_p2.local_inputs import check
from json_recovery_p2.tokenizer_cpu import retokenize, check_grammar


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--runtime', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    import json
    runtime = json.loads(args.runtime.read_text()); snapshot = Path(runtime['model']['path'])
    check(args.runtime, args.output / 'local_file_identity.json')
    retokenize(snapshot, args.output / 'retokenization.json')
    check_grammar(snapshot, args.output / 'grammar_token_mask.json')
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise RuntimeError('CPU import boundary failed')
    print('CPU_PREFLIGHT_PASS')


if __name__ == '__main__':
    main()
