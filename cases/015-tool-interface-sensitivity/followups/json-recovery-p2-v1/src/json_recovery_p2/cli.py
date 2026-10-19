"""CPU preparation CLI; run is deliberately closed until a separate GPU stage."""
import argparse
import json
import sys
from pathlib import Path
from .cpu_only import install

install()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('validate', help='verify original bytes and P2 registration, CPU only')
    for name, description in [('prepare', 'reconstruct historical evidence into a new external directory'),
                              ('audit', 'recompute and compare frozen checkpoint/diagnostic data'),
                              ('dry-run', 'write the conditional future request plan; no model calls')]:
        sub = subs.add_parser(name, help=description); sub.add_argument('--output', type=Path, required=True)
    base = subs.add_parser('baseline', help='restore the original closed view and run its unchanged checks')
    base.add_argument('--output', type=Path, required=True); base.add_argument('--tests', action='store_true')
    for name in ['tokenizer-check', 'grammar-check']:
        sub = subs.add_parser(name, help='optional installed CPU dependencies and local tokenizer only')
        sub.add_argument('--snapshot', type=Path, required=True); sub.add_argument('--output', type=Path, required=True)
    local = subs.add_parser('local-inputs-check', help='hash registered local files without loading a model')
    local.add_argument('--runtime', type=Path, required=True); local.add_argument('--output', type=Path, required=True)
    run = subs.add_parser('run', help='REFUSED in this CPU stage; --dry-run only')
    run.add_argument('--dry-run', action='store_true'); run.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'run' and not args.dry_run:
        print(json.dumps({'status': 'CPU_SCOPE_ONLY', 'GPU_execution': 'REFUSED', 'model_forwards': 0,
                          'reason': 'separate explicit authorization and a reviewed GPU adapter are required'}))
        return 3
    from .io import write
    from .verify import audit, baseline, validate
    if args.command == 'validate':
        result = validate()
    elif args.command in ['dry-run', 'run']:
        if args.output is None:
            parser.error('--output is required for a dry-run')
        validate()
        from .plan import dry_run
        result = dry_run(); write(args.output, result)
    elif args.command == 'prepare':
        from .verify import originals
        originals()
        from .evidence import derive
        result = derive(args.output)
    elif args.command == 'audit':
        result = audit(args.output)
    elif args.command == 'baseline':
        result = baseline(args.output, args.tests)
    elif args.command == 'local-inputs-check':
        validate()
        from .local_inputs import check
        result = check(args.runtime, args.output)
    else:
        validate()
        from .tokenizer_cpu import retokenize, check_grammar
        result = (retokenize if args.command == 'tokenizer-check' else check_grammar)(args.snapshot, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'cells', 'examples']}, ensure_ascii=False))
    if any(name.split('.')[0] in {'torch', 'transformers'} for name in sys.modules):
        raise RuntimeError('CPU import boundary failed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

