"""Optional local-tokenizer CPU audit of new inputs, decoded outputs and G masks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

P2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P2 / 'src'))
from json_recovery_p2.tokenizer_cpu import local_tokenizer, render, tokenizer_data
from json_recovery_p2.grammar import token_enforcer
from json_recovery_p2.upstream import token_hash, digest


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--results', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    tokenizer, files = local_tokenizer(args.snapshot)
    config = json.loads((args.snapshot / 'tokenizer_config.json').read_text())
    _, data, tokenizer_identity = tokenizer_data(args.snapshot)
    requests = [json.loads(x) for x in (args.results / 'requests.jsonl').read_text().splitlines() if x]
    masks = tokens_checked = 0
    for r in requests:
        request = r['model_request']
        if set(request) != {'messages', 'tools'} or digest(request) != r['messages_tools_hash']:
            raise ValueError('model-visible request allowlist/hash differs')
        ids = tokenizer.encode(render(config, request), add_special_tokens=False).ids
        if ids != r['input_token_ids'] or token_hash(ids) != r['input_token_sha256']:
            raise ValueError('independent retokenization differs: ' + r['request_id'])
        if tokenizer.decode(r['output_token_ids'], skip_special_tokens=False) != r['raw_text'] or tokenizer.decode(r['output_token_ids'], skip_special_tokens=True) != r['text']:
            raise ValueError('output IDs do not decode to saved raw/parser text')
        if r['grammar_applied']:
            enforcer = token_enforcer(data, request['tools']); prefix = []
            if len(r['grammar_steps']) != len(r['output_token_ids']):
                raise ValueError('mask/token count mismatch')
            for token, step in zip(r['output_token_ids'], r['grammar_steps']):
                allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
                fingerprint = hashlib.sha256(json.dumps(sorted(allowed), separators=(',', ':')).encode()).hexdigest()
                if token not in allowed or len(allowed) != step['allowed_count'] or fingerprint != step['allowed_ids_sha256']:
                    raise ValueError('independent actual-tokenizer mask differs')
                if token != step['masked_argmax_token_id']:
                    raise ValueError('generated token differs from recorded masked greedy decision')
                prefix.append(token); tokens_checked += 1
            masks += 1
        elif r['grammar_steps']:
            raise ValueError('unexpected grammar trace in unmasked continuation')
    value = {'status': 'PASS', 'input_requests_retokenized': len(requests), 'output_ID_decodes_checked': len(requests),
        'G_focal_masks_recomputed': masks, 'G_generated_tokens_verified_in_mask': tokens_checked,
        'tokenizer_identity': tokenizer_identity, 'model_forwards': 0, 'torch_imported': 'torch' in sys.modules,
        'transformers_imported': 'transformers' in sys.modules,
        'scope': 'CPU native tokenizer/template and allowed-token sets; original model logits/argmax scores not independently regenerated'}
    with args.output.open('x') as f:json.dump(value, f, indent=2);f.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':main()
