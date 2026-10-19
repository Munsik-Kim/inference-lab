"""Isolated GPU-only process. Reuse the pinned original read-only Adapter."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

P2 = Path(__file__).resolve().parents[1]
CASE = P2.parents[1]
sys.path.insert(0, str(CASE / 'src'))


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def tokenizer_data(adapter, snapshot):
    from lmformatenforcer import TokenEnforcerTokenizerData
    tokenizer = adapter.tokenizer
    doc = json.loads((snapshot / 'tokenizer.json').read_text())
    specials = {t['id'] for t in doc['added_tokens'] if t['special']}
    token_zero = tokenizer.encode('0', add_special_tokens=False)[-1]
    regular = []
    for token in range(len(tokenizer)):
        if token in specials:
            continue
        after = tokenizer.decode([token_zero, token], skip_special_tokens=False)[1:]
        single = tokenizer.decode([token], skip_special_tokens=False)
        regular.append((token, after, len(after) > len(single)))
    decoder = lambda ids: tokenizer.decode(ids, skip_special_tokens=False).rstrip('�')
    stops = adapter.model.generation_config.eos_token_id
    if stops != [151645, 151643]:
        raise ValueError('original EOS stop policy differs')
    data = TokenEnforcerTokenizerData(regular, decoder, stops, use_bitmask=False, vocab_size=adapter.model.config.vocab_size)
    return data


class GrammarProcessor:
    def __init__(self, adapter, data, tools, input_length):
        from grammar_gpu import token_enforcer
        self.adapter = adapter; self.enforcer = token_enforcer(data, tools)
        self.input_length = input_length; self.steps = []

    def __call__(self, ids, scores):
        torch = self.adapter.torch
        # Exact original deadline and finite UNMASKED logits checks first.
        scores = self.adapter.processor_original(ids, scores)
        prefix = ids[0, self.input_length:].tolist()
        allowed = self.enforcer.get_allowed_tokens(prefix).allowed_tokens
        if not allowed or min(allowed) < 0 or max(allowed) >= scores.shape[-1]:
            raise ValueError('invalid grammar token mask')
        mask = torch.full_like(scores, -torch.inf)
        permitted = torch.tensor(allowed, device=scores.device, dtype=torch.long)
        mask[:, permitted] = scores[:, permitted]
        chosen = int(mask[0].argmax())
        if chosen not in allowed:
            raise ValueError('masked argmax outside allowed tokens')
        self.steps.append({'step': len(prefix), 'allowed_count': len(allowed),
            'allowed_ids_sha256': hashlib.sha256(json.dumps(sorted(allowed), separators=(',', ':')).encode()).hexdigest(),
            'masked_argmax_token_id': chosen, 'unmasked_finite_before_mask': True,
            'incomplete_EOS_blocked': not self.enforcer.prefix_states[tuple(prefix)].parser.can_end()})
        return mask


def grammar_gpu_gate(adapter, data, tools):
    """Synthetic CUDA score tensors and real tokenizer; no model forward."""
    torch = adapter.torch
    adapter.episode_deadline = min(adapter.deadline, time.monotonic() + 900)
    processor = GrammarProcessor(adapter, data, tools, 1)
    dummy = torch.tensor([[151644]], device='cuda')
    scores = torch.zeros((1, adapter.model.config.vocab_size), device='cuda', dtype=torch.float32)
    masked = processor(dummy, scores)
    stops = adapter.model.generation_config.eos_token_id
    if not all(bool(torch.isneginf(masked[0, eos])) for eos in stops):
        raise ValueError('incomplete grammar must block original EOS IDs')
    name = tools[2]['function']['name']
    text = '<tool_call>' + json.dumps({'name': name, 'arguments': {'order_id': 'NOT-A-TARGET', 'total_minor': -1}})
    tokens = adapter.tokenizer.encode(text, add_special_tokens=False)
    prefix = []
    for token in tokens:
        if token not in processor.enforcer.get_allowed_tokens(prefix).allowed_tokens:
            raise ValueError('GPU grammar backend semantically restricted a wrong ID/amount')
        prefix.append(token)
    complete = processor(torch.tensor([[151644] + tokens], device='cuda'), scores)
    if not all(bool(torch.isfinite(complete[0, eos])) for eos in stops):
        raise ValueError('complete call dict without closing marker must allow original EOS')
    bad = scores.clone(); bad[0, 0] = torch.nan
    rejected = False
    try:
        processor(dummy, bad)
    except Exception as error:
        rejected = getattr(error, 'label', None) == 'INFRA_ERROR'
    if not rejected:
        raise ValueError('original FiniteBudget did not reject nonfinite unmasked logits')
    return {'status': 'PASS', 'evidence_kind': 'synthetic_contract_test', 'model_forwards': 0,
            'actual_CUDA_score_mask': True, 'wrong_ID_and_amount_allowed': True,
            'incomplete_EOS_blocked_complete_EOS_allowed': True,
            'original_finite_check_before_intentional_negative_infinity': True}


def loaded_gate(adapter, runtime, output):
    """Actual module/shape/W_pre/beta and unchanged nonprojection tensor checks."""
    import torch
    from safetensors import safe_open
    import reference_int4
    names = json.loads((CASE / 'provenance/artifact_identity.json').read_text())
    expected = names['RTN']['per_projection_identity']
    modules = {e['layer']: e for e in expected}
    snapshot = Path(runtime['model']['path'])
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    historical_environment = json.loads((CASE / 'followups/order-api-regression-v1/results/a1_r1/workers' / adapter.method / 'attempt01/environment.json').read_text())
    actual_environment = json.loads((output / 'environment.json').read_text())
    environment_fields = ['python', 'torch', 'transformers', 'cuda', 'gpu', 'capability', 'TF32',
                          'BF16_reduced_reduction', 'FP16_reduced_reduction', 'attention', 'threads', 'model_generation_config']
    for field in environment_fields:
        if actual_environment[field] != historical_environment[field]:
            raise ValueError('original A1 runtime differs: ' + field)
    actual_config = {k: v for k, v in actual_environment['model_config'].items() if k != '_name_or_path'}
    historical_config = {k: v for k, v in historical_environment['model_config'].items() if k != '_name_or_path'}
    if actual_config != historical_config:
        raise ValueError('original model config differs beyond sanitized path label')
    checked = []
    if adapter.method != 'BF16':
        entries = names[adapter.method]['per_projection_identity']
        for entry in entries:
            layer = adapter.model.get_submodule(entry['layer'])
            q = reference_int4.QuantizedWeight.deserialize((Path(runtime['artifact_root']) / adapter.method / (entry['layer'] + '.rpi4')).read_bytes())
            if list(layer.weight.shape) != entry['shape'] or layer.weight.dtype != torch.bfloat16 or layer.activation4:
                raise ValueError('loaded module shape/dtype/activation policy mismatch')
            if not torch.equal(layer.weight.detach().cpu(), q.weight_pre().to(torch.bfloat16)):
                raise ValueError('installed W_pre differs from saved artifact')
            if layer.beta_inverse.dtype != torch.float32 or not torch.equal(layer.beta_inverse.cpu(), q.beta_inv):
                raise ValueError('installed beta_inverse differs from saved artifact')
            checked.append({'layer': entry['layer'], 'shape': entry['shape'], 'artifact_sha256': entry['sha256'],
                'W_pre_BF16_sha256': tensor_sha(layer.weight), 'beta_inverse_FP32_sha256': tensor_sha(layer.beta_inverse)})
    else:
        for name, entry in modules.items():
            layer = adapter.model.get_submodule(name)
            if list(layer.weight.shape) != entry['shape'] or not isinstance(layer, torch.nn.Linear) or layer.weight.dtype != torch.bfloat16:
                raise ValueError('BF16 module coverage/shape mismatch')
    small = []
    for name, param in adapter.model.named_parameters():
        if any(name == module + '.weight' for module in modules):
            continue
        with safe_open(str(snapshot / index[name]), framework='pt', device='cpu') as file:
            original = file.get_tensor(name).to(param.dtype)
        if tensor_sha(original) != tensor_sha(param):
            raise ValueError('unchanged parameter differs from original snapshot: ' + name)
        small.append({'name': name, 'shape': list(param.shape), 'sha256': tensor_sha(param)})
    gate = {'status': 'PASS', 'method': adapter.method, 'actual_projection_modules': len(modules),
        'saved_W_pre_and_beta_checked': len(checked), 'projection_readbacks': checked,
        'unchanged_nonprojection_parameters': small, 'unchanged_parameters_checked': len(small),
        'embedding_head_tie_preserved': True, 'model_dtype': 'BF16', 'attention': 'sdpa',
        'TF32': torch.backends.cuda.matmul.allow_tf32,
        'BF16_reduced_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'FP16_reduced_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'model_revision': 'cdbee75f17c01a7cc42f958dc650907174af0554',
        'original_A1_runtime_fields_matched': environment_fields,
        'original_model_config_matched_except_sanitized_path_label': True,
        'new_solver_calls': 0, 'new_encode_calls': 0, 'artifact_writes': 0}
    if any(gate[k] for k in ['TF32', 'BF16_reduced_reduction', 'FP16_reduced_reduction']):
        raise ValueError('original numerical flags differ')
    (output / 'loaded_gate.json').write_text(json.dumps(gate, indent=2) + '\n')
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True); parser.add_argument('--method', choices=['BF16', 'RTN', 'D16'], required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--worker-deadline', type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from interface_sensitivity.model_adapter import Adapter
    from interface_sensitivity.common import digest, token_hash
    runtime = json.loads(args.runtime.read_text())
    original_manifest = json.loads((Path(runtime['original_result_root']) / 'experiment_manifest.json').read_text())
    if Path(original_manifest['models']['4b']['path']).resolve() != Path(runtime['model']['path']).resolve():
        raise ValueError('original Worker and runtime resolve to different snapshots')
    adapter = Adapter(CASE, runtime, args.method, args.output, args.worker_deadline)
    gate = loaded_gate(adapter, runtime, args.output)
    adapter.processor_original = adapter.processor
    data = None
    static_buffers = [(n, id(v), v._version) for n, v in adapter.model.named_buffers() if n.endswith('beta_inverse')]
    print(json.dumps({'kind': 'READY', 'gate': {k: gate[k] for k in ['status', 'method', 'actual_projection_modules', 'saved_W_pre_and_beta_checked', 'unchanged_parameters_checked']}}), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        if message['command'] == 'STOP':
            break
        try:
            payload = message['payload']
            if set(payload) != {'messages', 'tools'}:
                raise ValueError('model input allowlist must be messages/tools only')
            ids = adapter.tokenizer.apply_chat_template(payload['messages'], tools=payload['tools'], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
            if message.get('expected_input_hash') and token_hash(ids) != message['expected_input_hash']:
                raise ValueError('actual generation tokenizer path differs from registered focal input')
            grammar = None
            if message['grammar']:
                if data is None:
                    data = tokenizer_data(adapter, Path(runtime['model']['path']))
                    probe = grammar_gpu_gate(adapter, data, payload['tools'])
                    (args.output / 'grammar_gpu_gate.json').write_text(json.dumps(probe, indent=2) + '\n')
                grammar = GrammarProcessor(adapter, data, payload['tools'], len(ids))
                adapter.processor = grammar
            try:
                response = adapter.generate(payload['messages'], payload['tools'], message['episode_deadline'])
            finally:
                adapter.processor = adapter.processor_original
            if response['input_token_sha256'] != token_hash(ids):
                raise ValueError('actual Adapter and prechecked tokenizer IDs differ')
            if static_buffers != [(n, id(v), v._version) for n, v in adapter.model.named_buffers() if n.endswith('beta_inverse')]:
                raise ValueError('beta_inverse changed between branches')
            response.update({'input_token_ids': ids, 'grammar_applied': grammar is not None,
                'grammar_steps': grammar.steps if grammar else [], 'parameters_unchanged': True,
                'request_id': message['request_id'], 'method': args.method,
                'EOS': bool(response['output_token_ids'] and response['output_token_ids'][-1] in adapter.model.generation_config.eos_token_id),
                'reached_token_cap': response['output_tokens'] == 192})
            if grammar and [s['masked_argmax_token_id'] for s in grammar.steps] != response['output_token_ids']:
                raise ValueError('actual output tokens differ from recorded masked greedy decisions')
            print(json.dumps({'kind': 'RESPONSE', 'response': response}, ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps({'kind': 'ERROR', 'label': getattr(e, 'label', 'INFRA_ERROR'), 'type': type(e).__name__, 'message': str(e),
                              'request_id': message.get('request_id')}), flush=True)


if __name__ == '__main__':
    main()
