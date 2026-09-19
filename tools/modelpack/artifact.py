"""Standalone dense BF16 checkpoints with strict per-layer shape restoration."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from . import LOADER_VERSION
from .common import checksums, inventory, new_external, read, require, sha, verify_checksums, write

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json',
                   'special_tokens_map.json', 'chat_template.jinja', 'generation_config.json')


def tensor_hash(tensor) -> str:
    import hashlib
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def tensor_inventory(state: dict) -> dict:
    import torch
    result = {}
    for name, tensor in sorted(state.items()):
        require(bool(torch.isfinite(tensor).all()), 'Nonfinite tensor: ' + name)
        result[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                        'numel': tensor.numel(), 'bytes': tensor.numel()*tensor.element_size(),
                        'sha256': tensor_hash(tensor)}
    return result


def save_dense(model, output: Path, source: dict, retained: list[int] | None = None,
               layer: int = 13, tokenizer_source: Path | None = None, recipe: dict | None = None) -> dict:
    """Export all represented weights; no dependency on the original checkpoint."""
    import torch
    from safetensors.torch import save_file
    require(not output.exists(), 'Artifact target exists')
    out = new_external(output.with_name(output.name + '.partial'))
    config = model.config.to_dict()
    require(config['model_type'] == 'qwen3', 'Only audited Qwen3 loader supported')
    if retained is not None:
        require(len(retained) == len(set(retained)) and retained == sorted(retained), 'Invalid retained order')
        require(all(0 <= i < config['intermediate_size'] for i in retained), 'Bad retained indices')
    structure = {'layer': layer if retained is not None else None, 'retained': retained,
                 'width': len(retained) if retained is not None else None,
                 'native_intermediate_size': config['intermediate_size']}
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    require(all(v.dtype == torch.bfloat16 for v in state.values()), 'Dense export requires BF16 weights')
    tied = bool(config.get('tie_word_embeddings'))
    if tied:
        require(torch.equal(state['lm_head.weight'], state['model.embed_tokens.weight']), 'Broken tied weights')
        del state['lm_head.weight']
    info = tensor_inventory(state)
    (out/'weights').mkdir()
    save_file(state, out/'weights/model.safetensors', metadata={'format': 'pt'})
    write(out/'config.json', config)
    write(out/'structure.json', structure)
    write(out/'recipe.json', recipe or {'method': 'identity'})
    if tokenizer_source:
        for name in TOKENIZER_FILES:
            if (tokenizer_source/name).is_file():
                shutil.copyfile(tokenizer_source/name, out/name)
    manifest = {'schema': 1, 'artifact_type': 'pruned_layerwise' if retained else 'dense_bf16',
                'loader_version': LOADER_VERSION, 'loader_source_sha256': sha(Path(__file__)),
                'source': source, 'weights': info, 'tied_aliases': {'lm_head.weight': 'model.embed_tokens.weight'} if tied else {},
                'build': 'BUILD_COMPLETE', 'reload': 'NOT_RUN', 'quality': 'NOT_RUN', 'performance': 'NOT_RUN',
                'scope': 'local weights; no public weight upload', 'payload_files': inventory(out)}
    write(out/'artifact.json', manifest)
    checksums(out)
    inspect_artifact(out)
    out.rename(output)
    return manifest


def inspect_artifact(path: Path) -> dict:
    from safetensors import safe_open
    verify_checksums(path)
    manifest = read(path/'artifact.json')
    require(manifest['schema'] == 1 and manifest['loader_version'] == LOADER_VERSION, 'Unsupported loader/schema')
    require(manifest['artifact_type'] in ('dense_bf16', 'pruned_layerwise'), 'Unsupported artifact type')
    require(manifest['build'] == 'BUILD_COMPLETE', 'Incomplete build')
    require(manifest['payload_files'] == inventory(path, ('artifact.json', 'checksums.sha256')), 'Payload mismatch')
    state = {}
    with safe_open(path/'weights/model.safetensors', framework='pt', device='cpu') as stream:
        for key in stream.keys():
            state[key] = stream.get_tensor(key)
    require(tensor_inventory(state) == manifest['weights'], 'Tensor inventory/value mismatch')
    config, structure = read(path/'config.json'), read(path/'structure.json')
    require(structure['native_intermediate_size'] == config['intermediate_size'], 'Base width mismatch')
    if structure['layer'] is not None:
        layer, ids, width = structure['layer'], structure['retained'], structure['width']
        require(0 <= layer < config['num_hidden_layers'] and ids == sorted(set(ids)) and len(ids) == width,
                'Invalid layer/index mapping')
        require(width > 0 and all(0 <= x < config['intermediate_size'] for x in ids), 'Invalid retained channel')
    # Check every layer, including layers that were not pruned.
    for layer in range(config['num_hidden_layers']):
        width = structure['width'] if layer == structure['layer'] else config['intermediate_size']
        prefix = f'model.layers.{layer}.mlp.'
        for name, shape in [('gate_proj', [width, config['hidden_size']]),
                            ('up_proj', [width, config['hidden_size']]),
                            ('down_proj', [config['hidden_size'], width])]:
            require(manifest['weights'].get(prefix+name+'.weight', {}).get('shape') == shape, 'MLP shape mismatch')
    return {'status': 'PASS', 'artifact_type': manifest['artifact_type'],
            'tensor_count': len(state), 'parameter_bytes': sum(t['bytes'] for t in manifest['weights'].values()),
            'total_file_bytes': sum(f.stat().st_size for f in path.rglob('*') if f.is_file()),
            'manifest_sha256': sha(path/'artifact.json')}


def load_dense(path: Path, device: str = 'cpu'):
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RotaryEmbedding
    inspect_artifact(path)
    info, structure = read(path/'artifact.json'), read(path/'structure.json')
    cfg = Qwen3Config.from_dict(read(path/'config.json'))
    cfg._attn_implementation = 'sdpa'
    # Meta construction avoids loading a source model or allocating the full-width weights.
    with torch.device('meta'):
        model = Qwen3ForCausalLM(cfg)
        if structure['layer'] is not None:
            local = copy.deepcopy(cfg); local.intermediate_size = structure['width']
            model.model.layers[structure['layer']].mlp = Qwen3MLP(local)
    # These two nonpersistent buffers are derived from config, not checkpoint keys.
    model.model.rotary_emb = Qwen3RotaryEmbedding(cfg, device='cpu')
    state = load_file(path/'weights/model.safetensors', device='cpu')
    for alias, canonical in info['tied_aliases'].items():
        require(alias == 'lm_head.weight' and canonical == 'model.embed_tokens.weight', 'Unexpected tied alias')
        state[alias] = state[canonical]
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    require(all(p.dtype == torch.bfloat16 and not p.is_meta for p in model.parameters()), 'Unloaded/wrong dtype state')
    return model.eval().to(device)


def slice_mlp(original, retained: list[int]):
    """Aligned gate/up rows and down columns, adapted from Case007 surgery.py."""
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    width = original.intermediate_size
    require(retained and retained == sorted(set(retained)) and min(retained) >= 0 and max(retained) < width,
            'Invalid retained mapping')
    cfg = copy.deepcopy(original.config); cfg.intermediate_size = len(retained)
    result = Qwen3MLP(cfg).to(device=original.gate_proj.weight.device, dtype=original.gate_proj.weight.dtype)
    indices = torch.tensor(retained, device=original.gate_proj.weight.device)
    with torch.no_grad():
        result.gate_proj.weight.copy_(original.gate_proj.weight.index_select(0, indices))
        result.up_proj.weight.copy_(original.up_proj.weight.index_select(0, indices))
        result.down_proj.weight.copy_(original.down_proj.weight.index_select(1, indices))
    return result.eval()
