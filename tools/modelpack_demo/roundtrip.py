"""CPU-only, random-initialized tiny Qwen serialization demo; no quality benchmark."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.modelpack.common import new_external, read, require, verify_checksums, write

KIND = 'TINY_RANDOM_CPU_FIXTURE'


def offline_environment() -> dict[str, str]:
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
            'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false'}


def observe(model) -> dict:
    import torch
    from tools.modelpack.artifact import tensor_hash
    checks = []
    def finite(module, args, output):
        tensors = output if isinstance(output, tuple) else (output,)
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                ok = bool(torch.isfinite(tensor).all()); checks.append(ok)
                require(ok, 'Nonfinite full hidden output')
    hooks = [block.register_forward_hook(finite) for block in model.model.layers]
    hooks.append(model.model.norm.register_forward_hook(finite))
    try:
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([[1, 2, 3, 4]]),
                           use_cache=False, logits_to_keep=1).logits
    finally:
        for hook in hooks: hook.remove()
    require(len(checks) == 3 and all(checks), 'Missing full-output evidence')
    require(bool(torch.isfinite(logits).all()), 'Nonfinite vocabulary output')
    return {'kind': KIND, 'layer_widths': [m.mlp.intermediate_size for m in model.model.layers],
            'shapes': {name: list(getattr(model.model.layers[1].mlp, name).weight.shape)
                       for name in ('gate_proj', 'up_proj', 'down_proj')},
            'tied_weights': model.lm_head.weight is model.model.embed_tokens.weight,
            'logits_shape': list(logits.shape), 'logits_sha256': tensor_hash(logits),
            'full_hidden_checks': len(checks), 'full_outputs_finite': True,
            'all_parameters_cpu_bf16': all(p.device.type == 'cpu' and p.dtype == torch.bfloat16
                                            and not p.is_meta for p in model.parameters()),
            'all_buffers_materialized': all(not p.is_meta for p in model.buffers())}


def compare(before: dict, after: dict) -> None:
    for value in (before, after):
        require(value.get('kind') == KIND, 'Wrong evidence kind')
        require(value.get('layer_widths') == [64, 48], 'Wrong layer widths')
        require(value.get('shapes') == {'gate_proj': [48, 32], 'up_proj': [48, 32],
                                       'down_proj': [32, 48]}, 'Wrong gate/up/down mapping')
        for flag in ('tied_weights', 'full_outputs_finite', 'all_parameters_cpu_bf16',
                     'all_buffers_materialized'):
            require(value.get(flag) is True, 'Missing/failed validity: ' + flag)
        require(value.get('full_hidden_checks') == 3, 'Incomplete full-output checks')
        require(value.get('logits_shape') == [1, 1, 64], 'Wrong output boundary')
        require(isinstance(value.get('logits_sha256'), str) and len(value['logits_sha256']) == 64,
                'Missing output digest')
    require(before == after, 'Saved/reloaded computation differs')


def worker(stage: str, directory: Path, blocked: Path | None) -> None:
    os.environ.update(offline_environment())
    if blocked is not None:
        blocked = blocked.resolve()
        def forbid_original(event, args):
            if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
                raw = os.fsdecode(args[0])
                if Path(raw).resolve().is_relative_to(blocked):
                    raise PermissionError('Reload cannot access the build-process directory')
        sys.addaudithook(forbid_original)
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from tools.modelpack.artifact import save_dense, load_dense, slice_mlp
    torch.set_num_threads(2)
    if stage == 'build':
        torch.manual_seed(8008)
        cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, tie_word_embeddings=True)
        cfg._attn_implementation = 'sdpa'
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            model = Qwen3ForCausalLM(cfg).eval()
        finally:
            torch.set_default_dtype(old)
        model.model.layers[1].mlp = slice_mlp(model.model.layers[1].mlp, list(range(48)))
        observation = observe(model)
        save_dense(model, directory/'artifact', {'kind': KIND, 'seed': 8008},
                   retained=list(range(48)), layer=1,
                   recipe={'kind': KIND, 'method': 'fixed slice; no quality selection'})
    else:
        from unittest.mock import patch
        # The loader must reconstruct from the copied local artifact, not fetch a checkpoint.
        with patch.object(Qwen3ForCausalLM, 'from_pretrained', side_effect=AssertionError('No checkpoint download')):
            observation = observe(load_dense(directory/'artifact', device='cpu'))
    write(directory/'observation.json', observation)
    write(directory/'process.json', {'pid': os.getpid(), 'stage': stage,
        'device': 'cpu', 'original_build_access_blocked': blocked is not None})


def run(output: Path) -> dict:
    out = new_external(output)
    build, reload = out/'build-process', out/'reload-process'
    build.mkdir(); reload.mkdir()
    script = str(Path(__file__).resolve())
    def call(stage: str, directory: Path):
        cmd = [sys.executable, '-B', script, '--worker', stage, '--output', str(directory)]
        if stage == 'reload': cmd += ['--blocked-build', str(build)]
        result = subprocess.run(cmd, cwd=directory, env=offline_environment(),
                                capture_output=True, text=True, timeout=120)
        (directory/'process.log').write_text(result.stdout + result.stderr)
        require(result.returncode == 0, stage + ' worker failed; see process.log')
    call('build', build)  # Process exits before copying and loading.
    before = verify_checksums(build/'artifact')
    shutil.copytree(build/'artifact', reload/'artifact')
    require(verify_checksums(reload/'artifact') == before, 'Copied artifact changed')
    call('reload', reload)
    first, second = read(build/'observation.json'), read(reload/'observation.json')
    compare(first, second)
    processes = [read(folder/'process.json') for folder in (build, reload)]
    require(processes[0]['pid'] != processes[1]['pid'], 'Expected separate processes')
    require(processes[1]['original_build_access_blocked'], 'Missing source isolation')
    result = {'status': 'PASS', 'evidence_kind': KIND, 'trained_weights': False,
        'model_downloads': 0, 'gpu_runs': 0, 'original_width': 64, 'retained_width': 48,
        'layer': 1, 'observation': second, 'separate_processes': True,
        'different_directories': True, 'build_process_exited_before_reload': True,
        'original_build_access_blocked': True, 'copy_checksum_identity': True,
        'scope': 'Serialization/structure/output contract for a random tiny CPU model; no task quality or speed claim.'}
    write(out/'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path, help='New directory outside the repository')
    p.add_argument('--worker', choices=('build', 'reload'), help=argparse.SUPPRESS)
    p.add_argument('--blocked-build', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker: worker(a.worker, a.output.resolve(), a.blocked_build)
    else:
        result = run(a.output.resolve())
        print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
