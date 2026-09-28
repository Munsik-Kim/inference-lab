"""Serial individual-head complete-call CPU timing; never a shared-three-head timer.

Run only once fitting/evaluation workers have stopped. Reuses Case010's frozen
16x128 timing inputs and original complete-call boundary, with one four-token
warmup per cell and three predeclared repetitions. No model or patch fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from source.adapter import freeze_model, parameter_manifest
from source.data import write_json
from source.head_patch import apply_patch, load_patch
from source.references import load_references, load_storage, load_verified_model
from source.reload_check import checkpoint_path, digest, patch_path, READOUTS, STORAGES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("upstream", "checkpoint-root", "patch-root", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    ref = load_references()
    tokens, _, _, input_identity = ref.common.load_cohort("timing")
    if tokens.shape != (16, 128):
        raise ValueError("Historical timing input shape differs from frozen 16x128")
    rows = []
    for seed in range(3):
        model, table = load_verified_model(args.upstream, checkpoint_path(args.checkpoint_root, seed), seed)
        original_manifest = freeze_model(model)
        original = {name: p.detach().clone() for name, p in model.named_parameters() if name in ("mlp.2.weight", "mlp.2.bias")}
        patches = {name: load_patch(patch_path(args.patch_root, seed, name), expected_base_sha256=ref.common.CHECKPOINTS[seed])
                   for name in READOUTS}
        adapters = {storage: load_storage(storage, seed) for storage in STORAGES}

        def choose(name):
            if name == "ORIGINAL":
                with torch.no_grad():
                    for key, value in original.items():
                        dict(model.named_parameters())[key].copy_(value)
            else:
                apply_patch(model, patches[name], expected_base_sha256=ref.common.CHECKPOINTS[seed])
            expected = parameter_manifest(model)
            if any(expected[k] != original_manifest[k] for k in expected if k not in original):
                raise ValueError("Timing model changed outside final head")

        for storage in STORAGES:
            for readout in ("ORIGINAL",) + READOUTS:
                choose(readout)
                ref.evaluation.sequence_call(model, table, tokens[:, :4], adapters[storage], diagnostics=False)
        cache_hashes = {}
        for repeat in range(3):
            for storage in STORAGES:
                for readout in ("ORIGINAL",) + READOUTS:
                    choose(readout)
                    predictions, state, info = ref.evaluation.sequence_call(model, table, tokens, adapters[storage], diagnostics=False)
                    # Timing ends inside sequence_call, before these checks or file I/O.
                    cache_hash = digest(state.payload.tobytes())
                    if storage in cache_hashes and cache_hashes[storage] != cache_hash:
                        raise ValueError("Individual head altered the timing stream cache")
                    cache_hashes[storage] = cache_hash
                    row = {"model_seed": seed, "storage": storage, "readout": readout, "repeat": repeat,
                           "seconds": info["elapsed_seconds"],
                           "ms_per_group_token": info["elapsed_seconds"] * 1000 / (16 * 128),
                           "active_updates": info["active_update_attempts"], "terminal_noops": info["terminal_noop_steps"],
                           "terminal_streams": sum(x != 0 for x in info["terminal_codes"]),
                           "invalid_readout_tokens": sum(info["invalid_readout_counts"]),
                           "final_cache_sha256": cache_hash, "prediction_sha256": digest(predictions.tobytes()),
                           "one_final_linear_call_per_write": True,
                           "base_checkpoint_sha256": ref.common.CHECKPOINTS[seed]}
                    rows.append(row)
                    write_json(out / "timing.json", {"schema": "case011-individual-head-cpu-timing-v1", "status": "IN_PROGRESS", "rows": rows})
                    print(json.dumps(row), flush=True)
    write_json(out / "timing.json", {"schema": "case011-individual-head-cpu-timing-v1", "status": "COMPLETE",
        "threads": 2, "sequences": 16, "group_tokens": 128, "generator_seed": 2002,
        "input_sha256": input_identity["tokens_file_sha256"], "repeats": 3,
        "warmup": "one four-group-token call per seed/storage/readout, excluded",
        "timed": "state initialization, BOS, recurrence, codec, status, original module path with one selected final Linear",
        "excluded": "checkpoint/table/patch loading, patch assignment, gold/shadow/diagnostics, file I/O",
        "denominator": "16*128 group tokens; BOS work included in numerator",
        "scope": "CPU Python/NumPy/Torch complete-call prototype; not individual-request latency or GPU speed",
        "temporary_memory": "transient represented state and readout features; process RAM peak not measured",
        "rows": rows})


if __name__ == "__main__":
    main()
