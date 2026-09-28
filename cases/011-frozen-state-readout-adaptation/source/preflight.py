"""Reproduce the small model-backed feature/cache boundary check; no fitting.

Use the retained local checkpoints and pinned upstream checkout. This is an
8x32 SMOKE execution test, not a memory-horizon evaluation or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import data
from source.adapter import FEATURE_BOUNDARY, FrozenRollout
from source.references import load_references, load_verified_model


def run(upstream, checkpoint_root):
    tokens, _, identity = data.load("smoke")
    if tokens.shape != (8, 32):
        raise ValueError("Preflight is fixed to the 8x32 SMOKE cohort")
    tokens = tokens.astype(np.int64)
    torch.set_num_threads(2)
    ref = load_references()
    rows = []
    for seed in range(3):
        checkpoint = Path(checkpoint_root) / f"training-seed{seed}" / "final.pt"
        model, table = load_verified_model(upstream, checkpoint, seed)
        for storage in ("NATIVE_FP32", "UNIFORM_8"):
            start = time.perf_counter()
            rollout = FrozenRollout(model, table, storage, model_seed=seed, batch_size=8)
            all_tokens = np.column_stack((np.full(8, 6), tokens))
            predictions = []
            maximum = 0.0
            saved = None
            for index, ids in enumerate(all_tokens.T):
                output = rollout.step(ids)
                coeff = ref.v1.gather(table, ids)
                original = ref.v1.logits_from_numpy(model, rollout.adapter.represented(rollout.state), coeff)
                if not torch.equal(original, output.logits_original) or not torch.equal(model.mlp[2](output.phi), original):
                    raise AssertionError("Original feature/head boundary is not bitwise identical")
                maximum = max(maximum, float((original - output.logits_original).abs().max()))
                predictions.append(output.predictions_original)
                if index == 16:
                    saved = rollout.export_bytes()
            rollout.assert_frozen()
            baseline, state, _ = ref.evaluation.sequence_call(model, table, tokens, rollout.adapter)
            if not np.array_equal(np.stack(predictions, axis=1), baseline):
                raise AssertionError("Original Case010 v2 predictions differ")
            if state.payload.tobytes() != rollout.export_bytes():
                raise AssertionError("Original Case010 v2 persistent cache differs")
            resumed = FrozenRollout(model, table, storage, model_seed=seed).import_bytes(saved, 8)
            suffix = [resumed.step(ids).predictions_original for ids in all_tokens[:, 17:].T]
            if not np.array_equal(np.stack(suffix, axis=1), baseline[:, 17:]):
                raise AssertionError("Split/resume predictions differ")
            if resumed.export_bytes() != rollout.export_bytes():
                raise AssertionError("Split/resume final state bytes differ")
            rows.append({"model_seed": seed, "storage": storage, "group_tokens": 32,
                "sequences": 8, "steps_including_BOS": 33, "readout_bitwise_equal": True,
                "maximum_logit_difference": maximum, "reference_v2_predictions_equal": True,
                "reference_v2_cache_equal": True, "split_predictions_equal": True,
                "split_final_bytes_equal": True, "terminal_streams": int((~output.status["active"]).sum()),
                "frozen_parameters_unchanged": True, "checkpoint_sha256": ref.common.CHECKPOINTS[seed],
                "token_table_sha256": rollout.table_sha256,
                "final_state_sha256": hashlib.sha256(rollout.export_bytes()).hexdigest(),
                "elapsed_seconds": time.perf_counter() - start})
    return {"schema": "case011-adapter-real-smoke-v1",
        "scope": "Fixed SMOKE cohort only; boundary and historical execution parity, not performance selection",
        "input_file_sha256": identity["tokens_sha256"], "torch_version": str(torch.__version__),
        "device": "cpu", "cpu_threads": 2, "feature_boundary": FEATURE_BOUNDARY,
        "cases": rows, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="Pinned ComplexKDA checkout")
    parser.add_argument("--checkpoint-root", required=True, type=Path,
                        help="Directory containing training-seed0/1/2/final.pt")
    parser.add_argument("--output", required=True, type=Path, help="New receipt JSON; refuses overwrite")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing preflight receipt")
    result = run(args.upstream, args.checkpoint_root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "storage_seed_checks": len(result["cases"])}))


if __name__ == "__main__":
    main()
