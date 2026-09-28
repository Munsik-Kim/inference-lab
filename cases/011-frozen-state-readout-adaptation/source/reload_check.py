"""Real-checkpoint fresh-process reload check on the frozen eight-input SMOKE.

Requires private original checkpoints and the six selected two-tensor patches.
It does not fit, select, or evaluate new TEST inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from source.adapter import FrozenRollout, HEAD_TENSORS
from source.data import load, write_json
from source.head_patch import apply_patch, load_patch
from source.references import load_references, load_verified_model

STORAGES = ("NATIVE_FP32", "UNIFORM_8")
READOUTS = ("SHORT_REFIT", "MIXED_REFIT")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checkpoint_path(root, seed):
    root = Path(root)
    choices = [root / f"training-seed{seed}" / "final.pt",
               root / f"seed{seed}" / "final.pt", root / f"seed{seed}.pt"]
    found = [p for p in choices if p.is_file() and not p.is_symlink()]
    if len(found) != 1:
        raise ValueError(f"expected exactly one original checkpoint location for seed {seed}")
    return found[0]


def patch_path(root, seed, readout):
    root = Path(root)
    choices = (root / f"seed{seed}-{readout}", root / f"seed{seed}" / readout)
    found = [p for p in choices if p.is_dir() and not p.is_symlink()]
    if len(found) != 1:
        raise ValueError(f"missing selected patch for seed {seed} / {readout}")
    return found[0]


def _plain(status):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in status.items()}


@torch.inference_mode()
def collect_smoke(model, table, storage, seed, tokens, patches=None):
    """One shared recurrence, detached candidate heads, and every cache boundary."""
    if tokens.shape != (8, 32):
        raise ValueError("Reload verification requires the frozen 8x32 SMOKE")
    runner = FrozenRollout(model, table, storage, model_seed=seed, batch_size=len(tokens))
    hashes = []
    features = hashlib.sha256()
    logits = {name: [] for name in (patches or {"ACTUAL": None})}
    labels = {name: [] for name in logits}
    for pos in range(tokens.shape[1] + 1):
        ids = np.full(len(tokens), 6, dtype=np.int64) if pos == 0 else tokens[:, pos - 1].astype(np.int64)
        step = runner.step(ids)
        if not bool(step.status["active"].all() and step.status["finite_features"].all()):
            raise ValueError("Actual SMOKE encountered terminal/nonfinite features; parity did not pass")
        features.update(step.phi.numpy().tobytes())
        for name in logits:
            if patches is None:
                value = step.logits_original
            else:
                tensors = patches[name]["tensors"]
                value = F.linear(step.phi, tensors["mlp.2.weight"], tensors["mlp.2.bias"])
            if not torch.isfinite(value).all():
                raise ValueError("Actual SMOKE readout is nonfinite")
            logits[name].append(value.numpy().copy())
            labels[name].append(value.argmax(-1).numpy().astype(np.int16))
        hashes.append(digest(runner.export_bytes()))
    runner.assert_frozen()
    return {"logits": {k: np.stack(v, axis=1) for k, v in logits.items()},
            "labels": {k: np.stack(v, axis=1) for k, v in labels.items()},
            "cache_boundary_sha256": hashes, "feature_sha256": features.hexdigest(),
            "final_payload": runner.export_bytes(), "final_status": _plain(step.status),
            "table_sha256": runner.table_sha256,
            "no_gradients": all(not p.requires_grad and p.grad is None for p in model.parameters())}


def child(args):
    torch.set_num_threads(2)
    tokens, _, item = load("smoke")
    checkpoint = checkpoint_path(args.checkpoint_root, args.seed)
    model, table = load_verified_model(args.upstream, checkpoint, args.seed)
    base = load_references().common.CHECKPOINTS[args.seed]
    patch = load_patch(patch_path(args.patch_root, args.seed, args.readout), expected_base_sha256=base)
    applied = apply_patch(model, patch, expected_base_sha256=base)
    output = collect_smoke(model, table, args.storage, args.seed, tokens)
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    np.save(out / "logits.npy", output["logits"]["ACTUAL"], allow_pickle=False)
    np.save(out / "labels.npy", output["labels"]["ACTUAL"], allow_pickle=False)
    (out / "final_payload.bin").write_bytes(output["final_payload"])
    write_json(out / "receipt.json", {
        k: output[k] for k in ("cache_boundary_sha256", "feature_sha256", "final_status",
                                "table_sha256", "no_gradients")
    } | {"applied": applied, "base_checkpoint_sha256": base,
         "patch_manifest_sha256": patch["manifest_sha256"], "input_sha256": item["tokens_sha256"]})


def run(args):
    torch.set_num_threads(2)
    destination = Path(args.output)
    destination.mkdir(exist_ok=False, parents=True)
    tokens, _, item = load("smoke")
    rows = []
    for seed in range(3):
        checkpoint = checkpoint_path(args.checkpoint_root, seed)
        model, table = load_verified_model(args.upstream, checkpoint, seed)
        base = load_references().common.CHECKPOINTS[seed]
        patches = {name: load_patch(patch_path(args.patch_root, seed, name), expected_base_sha256=base)
                   for name in READOUTS}
        for storage in STORAGES:
            expected = collect_smoke(model, table, storage, seed, tokens, patches)
            for readout in READOUTS:
                with tempfile.TemporaryDirectory(prefix="case011-reload-") as tmp:
                    out = Path(tmp) / "child"
                    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child",
                               "--upstream", str(args.upstream), "--checkpoint-root", str(args.checkpoint_root),
                               "--patch-root", str(args.patch_root), "--seed", str(seed),
                               "--storage", storage, "--readout", readout, "--output", str(out)]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2")
                    child_run = subprocess.run(command, cwd=tmp, env=env, text=True,
                                               capture_output=True, timeout=180)
                    if child_run.returncode != 0:
                        stderr = child_run.stderr
                        for private in (str(args.upstream), str(args.checkpoint_root), str(args.patch_root), tmp, str(ROOT)):
                            stderr = stderr.replace(private, "<local-path>")
                        write_json(destination / "child_failure.json", {"model_seed": seed, "storage": storage,
                            "readout": readout, "exit_code": child_run.returncode, "stderr_sanitized": stderr,
                            "completed_rows": rows, "status": "FAILED_CHILD"})
                        raise RuntimeError(f"reload child failed: seed={seed}, storage={storage}, head={readout}, exit={child_run.returncode}")
                    actual = json.loads((out / "receipt.json").read_text())
                    found_logits = np.load(out / "logits.npy", allow_pickle=False)
                    found_labels = np.load(out / "labels.npy", allow_pickle=False)
                    checks = {
                        "logits_bitwise": np.array_equal(found_logits, expected["logits"][readout]),
                        "labels_bitwise": np.array_equal(found_labels, expected["labels"][readout]),
                        "entire_logits_finite": bool(np.isfinite(found_logits).all()),
                        "feature_bitwise": actual["feature_sha256"] == expected["feature_sha256"],
                        "all_cache_boundaries_identical": actual["cache_boundary_sha256"] == expected["cache_boundary_sha256"],
                        "final_payload_bytes_identical": (out / "final_payload.bin").read_bytes() == expected["final_payload"],
                        "cursor_terminal_rng_state_identical": actual["final_status"] == expected["final_status"],
                        "table_identity": actual["table_sha256"] == expected["table_sha256"],
                        "only_final_tensor_changes": set(actual["applied"]["changed_tensors"]) <= set(HEAD_TENSORS),
                        "no_gradients": actual["no_gradients"] and expected["no_gradients"],
                    }
                    row = {"model_seed": seed, "storage": storage, "readout": readout,
                           "child_exit_code": child_run.returncode, "checks": checks,
                           "status": "PASS" if all(checks.values()) else "FAIL",
                           "max_abs_logit_difference": float(np.abs(found_logits.astype(np.float64) - expected["logits"][readout].astype(np.float64)).max()),
                           "base_checkpoint_sha256": base, "patch_manifest_sha256": patches[readout]["manifest_sha256"],
                           "final_payload_sha256": digest(expected["final_payload"]),
                           "final_payload_bytes": len(expected["final_payload"]),
                           "changed_tensors": actual["applied"]["changed_tensors"]}
                    rows.append(row)
                    write_json(destination / "receipt.json", {"schema": "case011-fresh-process-reload-v1",
                        "evidence_kind": "ACTUAL_CHECKPOINT_SMOKE", "status": "IN_PROGRESS",
                        "input_sha256": item["tokens_sha256"], "sequences": 8, "group_tokens": 32,
                        "BOS_included_in_logits_and_cache_checks": True, "rows": rows})
                    if not all(checks.values()):
                        raise ValueError(f"fresh-process parity failed: {seed}/{storage}/{readout}")
    receipt = json.loads((destination / "receipt.json").read_text())
    receipt.update(status="PASS", fresh_child_processes=len(rows),
                   model_weights_uploaded=False, recurrent_rollouts_for_quality_inference=False)
    write_json(destination / "receipt.json", receipt)
    print(json.dumps({"status": "PASS", "fresh_child_processes": len(rows)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("upstream", "checkpoint-root", "patch-root", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, choices=range(3))
    p.add_argument("--storage", choices=STORAGES)
    p.add_argument("--readout", choices=READOUTS)
    args = p.parse_args()
    if args.child and None in (args.seed, args.storage, args.readout):
        p.error("child mode requires seed, storage and readout")
    (child if args.child else run)(args)


if __name__ == "__main__":
    main()
