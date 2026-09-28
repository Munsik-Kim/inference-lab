"""Supporting DEV and parameter diagnostics of the already selected heads.

No recurrence, feature extraction, fitting, or TEST selection is performed.
Only the cached DEV features pass through the original/selected final Linear.
Private FIT arrays contribute byte/hash metadata only, never exported values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source.head_patch import load_patch

BANDS = ((1, 32), (33, 64), (65, 128), (129, 256))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_inventory(path, receipt):
    if sha(path) != receipt["private_features_sha256"]:
        raise ValueError("Frozen feature archive checksum mismatch")
    with np.load(path, allow_pickle=False) as contents:
        arrays = {}
        for name in contents.files:
            array = contents[name]
            arrays[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                            "array_bytes": int(array.nbytes),
                            "array_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    return {"file_sha256": sha(path), "file_bytes": path.stat().st_size,
            "arrays": arrays, "role": "private fitting/evaluation workspace; not recurrent cache",
            "array_values_in_review_zip": False}


def linear_scores(phi, gold, weight, bias):
    if phi.dtype != np.float32 or phi.ndim != 2 or phi.shape[1] != 192:
        raise ValueError("Expected original FP32 cached features")
    if not np.isfinite(phi).all():
        raise ValueError("Nonfinite cached feature; no rows are silently removed")
    with torch.inference_mode():
        logits = F.linear(torch.from_numpy(phi), weight, bias).numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite FP32 DEV logits")
    # Independent stable scalar scoring, after the actual FP32 execution boundary.
    double = logits.astype(np.float64)
    shifted = double - double.max(axis=1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(gold)), gold]
    predicted = logits.argmax(axis=1)
    return predicted, ce


def band_summary(predicted, ce, gold, positions, original_predicted):
    rows = []
    for low, high in BANDS:
        selected = (positions >= low) & (positions <= high)
        if not selected.any():
            raise ValueError("Missing declared DEV position band")
        correct = predicted[selected] == gold[selected]
        original_correct = original_predicted[selected] == gold[selected]
        rows.append({"position_start": low, "position_end": high,
                     "rows": int(selected.sum()), "gold_ce_nats": float(ce[selected].mean()),
                     "correct": int(correct.sum()), "accuracy": float(correct.mean()),
                     "original_correct_to_head_wrong": int((original_correct & ~correct).sum()),
                     "original_wrong_to_head_correct": int((~original_correct & correct).sum()),
                     "all_answer_disagreement": int((predicted[selected] != original_predicted[selected]).sum())})
    return {"bands": rows, "equal_band_gold_ce_nats": float(np.mean([r["gold_ce_nats"] for r in rows])),
            "equal_band_accuracy": float(np.mean([r["accuracy"] for r in rows])),
            "all_rows_accuracy": float((predicted == gold).mean()),
            "note": "DEV descriptions only; fitting and head choices remain unchanged"}


def displacement(base, candidate):
    delta = candidate.double() - base.double()
    return {"shape": list(base.shape), "dtype": "float32", "parameter_count": base.numel(),
            "tensor_bytes_before": base.numel() * base.element_size(),
            "tensor_bytes_after": candidate.numel() * candidate.element_size(),
            "changed_values": int((candidate != base).sum()),
            "l2_displacement": float(torch.linalg.vector_norm(delta)),
            "maximum_absolute_displacement": float(delta.abs().max()),
            "base_l2": float(torch.linalg.vector_norm(base.double())),
            "candidate_l2": float(torch.linalg.vector_norm(candidate.double()))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Supporting diagnostics do not overwrite a prior result")
    torch.set_num_threads(2)
    selection_path = ROOT / "configs/selected_heads.json"
    selected_bytes = selection_path.read_bytes()
    selection = json.loads(selected_bytes)
    selected = {(x["model_seed"], x["condition"]): x for x in selection["heads"]}
    rows = []
    for seed in range(3):
        path = args.checkpoint_root / f"training-seed{seed}/final.pt"
        base_sha = sha(path)
        if any(selected[(seed, name)]["base_checkpoint_sha256"] != base_sha
               for name in ("SHORT_REFIT", "MIXED_REFIT")):
            raise ValueError("Checkpoint differs from the already selected patch identity")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model_state_dict"]
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Expected finite original FP32 checkpoint tensors")
        w0, b0 = state["mlp.2.weight"], state["mlp.2.bias"]
        if w0.shape != (6, 192) or b0.shape != (6,):
            raise ValueError("Original final head shape changed")
        archives = {}
        for role in ("fit", "dev"):
            receipt = json.loads((ROOT / f"results/seed{seed}-{role}-features.json").read_text())
            archives[role] = cache_inventory(args.private / f"seed{seed}-{role}.npz", receipt)
        with np.load(args.private / f"seed{seed}-dev.npz", allow_pickle=False) as cache:
            phi, gold, positions = (cache[name] for name in ("phi", "gold", "positions"))
        pred0, ce0 = linear_scores(phi, gold, w0, b0)
        heads = {"ORIGINAL": {"parameter_changes": 0,
            "DEV": band_summary(pred0, ce0, gold, positions, pred0)}}
        for name in ("SHORT_REFIT", "MIXED_REFIT"):
            identity = selected[(seed, name)]
            patch_dir = args.private / "patches" / identity["patch_directory"]
            patch = load_patch(patch_dir, expected_base_sha256=base_sha)
            if patch["manifest_sha256"] != identity["manifest_sha256"]:
                raise ValueError("Selected patch manifest changed")
            w, b = patch["tensors"]["mlp.2.weight"], patch["tensors"]["mlp.2.bias"]
            pred, ce = linear_scores(phi, gold, w, b)
            metrics = band_summary(pred, ce, gold, positions, pred0)
            prior = json.loads((ROOT / f"results/fit-seed{seed}-{name}.json").read_text())
            chosen = next(row for row in prior["selection"]["dev_scores"] if row["regularization"] == identity["regularization"])
            if abs(chosen["equal_band_ce"] - metrics["equal_band_gold_ce_nats"]) > 1e-12:
                raise ValueError("Independent DEV score differs from frozen lambda-selection score")
            tensor_rows = {"mlp.2.weight": displacement(w0, w), "mlp.2.bias": displacement(b0, b)}
            heads[name] = {"tensor_displacements": tensor_rows,
                "parameter_changes": sum(x["changed_values"] for x in tensor_rows.values()),
                "changed_tensor_count": sum(x["changed_values"] > 0 for x in tensor_rows.values()),
                "combined_l2_displacement": float(np.sqrt(sum(x["l2_displacement"] ** 2 for x in tensor_rows.values()))),
                "lambda": identity["regularization"], "solver_status": patch["manifest"]["metadata"]["solver"]["status"],
                "patch_payload_bytes": patch["manifest"]["payload_bytes"],
                "patch_manifest_bytes": (patch_dir / "manifest.json").stat().st_size,
                "patch_total_file_bytes": sum(file.stat().st_size for file in patch_dir.iterdir()),
                "patch_manifest_sha256": patch["manifest_sha256"], "DEV": metrics,
                "DEV_score_matches_frozen_selection_atol": 1e-12}
        rows.append({"model_seed": seed, "base_checkpoint_sha256": base_sha,
            "model_tensor_count": len(state), "model_parameters": sum(v.numel() for v in state.values()),
            "model_tensor_bytes_before_and_after": sum(v.numel() * v.element_size() for v in state.values()),
            "final_head_parameters": 1158, "final_head_tensor_bytes_before_and_after": 4632,
            "additional_recurrent_state_bytes": 0, "extra_final_linear_calls": 0,
            "feature_archives": archives, "heads": heads})
    if selection_path.read_bytes() != selected_bytes:
        raise ValueError("Head selection was modified during supporting analysis")
    result = {"schema": "case011-head-supporting-diagnostics-v1", "status": "COMPLETE",
        "evidence_kind": "SUPPORTING_DEV_AND_PARAMETER_ANALYSIS",
        "new_recurrence_or_feature_extraction": 0, "new_fits": 0, "TEST_used": False,
        "head_selection_changes": 0, "logit_execution_dtype": "float32", "score_dtype": "float64",
        "selected_heads_sha256": hashlib.sha256(selected_bytes).hexdigest(),
        "source_sha256": sha(Path(__file__)), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": "COMPLETE", "checkpoint_rows": 3,
                     "output_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
