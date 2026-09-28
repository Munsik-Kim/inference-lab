#!/usr/bin/env python3
"""Independent saved-label audit. Does not import the study metrics/runner.

This requires NumPy only. It never loads a model, checkpoint or feature array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HEADS = ("ORIGINAL", "SHORT_REFIT", "MIXED_REFIT")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def strict_json(path):
    def invalid(value):
        raise ValueError(f"Non-JSON finite value {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def independently_derive(pred, gold):
    """Scan each sequence for first mismatch; later recovery stays separate."""
    n, length = gold.shape
    tau, lengths = [], []
    histogram = np.zeros(length + 1, dtype=np.int64)
    recovered_rows = 0
    recovered_tokens = 0
    post_error_tokens = 0
    for row in range(n):
        errors = np.flatnonzero(pred[row] != gold[row])
        if len(errors):
            first = int(errors[0])
            tau.append(first + 1)
            lengths.append(first)
            histogram[first] += 1
            later = pred[row, first+1:] == gold[row, first+1:]
            recovered_rows += int(later.any())
            recovered_tokens += int(later.sum())
            post_error_tokens += len(later)
        else:
            tau.append(None)
            lengths.append(length)
            histogram[length] += 1
    failure_counts = np.cumsum(histogram[:-1])
    empirical = 0
    for t, count in enumerate(failure_counts, 1):
        if int(count) <= 0.05 * n:
            empirical = t
    correct = pred == gold
    result = {
        "n_sequences": n, "max_group_tokens": length, "tau": tau,
        "right_censored_count": int(histogram[-1]),
        "rmst0": sum(lengths) / n,
        "empirical_t005": empirical,
        "empirical_horizon_at_observation_limit": empirical == length,
        "confidence_supported_horizon": None,
        "token_accuracy": int(correct.sum()) / (n * length),
        "invalid_prediction_count": int(np.count_nonzero(pred == -1)),
        "failed_sequence_count": n - int(histogram[-1]),
        "sequences_with_correct_token_after_first_error": recovered_rows,
        "correct_tokens_after_first_error": recovered_tokens,
        "tokens_after_first_error": post_error_tokens,
        "survival_by_token": [1.0 - int(c)/n for c in failure_counts],
        "token_accuracy_by_token": [int(correct[:, t].sum())/n for t in range(length)],
        "prefix_summary": [],
    }
    for t in (32, 128, 256, 512, 1024, 2048):
        if t > length:
            continue
        count = int(failure_counts[t-1])
        result["prefix_summary"].append({
            "group_token": t, "failure_count": count, "failure_fraction": count/n,
            "survival": 1.0-count/n,
            "token_accuracy_at_position": int(correct[:, t-1].sum())/n,
            "token_accuracy_through_position": int(correct[:, :t].sum())/(n*t),
            "rmst0_restricted": sum(min(v, t) for v in lengths)/n,
            "denominator_sequences": n,
        })
    return result, np.asarray(lengths, dtype=np.int64)


def assert_matches(expected, actual, where="summary"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{where}: expected object")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{where}: missing {key}")
            assert_matches(value, actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"{where}: array length differs")
        for i, (left, right) in enumerate(zip(expected, actual)):
            assert_matches(left, right, f"{where}[{i}]")
    elif isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not np.isfinite(actual):
            raise ValueError(f"{where}: expected finite scalar")
        if not np.isclose(expected, actual, rtol=0.0, atol=1e-12):
            raise ValueError(f"{where}: {actual!r} differs from recomputed {expected!r}")
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"{where}: {actual!r} differs from recomputed {expected!r}")


def independent_primary(lengths, checkpoint_seed):
    difference = lengths[2] - lengths[0]
    generator = np.random.default_rng(63001 + checkpoint_seed)
    means = []
    for _ in range(5000):
        selected = generator.integers(0, len(difference), size=len(difference))
        means.append(float(np.sum(difference[selected], dtype=np.int64) / len(difference)))
    alpha = 0.05 / 3
    bounds = np.quantile(means, [alpha/2, 1-alpha/2], method="linear")
    return {"mean_delta_tokens": float(difference.mean()),
            "interval_tokens": bounds.tolist(), "confidence": 1-alpha,
            "bootstrap_repetitions": 5000, "bootstrap_seed": 63001+checkpoint_seed,
            "n_paired_sequences": len(difference), "quantile_method": "linear",
            "family_size": 3, "family_alpha": 0.05}


def audit(prediction_path, summary_path, *, checkpoint_seed, primary_path=None):
    path = Path(prediction_path)
    with np.load(path, allow_pickle=False) as raw:
        required = {"predictions", "gold", "head_names", "sample_ids", "input_hash",
                    "checkpoint_sha256", "checkpoint_seed", "storage"}
        if not required.issubset(raw.files):
            raise ValueError(f"prediction archive missing {sorted(required-set(raw.files))}")
        predictions, gold = raw["predictions"], raw["gold"]
        names = raw["head_names"]
        if names.dtype.kind != "U" or names.shape != (3,) or tuple(names.tolist()) != HEADS:
            raise ValueError("head identity/order must be ORIGINAL, SHORT_REFIT, MIXED_REFIT")
        if predictions.ndim != 3 or predictions.shape[0] != 3 or gold.ndim != 2 or predictions.shape[1:] != gold.shape:
            raise ValueError("prediction/gold dimensions differ")
        if not gold.size or predictions.dtype.kind not in "iu" or gold.dtype.kind not in "iu":
            raise ValueError("empty or noninteger label arrays")
        if np.any(predictions < -1) or np.any(predictions > 5) or np.any(gold < 0) or np.any(gold > 5):
            raise ValueError("labels outside S3 range, including INVALID=-1")
        identity = {}
        for key in ("checkpoint_seed", "storage", "checkpoint_sha256", "input_hash"):
            value = raw[key]
            if value.shape != ():
                raise ValueError(f"{key} must be scalar metadata")
            identity[key] = value.item()
        if identity["checkpoint_seed"] != checkpoint_seed:
            raise ValueError("checkpoint seed does not match raw metadata")
        if identity["storage"] not in ("NATIVE_FP32", "UNIFORM_8"):
            raise ValueError("storage identity outside the frozen two-codec scope")
        sample_ids = raw["sample_ids"]
        if sample_ids.shape != (gold.shape[0],) or sample_ids.dtype.kind not in "Uiu" or len(set(sample_ids.tolist())) != len(sample_ids):
            raise ValueError("missing, duplicate or malformed sample IDs")
        identity["sample_ids_sha256"] = hashlib.sha256(sample_ids.tobytes()).hexdigest()
        identity["gold_sha256"] = hashlib.sha256(gold.tobytes()).hexdigest()
        for key in ("checkpoint_sha256", "input_hash"):
            if key in identity and (not isinstance(identity[key], str) or len(identity[key]) != 64 or any(c not in "0123456789abcdef" for c in identity[key])):
                raise ValueError(f"invalid {key}")
    stated = strict_json(summary_path)
    if "heads" in stated:
        stated = stated["heads"]
    recalculated, lengths = {}, []
    for i, name in enumerate(HEADS):
        metrics, consecutive = independently_derive(predictions[i], gold)
        if name not in stated:
            raise ValueError(f"missing summary for {name}")
        assert_matches(metrics, stated[name], name)
        recalculated[name] = metrics
        lengths.append(consecutive)
    primary = independent_primary(lengths, checkpoint_seed) if identity["storage"] == "UNIFORM_8" else None
    if primary_path and identity["storage"] != "UNIFORM_8":
        raise ValueError("the primary comparison is defined only for UNIFORM_8")
    if primary_path:
        assert_matches(primary, strict_json(primary_path), "primary")
    return {"status": "PASS", "scope": "independent saved-prediction/gold scalar recomputation; no model forward",
            "predictions_sha256": sha(path), "summary_sha256": sha(summary_path),
            "identity": identity, "checkpoint_seed": checkpoint_seed,
            "n_sequences": gold.shape[0], "max_group_tokens": gold.shape[1],
            "per_head_rmst0": {k: v["rmst0"] for k, v in recalculated.items()},
            "primary_recomputed": primary,
            "primary_interval_compared": primary_path is not None}


def check_pairing(receipts):
    """Validate a complete 3-checkpoint by 2-storage table after cell audits."""
    if len(receipts) != 6:
        raise ValueError("expected exactly six recurrent rollout cells")
    seen = set()
    checkpoint_hashes = {}
    shared_input = None
    for receipt in receipts:
        if receipt.get("status") != "PASS":
            raise ValueError("cannot pair an unaudited or failed cell")
        identity = receipt["identity"]
        seed, storage = receipt["checkpoint_seed"], identity["storage"]
        key = (seed, storage)
        if seed not in (0, 1, 2) or storage not in ("NATIVE_FP32", "UNIFORM_8") or key in seen:
            raise ValueError("duplicate or unexpected rollout identity")
        seen.add(key)
        current_input = tuple(identity[k] for k in ("input_hash", "sample_ids_sha256", "gold_sha256"))
        current_input += (receipt["n_sequences"], receipt["max_group_tokens"])
        if shared_input is None:
            shared_input = current_input
        elif current_input != shared_input:
            raise ValueError("base-sequence IDs, input hash, gold or evaluation dimensions differ across cells")
        prior = checkpoint_hashes.setdefault(seed, identity["checkpoint_sha256"])
        if prior != identity["checkpoint_sha256"]:
            raise ValueError("storage arms do not share the same frozen checkpoint")
    if len(set(checkpoint_hashes.values())) != 3:
        raise ValueError("checkpoint seed identities unexpectedly reuse the same bytes")
    return {"status": "PASS", "rollout_cells": 6, "logical_readout_arms": 18,
            "n_paired_base_sequences": shared_input[-2], "max_group_tokens": shared_input[-1],
            "input_hash": shared_input[0], "checkpoint_sha256": checkpoint_hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--checkpoint-seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new audit receipt path")
    result = audit(args.predictions, args.summary, checkpoint_seed=args.checkpoint_seed, primary_path=args.primary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "n_sequences": result["n_sequences"]}))


if __name__ == "__main__":
    main()
