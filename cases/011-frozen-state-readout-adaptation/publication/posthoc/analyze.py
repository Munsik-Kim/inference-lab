#!/usr/bin/env python3
"""Recompute publication diagnostics from frozen Case011 records (NumPy only).

No model, feature extraction, optimizer or checkpoint loader is imported.
The output directory must be new. The protocol beside this file is an input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parents[2]
HEADS = ("ORIGINAL", "SHORT_REFIT", "MIXED_REFIT")
STORAGES = ("NATIVE_FP32", "UNIFORM_8")
BANDS = ((1, 32), (33, 128), (129, 256), (257, 512), (513, 1024), (1025, 2048))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def bad(value):
        raise ValueError(f"Nonfinite JSON literal: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(value, message):
    if not value:
        raise ValueError(message)


def lengths(prediction, gold):
    """Independent first-error scan; a later correct answer never clears tau."""
    require(prediction.shape == gold.shape and gold.ndim == 2 and gold.size, "prediction/gold shape")
    require(prediction.dtype.kind in "iu" and gold.dtype.kind in "iu", "integer labels required")
    require(np.all((gold >= 0) & (gold < 6)) and np.all((prediction >= -1) & (prediction < 6)), "label range")
    failures = prediction != gold
    ever = failures.any(axis=1)
    result = np.where(ever, failures.argmax(axis=1), gold.shape[1]).astype(np.int64)
    return result


def empirical_horizon(consecutive, tmax):
    # Failure at group token t iff the uninterrupted correct prefix is < t.
    risks = np.asarray([(consecutive < t).sum() for t in range(1, tmax + 1)])
    acceptable = np.flatnonzero(risks <= 0.05 * len(consecutive))
    return int(acceptable[-1] + 1) if len(acceptable) else 0


def paired_interval(delta, seed, confidence=.95, repetitions=5000):
    values = np.asarray(delta)
    require(values.ndim == 1 and values.size and np.isfinite(values).all(), "finite paired differences required")
    require(0 < confidence < 1 and repetitions > 0, "bootstrap arguments")
    generator = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        indices = generator.integers(0, len(values), size=len(values))
        means[i] = values[indices].mean()
    alpha = 1 - confidence
    return {"mean_delta_tokens": float(values.mean()),
            "interval_tokens": np.quantile(means, [alpha/2, 1-alpha/2], method="linear").tolist(),
            "confidence": confidence, "bootstrap_repetitions": repetitions,
            "bootstrap_seed": seed, "quantile_method": "linear",
            "n_paired_sequences": len(values), "sampling_unit": "paired base sequence"}


def transitions(original, mixed, gold):
    require(original.shape == mixed.shape == gold.shape and gold.size, "transition shape")
    a, b = original == gold, mixed == gold
    counts = {"both_correct": int((a & b).sum()),
              "original_only_correct": int((a & ~b).sum()),
              "mixed_only_correct": int((~a & b).sum()),
              "both_wrong_same_answer": int((~a & ~b & (original == mixed)).sum()),
              "wrong_to_wrong_different_answer": int((~a & ~b & (original != mixed)).sum())}
    require(sum(counts.values()) == gold.size, "transition partition lost observations")
    counts["all_answer_disagreement"] = int((original != mixed).sum())
    counts["both_wrong_total"] = counts["both_wrong_same_answer"] + counts["wrong_to_wrong_different_answer"]
    return counts


def difference_distribution(delta):
    require(delta.ndim == 1 and delta.size and np.isfinite(delta).all(), "difference distribution input")
    return {"n_sequences": len(delta), "longer": int((delta > 0).sum()),
            "shorter": int((delta < 0).sum()), "same": int((delta == 0).sum()),
            "mean_delta_tokens": float(delta.mean()), "minimum_tokens": int(delta.min()),
            "q25_tokens": float(np.quantile(delta, .25, method="linear")),
            "median_tokens": float(np.median(delta)), "q75_tokens": float(np.quantile(delta, .75, method="linear")),
            "maximum_tokens": int(delta.max()), "quantile_method": "linear"}


def load_cell(case, seed, storage):
    folder = case / "results" / "fresh" / f"seed{seed}" / storage
    with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    require(data["storage"].item() == storage and data["checkpoint_seed"].item() == seed, "cell identity mismatch")
    require(tuple(data["head_names"].tolist()) == HEADS, "head order mismatch")
    gold, predictions = data["gold"], data["predictions"]
    require(gold.shape == (1024, 2048) and predictions.shape == (3, *gold.shape), "incomplete frozen TEST cell")
    ids = data["sample_ids"]
    require(ids.shape == (1024,) and len(set(ids.tolist())) == len(ids), "duplicate/missing sample ID")
    data["lengths"] = np.stack([lengths(p, gold) for p in predictions])
    stored = read_json(folder / "summary.json")
    for index, name in enumerate(HEADS):
        actual = data["lengths"][index]
        expected_tau = [int(v + 1) if v < gold.shape[1] else None for v in actual]
        require(stored[name]["tau"] == expected_tau, f"tau mismatch {seed}/{storage}/{name}")
        require(abs(stored[name]["rmst0"] - float(actual.mean())) < 1e-12, "RMST mismatch")
        require(stored[name]["empirical_t005"] == empirical_horizon(actual, gold.shape[1]), "horizon mismatch")
        require(stored[name]["right_censored_count"] == int((actual == gold.shape[1]).sum()), "censor mismatch")
    count = data["score_valid_count"]
    require(count.shape == (3, 2048) and count.dtype.kind in "iu", "score count shape")
    require(np.array_equal(count, (predictions != -1).sum(axis=1)), "score/prediction denominators differ")
    for key in ("ce_sum", "gold_margin_sum", "top_margin_sum"):
        value = data[key]
        require(value.shape == count.shape and np.isfinite(value).all(), f"nonfinite score sum: {key}")
        require(np.all(value[count == 0] == 0), "nonzero sum without valid observations")
    require(np.all(data["ce_sum"] >= -1e-10), "negative CE")
    require(np.array_equal(predictions[0], predictions[1]), "SHORT predictions changed")
    receipt = read_json(folder / "receipt.json")
    require(receipt["status"] == "COMPLETE" and receipt["N"] == 1024 and receipt["T"] == 2048, "incomplete receipt")
    require(receipt["raw_sha256"] == sha(folder / "predictions.npz"), "raw receipt hash mismatch")
    require(receipt["terminal_count"] == int((data["terminal_code"] != 0).sum()), "terminal count mismatch")
    data["receipt"] = receipt
    return data


def check_pairing(left, right):
    for key in ("sample_ids", "gold", "input_hash"):
        require(np.array_equal(left[key], right[key]), f"pairing mismatch: {key}")
    require(left["checkpoint_sha256"].item() == right["checkpoint_sha256"].item(), "checkpoint pairing mismatch")


def band_rows(data, seed, storage):
    rows = []
    gold, pred, ls = data["gold"], data["predictions"], data["lengths"]
    for low, high in BANDS:
        sl = slice(low - 1, high)
        denominator = len(gold) * (high - low + 1)
        row = {"checkpoint_seed": seed, "storage": storage,
               "first_group_token": low, "last_group_token": high,
               "n_sequences": len(gold), "denominator_tokens": denominator,
               "weighting": "equal token observations inside this band; overall means weight bands by their token counts"}
        for name, head in (("original", 0), ("mixed", 2)):
            count = int(data["score_valid_count"][head, sl].sum())
            ce = float(data["ce_sum"][head, sl].sum(dtype=np.float64))
            correct = int((pred[head, :, sl] == gold[:, sl]).sum())
            first_count = int(((ls[head] >= low-1) & (ls[head] < high)).sum())
            row[name] = {"gold_ce_nats": ce/count if count else None,
                         "gold_ce_sum_nats": ce, "valid_score_observations": count,
                         "invalid_score_observations": denominator-count,
                         "correct_tokens": correct, "token_accuracy": correct/denominator,
                         "first_error_count": first_count, "first_error_fraction_of_all_sequences": first_count/len(gold),
                         "first_error_denominator": len(gold),
                         "first_error_is_conditional_hazard": False,
                         "all_item_mean_gold_margin": float(data["gold_margin_sum"][head, sl].sum())/count if count else None}
        row["gold_ce_delta_nats"] = row["mixed"]["gold_ce_nats"] - row["original"]["gold_ce_nats"] if row["mixed"]["gold_ce_nats"] is not None and row["original"]["gold_ce_nats"] is not None else None
        row["token_accuracy_delta"] = row["mixed"]["token_accuracy"] - row["original"]["token_accuracy"]
        row["transitions"] = transitions(pred[0, :, sl], pred[2, :, sl], gold[:, sl])
        rows.append(row)
    return rows


def control_audit(case, cells):
    require(all(not np.any(cell["terminal_code"]) for cell in cells.values()),
            "conditional shuffle audit requires the recorded zero-terminal dataset; do not infer historical active masks from final status")
    tokens = np.load(case / "inputs/fresh/tokens.npy", allow_pickle=False)
    fit_tokens = np.load(case / "inputs/fit/tokens.npy", allow_pickle=False)
    fit_gold = np.load(case / "inputs/fit/gold.npy", allow_pickle=False)
    bands = np.searchsorted([32, 64, 128], np.arange(1, 257), side="left")
    counts = np.zeros((4, 6, 6), dtype=np.int64)
    for b in range(4):
        selected = bands == b
        np.add.at(counts[b], (fit_tokens[:, selected].ravel(), fit_gold[:, selected].ravel()), 1)
    recorded = read_json(case / "results/input_only_fit.json")
    require(np.array_equal(counts, recorded["counts"]), "input-only FIT counts differ")
    prediction = (counts + 1).argmax(axis=-1)[np.searchsorted([32, 64, 128], np.arange(1, 2049), side="left"), tokens]
    singleton_count = 0
    for position in range(tokens.shape[1]):
        permutation = np.arange(len(tokens))
        for token in range(6):
            bucket = np.flatnonzero(tokens[:, position] == token)
            if len(bucket) < 2:
                singleton_count += int(len(bucket) == 1)
            else:
                rng = np.random.default_rng(np.random.SeedSequence([61401, position+1, token]))
                order = rng.permutation(bucket)
                permutation[order] = np.roll(order, 1)
        for cell in cells.values():
            require(np.array_equal(cell["predictions"][:, permutation, position], cell["shuffled_predictions"][:, :, position]), "shuffle permutation does not match frozen predictions")
    results = []
    reference = cells[(0, STORAGES[0])]
    for (seed, storage), cell in cells.items():
        require(np.array_equal(prediction, cell["input_only_predictions"]), "input-only TEST output differs")
        require(np.array_equal(reference["gold"], cell["gold"]), "control gold pairing")
        require(singleton_count == cell["receipt"]["shuffle_singleton_unchanged_rows_this_attempt"], "shuffle singleton ledger mismatch")
        for index, name in enumerate(HEADS):
            pred = cell["shuffled_predictions"][index]
            results.append({"checkpoint_seed": seed, "storage": storage, "head": name,
                            "kind": "STATE_SHUFFLE", "denominator_tokens": pred.size,
                            "n_sequences": pred.shape[0], "positions": pred.shape[1],
                            "token_accuracy": float((pred == cell["gold"]).mean()),
                            "accuracy_at_first_position": float((pred[:, 0] == cell["gold"][:, 0]).mean())})
    results.append({"checkpoint_seed": None, "storage": None, "head": None,
                    "kind": "INPUT_ONLY", "denominator_tokens": prediction.size,
                    "n_sequences": prediction.shape[0], "positions": prediction.shape[1],
                    "token_accuracy": float((prediction == reference["gold"]).mean()),
                    "accuracy_at_first_position": float((prediction[:, 0] == reference["gold"][:, 0]).mean())})
    return {"status": "PASS", "rows": results, "prediction_denominator_matches_each_actual_readout": True,
            "shuffle_singleton_rows_total": singleton_count,
            "labels_used_in_permutation": False, "shuffle_bucket": "same position and current group token",
            "fixed_points": "only singleton buckets; counted without claiming feature destruction",
            "input_only_fit_rows": int(counts.sum()), "input_only_fit_class_token_band_counts_recomputed": True,
            "source_contract": "Frozen rowwise final Linear commutes with the within-position feature-row permutation; stored predictions apply the same permutation. All six raw cells contain zero terminal streams.",
            "interpretation": "Overall accuracy near 1/6 is not a per-position chance assertion. At position1 the group label follows the current token and the shuffle retains it."}


def solver_audit(case):
    candidates, selected = [], []
    diagnostics = {row["model_seed"]: row for row in read_json(case / "results/head_diagnostics.json")["rows"]}
    current_audit_path = case / "publication/checkpoint_head_audit.json"
    current_audit = read_json(current_audit_path)
    require(current_audit["status"] == "PASS" and len(current_audit["rows"]) == 6, "current read-only head audit missing")
    current = {(row["seed"], row["head"]): row for row in current_audit["rows"]}
    for seed in range(3):
        for head in HEADS[1:]:
            path = case / f"results/fit-seed{seed}-{head}.json"
            raw = read_json(path)
            require(len(raw["solvers"]) == 3, "missing fitting candidates")
            eligible = [row for row in raw["selection"]["dev_scores"] if row["eligible_for_dev"]]
            winner = min(eligible, key=lambda x: (x["equal_band_ce"], -x["regularization"]))
            require(winner["candidate_index"] == raw["selection"]["candidate_index"], "DEV selection rule mismatch")
            for i, log in enumerate(raw["solvers"]):
                trace = log["objective_trace"]
                require(len(trace) == log["objective_evaluations"] and len(trace) > 0, "solver trace count")
                require(log["iterations"] <= 200 and len(trace) <= 1000, "solver budget exceeded")
                require(log["feature_rows"] == 16384 and log["trainable_parameter_count"] == 1158, "fitting shape")
                require(log["fitting_source_sha256"] == sha(case / "source/fitting.py"), "fitting source identity")
                for j, point in enumerate(trace):
                    require(point["evaluation"] == j+1, "trace sequence")
                    require(all(np.isfinite(point[k]) for k in ("objective", "ce", "centered_l2", "max_abs_gradient")), "nonfinite solver trace")
                    require(abs(point["objective"] - point["ce"] - point["centered_l2"]) < 1e-10, "objective differs from CE plus lambda-weighted penalty")
                require(any(all(abs(point[k] - log["final_"+k]) < 1e-12 for k in ("objective", "ce")) and abs(point["max_abs_gradient"]-log["final_max_abs_gradient"]) < 1e-12 for point in trace), "final iterate missing in trace")
                row = {"checkpoint_seed": seed, "head": head, "candidate_index": i,
                       "selected": i == winner["candidate_index"], "lambda": log["regularization"],
                       "status": log["status"], "iterations": log["iterations"],
                       "objective_evaluations": len(trace), "fitting_dtype": log["fitting_dtype"],
                       "export_dtype": log["execution_dtype"], "initial_objective": trace[0]["objective"],
                       "final_objective": log["final_objective"], "initial_ce": trace[0]["ce"],
                       "final_ce": log["final_ce"], "initial_max_abs_gradient": trace[0]["max_abs_gradient"],
                       "final_max_abs_gradient": log["final_max_abs_gradient"],
                       "gradient_tolerance": log["options"]["tolerance_grad"],
                       "gradient_tolerance_met": log["gradient_tolerance_met"],
                       "dev_equal_band_ce": raw["selection"]["dev_scores"][i].get("equal_band_ce"),
                       "source": path.relative_to(case).as_posix(), "source_sha256": sha(path)}
                candidates.append(row)
                if row["selected"]:
                    diag = diagnostics[seed]["heads"][head]
                    manifest_path = case / f"provenance/head_patches/seed{seed}-{head}.json"
                    manifest = read_json(manifest_path)
                    require(manifest["metadata"]["solver"] == log, "selected manifest solver differs")
                    require(manifest["trainable_parameters"] == 1158 and manifest["additional_recurrent_state_bytes"] == 0, "patch contract")
                    require(sha(manifest_path) == raw["selected"]["manifest_sha256"] == diag["patch_manifest_sha256"], "selected manifest hash")
                    actual = current[(seed, head)]
                    require(actual["base_checkpoint_sha256"] == manifest["metadata"]["base_checkpoint_sha256"], "current checkpoint identity differs")
                    for tensor in manifest["tensors"]:
                        check = next(row for row in actual["tensors"] if row["name"] == tensor["name"])
                        require(check["patch_sha256"] == tensor["sha256"], "current tensor hash differs from selected patch")
                    require(sum(row["changed_values"] for row in actual["tensors"]) == diag["parameter_changes"], "current tensor displacement differs from historical diagnostic")
                    selected.append({**row, "fp32_changed_values_from_original_receipt": diag["parameter_changes"],
                                     "fp32_changed_tensors_from_original_receipt": diag["changed_tensor_count"],
                                     "fp32_tensor_equality_evidence": "historical head_diagnostics.json plus current publication/checkpoint_head_audit.json read-only private tensor comparison; no new weight exposure",
                                     "fp64_parameter_delta": None,
                                     "fp64_parameter_delta_reason": "FP64 solution tensors were not retained in the public records; for SHORT, zero iterations and zero initial centered penalty establish no optimizer update in the inspected implementation.",
                                     "fp32_cast_dev_evaluated": True, "patch_manifest_sha256": sha(manifest_path)})
    require(len(candidates) == 18 and len(selected) == 6, "solver inventory")
    return {"status": "PASS", "candidate_count": 18, "selected_count": 6,
            "objective": "mean CE + lambda*(sum((W-W0)^2)+sum((b-b0)^2)); penalty not divided by parameter count",
            "no_new_fitting": True, "candidates": candidates, "selected": selected,
            "short_interpretation": "Final FP32 weights were unchanged under this fixed termination rule, not a general null effect of short refitting.",
            "mixed_interpretation": "Selected seeds0/1 reached the iteration cap; seed2 stopped by the optimizer small-change/direction rule. A converged optimum is not certified."}


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_start(title, description, width=840, height=390):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(description)}</desc>',
            '<rect width="100%" height="100%" fill="#faf8f4"/>',
            '<g font-family="system-ui,sans-serif" fill="#253440" font-size="15">',
            f'<text x="30" y="32" font-size="19">{html.escape(title)}</text>']


def figures(summary, destination):
    rows = summary["primary_recomputed"]
    parts = svg_start("INT8: change in the uninterrupted correct prefix", "MIXED minus ORIGINAL RMST0, group tokens. Positive is longer. Original 98.333% paired bootstrap intervals.")
    x = lambda value: 160 + (value + 22) / 25 * 620
    parts.append('<text x="30" y="60">MIXED − ORIGINAL; original 98.333% intervals; positive means longer</text>')
    parts.append(f'<line x1="{x(0):.3f}" y1="85" x2="{x(0):.3f}" y2="287" stroke="#69747a" stroke-dasharray="5 4"/>')
    for i, row in enumerate(rows):
        y = 110 + i*72
        lo, hi = row["interval_tokens"]
        parts.extend([f'<text x="30" y="{y+5}">Checkpoint {i}</text>',
                      f'<line x1="{x(lo):.3f}" y1="{y}" x2="{x(hi):.3f}" y2="{y}" stroke="#a66740" stroke-width="4"/>',
                      f'<circle cx="{x(row["mean_delta_tokens"]):.3f}" cy="{y}" r="6" fill="#253440"/>',
                      f'<text x="{x(row["mean_delta_tokens"]):.3f}" y="{y+27}" text-anchor="middle">{row["mean_delta_tokens"]:+.2f} [{lo:.2f}, {hi:.2f}]</text>'])
    for tick in (-20, -15, -10, -5, 0):
        parts.append(f'<text x="{x(tick):.3f}" y="320" text-anchor="middle">{tick}</text>')
    parts.append('<text x="440" y="354" text-anchor="middle">Change in mean consecutive correct group tokens</text></g></svg>')
    (destination / "primary_rmst0.svg").write_text("\n".join(parts)+"\n", encoding="utf-8")
    bands = [row for row in summary["bands"] if row["storage"] == "UNIFORM_8"]
    max_abs = max(abs(row["gold_ce_delta_nats"]) for row in bands)
    lower = -max_abs*1.1
    upper = max(.5, max(row["gold_ce_delta_nats"] for row in bands)*1.3)
    y = lambda value: 290 - (value-lower)/(upper-lower)*205
    parts = svg_start("INT8: gold-label CE change by position band", "MIXED minus ORIGINAL mean CE in nats per valid token; negative is lower loss. Descriptive post-hoc analysis of unchanged score sums.")
    parts.append('<text x="30" y="60">MIXED − ORIGINAL; lower CE is better; saved-record post-hoc analysis</text>')
    parts.append(f'<line x1="85" y1="{y(0):.3f}" x2="785" y2="{y(0):.3f}" stroke="#69747a" stroke-dasharray="4 3"/>')
    colors = ("#253440", "#a66740", "#267a7c")
    for seed, color in enumerate(colors):
        points = [(115+i*130, y(row["gold_ce_delta_nats"])) for i, row in enumerate(r for r in bands if r["checkpoint_seed"] == seed)]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{("none", "7 3", "2 3")[seed]}" points="'+" ".join(f"{px:.3f},{py:.3f}" for px,py in points)+'"/>')
        for px, py in points:
            if seed == 1:
                parts.append(f'<rect x="{px-4:.3f}" y="{py-4:.3f}" width="8" height="8" fill="{color}"/>')
            else:
                parts.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="{4+seed}" fill="{color}"/>')
        parts.append(f'<text x="{100+seed*235}" y="376" fill="{color}">Checkpoint {seed} {("● solid", "■ dashed", "● dotted")[seed]}</text>')
    for i, (low, high) in enumerate(BANDS):
        parts.append(f'<text x="{115+i*130}" y="319" text-anchor="middle">{low}–{high}</text>')
    for value in (0, lower/2, lower):
        parts.append(f'<text x="75" y="{y(value)+5:.3f}" text-anchor="end">{value:.1f}</text>')
    parts.append('<text x="440" y="347" text-anchor="middle">Group-token position band; CE delta in nats/token</text></g></svg>')
    (destination / "ce_by_band.svg").write_text("\n".join(parts)+"\n", encoding="utf-8")


def analyze(case, destination, make_figures=True):
    case, destination = Path(case).resolve(), Path(destination).resolve()
    require(not destination.exists(), "output exists; choose a new directory")
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = read_json(protocol_path)
    require(protocol["posthoc_interaction_interval"]["bootstrap_seed_rule"] == "71101 + checkpoint_seed", "post-hoc protocol changed")
    require(protocol["bands_inclusive"] == [list(b) for b in BANDS], "band protocol mismatch")
    cells = {(seed, storage): load_cell(case, seed, storage) for seed in range(3) for storage in STORAGES}
    reference = cells[(0, STORAGES[0])]
    fresh_ids = read_json(case / "inputs/fresh/sample_ids.json")
    require(reference["sample_ids"].tolist() == fresh_ids, "raw IDs differ from frozen manifest IDs")
    require(np.array_equal(reference["gold"], np.load(case / "inputs/fresh/gold.npy", allow_pickle=False)), "frozen gold differs")
    interaction, bands, shifts, items, primary, arm_rows = [], [], [], [], [], []
    old_primary = read_json(case / "results/derived/aggregate.json")["primary"]
    for seed in range(3):
        native, int8 = (cells[(seed, storage)] for storage in STORAGES)
        check_pairing(native, int8)
        require(np.array_equal(native["sample_ids"], reference["sample_ids"]) and np.array_equal(native["gold"], reference["gold"]), "cross-checkpoint pairing")
        a = native["lengths"][2] - native["lengths"][0]
        b = int8["lengths"][2] - int8["lengths"][0]
        interval = paired_interval(b-a, 71101+seed)
        interaction.append({"checkpoint_seed": seed, "native_delta_tokens": float(a.mean()),
                            "int8_delta_tokens": float(b.mean()), "interaction_tokens": float((b-a).mean()),
                            **interval, "evidence_kind": "POST_HOC_SAME_RECORDS",
                            "multiplicity": "descriptive pointwise95%, distinct from primary corrected intervals"})
        original_interval = paired_interval(b, 63001+seed, confidence=1-.05/3)
        require(np.allclose(original_interval["interval_tokens"], old_primary[seed]["interval_tokens"], atol=1e-12, rtol=0), "primary interval differs")
        require(original_interval["mean_delta_tokens"] == old_primary[seed]["mean_delta_tokens"], "primary mean differs")
        primary.append({"checkpoint_seed": seed, **original_interval, "evidence_kind": "ORIGINAL_PRIMARY_RECOMPUTED", "family_size": 3})
        for storage, cell in ((STORAGES[0], native), (STORAGES[1], int8)):
            ls = cell["lengths"]
            delta = ls[2]-ls[0]
            bands.extend(band_rows(cell, seed, storage))
            shifts.append({"checkpoint_seed": seed, "storage": storage, **difference_distribution(delta)})
            for head_index, name in enumerate(HEADS):
                count = int(cell["score_valid_count"][head_index].sum())
                arm_rows.append({"checkpoint_seed": seed, "storage": storage, "readout": name,
                                 "rmst0_tokens": float(ls[head_index].mean()), "empirical_t005": empirical_horizon(ls[head_index], 2048),
                                 "gold_ce_nats": float(cell["ce_sum"][head_index].sum())/count,
                                 "token_accuracy": float((cell["predictions"][head_index] == cell["gold"]).mean()),
                                 "n_sequences": 1024, "tmax": 2048, "censored_count": int((ls[head_index]==2048).sum()),
                                 "terminal_stream_count": int((cell["terminal_code"] != 0).sum()),
                                 "valid_score_observations": count, "denominator_tokens": 1024*2048})
            for index, sample in enumerate(cell["sample_ids"].tolist()):
                items.append({"checkpoint_seed": seed, "storage": storage, "sample_id": sample,
                              "tau_original": int(ls[0,index]+1) if ls[0,index] < 2048 else None,
                              "tau_mixed": int(ls[2,index]+1) if ls[2,index] < 2048 else None,
                              "rmst_delta": int(delta[index]), "tmax": 2048,
                              "category": "longer" if delta[index] > 0 else "shorter" if delta[index] < 0 else "same"})
    controls = control_audit(case, cells)
    solver = solver_audit(case)
    input_files = {"source/fitting.py", "source/controls.py", "source/run_eval.py", "configs/protocol.json", "configs/selected_heads.json", "results/head_diagnostics.json", "results/input_only_fit.json", "results/derived/aggregate.json", "inputs/fresh/tokens.npy", "inputs/fresh/gold.npy", "inputs/fresh/sample_ids.json", "inputs/fit/tokens.npy", "inputs/fit/gold.npy", "inputs/manifest.json", "publication/checkpoint_head_audit.json"}
    for seed in range(3):
        for storage in STORAGES:
            for file in ("predictions.npz", "summary.json", "receipt.json"):
                input_files.add(f"results/fresh/seed{seed}/{storage}/{file}")
        for head in HEADS[1:]:
            input_files.add(f"results/fit-seed{seed}-{head}.json")
            input_files.add(f"provenance/head_patches/seed{seed}-{head}.json")
    identities = [{"path": name, "bytes": (case/name).stat().st_size, "sha256": sha(case/name)} for name in sorted(input_files)]
    summary = {"schema": "case011-publication-posthoc-v1", "status": "PASS", "evidence_kind": "POST_HOC_SAME_RECORDS",
               "new_model_forwards": 0, "new_fits": 0, "protocol_sha256": sha(protocol_path), "analysis_source_sha256": sha(__file__),
               "independent_input_sequences": 1024, "scored_tokens_per_sequence": 2048,
               "checkpoint_count": 3, "actual_recurrent_rollouts": 6, "logical_readout_conditions": 18,
               "sample_pairing": "same1024base-sequence IDs and gold, within and across all checkpoint/storage cells",
               "units": {"rmst0": "group tokens before first wrong/INVALID; larger is longer", "ce": "nats per valid token; lower is less loss", "token_accuracy": "correct/all scored token positions; higher is more correct"},
               "primary_recomputed": primary, "arms": arm_rows, "interaction": interaction, "bands": bands,
               "error_shifts": shifts, "controls": controls, "solver": solver,
               "conditional_margin": {"status": "UNAVAILABLE_NOT_RECORDED", "result": None,
                   "reason": "Public records retain per-head/per-position margin sums across all sequences, not per-item margins or logits. Conditional margins at ORIGINAL-correct/MIXED-wrong events and full-vocabulary KL cannot be reconstructed.",
                   "available": "all-item mean gold margin by position band is descriptive and is not a selected-error margin"},
               "sources": identities}
    destination.mkdir(parents=True)
    dump(destination / "summary.json", summary)
    dump(destination / "items.json", {"schema": "case011-first-error-movement-v1", "evidence_kind": "POST_HOC_SAME_RECORDS",
                                      "row_count": len(items), "definition": "per-sequence uninterrupted-prefix difference, MIXED minus ORIGINAL",
                                      "sources": [r for r in identities if r["path"].endswith("predictions.npz")], "rows": items})
    write_csv(destination / "interaction.csv", [{"checkpoint_seed": r["checkpoint_seed"], "native_delta_tokens": r["native_delta_tokens"], "int8_delta_tokens": r["int8_delta_tokens"], "interaction_tokens": r["interaction_tokens"], "interval_low_tokens": r["interval_tokens"][0], "interval_high_tokens": r["interval_tokens"][1], "confidence": .95, "bootstrap_seed": r["bootstrap_seed"], "n_paired_sequences": 1024} for r in interaction])
    write_csv(destination / "error_shifts.csv", shifts)
    write_csv(destination / "bands.csv", [{"checkpoint_seed": r["checkpoint_seed"], "storage": r["storage"], "first_group_token": r["first_group_token"], "last_group_token": r["last_group_token"], "denominator_tokens": r["denominator_tokens"], "original_ce_nats": r["original"]["gold_ce_nats"], "mixed_ce_nats": r["mixed"]["gold_ce_nats"], "ce_delta_nats": r["gold_ce_delta_nats"], "original_accuracy": r["original"]["token_accuracy"], "mixed_accuracy": r["mixed"]["token_accuracy"], "original_first_errors": r["original"]["first_error_count"], "mixed_first_errors": r["mixed"]["first_error_count"], **r["transitions"]} for r in bands])
    write_csv(destination / "solver.csv", solver["candidates"])
    if make_figures:
        figures(summary, destination)
    dump(destination / "output_manifest.json", {"schema": "case011-posthoc-output-manifest-v1", "protocol_sha256": sha(protocol_path), "analysis_source_sha256": sha(__file__), "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(destination.iterdir()) if p.is_file()]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    result = analyze(args.case, args.output, not args.no_figures)
    print(json.dumps({"status": result["status"], "interaction": result["interaction"], "source_files": len(result["sources"])}))


if __name__ == "__main__":
    main()
