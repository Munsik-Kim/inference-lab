#!/usr/bin/env python3
"""Collect the frozen six Case 011 rollout cells; no model/feature forward.

All output goes to a new directory. A failed partial directory is retained and
never treated as a completed aggregate. Input files are read-only throughout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np

CASE = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = load_module("case011_aggregation_metrics", CASE / "source" / "metrics.py")
independent = load_module("case011_independent_auditor", CASE / "analysis" / "audit.py")

HEADS = ("ORIGINAL", "SHORT_REFIT", "MIXED_REFIT")
STORAGES = ("NATIVE_FP32", "UNIFORM_8")
RANGES = ((1, 32), (33, 128), (129, 256), (257, 512), (513, 1024), (1025, 2048))
ENDPOINTS = (32, 128, 256, 512, 1024, 2048)
SECONDARY = (
    (0, "NATIVE_FP32/MIXED_REFIT - NATIVE_FP32/ORIGINAL", ("NATIVE_FP32", 2), ("NATIVE_FP32", 0)),
    (1, "UNIFORM_8/MIXED_REFIT - UNIFORM_8/SHORT_REFIT", ("UNIFORM_8", 2), ("UNIFORM_8", 1)),
    (2, "UNIFORM_8/ORIGINAL - NATIVE_FP32/ORIGINAL", ("UNIFORM_8", 0), ("NATIVE_FP32", 0)),
    (3, "UNIFORM_8/SHORT_REFIT - NATIVE_FP32/SHORT_REFIT", ("UNIFORM_8", 1), ("NATIVE_FP32", 1)),
    (4, "UNIFORM_8/MIXED_REFIT - NATIVE_FP32/MIXED_REFIT", ("UNIFORM_8", 2), ("NATIVE_FP32", 2)),
)


def sha(path):
    return independent.sha(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("refusing an empty result table")
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise ValueError("CSV row fields differ")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def compact_head(summary):
    """Keep scalar/prefix definitions; full first-error records remain in cells."""
    return {key: value for key, value in summary.items()
            if key not in ("tau", "survival_by_token", "token_accuracy_by_token")}


def load_cell(case, seed, storage, expected_shape=(1024, 2048)):
    directory = Path(case) / "results" / "fresh" / f"seed{seed}" / storage
    raw_path = directory / "predictions.npz"
    summary_path = directory / "summary.json"
    receipt_path = directory / "receipt.json"
    required = {"predictions", "gold", "head_names", "sample_ids", "input_hash", "checkpoint_sha256",
                "checkpoint_seed", "storage", "shuffled_predictions", "input_only_predictions",
                "ce_sum", "gold_margin_sum", "top_margin_sum", "score_valid_count"}
    with np.load(raw_path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError(f"{seed}/{storage}: missing fields {sorted(required-set(archive.files))}")
        data = {key: archive[key] for key in required}
    if tuple(data["head_names"].tolist()) != HEADS or data["storage"].shape != () or data["storage"].item() != storage:
        raise ValueError("head or storage identity differs from the frozen cell")
    if data["checkpoint_seed"].shape != () or data["checkpoint_seed"].item() != seed:
        raise ValueError("checkpoint seed differs from the cell path")
    gold = data["gold"]
    if gold.shape != expected_shape:
        raise ValueError(f"incomplete TEST cell: expected {expected_shape}, found {gold.shape}")
    for name in ("predictions", "shuffled_predictions"):
        value = data[name]
        if value.shape != (3,) + gold.shape or value.dtype.kind not in "iu" or np.any(value < -1) or np.any(value > 5):
            raise ValueError(f"malformed {name}")
    control = data["input_only_predictions"]
    if control.shape != gold.shape or control.dtype.kind not in "iu" or np.any(control < 0) or np.any(control > 5):
        raise ValueError("malformed input-only predictions")
    count = data["score_valid_count"]
    if count.shape != (3, gold.shape[1]) or count.dtype.kind not in "iu" or np.any(count < 0) or np.any(count > gold.shape[0]):
        raise ValueError("invalid score denominators")
    expected_count = np.count_nonzero(data["predictions"] != -1, axis=1)
    if not np.array_equal(count, expected_count):
        raise ValueError("score valid counts disagree with finite predictions")
    for name in ("ce_sum", "gold_margin_sum", "top_margin_sum"):
        value = data[name]
        if value.shape != count.shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"invalid public score sums: {name}")
        if np.any(value[count == 0] != 0):
            raise ValueError(f"{name}: no-valid-observation sum must be zero")
    if np.any(data["ce_sum"] < -1e-10) or np.any(data["top_margin_sum"] < -1e-10):
        raise ValueError("CE and top-two margin sums cannot be negative")
    independent.strict_json(receipt_path)
    stored_summary = independent.strict_json(summary_path)
    if "heads" in stored_summary:
        stored_summary = stored_summary["heads"]
    summaries = metrics.summarize_predictions(data["predictions"], gold)
    for name in HEADS:
        independent.assert_matches(summaries[name], stored_summary[name], f"{seed}/{storage}/{name}")
    data.update({"summary": summaries, "path": raw_path, "summary_path": summary_path,
                 "source_identity": {"predictions_sha256": sha(raw_path), "summary_sha256": sha(summary_path),
                                     "receipt_sha256": sha(receipt_path),
                                     "cell": directory.relative_to(case).as_posix()}})
    return data


def score_rows(cell, seed, storage, ranges=RANGES, endpoints=ENDPOINTS):
    rows = []
    n, length = cell["gold"].shape
    slices = [("position_range", lo, min(hi, length)) for lo, hi in ranges if lo <= length]
    slices += [("single_position", t, t) for t in endpoints if t <= length]
    for head, name in enumerate(HEADS):
        for kind, first, last in slices:
            segment = slice(first-1, last)
            count = int(cell["score_valid_count"][head, segment].sum(dtype=np.int64))
            total = n * (last-first+1)
            sums = {key: float(cell[key][head, segment].sum(dtype=np.float64))
                    for key in ("ce_sum", "gold_margin_sum", "top_margin_sum")}
            correct = cell["predictions"][head, :, segment] == cell["gold"][:, segment]
            rows.append({"checkpoint_seed": seed, "storage": storage, "head": name,
                         "position_scope": kind, "first_group_token": first, "last_group_token": last,
                         "valid_score_observations": count, "invalid_score_observations": total-count,
                         "total_token_observations": total,
                         "gold_ce_nats": sums["ce_sum"]/count if count else None,
                         "gold_margin": sums["gold_margin_sum"]/count if count else None,
                         "top1_top2_margin": sums["top_margin_sum"]/count if count else None,
                         "gold_ce_sum_nats": sums["ce_sum"], "gold_margin_sum": sums["gold_margin_sum"],
                         "top1_top2_margin_sum": sums["top_margin_sum"],
                         "token_accuracy": float(correct.mean()), "correct_tokens": int(correct.sum())})
    return rows


def contrast_row(seed, result, name, kind):
    return {"checkpoint_seed": seed, "comparison": name, "kind": kind,
            "rmst0_delta_tokens": result["mean_delta_tokens"],
            "interval_low_tokens": result["interval_tokens"][0],
            "interval_high_tokens": result["interval_tokens"][1],
            "interval_confidence": result["confidence"],
            "bootstrap_repetitions": result["bootstrap_repetitions"],
            "bootstrap_seed": result["bootstrap_seed"],
            "n_paired_sequences": result["n_paired_sequences"]}


def plot_outputs(cells, primary_rows, directory):
    """Recorded results only. Shaded bootstrap RMST intervals never label horizons."""
    directory.mkdir()
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="case011-matplotlib-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 120, "savefig.dpi": 160})
    colors = ("#273747", "#B27745", "#267A7C")
    ticks = [1, 32, 128, 256, 512, 1024, 2048]

    def survival_panel(axis, seed, storage):
        cell = cells[(seed, storage)]
        times = np.arange(1, cell["gold"].shape[1]+1)
        for head, color in zip(HEADS, colors):
            axis.plot(times, cell["summary"][head]["survival_by_token"], label=head, color=color, linewidth=1.8)
        axis.axvline(256, linestyle=":", color="#777777", linewidth=1)
        axis.set_xscale("log")
        axis.set_xticks(ticks, [str(t) for t in ticks], rotation=35)
        axis.set_ylim(-.015, 1.015)
        axis.set_title(f"Checkpoint {seed} | {storage}")
        axis.set_xlabel("Group-token position (log axis)")
        axis.set_ylabel("S(t): no readout error through t")
        axis.grid(alpha=.2)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True, constrained_layout=True)
    for seed, axis in enumerate(axes):
        survival_panel(axis, seed, "UNIFORM_8")
    axes[0].legend(fontsize=8)
    figure.suptitle("INT8 frozen states, three readouts | 1,024 paired sequences per checkpoint\nDotted line: maximum refitting position 256")
    figure.savefig(directory / "int8_survival.png")
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(12, 11), sharey=True, constrained_layout=True)
    for seed in range(3):
        for column, storage in enumerate(STORAGES):
            survival_panel(axes[seed, column], seed, storage)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Shared recurrent rollout within each panel; readout intervention only")
    figure.savefig(directory / "all_storage_survival.png")
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for seed in range(3):
        for column, storage in enumerate(STORAGES):
            cell = cells[(seed, storage)]
            axis = axes[seed, column]
            length = cell["gold"].shape[1]
            for head, color in enumerate(colors):
                x, y = [], []
                for lo in range(0, length, 32):
                    hi = min(lo+32, length)
                    count = int(cell["score_valid_count"][head, lo:hi].sum())
                    x.append((lo+1+hi)/2)
                    y.append(float(cell["ce_sum"][head, lo:hi].sum())/count if count else np.nan)
                axis.plot(x, y, label=HEADS[head], color=color, linewidth=1.7)
            axis.axvline(256, linestyle=":", color="#777777", linewidth=1)
            axis.set_title(f"Checkpoint {seed} | {storage}")
            axis.set_xlabel("Group-token position; 32-position bins")
            axis.set_ylabel("Mean gold CE (nats / valid token)")
            axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Gold CE on the same frozen TEST inputs; invalid score counts are retained in scores.csv")
    figure.savefig(directory / "gold_ce_by_position.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for row in primary_rows:
        seed = row["checkpoint_seed"]
        axis.hlines(seed, row["interval_low_tokens"], row["interval_high_tokens"], color=colors[2], linewidth=3)
        axis.scatter(row["rmst0_delta_tokens"], seed, color=colors[2], s=45, zorder=3)
    axis.axvline(0, color="#555555", linewidth=1)
    axis.set_yticks([0, 1, 2], ["Checkpoint 0", "Checkpoint 1", "Checkpoint 2"])
    axis.set_xlabel("RMST0 difference in consecutive correct group tokens\nUNIFORM_8 / MIXED_REFIT − UNIFORM_8 / ORIGINAL")
    axis.set_title("Primary paired comparison: 5,000 base-sequence resamples\n98.333% intervals per checkpoint; family 95% target\nBonferroni correction; bootstrap approximation", fontsize=11)
    axis.grid(axis="x", alpha=.2)
    figure.savefig(directory / "primary_rmst0.png")
    plt.close(figure)
    return ["figures/" + path.name for path in sorted(directory.glob("*.png"))]


def collect(case, input_only_fit, output, *, make_figures=True):
    case, output, input_only_fit = Path(case).resolve(), Path(output).resolve(), Path(input_only_fit).resolve()
    if output.exists():
        raise ValueError("aggregate output already exists; choose a new directory")
    if output == case / "results" / "fresh" or case / "results" / "fresh" in output.parents:
        raise ValueError("aggregate output must not be inside the frozen raw cells")
    fit_control = independent.strict_json(input_only_fit)
    counts = np.asarray(fit_control["counts"])
    if counts.shape != (4, 6, 6) or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("invalid FIT input-only label/token/band counts")
    if counts.sum() != 512*256 or counts.sum(axis=(1,2)).tolist() != [512*32, 512*32, 512*64, 512*128]:
        raise ValueError("input-only control FIT rows/bands do not match the frozen 512×256 design")
    cells = {(seed, storage): load_cell(case, seed, storage) for seed in range(3) for storage in STORAGES}
    first_control = cells[(0, STORAGES[0])]["input_only_predictions"]
    for cell in cells.values():
        if not np.array_equal(first_control, cell["input_only_predictions"]):
            raise ValueError("input-only control unexpectedly changes across checkpoint/storage")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        (stage / "independent_audit").mkdir()
        primary, primary_rows, secondary, secondary_rows, arms, scores, controls = [], [], [], [], [], [], []
        audits, case_summaries, sources = [], {}, {}
        for seed in range(3):
            int8 = cells[(seed, "UNIFORM_8")]
            comparison = metrics.primary_comparison(int8["predictions"][0], int8["predictions"][2], int8["gold"], seed)
            comparison["checkpoint_seed"] = seed
            primary.append(comparison)
            primary_rows.append(contrast_row(seed, comparison, comparison["comparison"], "primary_family_3"))
            primary_file = stage / "independent_audit" / f"seed{seed}_primary.json"
            write_json(primary_file, comparison)
            for index, name, candidate, baseline in SECONDARY:
                cand, base = cells[(seed, candidate[0])], cells[(seed, baseline[0])]
                delta = metrics.continuous_lengths(cand["predictions"][candidate[1]], cand["gold"]) - metrics.continuous_lengths(base["predictions"][baseline[1]], base["gold"])
                result = metrics.paired_bootstrap_rmst(delta, seed=63101+seed*10+index, confidence=.95)
                result.update({"checkpoint_seed": seed, "pair_index": index, "comparison": name,
                               "scope": "secondary pointwise interval; no familywise claim"})
                secondary.append(result)
                secondary_rows.append(contrast_row(seed, result, name, "secondary_pointwise"))
            for storage in STORAGES:
                cell = cells[(seed, storage)]
                key = f"seed{seed}/{storage}"
                sources[key] = cell["source_identity"]
                receipt = independent.audit(cell["path"], cell["summary_path"], checkpoint_seed=seed,
                                            primary_path=primary_file if storage == "UNIFORM_8" else None)
                audits.append(receipt)
                write_json(stage / "independent_audit" / f"seed{seed}_{storage}.json", receipt)
                case_summaries[key] = {name: compact_head(cell["summary"][name]) for name in HEADS}
                for head, name in enumerate(HEADS):
                    summary = cell["summary"][name]
                    arms.append({"checkpoint_seed": seed, "storage": storage, "head": name,
                                 "n_sequences": summary["n_sequences"], "max_group_tokens": summary["max_group_tokens"],
                                 "rmst0_tokens": summary["rmst0"], "empirical_t005_tokens": summary["empirical_t005"],
                                 "token_accuracy": summary["token_accuracy"], "right_censored_count": summary["right_censored_count"],
                                 "invalid_prediction_count": summary["invalid_prediction_count"],
                                 "sequences_recovering_after_first_error": summary["sequences_with_correct_token_after_first_error"]})
                    shuffled = metrics.summarize_head(cell["shuffled_predictions"][head], cell["gold"])
                    independently_shuffled, _ = independent.independently_derive(cell["shuffled_predictions"][head], cell["gold"])
                    independent.assert_matches(independently_shuffled, shuffled, f"{key}/{name}/shuffle")
                    controls.append({"checkpoint_seed": seed, "storage": storage, "head": name,
                                     "control": "same-position/current-token feature shuffle",
                                     **compact_head(shuffled)})
                scores.extend(score_rows(cell, seed, storage))
        input_summary = metrics.summarize_head(first_control, cells[(0, STORAGES[0])]["gold"])
        independently_input, _ = independent.independently_derive(first_control, cells[(0, STORAGES[0])]["gold"])
        independent.assert_matches(independently_input, input_summary, "input-only control")
        controls.append({"checkpoint_seed": None, "storage": "MODEL_INDEPENDENT", "head": "INPUT_ONLY",
                         "control": "FIT token/position-band categorical predictor", **compact_head(input_summary)})
        pairing = independent.check_pairing(audits)
        write_json(stage / "independent_audit" / "pairing.json", pairing)
        control_rows = [{"checkpoint_seed": c["checkpoint_seed"], "storage": c["storage"], "head": c["head"],
                         "control": c["control"], "n_sequences": c["n_sequences"], "max_group_tokens": c["max_group_tokens"],
                         "rmst0_tokens": c["rmst0"], "empirical_t005_tokens": c["empirical_t005"],
                         "token_accuracy": c["token_accuracy"], "invalid_prediction_count": c["invalid_prediction_count"]} for c in controls]
        for filename, rows in (("arms.csv", arms), ("primary.csv", primary_rows), ("secondary.csv", secondary_rows),
                               ("scores.csv", scores), ("controls.csv", control_rows)):
            write_csv(stage / filename, rows)
        figures = plot_outputs(cells, primary_rows, stage / "figures") if make_figures else []
        result = {"schema": "case011-readout-aggregate-v1", "status": "COMPLETE_SAVED_SCALAR_AGGREGATION",
                  "scope": "new Case 011 fixed TEST; frozen states shared by three readouts per storage/checkpoint",
                  "logical_arms": 18, "recurrent_rollout_cells": 6,
                  "primary": primary, "secondary": secondary, "arms": arms, "scores": scores,
                  "head_summaries": case_summaries, "controls": controls,
                  "fit_input_only_control": {"source_sha256": sha(input_only_fit), "counts": counts.tolist(),
                                             "fit_rows": int(counts.sum()), "definition": fit_control.get("definition")},
                  "input_pairing": pairing, "source_cells": sources, "figures": figures,
                  "independent_control_summary_checks": 19,
                  "interpretation": {"rmst0": "consecutive correct group tokens before first error, not all-token accuracy",
                                     "primary_intervals": "three 98.333...% paired percentile-bootstrap intervals; approximate Bonferroni family 95%",
                                     "secondary_intervals": "pointwise 95%; not confirmatory familywise intervals",
                                     "empirical_t005": "descriptive every-token threshold; no confidence-supported horizon calculated",
                                     "controls": "diagnostics, not six new model conditions; unchanged singleton shuffle rows are recorded by runner"}}
        write_json(stage / "aggregate.json", result)
        write_json(stage / "source_hashes.json", sources)
        # Detect accidental edits or concurrent replacement of measured inputs.
        for cell in cells.values():
            if sha(cell["path"]) != cell["source_identity"]["predictions_sha256"] or sha(cell["summary_path"]) != cell["source_identity"]["summary_sha256"]:
                raise ValueError("a source cell changed during aggregation")
        stage.rename(output)
        return result
    except Exception as exc:
        write_json(stage / "FAILED_AGGREGATION.json", {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--input-only-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true", help="run the same numeric audit without importing Matplotlib")
    args = parser.parse_args()
    result = collect(args.case, args.input_only_fit, args.output, make_figures=not args.no_figures)
    print(json.dumps({"status": result["status"], "rollout_cells": result["recurrent_rollout_cells"],
                      "logical_arms": result["logical_arms"], "independent_pairing": result["input_pairing"]["status"]}))


if __name__ == "__main__":
    main()
