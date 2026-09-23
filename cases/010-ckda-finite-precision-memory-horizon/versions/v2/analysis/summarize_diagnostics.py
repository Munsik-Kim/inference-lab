"""Model-free diagnostic tables from retained predictions and scalar trajectories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ("D00", "D10", "D01", "D11")
ARMS = ("NATIVE_FP32", "UNIFORM_8", "UNIFORM_5", "LOWRANK_4_8_R2", "MIXED_5_6_BUDGET")
TIMES = (128, 512, 2048)
TRACE_FIELDS = (
    "state_norm", "residual_norm", "native_state_norm", "fp64_shadow_state_norm",
    "state_scale_minimum", "state_scale_maximum", "residual_scale_minimum", "residual_scale_maximum",
    "self_writeback_mse", "projection_leakage_mse", "native_state_error_l2", "native_state_error_mse",
    "fp64_shadow_error_l2", "fp64_shadow_error_mse", "gold_margin", "top_two_margin",
)
EXTREMA_FIELDS = ("state_norm", "residual_norm", "native_state_error_l2", "fp64_shadow_error_l2",
                  "self_writeback_mse", "projection_leakage_mse")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_json(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_json(child)
    elif isinstance(value, list):
        for child in value:
            finite_json(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON number")


def read_json(path):
    value = json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    finite_json(value)
    return value


def write_json(path, value):
    finite_json(value)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refuse an empty measurement table")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("Inconsistent table schema")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def statistics(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return dict(valid_count=0, mean=None, median=None, minimum=None, maximum=None)
    scale = np.max(np.abs(finite))
    mean = 0. if scale == 0 else float(scale*np.mean(finite/scale))
    return dict(valid_count=int(finite.size), mean=mean, median=float(np.median(finite)),
                minimum=float(finite.min()), maximum=float(finite.max()))


class Inputs:
    def __init__(self, root):
        self.root = root
        self.inventory = {}
        self.checked_cells = []
        manifest = self.json(root/"inputs/manifest.json")
        self.cohort = manifest["cohorts"]["diagnostic"]
        self.gold = self.array_file(root/self.cohort["gold_file"], self.cohort["gold_file_sha256"])
        self.ids = self.json(root/self.cohort["sample_ids_file"])
        if file_sha(root/self.cohort["sample_ids_file"]) != self.cohort["sample_ids_sha256"]:
            raise ValueError("Sample-ID identity mismatch")
        self.track(root/self.cohort["tokens_file"], self.cohort["tokens_file_sha256"])
        self.protocol = self.json(root/"protocol_v2.json")
        freeze = self.json(root/"provenance/protocol_freeze.json")
        if file_sha(root/"protocol_v2.json") != freeze["protocol_sha256"]:
            raise ValueError("Protocol identity mismatch")
        if self.gold.shape != (32, 2048) or len(self.ids) != 32:
            raise ValueError("Expected frozen diagnostic dimensions")

    def track(self, path, expected=None):
        path = Path(path)
        digest = file_sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Hash mismatch: {path.relative_to(self.root)}")
        self.inventory[str(path.relative_to(self.root))] = {"sha256": digest, "bytes": path.stat().st_size}

    def json(self, path):
        self.track(path)
        return read_json(path)

    def array_file(self, path, expected):
        self.track(path, expected)
        return np.load(path, allow_pickle=False)

    def cell(self, kind, seed, arm):
        folder = self.root/"results"/kind/f"seed{seed}"/arm
        index = self.json(folder.parent/"index.json")
        self.track(folder/"summary.json", index["cells"][arm])
        summary = read_json(folder/"summary.json")
        if (index["status"] != "COMPLETE" or summary["execution"]["status"] != "COMPLETE"
                or summary["family_size"] is not None or summary["N"] != 32 or summary["T"] != 2048
                or summary["model_seed"] != seed or summary["arm"] != arm
                or summary["sample_ids"] != self.ids or summary["sample_ids_sha256"] != self.cohort["sample_ids_sha256"]
                or summary["protocol_sha256"] != file_sha(self.root/"protocol_v2.json")
                or summary["input_manifest_sha256"] != file_sha(self.root/"inputs/manifest.json")):
            raise ValueError("Cell scope/identity/completion mismatch")
        self.track(folder/"predictions.npz", summary["artifact_sha256"])
        with np.load(folder/"predictions.npz", allow_pickle=False) as z:
            pred, gold, packed = z["predictions"].copy(), z["gold"].copy(), z["correctness"].copy()
        if pred.shape != (32, 2049) or not np.array_equal(gold, self.gold):
            raise ValueError("Cells do not use exactly the same retained gold")
        wrong = pred[:, 1:] != gold
        tau = [int(np.flatnonzero(row)[0])+1 if row.any() else None for row in wrong]
        times = np.array([2049 if value is None else value for value in tau])
        if (tau != summary["tau"] or not np.array_equal(np.packbits(~wrong, axis=1, bitorder="little"), packed)
                or float(np.minimum(times-1, 2048).mean()) != summary["RMST0"]):
            raise ValueError("Prediction-to-tau/RMST/correctness inconsistency")
        scalar_file = folder/"scalar_trajectories.npz"
        for name, digest in summary["diagnostic_files_sha256"].items():
            self.track(folder/name, digest)
        with np.load(scalar_file, allow_pickle=False) as z:
            scalars = {key: z[key].copy() for key in z.files}
        if any(value.shape != (32, 2049) for value in scalars.values()):
            raise ValueError("Unexpected scalar trajectory dimensions")
        if any(np.isinf(value).any() for value in scalars.values()):
            raise ValueError("Infinite scalar should have been marked invalid")
        if not np.isnan(scalars["state_norm"][~scalars["active"]]).all():
            raise ValueError("Terminal placeholders leaked into state-norm diagnostics")
        self.checked_cells.append({"kind": kind, "seed": seed, "arm": arm,
                                   "summary_sha256": file_sha(folder/"summary.json")})
        return summary, pred, times, scalars


def endpoint(seed, arm, summary, pred, times, gold):
    correct = pred[:, 1:] == gold
    survival = (times[:, None] > np.arange(1, 2049)).mean(0)
    qualified = np.flatnonzero(1-survival <= .05)
    empirical = int(qualified[-1])+1 if qualified.size else None
    if (empirical != summary["empirical_T05_all_tokens"] or float(correct.mean()) != summary["token_accuracy"]
            or float(correct[:, -512:].mean()) != summary["final_quarter_accuracy"]):
        raise ValueError("Endpoint does not reconstruct from retained predictions")
    return {"model_seed": seed, "mode": arm, "N": 32, "scored_tokens_per_sequence": 2048,
            "RMST0": summary["RMST0"], "empirical_T05_all_tokens": empirical,
            "token_accuracy": float(correct.mean()), "final_quarter_accuracy": float(correct[:, -512:].mean()),
            "S128": float((times > 128).mean()), "S512": float((times > 512).mean()),
            "S2048": float((times > 2048).mean()), "BOS_accuracy": float((pred[:, 0] == 0).mean()),
            "terminal_sequences": summary["terminal_count"], "confidence_family": None}


def extreme(seed, arm, metric, values, times, ids):
    valid = np.isfinite(values)
    out = dict(model_seed=seed, arm=arm, metric=metric, valid_sample_writes=int(valid.sum()),
               maximum=None, sample_index=None, sample_id=None, write_position=None,
               first_failure_tau=None, is_BOS=None, at_or_after_first_failure=None,
               strictly_after_first_failure=None)
    if not valid.any():
        return out
    i, position = np.unravel_index(np.argmax(np.where(valid, values, -np.inf)), values.shape)
    tau = None if times[i] == 2049 else int(times[i])
    out.update(maximum=float(values[i, position]), sample_index=int(i), sample_id=ids[i],
               write_position=int(position), first_failure_tau=tau, is_BOS=bool(position == 0),
               at_or_after_first_failure=bool(tau is not None and position >= tau),
               strictly_after_first_failure=bool(tau is not None and position > tau))
    return out


def definitions():
    return {
        "scope": "32 frozen diagnostic sequences per fixed checkpoint; no pooling checkpoints, confidence intervals, fitted threshold, new inference, or causal attribution",
        "indexing": "BOS is write0 and unscored; group token t is write t, t=1..2048",
        "missing": "CSV empty field / JSON null means unavailable; raw NPZ NaN is an invalid or inapplicable scalar and is excluded with explicit finite counts",
        "precision_endpoints": {
            "RMST0": "mean(min(tau-1,2048)); censored contributes2048; units consecutive correct group tokens",
            "empirical_T05_all_tokens": "largest integer t in1..2048 whose observed first-failure fraction is <=0.05; empirical only, no confidence support",
            "token_accuracy": "correct predictions divided by32*2048 scored token positions, including recovery after first error",
            "final_quarter_accuracy": "correct predictions at1537..2048 divided by32*512",
            "S128/S512/S2048": "fraction of32 sequences with tau greater than the named horizon",
            "BOS_accuracy": "fraction of32 BOS predictions equal to identity class0; excluded from tau",
            "terminal_sequences": "number with an absorbing nonfinite recurrent-state failure, distinct from finite-state wrong labels",
        },
        "precision_pairs": "All6 unordered mode pairs per seed; compare complete integer predictions, not logits. Scored denominator65536; BOS denominator32; tau_equal_sequences denominator32.",
        "precision_D00_errors": {
            "population": "all2049 writes including BOS and post-label-failure rollout; each scalar has its own finite valid sample-write count",
            "state_relative_l2": "||S_mode-S_D00||F / ||S_D00||F, only when both stored scalars are finite and D00 norm strictly positive; no epsilon floor",
            "state_difference_l2_from_D00": "Frobenius distance between current precision-mode and D00 states, diagnostically accumulated inFP64",
            "state_difference_mse_from_D00": "mean squared difference over3072 state entries of one sample/current write",
            "logit_max_difference_from_D00": "maximum absolute difference among6 current class logits; state/readout precision varies as frozen",
        },
        "trace_metrics": {
            "state_norm": "Frobenius norm of the full represented after-write candidate state, all12*16*16 entries",
            "residual_norm": "Frobenius norm of the decoded residual expanded through its fixed basis; zero when no residual exists",
            "native_state_norm": "Frobenius norm of independently advanced evaluator FP32 state; not passed to candidate",
            "fp64_shadow_state_norm": "Frobenius norm of independent FP64 recurrence using promoted fixed FP32 coefficients; not arbitrary-precision ground truth",
            "state_scale_minimum/state_scale_maximum": "minimum/maximum of12 stored per-head max-absolute FP32 scales for the base state; absent for native floating codec",
            "residual_scale_minimum/residual_scale_maximum": "minimum/maximum of12 stored residual-coefficient scales; absent when residual codec absent",
            "self_writeback_mse": "mean over3072 entries of(candidate's own FP32 updated state - represented after-write state)^2, with subtraction/squaring diagnosed inFP64; not native-state error",
            "projection_leakage_mse": "mean over3072 entries of(error32 - expand64(project32(error32)))^2; error32=own update-decoded base; zero when no projection basis; distinct from residual-code rounding",
            "native_state_error_l2/native_state_error_mse": "Frobenius distance / mean squared per-entry difference from independent FP32 shadow at the same write; valid only if candidate and shadow active",
            "fp64_shadow_error_l2/fp64_shadow_error_mse": "corresponding distance / per-entry MSE from independent FP64 shadow",
            "gold_margin": "gold logit minus largest other-class logit; negative indicates strictly losing gold, zero permits a tie",
            "top_two_margin": "largest minus second-largest logit, independent of whether winner is correct",
        },
        "trace_fixed_times": "At each declared group position128/512/2048, statistics summarize only finite metric values among32 samples; valid_N is a stream count. Active_N counts nonterminal candidate streams; before/after counts refer to label tau and need not equal active_N.",
        "rank2_before_after": "BOS excluded. Before:1<=t<tau. At/after:tau<=t<=2048. Each row pools eligible sample-writes; valid_sample_writes is not an independent sample N. contributing_sequences counts samples with any finite eligible value; mean_of_sequence_means weights contributing samples equally, while mean weights each valid sample-write equally. Temporal dependence is not treated as independent evidence.",
        "trace_extrema": "Maximum finite scalar over all32*2049 writes including BOS/post-failure; first flattened(sample then write) maximizer retained. Flags identify whether the extremum occurs at/after or strictly after that sample's first wrong label; no causal interpretation.",
        "statistics": "mean, median, minimum, maximum over explicitly counted finite values; FP64 mean scales values before summation; no intervals or tests. Counts in fixed-time and pooled-time tables have different units.",
        "precision_modes": {"D00": "FP32 recurrence / original FP32 readout", "D10": "FP64 recurrence / state castFP32 then original readout", "D01": "FP32 recurrence / explicit trueFP64 readout", "D11": "FP64 recurrence / explicit trueFP64 readout"},
        "fixed_values": "Existing FP32 coefficient and parameter values are promoted, never reprojected or retrained; reduction order limitations remain in source precision metadata",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Derived output directory must be new")
    inputs = Inputs(args.root.resolve())
    tables = {name: [] for name in ("precision_endpoints", "precision_pairs", "precision_D00_errors",
                                    "trace_fixed_times", "rank2_before_after", "trace_extrema")}
    precision_predictions = {}
    for seed in range(3):
        cells = {mode: inputs.cell("precision", seed, mode) for mode in MODES}
        base_norm = cells["D00"][3]["state_norm"]
        precision_predictions[seed] = cells["D00"][1]
        for mode, (summary, pred, times, scalars) in cells.items():
            tables["precision_endpoints"].append(endpoint(seed, mode, summary, pred, times, inputs.gold))
            fields = ["state_difference_l2_from_D00", "state_difference_mse_from_D00", "logit_max_difference_from_D00"]
            delta = scalars[fields[0]]
            relative = np.divide(delta, base_norm, out=np.full_like(delta, np.nan),
                                 where=np.isfinite(base_norm) & (base_norm > 0) & np.isfinite(delta))
            row = dict(model_seed=seed, mode=mode, N=32, writes_per_sequence=2049,
                       includes_BOS=True, includes_post_failure=True)
            for metric, values in [(key, scalars[key]) for key in fields]+[("state_relative_l2", relative)]:
                row.update({metric+"__"+name: value for name, value in statistics(values).items()})
            tables["precision_D00_errors"].append(row)
        for left, right in itertools.combinations(MODES, 2):
            a, b = cells[left], cells[right]
            differences = a[1][:, 1:] != b[1][:, 1:]
            tables["precision_pairs"].append(dict(model_seed=seed, mode_a=left, mode_b=right,
                N=32, scored_prediction_denominator=65536, scored_prediction_disagreements=int(differences.sum()),
                scored_disagreement_fraction=float(differences.mean()), BOS_denominator=32,
                BOS_disagreements=int((a[1][:, 0] != b[1][:, 0]).sum()),
                tau_equal_sequences=int((a[2] == b[2]).sum()), tau_comparison_denominator=32))
        del cells
        for arm in ARMS:
            summary, pred, times, scalars = inputs.cell("trace", seed, arm)
            if arm == "NATIVE_FP32":
                if not np.array_equal(pred, precision_predictions[seed]) or not np.all(scalars["native_state_error_mse"] == 0):
                    raise ValueError("Native trace differs from D00 or its independent FP32 shadow")
            positions = np.arange(2049)[None, :]
            before = (positions > 0) & (positions < times[:, None])
            after = (positions > 0) & (positions >= times[:, None])
            for position in TIMES:
                for metric in TRACE_FIELDS:
                    stats = statistics(scalars[metric][:, position])
                    tables["trace_fixed_times"].append(dict(model_seed=seed, arm=arm, group_position=position,
                        metric=metric, N=32, active_N=int(scalars["active"][:, position].sum()),
                        before_first_failure_N=int(before[:, position].sum()), at_or_after_first_failure_N=int(after[:, position].sum()),
                        valid_N=stats.pop("valid_count"), **stats))
            for metric in EXTREMA_FIELDS:
                tables["trace_extrema"].append(extreme(seed, arm, metric, scalars[metric], times, inputs.ids))
            if arm == "LOWRANK_4_8_R2":
                for label, eligible in (("before_first_failure", before), ("at_or_after_first_failure", after)):
                    for metric in TRACE_FIELDS:
                        values = scalars[metric]
                        valid = eligible & np.isfinite(values)
                        per_sequence_means = [statistics(values[i][valid[i]])["mean"] for i in range(32) if valid[i].any()]
                        stats = statistics(values[valid])
                        tables["rank2_before_after"].append(dict(model_seed=seed, arm=arm, period=label, metric=metric,
                            N_sequences=32, eligible_sample_writes=int(eligible.sum()),
                            contributing_sequences=int(valid.any(1).sum()), valid_sample_writes=stats.pop("valid_count"),
                            mean_of_sequence_means=statistics(per_sequence_means)["mean"], **stats))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        write_csv(args.output/(name+".csv"), rows)
    write_json(args.output/"field_definitions.json", definitions())
    inputs.track(Path(__file__).resolve())
    write_json(args.output/"source_hashes.json", inputs.inventory)
    audit = dict(status="PASS", model_execution=False, tested_thresholds=False, confidence_intervals=False,
                 source_JSON_all_finite=True, same_retained_gold_in_all_cells=True,
                 prediction_tau_RMST_correctness_reconstructed=True, diagnostic_cell_count=len(inputs.checked_cells),
                 checked_cells=inputs.checked_cells,
                 scope="Model-free file identities, scalar dimensions/finite masking, and prediction-based metrics; no hidden-state replay or independent recomputation of floating-point diagnostics")
    output_files = {p.name: {"sha256": file_sha(p), "bytes": p.stat().st_size} for p in sorted(args.output.iterdir())}
    write_json(args.output/"summary.json", dict(schema="case010-v2-diagnostic-summary-v1", audit=audit,
        table_rows={name: len(rows) for name, rows in tables.items()}, N_per_checkpoint=32, T=2048,
        fixed_trace_times=list(TIMES), population="separate diagnostic cohort; no confidence family; do not add to fresh N",
        output_files=output_files))
    print(json.dumps({"status": "PASS", "table_rows": {name: len(rows) for name, rows in tables.items()},
                      "model_execution": False, "diagnostic_cells": len(inputs.checked_cells)}))


if __name__ == "__main__":
    main()
