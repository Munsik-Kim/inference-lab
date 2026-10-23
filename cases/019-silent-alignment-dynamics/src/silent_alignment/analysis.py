"""Raw-grid reductions; this module never integrates the ODE."""

import csv
from pathlib import Path
import numpy as np

from .io import read_json, write_json


def crossing(times, values, threshold, already=False):
    if already:
        return {"status": "ALREADY_ALIGNED", "time": 0.0,
                "bracket": [0.0, 0.0], "threshold": float(threshold)}
    found = np.flatnonzero(np.isfinite(values) & (values <= threshold))
    if not len(found):
        return {"status": "NOT_REACHED", "time": None, "observed_end": float(times[-1]),
                "threshold": float(threshold)}
    i = int(found[0])
    if i == 0:
        return {"status": "REACHED_AT_START", "time": float(times[0]),
                "bracket": [float(times[0]), float(times[0])], "threshold": float(threshold)}
    a, b = float(values[i-1]), float(values[i])
    fraction = (a - threshold) / (a - b) if a != b else 1.0
    t = float(times[i-1] + fraction * (times[i] - times[i-1]))
    return {"status": "REACHED", "time": t, "bracket": [float(times[i-1]), float(times[i])],
            "interpolation": "linear between saved observations", "threshold": float(threshold)}


def reduce_trajectory(meta, raw, config):
    times = raw["times"]
    d0 = float(raw["right_distance"][0])
    aligned = meta["initialization"] == "aligned" or d0 <= config["tolerances"]["aligned_distance"]
    a = crossing(times, raw["right_distance"], d0 * config["metrics"]["align_fraction_initial"], aligned)
    v = crossing(times, raw["learnable_residual"], config["metrics"]["visible_residual"])
    learn_at_align = None
    if a["time"] is not None:
        learn_at_align = float(np.interp(a["time"], times, raw["learnable_residual"]))
    row = {"id": meta["id"], "block": meta["block"], "initialization": meta["initialization"],
           "replicate": meta["replicate"], "rho": meta["rho"],
           "amplitude": meta["amplitude"], "p": meta["p"], "d": meta["d"], "m": meta["m"],
           "boundary_gap": meta["boundary_gap"], "loss_0": float(raw["loss"][0]),
           "loss_min": float(raw["loss_min"][0]), "loss_end": float(raw["loss"][-1]),
           "residual_end": float(raw["learnable_residual"][-1]),
           "right_distance_0": d0, "right_distance_end": float(raw["right_distance"][-1]),
           "T_align_half": a, "T_visible10": v, "residual_at_alignment_half": learn_at_align,
           "maximum_imbalance_absolute": float(np.max(raw["imbalance_absolute"])),
           "maximum_imbalance_relative_gram": float(np.max(raw["imbalance_relative_gram"])),
           "maximum_normalized_loss_increase": float(np.max(np.diff(raw["loss"])) / raw["loss"][0])}
    # Actual grid width is retained, rather than representing interpolation as exact time.
    row["maximum_saved_time_gap"] = float(np.max(np.diff(times)))
    return row


def regenerate(run, output):
    run, output = Path(run), Path(output)
    config = read_json(run / "config.json")
    plan = read_json(run / "input_manifest.json")
    rows = []
    for meta in plan["trajectories"]:
        path = run / "raw" / (meta["id"] + ".npz")
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as raw:
            rows.append(reduce_trajectory(meta, raw, config))
    scalar_fit = []
    for sigma in config["blocks"]["A"]["sigmas"]:
        selected = [r for r in rows if r["block"] == "A" and
                    np.isclose(read_json(run / "metadata" / (r["id"] + ".json"))["sigma_scalar"], sigma)]
        # Fixed-amplitude arrival differs from the residual-visible threshold.
        x, y, detailed = [], [], []
        for row in selected:
            with np.load(run / "raw" / (row["id"] + ".npz"), allow_pickle=False) as raw:
                frac = config["references"]["scalar_fixed_amplitude"]["fraction_target"]
                c = crossing(raw["times"], -raw["target_mode_signed"][:, 0], -frac * sigma)
            detailed.append({"id": row["id"], "rho": row["rho"], "crossing": c})
            if c["time"] is not None:
                x.append(np.log(1 / row["rho"]))
                y.append(c["time"])
        fitted = float(np.polyfit(x, y, 1)[0]) if len(x) == 3 else None
        scalar_fit.append({"sigma": sigma, "fraction_target": frac,
                           "predicted_leading_log_slope": 1 / (2 * sigma),
                           "three_rho_descriptive_slope": fitted,
                           "points": detailed, "interpretation": "finite pilot, not asymptotic certification"})
    summary = {"schema": "case019-p0-summary-v1", "source_bundle_sha256": read_json(run / "execution_freeze.json")["bundle_sha256"],
               "primary_trajectories": len(rows), "rows": rows, "scalar_fixed_amplitude_fits": scalar_fit,
               "phenomenon_status": "OBSERVATIONS_REPORTED_SEPARATELY_FROM_IMPLEMENTATION_GATE"}
    write_json(output / "summary.json", summary)
    columns = ["id", "block", "initialization", "replicate", "rho", "amplitude", "p", "d", "m", "boundary_gap",
               "loss_0", "loss_min", "loss_end", "residual_end", "right_distance_0", "right_distance_end",
               "T_align_half_status", "T_align_half_time", "T_visible10_status", "T_visible10_time",
               "residual_at_alignment_half", "maximum_imbalance_absolute", "maximum_imbalance_relative_gram"]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {k: row.get(k) for k in columns}
            for key in ("T_align_half", "T_visible10"):
                flat[key + "_status"], flat[key + "_time"] = row[key]["status"], row[key]["time"]
            writer.writerow(flat)
    return summary
