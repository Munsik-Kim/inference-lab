"""Read-only Phase A/learned result aggregation and scientific figures.

Budget frontiers describe already-observed TEST results.  They do not perform
DEV selection, compare paired confidence intervals, or establish statistical
superiority.  Existing simultaneous bounds are reused without changing their
prespecified comparison family.  Model seeds remain separate strata.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="case010-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np


TOY_NAMES = {
    "native_fp32": "NATIVE_FP32", "native_fp64": "NATIVE_FP64",
    "fullres_4_4": "FULL_RESIDUAL_4_4", "fullres_4_fp32": "FULL_RESIDUAL_4_FP32",
    "lowrank_4_8": "LOWRANK_4_8", "oldframe_4_4": "UNTRANSPORTED_4_4",
    "coefficient_int8_fp32": "COEFFICIENT_INT8_FP32",
}
CONDITION_LABELS = {"s4_signed_lattice": "S4: signed permutation lattice", "c31_rotation_2d": "C31: physical 2D rotations",
                    "s4_rotated": "S4: rotated noncommuting operators", "c31_moving_plane": "C31: moving coordinate plane",
                    "c31_switching_plane": "C31: switching coordinate plane"}
TOY_DISPLAY = ("NATIVE_FP32", "UNIFORM_4", "UNIFORM_8", "UNIFORM_16", "FULL_RESIDUAL_4_4",
               "FULL_RESIDUAL_4_FP32", "LOWRANK_4_8", "UNTRANSPORTED_4_4", "STOCHASTIC_4")
LEARNED_DISPLAY = ("NATIVE_FP32", "UNIFORM_4", "UNIFORM_6", "UNIFORM_8", "UNIFORM_16",
                   "FULL_RESIDUAL_4_4", "LOWRANK_4_8_R2", "LOWRANK_4_8_R4")
DESCRIPTIVE = "TEST descriptive best feasible baseline; not DEV-selected and not a statistical superiority test"
N_STREAMS = (1, 16, 128)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def horizon(summary, confidence=False, epsilon=0.05):
    key = "confidence_supported_T_epsilon_lower_bound" if confidence else "empirical_T_epsilon"
    entries = [entry for entry in summary[key] if abs(entry["epsilon"] - epsilon) < 1e-12]
    if len(entries) != 1:
        raise ValueError("exactly one requested epsilon entry is required")
    return entries[0]["horizon"]


def horizon_text(value, maximum=2048):
    return "no qualifying grid horizon (<32)" if value is None else f">={value}" if value == maximum else str(value)


def normalize(summary, ledger, *, domain, condition, seed, arm, original_arm=None, version=None, timing=None, diagnostics=None):
    if summary["n_sequences"] != 512 or summary["max_horizon"] != 2048:
        raise ValueError("this summary accepts only primary TEST N=512,Tmax=2048; smoke/DEV is excluded")
    expected_family = 875 if domain == "toy" else 546
    if summary["confidence"]["family_size"] != expected_family:
        raise ValueError(f"unexpected {domain} confidence family")
    stream = int(ledger["stream_payload_bytes"] if domain == "toy" else ledger["per_stream_persistent_bytes"])
    shared = int(ledger["total_shared_bytes"] if domain == "toy" else ledger["shared_bytes"])
    total = {str(n): shared + n * stream for n in N_STREAMS}
    amortized = {str(n): total[str(n)] / n for n in N_STREAMS}
    for n in N_STREAMS:
        if domain == "toy" and amortized[str(n)] != ledger["amortized_bytes_per_stream"][str(n)]:
            raise ValueError("toy amortized-byte ledger does not reconcile")
        if domain == "learned" and total[str(n)] != ledger["total_bytes"][str(n)]:
            raise ValueError("learned total-byte ledger does not reconcile")
    return {"domain": domain, "condition": condition, "model_seed": seed, "arm": arm,
            "original_arm": original_arm or arm, "evaluation_version": version,
            "summary": summary, "per_stream_bytes": stream, "shared_bytes": shared,
            "total_bytes": total, "amortized_bytes": amortized,
            "empirical_horizon": horizon(summary), "supported_horizon": horizon(summary, True),
            "empirical_horizon_01": horizon(summary, epsilon=0.01),
            "supported_horizon_01": horizon(summary, True, epsilon=0.01),
            "rmst": summary["restricted_mean_failure_free_length"],
            "final_failure_probability": summary["horizons"][-1]["failure_probability"],
            "observed_failures": summary["observed_failures"],
            "timing": timing, "diagnostics": diagnostics}


def read_toys(original, repair):
    original, repair = Path(original), Path(repair)
    old = load_jsonl(original / "results.jsonl")
    identity = load_json(repair / "erratum-identity.json")
    if sha(original / "protocol-frozen.json") != identity["protocol_file_sha256"]:
        raise ValueError("toy original/repair protocol provenance does not match")
    if sha(original / "results.jsonl") != identity["prior_results_file_sha256"]:
        raise ValueError("toy original attempt changed since repair")
    rows = load_jsonl(repair / "derived-results.jsonl")
    records, controls = [], []
    for row in rows:
        if row["arm"] == "exact_symbolic_id":
            controls.append(row)
            continue
        if row["status"] != "ok":
            raise ValueError("current toy primary arm must have a complete record")
        arm = TOY_NAMES.get(row["arm"], row["arm"].upper())
        records.append(normalize(row["summary"], row["ledger"], domain="toy", condition=row["condition"], seed=None,
                                 arm=arm, original_arm=row["arm"], version=row["evaluation_version"],
                                 diagnostics=row.get("diagnostics"),
                                 timing={"diagnostic_loop_seconds": row.get("runtime_seconds"),
                                         "complete_call": "NOT_MEASURED", "scope": "includes gold and descriptive diagnostics; no production throughput claim"}))
    if len(records) != 125 or len(controls) != 5:
        raise ValueError("expected 125 current toy primary arms and 5 symbolic capacity controls")
    return records, {"primary_arm_count": len(records), "capacity_control_count": len(controls),
                     "evaluation_versions": dict(Counter(row["evaluation_version"] for row in records)),
                     "historical_failed_arms": [{"condition": row["condition"], "arm": row["arm"], "error": row["error"]}
                                                for row in old if row["status"] == "error"],
                     "historical_stochastic_v1_arms_retained": sum(row["arm"] == "stochastic_4" for row in old),
                     "canonical_protocol_sha256": identity["canonical_protocol_sha256"],
                     "protocol_file_sha256": identity["protocol_file_sha256"],
                     "erratum_identity_file_sha256": sha(repair / "erratum-identity.json"),
                     "complete_call_timing": "NOT_MEASURED; diagnostic-loop seconds only",
                     "symbolic_capacity_controls": [{"condition": row["condition"], "failures": sum(t is not None for t in row["tau"]),
                                                     "stream_bytes": row["ledger"]["stream_payload_bytes"],
                                                     "shared_bytes": row["ledger"]["total_shared_bytes"],
                                                     "interpretation": "finite task-specific capacity control; not infinite memory"} for row in controls]}


def parse_isolated_arguments(arguments):
    result = {}
    for argument in arguments:
        seed_text, separator, filename = argument.partition("=")
        if not separator or not filename:
            raise ValueError("--isolated-timing must be SEED=FILE")
        seed = int(seed_text)
        if seed < 0 or seed in result:
            raise ValueError("isolated timing seeds must be nonnegative and unique")
        result[seed] = Path(filename)
    return result


def timing_sources(directory, manifest, isolated_path=None):
    """Keep embedded and isolated files distinct; only isolated is headline cost."""
    directory = Path(directory)
    embedded_path = directory / "timing.json"
    named_isolated = directory / "timing-isolated.json"
    choice = "explicit --isolated-timing" if isolated_path is not None else "named timing-isolated.json"
    if isolated_path is None and named_isolated.exists():
        isolated_path = named_isolated
    sources = {}
    for kind, path in (("embedded", embedded_path if embedded_path.exists() else None), ("isolated", isolated_path)):
        if path is None:
            continue
        path = Path(path)
        if kind == "isolated" and path.resolve() == embedded_path.resolve():
            raise ValueError("embedded timing.json cannot be relabeled as isolated")
        raw = load_json(path)
        benchmark = raw.get("benchmark", raw.get("timing", raw))
        if kind == "isolated":
            if "model_seed" in raw and raw["model_seed"] != manifest["model_seed"]:
                raise ValueError("isolated timing model seed does not match")
            if "checkpoint_sha256" in raw and raw["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
                raise ValueError("isolated timing checkpoint does not match")
            if raw.get("isolated") is False:
                raise ValueError("timing file explicitly declares non-isolated execution")
            expected = {"sequences": 16, "length": 128, "repeats": 3, "seed": 2002}
            if benchmark.get("config") != expected:
                raise ValueError("isolated timing must retain the fixed DEV2002/N16/L128/3-repeat benchmark")
        sources[kind] = {"data": benchmark, "metadata": {
            "file_name": path.name, "file_sha256": sha(path),
            "selection": choice if kind == "isolated" else "embedded runner artifact retained",
            "contention_scope": "reported isolated serial benchmark" if kind == "isolated" else "potential CPU contention; not isolated primary cost",
            "model_seed_in_file": raw.get("model_seed"),
            "checkpoint_sha256_in_file": raw.get("checkpoint_sha256"),
            "config": benchmark.get("config"), "scope": benchmark.get("scope"),
        }}
    return sources


def arm_timings(sources, arm):
    result = {"primary_kind": "NOT_MEASURED", "embedded": None, "isolated": None}
    for kind, source in sources.items():
        data = source["data"]
        if arm in data["arms"]:
            result[kind] = dict(data["arms"][arm], scope=data["scope"], cpu_threads=data["cpu_threads"],
                                config=data["config"], source=source["metadata"])
    if result["isolated"] is not None and result["isolated"].get("all_complete") is True:
        result["primary_kind"] = "isolated"
    return result


def timing_value(record, kind="isolated"):
    if record["domain"] != "learned" or not record["timing"]:
        return None
    timing = record["timing"].get(kind)
    if timing is None or not timing.get("all_complete", False):
        return None
    return timing.get("milliseconds_per_group_token")


def validate_learned_execution(summary, arm):
    execution = summary.get("execution", {})
    allowed = {"COMPLETE", "NONFINITE_ROWS_ABSORBED", "NONFINITE_LOGITS_WITH_FINITE_STATE"}
    if execution.get("status") not in allowed or execution.get("completed_writes") != summary["max_horizon"] + 1:
        raise ValueError(f"incomplete learned TEST execution: {arm}")
    return execution


def read_learned(directory, isolated_path=None):
    directory = Path(directory)
    manifest = load_json(directory / "manifest.json")
    if manifest["phase"] != "FROZEN_PRIMARY":
        raise ValueError("learned input must be FROZEN_PRIMARY; DEV/SMOKE cannot enter primary figures")
    files = sorted((directory / "TEST").glob("*.json"))
    if not files:
        raise ValueError("learned input has no completed TEST arm JSON files")
    sources = timing_sources(directory, manifest, isolated_path)
    records = []
    index_path = directory / "index.json"
    index = load_json(index_path) if index_path.exists() else None
    for path in files:
        summary = load_json(path)
        arm = path.stem
        if "n_sequences" not in summary:
            continue
        if summary["model_seed"] != manifest["model_seed"]:
            raise ValueError("model seeds must remain in distinct matching strata")
        execution = validate_learned_execution(summary, arm)
        if index is not None and sha(path) != index["splits"]["TEST"][arm]["sha256"]:
            raise ValueError("learned TEST arm hash does not match completed index")
        arm_timing = arm_timings(sources, arm)
        record = normalize(summary, summary["ledger"], domain="learned", condition="learned_s3", seed=manifest["model_seed"],
                           arm=arm, version="frozen-learned-primary", timing=arm_timing, diagnostics=execution.get("diagnostics"))
        record["execution_status"] = execution["status"]
        record["absorbed_sequences"] = execution.get("absorbed_sequences", 0)
        record["source_json_sha256"] = sha(path)
        record["source_json_name"] = f"TEST/{path.name}"
        record["source_directory_name"] = directory.name
        records.append(record)
    return records, {"model_seed": manifest["model_seed"], "arms_available": len(records),
                     "expected_arms": 26, "complete_index_available": index is not None,
                     "model_checkpoint_bytes_common": manifest["model_checkpoint_bytes"],
                     "checkpoint_sha256": manifest["checkpoint_sha256"], "manifest_file_sha256": sha(directory / "manifest.json"),
                     "protocol_sha256": manifest["protocol_sha256"], "timing_available": bool(sources),
                     "timing_sources": {kind: source["metadata"] for kind, source in sources.items()},
                     "primary_timing_kind": "isolated" if "isolated" in sources else "NOT_MEASURED",
                     "timing_policy": "isolated result selected explicitly; embedded result retained and labeled for potential contention",
                     "ledger_scope": "recurrent stream cache plus shared codec/basis/token table; common model checkpoint separate",
                     "completeness": "complete" if index is not None and len(records) == 26 else "partial; missing arms are not inferred"}


def baseline(arm):
    return arm.startswith(("NATIVE_", "UNIFORM_", "MIXED_", "STOCHASTIC_"))


def group_key(record):
    return record["domain"], record["condition"], record["model_seed"]


def metric_value(record, metric):
    value = record[metric]
    return -1 if value is None else value


def best_feasible(feasible, metric, n):
    if not feasible:
        return None
    return max(feasible, key=lambda record: (metric_value(record, metric), record["rmst"],
                                             -record["total_bytes"][str(n)], record["arm"]))


def budget_rows(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)
    output = []
    for peers in grouped.values():
        reference = [record for record in peers if baseline(record["arm"])]
        for candidate in peers:
            for n in N_STREAMS:
                feasible = [record for record in reference if record["total_bytes"][str(n)] <= candidate["total_bytes"][str(n)]]
                supported = best_feasible(feasible, "supported_horizon", n)
                empirical = best_feasible(feasible, "empirical_horizon", n)
                rmst = best_feasible(feasible, "rmst", n)
                native = next((record for record in feasible if record["arm"] == "NATIVE_FP32"), None)
                output.append({"domain": candidate["domain"], "condition": candidate["condition"], "model_seed": candidate["model_seed"],
                    "candidate": candidate["arm"], "evaluation_version": candidate["evaluation_version"], "N_streams": n,
                    "candidate_total_byte_cap": candidate["total_bytes"][str(n)], "candidate_amortized_bytes": candidate["amortized_bytes"][str(n)],
                    "candidate_stream_bytes": candidate["per_stream_bytes"], "candidate_shared_bytes": candidate["shared_bytes"],
                    "candidate_empirical_T05": candidate["empirical_horizon"], "candidate_supported_T05": candidate["supported_horizon"],
                    "candidate_empirical_T05_status": horizon_text(candidate["empirical_horizon"]),
                    "candidate_supported_T05_status": horizon_text(candidate["supported_horizon"]),
                    "candidate_rmst": candidate["rmst"], "candidate_failure_probability_2048": candidate["final_failure_probability"],
                    "feasible_baseline_count": len(feasible), "feasible_baselines": ";".join(sorted(row["arm"] for row in feasible)),
                    "native_fp32_feasible": native is not None,
                    "best_TEST_supported_baseline": None if supported is None else supported["arm"],
                    "best_TEST_supported_T05": None if supported is None else supported["supported_horizon"],
                    "best_TEST_empirical_baseline": None if empirical is None else empirical["arm"],
                    "best_TEST_empirical_T05": None if empirical is None else empirical["empirical_horizon"],
                    "best_TEST_rmst_baseline": None if rmst is None else rmst["arm"],
                    "best_TEST_baseline_rmst": None if rmst is None else rmst["rmst"],
                    "descriptive_rmst_delta_vs_best_feasible": None if rmst is None else candidate["rmst"] - rmst["rmst"],
                    "baseline_selection_scope": DESCRIPTIVE,
                    "horizon_comparison_scope": "comparison of reported lower bounds is not a test of true-horizon superiority"})
    return output


def compact(record):
    return {"failures": record["observed_failures"], "F2048": record["final_failure_probability"], "rmst": record["rmst"],
            "empirical_T05": record["empirical_horizon"], "supported_T05": record["supported_horizon"],
            "empirical_T01": record["empirical_horizon_01"], "supported_T01": record["supported_horizon_01"],
            "stream_bytes": record["per_stream_bytes"], "shared_bytes": record["shared_bytes"],
            "amortized_bytes": record["amortized_bytes"], "evaluation_version": record["evaluation_version"],
            "execution_status": record.get("execution_status"), "absorbed_sequences": record.get("absorbed_sequences"),
            "isolated_benchmark_ms_per_token": timing_value(record),
            "embedded_potentially_contended_ms_per_token": timing_value(record, "embedded")}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def native_reference_rows(records):
    """Native gold agreement and first-failure survival at every frozen length."""
    rows = []
    for record in records:
        if record["domain"] != "learned" or record["arm"] != "NATIVE_FP32":
            continue
        summary = record["summary"]
        n = summary["n_sequences"]
        count_rows = summary["token_counts_at_horizons"]
        counts = {entry["horizon"]: entry for entry in count_rows}
        if len(counts) != len(count_rows):
            raise ValueError("duplicate native token-count horizons")
        for point in summary["horizons"]:
            t = point["horizon"]
            item = counts[t]
            expected_prefix, expected_quarter = n * t, n * ((t + 3) // 4)
            if item["prefix_total"] != expected_prefix or item["final_quarter_total"] != expected_quarter:
                raise ValueError("native token denominator must be N*T or N*ceil(T/4)")
            for correct_key, total_key in (("prefix_correct", "prefix_total"), ("final_quarter_correct", "final_quarter_total")):
                if not 0 <= item[correct_key] <= item[total_key]:
                    raise ValueError("native correct-token count exceeds its denominator")
            if not 0 <= point["first_failures"] <= n or abs(point["survival_probability"] - (1 - point["first_failures"] / n)) > 1e-12:
                raise ValueError("native first-failure counts and survival disagree")
            quarter_accuracy = item["final_quarter_correct"] / expected_quarter
            rows.append({"model_seed": record["model_seed"], "horizon": t, "n_sequences": n,
                         "first_failures": point["first_failures"], "survival_probability": point["survival_probability"],
                         "all_token_correct": item["prefix_correct"], "all_token_total": expected_prefix,
                         "all_token_accuracy": item["prefix_correct"] / expected_prefix,
                         "final_quarter_correct": item["final_quarter_correct"], "final_quarter_total": expected_quarter,
                         "final_quarter_accuracy": quarter_accuracy,
                         "final_quarter_chance_scaled_accuracy": (quarter_accuracy - 1 / 6) / (1 - 1 / 6),
                         "chance_scaling_definition": "(raw_accuracy - 1/6)/(1 - 1/6); raw accuracy is the primary displayed column",
                         "failure_probability_upper_bound": point["failure_upper_bound"],
                         "confidence_family_size": summary["confidence"]["family_size"],
                         "bos_accuracy": summary["bos"]["accuracy"], "bos_excluded_from_scored_steps": True,
                         "source_directory_name": record["source_directory_name"], "source_json_name": record["source_json_name"],
                         "source_json_sha256": record["source_json_sha256"]})
    return sorted(rows, key=lambda row: (row["model_seed"], row["horizon"]))


def arm_label(arm):
    labels = {"NATIVE_FP32": "FP32 state", "NATIVE_FP64": "FP64 state", "FULL_RESIDUAL_4_4": "Full residual 4+4",
              "FULL_RESIDUAL_4_FP32": "Full residual 4+FP32", "UNTRANSPORTED_4_4": "Untransported 4+4",
              "LOWRANK_4_8": "Fixed low rank 4+8", "STOCHASTIC_4": "Stochastic 4 (v2)"}
    if arm in labels:
        return labels[arm]
    if arm.startswith("LOWRANK_4_8_R"):
        return "Low rank 4+8, r=" + arm.rsplit("R", 1)[1]
    return arm.replace("_", " ").capitalize()


def plot_style(arm):
    colors = {"NATIVE_FP32": "#111111", "UNIFORM_4": "#0072B2", "UNIFORM_6": "#56B4E9", "UNIFORM_8": "#009E73",
              "UNIFORM_16": "#999999", "FULL_RESIDUAL_4_4": "#D55E00", "FULL_RESIDUAL_4_FP32": "#CC79A7",
              "LOWRANK_4_8": "#8B4FA3", "LOWRANK_4_8_R2": "#8B4FA3", "LOWRANK_4_8_R4": "#A6761D",
              "UNTRANSPORTED_4_4": "#E69F00", "STOCHASTIC_4": "#4B6A91"}
    return {"color": colors.get(arm, "#666666"), "linestyle": "--" if arm.startswith(("LOWRANK", "UNTRANSPORTED", "STOCHASTIC")) else "-",
            "linewidth": 1.65 if arm != "NATIVE_FP32" else 2.1}


def setup_plot_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white"})


def save_figure(figure, output, stem):
    figure.savefig(output / f"{stem}.png", dpi=180)
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)
    return [f"figures/{stem}.png", f"figures/{stem}.pdf"]


def empirical_curve(record):
    times = np.arange(1, record["summary"]["max_horizon"] + 1)
    tau = np.asarray([2049 if value is None else value for value in record["summary"]["tau"]])
    events = np.bincount(tau[tau <= 2048], minlength=2049)
    return times, np.cumsum(events)[1:] / len(tau)


def failure_axis(axis):
    axis.set_xscale("log", base=2)
    axis.set_xlim(1, 2048)
    axis.set_ylim(-0.025, 1.04)
    axis.set_xticks([1, 8, 32, 128, 512, 2048], ["1", "8", "32", "128", "512", "2048"])
    axis.set_yticks([0, .25, .5, .75, 1])
    axis.axhline(.05, color="#BBBBBB", lw=.8, linestyle=":", zorder=0)
    axis.grid(axis="y", color="#E5E5E5", lw=.6)
    axis.set_xlabel("Predicted steps T (log scale)")
    axis.set_ylabel("First-failure probability F(T)")


def plot_toy_failure(records, output):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.7))
    for index, (condition, title) in enumerate(CONDITION_LABELS.items()):
        axis = axes.flat[index]
        available = {record["arm"]: record for record in records if record["condition"] == condition}
        for arm in TOY_DISPLAY:
            if arm in available:
                x, y = empirical_curve(available[arm])
                axis.plot(x, y, label=arm_label(arm), **plot_style(arm))
        failure_axis(axis)
        axis.set_title(f"({chr(97 + index)}) {title}", loc="left")
        if all(available[arm]["observed_failures"] == 0 for arm in TOY_DISPLAY if arm in available):
            axis.text(.04, .78, "All displayed arms: 0/512 first failures\nthrough 2048 steps", transform=axis.transAxes, fontsize=10)
    axes.flat[5].axis("off")
    handles = [Line2D([], [], label=arm_label(arm), **plot_style(arm)) for arm in TOY_DISPLAY]
    axes.flat[5].legend(handles=handles, loc="upper left", frameon=False)
    axes.flat[5].text(.02, .18, "N = 512 independent sequences per condition\nCurves are empirical; later recovery is not survival.\nAll 25 arms per condition remain in the CSV data.\nStochastic results use the corrected v2 seed key.",
                      transform=axes.flat[5].transAxes, fontsize=9, linespacing=1.5)
    figure.suptitle("Finite-precision toy memory: first-failure curves", fontsize=14, y=.985)
    figure.subplots_adjust(left=.065, right=.98, top=.92, bottom=.08, wspace=.28, hspace=.38)
    return save_figure(figure, output, "toy_first_failure")


def plot_learned_failure(records, output):
    seeds = sorted({record["model_seed"] for record in records})
    figure, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 5.5), squeeze=False)
    for axis, seed in zip(axes.flat, seeds):
        available = {record["arm"]: record for record in records if record["model_seed"] == seed}
        for arm in LEARNED_DISPLAY:
            if arm in available:
                axis.plot(*empirical_curve(available[arm]), label=arm_label(arm), **plot_style(arm))
        failure_axis(axis)
        completeness = "26/26 arms" if len(available) == 26 else f"{len(available)}/26 arms; INCOMPLETE"
        axis.set_title(f"Model seed {seed} ({completeness})")
    displayed = [arm for arm in LEARNED_DISPLAY if any(record["arm"] == arm for record in records)]
    figure.legend(handles=[Line2D([], [], label=arm_label(arm), **plot_style(arm)) for arm in displayed],
                  loc="lower center", bbox_to_anchor=(.5, .01), ncol=min(4, len(seeds) * 2), frameon=False)
    figure.suptitle("Learned S3: first-failure curves", fontsize=14, y=.98)
    figure.subplots_adjust(left=.10 if len(seeds) == 1 else .055, right=.98, top=.88, bottom=.27, wspace=.28)
    return save_figure(figure, output, "learned_first_failure")


def category(arm):
    if arm.startswith("NATIVE"): return "Native", "s", "#111111"
    if arm.startswith("UNIFORM"): return "Uniform", "o", "#0072B2"
    if arm.startswith("MIXED"): return "Mixed", "^", "#8B4FA3"
    if arm.startswith("STOCHASTIC"): return "Stochastic", "*", "#E69F00"
    if arm.startswith("FULL_RESIDUAL"): return "Full residual", "D", "#D55E00"
    if arm.startswith("LOWRANK"): return "Low rank", "P", "#009E73"
    return "Untransported", "X", "#777777"


def budget_axis(axis, metric):
    axis.set_yscale("log", base=2)
    ticks = [16, 32, 64, 128, 256, 512, 1024, 2048]
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_formatter(FixedFormatter(["No qualifying\nhorizon (<32)", "32", "64", "128", "256", "512", "1024", "≥2048"]))
    axis.set_ylim(12, 2900)
    axis.axhspan(12, 23, color="#F1F1F1", zorder=0)
    axis.grid(axis="y", color="#E0E0E0", lw=.6)
    axis.set_xlabel("Actual amortized persistent bytes / stream, N=128")
    axis.set_ylabel("5% supported lower horizon" if metric == "supported_horizon" else "Empirical 5% sample-grid horizon")


def annotate_learned_budget(axis, records, metric):
    names = {"NATIVE_FP32": "FP32", "UNIFORM_4": "U4", "UNIFORM_6": "U6", "UNIFORM_8": "U8",
             "UNIFORM_16": "U16", "FULL_RESIDUAL_4_4": "Full 4+4", "LOWRANK_4_8_R2": "Rank 2", "LOWRANK_4_8_R4": "Rank 4"}
    selected = sorted((record for record in records if record["arm"] in names), key=lambda record: record["amortized_bytes"]["128"])
    span = max((record["amortized_bytes"]["128"] for record in records), default=1) - min((record["amortized_bytes"]["128"] for record in records), default=0)
    previous = None
    level = 0
    for record in selected:
        x = record["amortized_bytes"]["128"]
        y = 16 if record[metric] is None else record[metric]
        if previous is not None and previous[1] == y and x - previous[0] < .13 * max(span, 1):
            level = (level + 1) % 4
        else:
            level = 0
        axis.annotate(names[record["arm"]], (x, y), xytext=(0, 10 + level * 16), textcoords="offset points",
                      ha="center", va="bottom", fontsize=8, color=category(record["arm"])[2],
                      arrowprops={"arrowstyle": "-", "color": "#AAAAAA", "linewidth": .5})
        previous = x, y


def plot_budget(records, output, domain):
    groups = sorted({record["model_seed"] for record in records}) if domain == "learned" else list(CONDITION_LABELS)
    figure, axes = plt.subplots(len(groups), 2, figsize=(13, 4.4 * len(groups)), squeeze=False)
    for row_index, group in enumerate(groups):
        peers = [record for record in records if (record["model_seed"] if domain == "learned" else record["condition"]) == group]
        # Secondary coefficient precision is not a recurrent-state budget arm.
        peers = [record for record in peers if not record["arm"].startswith("COEFFICIENT")]
        for column, metric in enumerate(("supported_horizon", "empirical_horizon")):
            axis = axes[row_index, column]
            for record in peers:
                _, marker, color = category(record["arm"])
                y = 16 if record[metric] is None else record[metric]
                x = record["amortized_bytes"]["128"]
                axis.scatter(x, y, marker=marker, color=color, s=47, edgecolor="white", linewidth=.4, zorder=3)
            references = sorted((record for record in peers if baseline(record["arm"])), key=lambda record: record["amortized_bytes"]["128"])
            if references:
                costs = sorted({record["amortized_bytes"]["128"] for record in peers})
                best_y = []
                for cost in costs:
                    feasible = [record for record in references if record["amortized_bytes"]["128"] <= cost]
                    best = best_feasible(feasible, metric, 128)
                    best_y.append(np.nan if best is None else 16 if best[metric] is None else best[metric])
                axis.step(costs, best_y, where="post", color="#444444", linestyle=":", linewidth=1.2, zorder=1)
            budget_axis(axis, metric)
            label = f"Model seed {group}" if domain == "learned" else CONDITION_LABELS[group]
            if domain == "learned":
                annotate_learned_budget(axis, peers, metric)
                if len(peers) != 26:
                    label += f" ({len(peers)}/26; INCOMPLETE)"
            axis.set_title(f"{label}: {'simultaneous support' if column == 0 else 'empirical description'}", loc="left")
    legend_arms = ("NATIVE_FP32", "UNIFORM_4", "MIXED_4_8", "STOCHASTIC_4", "FULL_RESIDUAL_4_4", "LOWRANK_4_8", "UNTRANSPORTED_4_4")
    handles = [Line2D([], [], marker=category(arm)[1], color=category(arm)[2], linestyle="none", label=category(arm)[0], markersize=7) for arm in legend_arms]
    handles.append(Line2D([], [], color="#444444", linestyle=":", label="TEST descriptive feasible-baseline frontier"))
    height_inches = figure.get_size_inches()[1]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .10 / height_inches), ncol=4, frameon=False)
    figure.suptitle(f"{'Learned S3' if domain == 'learned' else 'Toy controls'}: memory horizon at actual byte cost", fontsize=14, y=.993)
    figure.text(.5, .72 / height_inches, "Gray bottom band is a category, not a measured T=0 or T=16. Bounds use the frozen comparison family.\nDotted frontier is selected descriptively from TEST; it is neither DEV-selected nor evidence of statistical superiority.",
                ha="center", fontsize=9)
    figure.subplots_adjust(left=.14, right=.98, top=.91 if len(groups) == 1 else .955,
                           bottom=1.4 / height_inches, hspace=.48, wspace=.36)
    return save_figure(figure, output, f"{domain}_budget_horizons_N128")


def summarize(toy_original, toy_repair, learned, output, isolated_timings=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    toy, toy_meta = read_toys(toy_original, toy_repair)
    learned_records, learned_meta = [], []
    isolated_timings = {} if isolated_timings is None else isolated_timings
    for directory in learned:
        seed = load_json(Path(directory) / "manifest.json")["model_seed"]
        records, metadata = read_learned(directory, isolated_timings.get(seed))
        learned_records.extend(records)
        learned_meta.append(metadata)
    seeds = [metadata["model_seed"] for metadata in learned_meta]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate learned model seeds; do not pool or duplicate a seed")
    if set(isolated_timings) - set(seeds):
        raise ValueError("isolated timing supplied for a model seed absent from --learned inputs")
    records = toy + learned_records
    comparisons = budget_rows(records)
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"
    figures.mkdir()
    write_csv(output / "budget_comparisons.csv", comparisons)
    all_arms = []
    curves = []
    for record in records:
        row = {key: record[key] for key in ("domain", "condition", "model_seed", "arm", "evaluation_version", "observed_failures", "final_failure_probability", "rmst", "empirical_horizon", "supported_horizon", "empirical_horizon_01", "supported_horizon_01", "per_stream_bytes", "shared_bytes")}
        row.update(execution_status=record.get("execution_status"), absorbed_sequences=record.get("absorbed_sequences"))
        row.update({f"total_bytes_N{n}": record["total_bytes"][str(n)] for n in N_STREAMS})
        row.update({f"amortized_bytes_N{n}": record["amortized_bytes"][str(n)] for n in N_STREAMS})
        timing = record["timing"] or {}
        row.update(complete_call_ms_per_group_token=timing_value(record),
                   isolated_complete_call_ms_per_group_token=timing_value(record),
                   embedded_potentially_contended_ms_per_group_token=timing_value(record, "embedded"),
                   primary_timing_kind=timing.get("primary_kind", "NOT_MEASURED"),
                   diagnostic_loop_seconds=timing.get("diagnostic_loop_seconds"),
                   timing_scope=timing.get("scope", "isolated and embedded benchmarks are separate; embedded is not primary cost"))
        all_arms.append(row)
        for point in record["summary"]["horizons"]:
            curves.append({"domain": record["domain"], "condition": record["condition"], "model_seed": record["model_seed"],
                           "arm": record["arm"], "evaluation_version": record["evaluation_version"], **point,
                           "family_size": record["summary"]["confidence"]["family_size"]})
    write_csv(output / "all_arms.csv", all_arms)
    write_csv(output / "horizon_curves.csv", curves)
    native_rows = native_reference_rows(learned_records)
    write_csv(output / "native_reference.csv", native_rows)
    timing_rows = []
    for record in learned_records:
        for kind in ("embedded", "isolated"):
            benchmark = record["timing"].get(kind)
            if benchmark is None:
                continue
            timing_rows.append({"model_seed": record["model_seed"], "arm": record["arm"], "timing_kind": kind,
                "primary_cost": kind == "isolated" and benchmark.get("all_complete", False),
                "median_seconds": benchmark["median_seconds"], "all_seconds": json.dumps(benchmark["all_seconds"]),
                "milliseconds_per_group_token": benchmark["milliseconds_per_group_token"], "all_complete": benchmark["all_complete"],
                "cpu_threads": benchmark["cpu_threads"], "config": json.dumps(benchmark["config"], sort_keys=True),
                "source_file_name": benchmark["source"]["file_name"], "source_file_sha256": benchmark["source"]["file_sha256"],
                "contention_scope": benchmark["source"]["contention_scope"], "call_scope": benchmark["scope"]})
    write_csv(output / "timings.csv", timing_rows)
    setup_plot_style()
    plot_paths = plot_toy_failure(toy, figures)
    plot_paths += plot_budget(toy, figures, "toy")
    if learned_records:
        plot_paths += plot_learned_failure(learned_records, figures)
        plot_paths += plot_budget(learned_records, figures, "learned")
    grouped = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)
    group_summaries = []
    for (domain, condition, seed), peers in grouped.items():
        selected = TOY_DISPLAY if domain == "toy" else LEARNED_DISPLAY
        at_cap = [row for row in comparisons if row["domain"] == domain and row["condition"] == condition and row["model_seed"] == seed and row["N_streams"] == 128
                  and row["candidate"] in selected and not baseline(row["candidate"])]
        group_summaries.append({"domain": domain, "condition": condition, "model_seed": seed, "arms": len(peers),
                                "selected_numeric_results": {record["arm"]: compact(record) for record in peers if record["arm"] in selected},
                                "candidate_caps_N128": [{key: row[key] for key in ("candidate", "candidate_amortized_bytes", "native_fp32_feasible", "best_TEST_supported_baseline", "best_TEST_supported_T05", "best_TEST_empirical_baseline", "best_TEST_empirical_T05", "best_TEST_rmst_baseline", "best_TEST_baseline_rmst", "descriptive_rmst_delta_vs_best_feasible")} for row in at_cap]})
    summary = {"schema": "case010-results-summary-v1", "toy": toy_meta, "learned_seeds": learned_meta,
               "groups": group_summaries, "budget_comparison_rows": len(comparisons), "all_arm_rows": len(all_arms),
               "native_reference_rows": len(native_rows),
               "interpretation": {"budget_selection": DESCRIPTIVE,
                                  "null_horizon": "No evaluated grid horizon >=32 qualifies; not measured zero memory",
                                  "censored_maximum": "Empirical maximum is a sample statement; population support comes only from the simultaneous confidence field",
                                  "model_seeds": "reported separately, never pooled as new independent input sequences",
                                  "byte_scope": "shared serialized bytes plus N times per-stream bytes, divided by N; no baseline padding",
                                  "toy_timing": "diagnostic loop only; complete-call timing NOT_MEASURED",
                                  "learned_timing": "isolated complete-call result is primary only when separately supplied; embedded potentially-contended timing retained explicitly; no GPU or whole-model speedup claim"},
               "figures": plot_paths, "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__, "style": "plain matplotlib; no SciencePlots"},
               "sources": {"toy_original_results_sha256": sha(Path(toy_original) / "results.jsonl"),
                           "toy_current_derived_results_sha256": sha(Path(toy_repair) / "derived-results.jsonl"),
                           "summarizer_source_sha256": sha(__file__)}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return {"output_directory_name": output.name, "primary_arm_rows": len(records), "budget_rows": len(comparisons),
            "learned_model_seeds": seeds, "figures": plot_paths, "matplotlib": matplotlib.__version__}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy-original", type=Path, required=True)
    parser.add_argument("--toy-repair", type=Path, required=True)
    parser.add_argument("--learned", type=Path, action="append", default=[])
    parser.add_argument("--isolated-timing", action="append", default=[], metavar="SEED=FILE",
                        help="separate isolated benchmark artifact; repeat per seed; embedded timing.json remains retained")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.toy_original, args.toy_repair, args.learned, args.output,
                               parse_isolated_arguments(args.isolated_timing)), sort_keys=True))


if __name__ == "__main__":
    main()
