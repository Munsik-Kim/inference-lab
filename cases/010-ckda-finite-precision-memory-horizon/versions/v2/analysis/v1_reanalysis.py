"""Model-free, post-hoc reanalysis of immutable Case010 v1 retained records.

Run with --v1 CASE_DIRECTORY --output NEW_DIRECTORY. Optional --audit-output
keeps raw auditor receipts outside the report tree; --figures writes the figure
to a separate directory. No model, codec, checkpoint, or GPU is imported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

BOOTSTRAP_SEED = 41001
BOOTSTRAP_REPEATS = 5000
RESIDUALS = {"FULL_RESIDUAL_4_4", "FULL_RESIDUAL_4_8", "FULL_RESIDUAL_4_FP32",
             "LOWRANK_4_8_R1", "LOWRANK_4_8_R2", "LOWRANK_4_8_R4"}
REVIEW_NAMES = ["CASE010_REVIEW_KO.md", "diova_case010_independent_review_20260923.zip",
                "probe_restart_failure.py"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    require(bool(rows), "cannot write empty table")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def failure_free_lengths(tau, horizon):
    require(type(horizon) is int and horizon > 0, "invalid horizon")
    require(len(tau) > 0, "empty tau")
    require(all(t is None or (type(t) is int and 1 <= t <= horizon) for t in tau),
            "tau must contain observed positive failure positions or None")
    return np.asarray([horizon if t is None else t - 1 for t in tau], dtype=np.int64)


def empirical_token_horizon(tau, horizon, epsilon):
    """Largest integer t with empirical P(tau <= t) <= epsilon, including t=0."""
    require(0 <= epsilon < 1, "invalid epsilon")
    length = failure_free_lengths(tau, horizon)
    positions = np.arange(1, horizon + 1)
    failures = (length[:, None] < positions).sum(axis=0)
    eligible = positions[failures / len(length) <= epsilon]
    return int(eligible[-1]) if len(eligible) else 0


def paired_difference(candidate, baseline):
    """Pair by unique sequence identity, preserving candidate order."""
    require(candidate["max_horizon"] == baseline["max_horizon"], "horizon mismatch")
    require(candidate["model_seed"] == baseline["model_seed"], "model seed mismatch")
    cid, bid = candidate["sequence_ids"], baseline["sequence_ids"]
    require(len(set(cid)) == len(cid) and len(set(bid)) == len(bid), "duplicate sequence identity")
    require(len(cid) == len(candidate["tau"]) and len(bid) == len(baseline["tau"]), "ID/tau mismatch")
    require(set(cid) == set(bid), "paired sequence identities differ")
    c = failure_free_lengths(candidate["tau"], candidate["max_horizon"])
    b = failure_free_lengths(baseline["tau"], baseline["max_horizon"])
    lookup = dict(zip(bid, b))
    return c - np.asarray([lookup[key] for key in cid], dtype=np.int64)


def paired_bootstrap(difference, *, model_seed, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED):
    """Resample whole paired sequences within a model seed, percentile CI."""
    difference = np.asarray(difference)
    require(difference.ndim == 1 and difference.size > 0 and np.isfinite(difference).all(),
            "invalid paired differences")
    require(type(repeats) is int and repeats > 0, "invalid repeat count")
    rng = np.random.default_rng(np.random.SeedSequence([seed, model_seed]))
    samples = np.empty(repeats)
    for begin in range(0, repeats, 250):
        end = min(begin + 250, repeats)
        indices = rng.integers(0, difference.size, size=(end - begin, difference.size))
        samples[begin:end] = difference[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return {"rmst_delta_tokens": float(difference.mean()), "pointwise_ci95_low": float(low),
            "pointwise_ci95_high": float(high), "paired_sequences": int(difference.size),
            "bootstrap_repeats": repeats, "bootstrap_seed": seed,
            "rng_substream": f"SeedSequence([{seed},{model_seed}])"}


def safe_public_json(value):
    """Auditor receipts use logical paths; reject accidental private paths/secrets."""
    raw = json.dumps(value, ensure_ascii=False)
    patterns = [r"/home/", r"/mnt/[a-z]/", r"/Users/", r"[A-Za-z]:[\\/](?:Users|home)[\\/]",
                r"(?:sk-(?:proj-)?|ghp_|github_pat_|hf_)[A-Za-z0-9_-]{20,}",
                r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"]
    require(not any(re.search(pattern, raw) for pattern in patterns), "private path or credential-shaped string in public receipt")
    return value


class Sources:
    def __init__(self, root):
        self.root, self.hashes = Path(root), {}

    def path(self, relative):
        path = self.root / relative
        require(path.resolve().is_relative_to(self.root.resolve()), "source escaped v1")
        self.hashes[relative] = digest(path)
        return path

    def json(self, relative):
        return json.loads(self.path(relative).read_text())

    def csv(self, relative):
        with self.path(relative).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def unchanged(self):
        require(all(digest(self.root / name) == sha for name, sha in self.hashes.items()),
                "a consumed v1 source changed during reanalysis")


def run_original_audits(sources, raw_dir, output):
    raw_dir.mkdir(parents=True, exist_ok=False)
    receipt_dir = output / "receipts"
    receipt_dir.mkdir()
    v1 = sources.root
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    scripts = ["scripts/audit_records.py", "scripts/audit_budget_supplement.py"]
    args = [["--toy-original", str(v1 / "results/toy-run"),
             "--toy-repair", str(v1 / "results/toy-repair-v1b")], []]
    for model_seed in (0, 1, 2):
        args[0] += ["--learned", str(v1 / f"results/learned-seed{model_seed}")]
        args[1] += ["--supplement", str(v1 / f"results/budget-supplement-seed{model_seed}")]
    names = ["primary_audit.json", "supplement_audit.json"]
    receipts = []
    for script, arguments, name in zip(scripts, args, names):
        path = sources.path(script)
        result = subprocess.run([sys.executable, str(path), *arguments, "--output", str(raw_dir / name)],
                                env=env, capture_output=True, text=True, check=False)
        (raw_dir / name.replace(".json", ".log")).write_text(result.stdout + result.stderr)
        require(result.returncode == 0, f"original auditor failed: {script}; see external audit log")
        receipt = safe_public_json(json.loads((raw_dir / name).read_text()))
        write_json(receipt_dir / name, receipt)
        receipts.append(receipt)
    primary, supplement = receipts
    counts = {"toy_current_records": primary["toy"]["current_verified_records"],
              "learned_records": sum(r["verified_test_arms"] for r in primary["learned"]),
              "supplement_records": sum(r["verified_test_arms"] for r in supplement["seeds"])}
    require(counts == {"toy_current_records": 125, "learned_records": 78, "supplement_records": 12},
            "unexpected audit record counts")
    return {"status": "PASS", **counts, "model_inference_executed": False,
            "checkpoint_bytes_reverified": False,
            "receipts": {name: digest(receipt_dir / name) for name in names},
            "original_auditor_sha256": {script: sources.hashes[script] for script in scripts}}


def load_records(sources, arms):
    records = {}
    for arm in arms:
        seed, name = int(arm["model_seed"]), arm["arm"]
        prefix = "learned" if arm["source_kind"] == "original" else "budget-supplement"
        rel = f"results/{prefix}-seed{seed}/TEST/{name}.json"
        record = sources.json(rel)
        require(sources.hashes[rel] == arm["source_json_sha256"], "joint CSV source hash mismatch")
        require(record["n_sequences"] == len(record["tau"]) == 512 and record["max_horizon"] == 2048,
                "historical denominator changed")
        require(len(set(record["sequence_ids"])) == 512, "duplicate historical sequence IDs")
        length = failure_free_lengths(record["tau"], 2048)
        require(float(length.mean()) == record["restricted_mean_failure_free_length"] == float(arm["rmst"]),
                "retained RMST differs from tau")
        correctness = record["correctness_artifact"]
        crel = f"results/{prefix}-seed{seed}/TEST/{correctness['file']}"
        require(digest(sources.path(crel)) == correctness["sha256"], "correctness file hash mismatch")
        with np.load(sources.root / crel, allow_pickle=False) as saved:
            shape = tuple(saved["shape"])
            require(shape == (512, 2048), "correctness shape mismatch")
            bits = np.unpackbits(saved["packed"], axis=1, count=2048, bitorder="little")
        observed = [int(np.flatnonzero(row == 0)[0]) if (row == 0).any() else 2048 for row in bits]
        require(np.array_equal(length, observed), "tau does not match correctness")
        for streams in (1, 16, 128):
            total = record["ledger"]["shared_bytes"] + streams * record["ledger"]["per_stream_persistent_bytes"]
            require(total == record["ledger"]["total_bytes"][str(streams)] == int(arm[f"total_bytes_N{streams}"]),
                    "joint CSV byte ledger mismatch")
        records[(seed, name)] = record
    require(len(records) == 90 and all(sum(s == seed for s, _ in records) == 30 for seed in (0, 1, 2)),
            "expected exactly 30 arms per model seed")
    return records


def supported_value(value):
    return int(value) if value not in (None, "") else 0


def create_figure(arms, directory):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/case010-historical-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    directory.mkdir(parents=True, exist_ok=True)
    categories = {"Native": ("#141e30", "*"), "Uniform": ("#5c7fa3", "o"),
                  "Original mixed": ("#398879", "s"), "Added mixed": ("#d5952c", "D"),
                  "Residual": ("#b84561", "^"), "Other controls": ("#8d7d9c", "x")}
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 10), constrained_layout=True)
    for seed in (0, 1, 2):
        for col, streams in enumerate((1, 16, 128)):
            ax = axes[seed, col]
            for arm in [a for a in arms if int(a["model_seed"]) == seed]:
                name = arm["arm"]
                category = ("Native" if name == "NATIVE_FP32" else "Residual" if name in RESIDUALS else
                            "Added mixed" if arm["source_kind"] != "original" else "Uniform" if name.startswith("UNIFORM_")
                            else "Original mixed" if name.startswith("MIXED_") else "Other controls")
                color, marker = categories[category]
                ax.scatter(int(arm[f"total_bytes_N{streams}"]) / 1024, float(arm["rmst"]),
                           color=color, marker=marker, s=54 if category == "Native" else 29, alpha=.85)
            ax.set_xscale("log")
            ax.set_title(f"Model seed {seed} · N = {streams}", fontsize=11)
            ax.grid(alpha=.18)
            if col == 0:
                ax.set_ylabel("Restricted mean failure-free tokens")
            if seed == 2:
                ax.set_xlabel("Total serialized cache + shared bytes (KiB; log)")
    handles = [Line2D([], [], color=color, marker=marker, linestyle="None", label=name)
               for name, (color, marker) in categories.items()]
    fig.legend(handles=handles, loc="outside lower center", ncols=6, frameon=False)
    fig.suptitle("Historical v1: all 30 arms, each model seed shown separately\n"
                 "Post-hoc reused TEST · 512 paired sequences per seed · 2,048-token restriction", fontsize=14)
    paths = []
    for ext in ("png", "pdf"):
        path = directory / f"historical_v1_30arm_budget.{ext}"
        require(not path.exists(), "figure output already exists")
        fig.savefig(path, dpi=160, metadata={"Creator": "Case010 model-free v1 reanalysis",
                                             "CreationDate": None, "ModDate": None} if ext == "pdf" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def run(v1, output, audit_output=None, figures=None):
    v1, output = Path(v1).resolve(), Path(output).resolve()
    for target in (output, audit_output, figures):
        if target is not None:
            require(not Path(target).resolve().is_relative_to(v1), "all outputs must be outside immutable v1")
    require(not output.exists(), "output must be a new directory")
    output.mkdir(parents=True)
    sources = Sources(v1)
    audits = run_original_audits(sources, Path(audit_output) if audit_output else output / "raw-audits", output)
    arms = sources.csv("results/supplement-summary/all_arms.csv")
    comparisons = sources.csv("results/supplement-summary/budget_comparisons.csv")
    primary_comparisons = sources.csv("results/summary/budget_comparisons.csv")
    sources.path("results/supplement-summary/horizon_curves.csv")
    records = load_records(sources, arms)
    current = [r for r in comparisons if r["candidate"] in RESIDUALS]
    primary = [r for r in primary_comparisons if r["domain"] == "learned" and r["candidate"] in RESIDUALS]
    require(len(current) == len(primary) == 54, "expected 54 residual budget comparisons")
    # The original 26-arm and supplemented 30-arm views are distinct historical analyses.
    counts = {}
    for label, rows in [("original_26arm_family546", primary), ("joint_30arm_family630", current)]:
        counts[label] = {"comparisons": len(rows), "supported_T05_strict_improvements": sum(
            supported_value(r["candidate_supported_T05"]) > supported_value(r["best_TEST_supported_T05"]) for r in rows)}
    native = []
    for seed in (0, 1, 2):
        record = records[(seed, "NATIVE_FP32")]
        length = failure_free_lengths(record["tau"], 2048)
        for epsilon in (.05, .01):
            t = empirical_token_horizon(record["tau"], 2048, epsilon)
            native.append({"model_seed": seed, "epsilon": epsilon, "n_sequences": 512, "max_horizon": 2048,
                           "empirical_per_token_T_epsilon": t, "failures_at_T": int((length < t).sum()),
                           "failures_at_T_plus_1": int((length < t + 1).sum()) if t < 2048 else "",
                           "rmst_tokens": float(length.mean()), "scope": "post-hoc empirical integer-token grid; no population bound"})
    paired, cache = [], {}
    for row in current:
        seed, candidate, baseline = int(row["model_seed"]), row["candidate"], row["best_TEST_rmst_baseline"]
        key = (seed, candidate, baseline)
        if key not in cache:
            cache[key] = paired_bootstrap(paired_difference(records[(seed, candidate)], records[(seed, baseline)]), model_seed=seed)
        result = cache[key]
        require(np.isclose(result["rmst_delta_tokens"], float(row["candidate_rmst"]) - float(row["best_TEST_rmst"])),
                "paired mean differs from retained RMST difference")
        paired.append({"model_seed": seed, "N_streams": int(row["N_streams"]), "candidate": candidate,
                       "baseline": baseline, "candidate_total_bytes": int(row["candidate_total_byte_cap"]),
                       "baseline_total_bytes": int(row["best_TEST_rmst_total_bytes"]), **result,
                       "scope": "post-hoc reused-TEST selected comparator; pointwise conditional CI; no multiplicity or selection correction"})
    ledger = [{"model_seed": int(a["model_seed"]), "arm": a["arm"], "source_kind": a["source_kind"],
               "joint_family_size": int(a["joint_family_size"]), "n_sequences": int(a["n_sequences"]),
               "per_stream_bytes": int(a["per_stream_bytes"]), "shared_bytes": int(a["shared_bytes"]),
               **{f"total_bytes_N{n}": int(a[f"total_bytes_N{n}"]) for n in (1, 16, 128)},
               "rmst_tokens": float(a["rmst"]), "supported_grid_T05": a["supported_T05"],
               "empirical_grid_T05": a["empirical_T05"]} for a in arms]
    write_csv(output / "native_per_token_horizons.csv", native)
    write_csv(output / "paired_rmst_bootstrap.csv", paired)
    write_csv(output / "all_30arm_byte_ledgers.csv", ledger)
    write_csv(output / "residual_budget_comparisons_joint630.csv", current)
    plots = create_figure(arms, Path(figures) if figures else output / "figures")
    sources.unchanged()
    write_json(output / "source_hashes.json", sources.hashes)
    summary = {"schema": "case010-v2-model-free-v1-reanalysis-v1", "status": "PASS",
               "phase": "POST_HOC_HISTORICAL_V1_SAME_TEST", "audit": audits,
               "historical_comparisons": counts, "model_seeds": [0, 1, 2], "arms_per_seed": 30,
               "sequences_per_seed": 512, "seed_pooling": False, "max_horizon": 2048,
               "bootstrap": {"seed": BOOTSTRAP_SEED, "repeats": BOOTSTRAP_REPEATS,
                             "method": "paired sequence percentile bootstrap, 95% pointwise CI; SeedSequence([41001, model_seed]); NumPy linear quantiles",
                             "selection_limitation": "comparator maximizes retained TEST RMST within cap; bootstrap conditions on this selected comparator; neither selection nor multiplicity corrected"},
               "native_per_token": native, "immutable_consumed_sources_unchanged": True,
               "source_file_count": len(sources.hashes), "source_hashes_file": "source_hashes.json",
               "analysis_file_sha256": digest(__file__), "numpy_version": np.__version__,
               "figures": {p.name: digest(p) for p in plots},
               "new_model_inference": False, "new_gpu_execution": False,
               "scientific_scope": "historical descriptive reanalysis only; no independent confirmation or superiority claim"}
    safe_public_json(summary)
    write_json(output / "summary.json", summary)
    print(json.dumps({"status": "PASS", "audit": audits, "historical_comparisons": counts,
                      "native_per_token": native}, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    run(args.v1, args.output, args.audit_output, args.figures)


if __name__ == "__main__":
    main()
