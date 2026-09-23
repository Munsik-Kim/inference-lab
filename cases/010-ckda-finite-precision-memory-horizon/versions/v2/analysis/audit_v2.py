"""Independent, model-free scalar audit of complete failure-aware v2 cells.

The audit reads saved predictions and gold; it imports neither the runtime nor
the metric implementation. A fresh-cohort directory must contain every one of
the fixed 3-seed x 5-arm cells. Missing cells are errors, never intersections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

import numpy as np

ARMS = ("NATIVE_FP32", "UNIFORM_8", "UNIFORM_5", "LOWRANK_4_8_R2", "MIXED_5_6_BUDGET")
GRID = (32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 512, 1024, 2048)
FRESH_N, FRESH_T, FRESH_FAMILY = 1024, 2048, 195
TERMINAL_CODES = set(range(7))


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label, tolerance=2e-11):
    require(isinstance(actual, (float, int)) and not isinstance(actual, bool) and math.isfinite(actual),
            f"{label}: expected finite scalar")
    require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), f"{label}: scalar mismatch")


def log_binomial_cdf(failures, count, probability):
    """Direct finite binomial sum, with log-combinations to avoid overflow."""
    require(type(failures) is int and type(count) is int and 0 <= failures <= count and count > 0,
            "invalid binomial counts")
    require(0 <= probability <= 1, "invalid binomial probability")
    if failures == count or probability == 0:
        return 0.0
    if probability == 1:
        return -math.inf
    logp, logq = math.log(probability), math.log1p(-probability)
    logn = math.lgamma(count + 1)
    terms = [logn - math.lgamma(j + 1) - math.lgamma(count - j + 1) + j * logp + (count - j) * logq
             for j in range(failures + 1)]
    top = max(terms)
    return top + math.log(math.fsum(math.exp(term - top) for term in terms))


def check_cp_upper(reported, failures, count, alpha):
    require(0 < alpha < 1, "invalid CP alpha")
    close(reported, reported, "CP endpoint")
    require(0 <= reported <= 1, "CP endpoint outside [0,1]")
    if failures == count:
        close(reported, 1.0, "all-failure CP endpoint")
    elif failures == 0:
        close(reported, -math.expm1(math.log(alpha) / count), "zero-failure CP endpoint")
    else:
        # At the exact one-sided CP upper endpoint, P_p[X <= failures] = alpha.
        value = log_binomial_cdf(failures, count, reported)
        require(math.isfinite(value) and abs(value - math.log(alpha)) <= 2e-7,
                "reported CP endpoint fails independent finite-binomial equation")


def reconstruct(predictions, gold):
    require(isinstance(predictions, np.ndarray) and predictions.dtype == np.int8 and predictions.ndim == 2,
            "predictions must be a two-dimensional int8 array")
    require(isinstance(gold, np.ndarray) and gold.dtype == np.uint8 and gold.ndim == 2,
            "gold must be a two-dimensional uint8 array")
    n, horizon = gold.shape
    require(n > 0 and horizon > 0 and predictions.shape == (n, horizon + 1), "prediction/BOS/gold shape mismatch")
    require(np.all((predictions >= -1) & (predictions <= 5)) and np.all(gold <= 5), "invalid S3 label or invalid sentinel")
    correct = predictions[:, 1:] == gold
    tau, lengths = [], []
    for row in correct:
        bad = np.flatnonzero(~row)
        first = int(bad[0]) + 1 if bad.size else None
        tau.append(first)
        lengths.append(horizon if first is None else first - 1)
    # Counting at every integer independently also exposes off-by-one errors.
    failure_counts = [sum(t is not None and t <= position for t in tau) for position in range(1, horizon + 1)]
    empirical = {}
    for epsilon in (.05, .01):
        eligible = [position for position, count in enumerate(failure_counts, 1) if count / n <= epsilon]
        empirical[epsilon] = max(eligible) if eligible else None
    quarter = max(horizon // 4, 1)
    return {"N": n, "T": horizon, "tau": tau, "RMST0": math.fsum(lengths) / n,
            "empirical": empirical, "failure_counts": failure_counts, "correct": correct,
            "token_accuracy": int(correct.sum()) / (n * horizon),
            "final_quarter_accuracy": int(correct[:, -quarter:].sum()) / (n * quarter),
            "invalid_predictions_including_BOS": int((predictions == -1).sum())}


def check_packed(packed, correct):
    n, horizon = correct.shape
    require(isinstance(packed, np.ndarray) and packed.dtype == np.uint8 and
            packed.shape == (n, (horizon + 7) // 8), "packed correctness layout mismatch")
    if horizon % 8:
        require(not np.any(packed[:, -1] >> (horizon % 8)), "nonzero correctness padding bits")
    unpacked = np.unpackbits(packed, axis=1, count=horizon, bitorder="little")
    require(np.array_equal(unpacked, correct), "packed correctness differs from predictions versus gold")


def check_terminal(predictions, codes, first_writes):
    n, cursor = predictions.shape
    require(isinstance(codes, list) and isinstance(first_writes, list) and len(codes) == len(first_writes) == n,
            "terminal metadata must retain every stream")
    for row, (code, first) in enumerate(zip(codes, first_writes)):
        require(type(code) is int and code in TERMINAL_CODES, "unknown terminal code")
        if code == 0:
            require(first is None, "active stream has terminal write")
        else:
            require(type(first) is int and 0 <= first < cursor, "terminal write outside consumed cursor")
            require(np.all(predictions[row, first:] == -1), "terminal stream predicts after its terminal write")
    return {str(code): codes.count(code) for code in sorted(TERMINAL_CODES)}


def check_ledger(ledger, n):
    integer_fields = ("per_stream_persistent_bytes", "state_code_bytes", "residual_code_bytes", "scale_bytes",
                      "rng_bytes", "cursor_bytes", "terminal_code_bytes", "first_terminal_write_bytes", "header_bytes",
                      "gauge_bytes", "padding_bytes", "shared_bytes", "shared_config_bytes", "shared_basis_bytes",
                      "shared_codec_bytes", "shared_token_table_bytes", "shared_runtime_metadata_bytes",
                      "payload_tensor_storage_bytes", "serialized_payload_bytes")
    require(isinstance(ledger, dict), "missing ledger")
    for key in integer_fields:
        require(type(ledger.get(key)) is int and ledger[key] >= 0, f"ledger {key}: missing/noninteger/negative")
    require((ledger["cursor_bytes"], ledger["terminal_code_bytes"], ledger["first_terminal_write_bytes"],
             ledger["header_bytes"]) == (8, 1, 8, 17), "ledger terminal header differs from 17-byte contract")
    stream = sum(ledger[key] for key in ("state_code_bytes", "residual_code_bytes", "scale_bytes", "rng_bytes",
                 "cursor_bytes", "terminal_code_bytes", "first_terminal_write_bytes", "gauge_bytes", "padding_bytes"))
    require(stream == ledger["per_stream_persistent_bytes"], "per-stream ledger arithmetic mismatch")
    require(ledger["shared_codec_bytes"] == ledger["shared_config_bytes"] + ledger["shared_basis_bytes"],
            "shared codec ledger mismatch")
    require(ledger["shared_bytes"] == ledger["shared_codec_bytes"] + ledger["shared_token_table_bytes"] +
            ledger["shared_runtime_metadata_bytes"], "shared ledger arithmetic mismatch")
    require(ledger["serialized_payload_bytes"] == ledger["payload_tensor_storage_bytes"] == n * stream,
            "actual-N payload ledger mismatch")
    totals = ledger.get("total_bytes")
    require(isinstance(totals, dict) and {"1", "16", "128"}.issubset(totals), "incomplete N=1/16/128 ledgers")
    for count, total in totals.items():
        require(isinstance(count, str) and count.isdecimal() and int(count) > 0 and type(total) is int,
                "invalid ledger count/total")
        require(total == ledger["shared_bytes"] + int(count) * stream, "amortization ledger mismatch")
    return {"per_stream_bytes": stream, "shared_bytes": ledger["shared_bytes"],
            "total_bytes": totals, "payload_bytes": n * stream,
            "scope": "independent component and denominator arithmetic; no actual cache/config bytes supplied to this check"}


def check_summary(summary, predictions, gold, packed, *, expected_fresh=True):
    require(isinstance(summary, dict), "summary is not an object")
    computed = reconstruct(predictions, gold)
    n, horizon = computed["N"], computed["T"]
    require(summary.get("N") == n and summary.get("T") == horizon, "summary denominator or horizon mismatch")
    require(summary.get("tau") == computed["tau"], "tau differs from predictions/gold")
    close(summary.get("RMST0"), computed["RMST0"], "RMST0")
    for suffix, epsilon in (("05", .05), ("01", .01)):
        require(summary.get(f"empirical_T{suffix}_all_tokens") == computed["empirical"][epsilon],
                "empirical all-token horizon mismatch")
    close(summary.get("token_accuracy"), computed["token_accuracy"], "token accuracy")
    close(summary.get("final_quarter_accuracy"), computed["final_quarter_accuracy"], "final quarter accuracy")
    close(summary.get("bos_accuracy"), float((predictions[:, 0] == 0).mean()), "unscored BOS accuracy")
    require(summary.get("bos_unscored_in_tau") is True, "BOS scoring contract changed")
    require(summary.get("first_failure_censored") == computed["tau"].count(None), "censor count mismatch")
    require(summary.get("invalid_prediction_tokens") == int((predictions[:, 1:] == -1).sum()), "invalid token count mismatch")
    check_packed(packed, computed["correct"])
    terminal = check_terminal(predictions, summary.get("terminal_codes"), summary.get("first_terminal_write"))
    require(summary.get("terminal_count") == n - terminal["0"], "terminal stream count mismatch")
    execution = summary.get("execution", {})
    require(execution.get("completed_writes") == horizon + 1, "completed write count mismatch")
    expected_status = "COMPLETE_WITH_TERMINAL_STREAMS" if terminal["0"] < n else "COMPLETE"
    require(execution.get("status") == expected_status, "execution terminal status mismatch")
    require(execution.get("active_update_attempts", 0) + execution.get("terminal_noop_steps", 0) == n * (horizon + 1),
            "execution dropped or added write attempts")
    if expected_fresh:
        require(n == FRESH_N and horizon == FRESH_T and summary.get("cohort") == "fresh", "wrong fresh cohort/denominator")
        require(summary.get("model_seed") in (0, 1, 2) and summary.get("arm") in ARMS, "unexpected fresh cell")
    rows = summary.get("grid_rows")
    require(isinstance(rows, list) and rows, "missing horizon grid")
    horizons = [r.get("horizon") for r in rows]
    require(all(type(h) is int and 1 <= h <= horizon for h in horizons) and horizons == sorted(set(horizons)),
            "invalid or duplicate grid positions")
    if expected_fresh:
        require(horizons == list(GRID), "fresh horizon grid changed")
    confidence = summary.get("confidence", {})
    family = summary.get("family_size", confidence.get("family_size"))
    if expected_fresh:
        require(family == FRESH_FAMILY, "fresh simultaneous family changed")
        alpha = confidence.get("family_alpha", summary.get("alpha"))
        close(alpha, .05, "family alpha")
    else:
        require(family is None, "diagnostic cell must not claim a simultaneous family")
        alpha = None
    for row in rows:
        failures = computed["failure_counts"][row["horizon"] - 1]
        require(row.get("failures") == failures, "grid failure count mismatch")
        close(row.get("F"), failures / n, "grid F")
        close(row.get("survival"), 1 - failures / n, "grid survival")
        if expected_fresh:
            check_cp_upper(row.get("p_upper"), failures, n, alpha / family)
        else:
            require(row.get("p_upper") is None, "diagnostic cell has an unsupported confidence bound")
    for suffix, epsilon in (("05", .05), ("01", .01)):
        eligible = [r["horizon"] for r in rows if r.get("p_upper") is not None and r["p_upper"] <= epsilon]
        expected = max(eligible) if eligible else None
        require(summary.get(f"supported_T{suffix}_grid") == expected, "supported grid horizon mismatch")
    ledger = check_ledger(summary.get("ledger"), n)
    return {"model_seed": summary.get("model_seed"), "arm": summary.get("arm"), "cohort": summary.get("cohort"),
            "N": n, "T": horizon, "first_failures": [r["failures"] for r in rows],
            "right_censored_sequences": computed["tau"].count(None), "RMST0": computed["RMST0"],
            "terminal_code_counts": terminal, "invalid_predictions_including_BOS": computed["invalid_predictions_including_BOS"],
            "ledger": ledger, "status": "PASS"}


def safe_artifact(root, name):
    require(isinstance(name, str) and bool(name) and "\\" not in name, "artifact path must be POSIX relative")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts and not re.match(r"^[A-Za-z]:", name),
            "unsafe artifact path")
    root = Path(root).resolve()
    path = root / name
    require(path.resolve().is_relative_to(root) and path.is_file(), "artifact is missing or escaped the case root")
    require(not any(part.is_symlink() for part in [path, *path.parents] if part != root and part.is_relative_to(root)),
            "artifact path contains a symlink")
    return path


def codec_byte_components(config):
    shape = config.get("shape")
    require(isinstance(shape, list) and len(shape) == 3 and all(type(x) is int and x > 0 for x in shape), "invalid codec shape")
    if config.get("format") == "ckda-native-v1":
        dtype = config.get("dtype")
        require(dtype in ("<f4", "<f8"), "unknown native dtype")
        return math.prod(shape) * int(dtype[-1]), 0, 0
    require(config.get("format") == "ckda-packed-v1", "unknown codec format")
    heads, keys, values = shape
    bits = config.get("mixed_bits")
    if bits is None:
        bits = [[config.get("bits")] * keys for _ in range(heads)]
    require(isinstance(bits, list) and len(bits) == heads and all(isinstance(r, list) and len(r) == keys for r in bits),
            "mixed allocation shape mismatch")
    require(all(type(b) is int and 2 <= b <= 16 for row in bits for b in row), "invalid packed width")
    require(config.get("scale_dtype") == "<f4" and config.get("bit_order") == "lsb_first", "unknown scale or bit layout")
    return (sum(map(sum, bits)) * values + 7) // 8, heads * 4, 16 if config.get("stochastic") else 0


def check_shared_artifacts(summary, case_root):
    declared = summary.get("shared_artifacts")
    required = {"initial_payload", "codec_config", "basis", "token_config", "token_data", "runtime_config"}
    require(isinstance(declared, dict) and set(declared) == required, "incomplete actual-byte artifact map")
    data, identities = {}, {}
    for name, entry in declared.items():
        require(isinstance(entry, dict) and type(entry.get("bytes")) is int and entry["bytes"] >= 0, "invalid artifact byte declaration")
        path = safe_artifact(case_root, entry.get("path"))
        raw = path.read_bytes()
        require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry.get("sha256"),
                "shared artifact length/hash mismatch")
        data[name] = raw
        identities[name] = entry
    ledger = summary["ledger"]
    initial = data["initial_payload"]
    require(len(initial) == ledger["per_stream_persistent_bytes"] and len(initial) >= 17, "one-stream actual payload length mismatch")
    require(int.from_bytes(initial[:8], "little") == 0 and initial[8] == 0 and
            int.from_bytes(initial[9:17], "little") == (1 << 64) - 1, "initial payload header is not active at cursor zero")
    cfg = json.loads(data["codec_config"])
    require(cfg.get("format") == "ckda-online-failure-aware-v2" and cfg.get("header_bytes") == 17,
            "wrong serialized cache format")
    require(cfg.get("bos_policy") == "present_write0" and cfg.get("cursor_semantics") == "next_zero_based_write",
            "wrong serialized BOS/cursor contract")
    state = codec_byte_components(cfg["state"])
    residual = (0, 0, 0) if cfg["residual"] is None else codec_byte_components(cfg["residual"])
    require((ledger["state_code_bytes"], ledger["residual_code_bytes"], ledger["scale_bytes"], ledger["rng_bytes"]) ==
            (state[0], residual[0], state[1] + residual[1], state[2] + residual[2]), "codec-derived byte components mismatch")
    basis_shape = cfg.get("basis_shape")
    if basis_shape is None:
        require(len(data["basis"]) == 0, "unexpected basis bytes")
    else:
        require(isinstance(basis_shape, list) and len(basis_shape) == 3 and all(type(v) is int and v > 0 for v in basis_shape),
                "invalid basis shape")
        require(len(data["basis"]) == 4 * math.prod(basis_shape), "basis byte length mismatch")
        require(np.isfinite(np.frombuffer(data["basis"], dtype="<f4")).all(), "nonfinite shared basis")
    require(hashlib.sha256(data["basis"]).hexdigest() == cfg.get("basis_sha256"), "basis config identity mismatch")
    require(ledger["shared_config_bytes"] == len(data["codec_config"]) and ledger["shared_basis_bytes"] == len(data["basis"]),
            "actual config/basis ledger mismatch")
    require(ledger["shared_token_table_bytes"] == len(data["token_config"]) + len(data["token_data"]) and
            ledger["shared_runtime_metadata_bytes"] == len(data["runtime_config"]), "actual table/runtime ledger mismatch")
    runtime = json.loads(data["runtime_config"])
    require(runtime.get("invalid_prediction") == -1 and runtime.get("terminal_output") == "always_invalid" and
            runtime.get("bos_policy") == "present_write0", "shared runtime output contract changed")
    return {"status": "PASS", "artifacts": identities,
            "scope": "actual one-stream initial payload and shared config/basis/table/runtime bytes; final hidden-state body excluded"}


def check_input_identity(summary, gold, case_root):
    manifest_path = safe_artifact(case_root, "inputs/manifest.json")
    require(sha(manifest_path) == summary.get("input_manifest_sha256"), "input manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    cohort = manifest["cohorts"][summary["cohort"]]
    require((cohort["sequences"], cohort["group_tokens"]) == gold.shape, "input cohort shape mismatch")
    gold_path = safe_artifact(case_root, cohort["gold_file"])
    require(sha(gold_path) == cohort["gold_file_sha256"], "input gold artifact hash mismatch")
    expected_gold = np.load(gold_path, allow_pickle=False)
    require(expected_gold.dtype == np.uint8 and np.array_equal(gold, expected_gold), "prediction artifact uses different gold")
    ids_path = safe_artifact(case_root, cohort["sample_ids_file"])
    require(sha(ids_path) == cohort["sample_ids_sha256"] == summary.get("sample_ids_sha256"), "sample identity hash mismatch")
    ids = json.loads(ids_path.read_text())
    require(isinstance(ids, list) and len(ids) == len(gold) and len(set(ids)) == len(ids), "invalid/missing/duplicate input identities")
    return {"manifest_sha256": sha(manifest_path), "gold_file_sha256": sha(gold_path), "sample_ids_sha256": sha(ids_path)}


def audit_cell(folder, *, expected_fresh=True, case_root=None):
    folder = Path(folder)
    summary_path, artifact = folder / "summary.json", folder / "predictions.npz"
    require(summary_path.is_file() and artifact.is_file(), "cell is missing summary or predictions")
    require(not summary_path.is_symlink() and not artifact.is_symlink(), "cell artifacts must not be symlinks")
    summary = json.loads(summary_path.read_text())
    require(summary.get("schema") == "case010-failure-aware-v2-cell-v1", "unknown result schema")
    require(summary.get("execution", {}).get("status") in ("COMPLETE", "COMPLETE_WITH_TERMINAL_STREAMS"), "cell is not complete")
    declared = summary.get("artifact_sha256")
    expected = declared.get("predictions.npz") if isinstance(declared, dict) else declared
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            "missing predictions artifact identity")
    require(sha(artifact) == expected, "predictions artifact hash mismatch")
    with np.load(artifact, allow_pickle=False) as saved:
        require(set(saved.files) == {"predictions", "correctness", "gold"}, "unexpected prediction artifact fields")
        result = check_summary(summary, saved["predictions"], saved["gold"], saved["correctness"], expected_fresh=expected_fresh)
        result["gold_bytes_sha256"] = hashlib.sha256(saved["gold"].tobytes()).hexdigest()
        if case_root is not None:
            result["input_identity"] = check_input_identity(summary, saved["gold"], case_root)
    require(case_root is not None, "case root is required for actual byte and input identity checks")
    result["actual_byte_verification"] = check_shared_artifacts(summary, case_root)
    result["artifacts"] = {"summary.json": sha(summary_path), "predictions.npz": expected}
    return result


def audit_results(results, *, allow_partial=False, case_root=None):
    results = Path(results).resolve()
    require(results.is_dir(), "fresh results directory is missing")
    # A manifest cannot turn partial discovery into an implicit smaller family.
    summaries = sorted(results.rglob("summary.json"))
    if case_root is None:
        case_root = next((p for p in [results, *results.parents] if (p / "inputs/manifest.json").is_file()), None)
    require(case_root is not None, "cannot locate case root with frozen input manifest")
    cells = []
    pending = []
    for artifact in results.rglob("predictions.npz"):
        if not (artifact.parent / "summary.json").is_file():
            require(allow_partial, "prediction artifact has no completed summary")
            pending.append(artifact.parent.relative_to(results).as_posix())
    for summary in summaries:
        item = json.loads(summary.read_text())
        require(item.get("cohort") == "fresh", "non-fresh summary in fresh results directory")
        if item.get("execution", {}).get("status") not in ("COMPLETE", "COMPLETE_WITH_TERMINAL_STREAMS"):
            require(allow_partial, "fresh cell is incomplete")
            pending.append(summary.parent.relative_to(results).as_posix())
            continue
        cell = audit_cell(summary.parent, case_root=case_root)
        cell["cell"] = summary.parent.relative_to(results).as_posix()
        cells.append(cell)
    expected = {(seed, arm) for seed in (0, 1, 2) for arm in ARMS}
    observed = [(r["model_seed"], r["arm"]) for r in cells]
    require(len(observed) == len(set(observed)), "duplicate fresh cell")
    require(set(observed).issubset(expected), "unexpected fresh cell")
    complete = set(observed) == expected and not pending
    require(allow_partial or complete, "incomplete fresh 3-seed x 5-arm cell set")
    require(len({r["gold_bytes_sha256"] for r in cells}) <= 1, "fresh gold differs between cells")
    return {"schema": "case010-v2-independent-scalar-audit-v1", "status": "PASS" if complete else "INCOMPLETE", "cells": cells,
            "missing_cells": [{"model_seed": seed, "arm": arm} for seed, arm in sorted(expected - set(observed))],
            "pending_cell_paths": pending,
            "verified_cells": len(cells), "sequences_per_cell": FRESH_N, "family_size": FRESH_FAMILY,
            "auditor_sha256": sha(__file__), "numpy_version": np.__version__,
            "CP_validation": "independent finite-binomial endpoint equation with log-combinations; no root metrics import",
            "model_inference_executed": False, "excluded_streams": 0,
            "scope": "retained predictions/gold and frozen input identities, packed correctness, absorbing terminal outputs, scalar reconstruction, actual fixed-size initial cache/shared bytes, and ledger arithmetic"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly emit INCOMPLETE for missing cells; never PASS on a subset")
    parser.add_argument("--case-root", type=Path)
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.results.resolve()), "audit receipt must be outside audited results")
    require(not args.output.exists(), "audit receipt already exists")
    receipt = audit_results(args.results, allow_partial=args.allow_partial, case_root=args.case_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "verified_cells": receipt["verified_cells"], "receipt_sha256": sha(args.output)}))


if __name__ == "__main__":
    try:
        main()
    except (AuditError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
