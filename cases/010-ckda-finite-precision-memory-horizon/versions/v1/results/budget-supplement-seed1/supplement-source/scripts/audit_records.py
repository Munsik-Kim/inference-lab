"""Independently audit retained Case010 scalars, packed correctness, and byte ledgers.

No model execution, codec import, or summary implementation is used. The audit
checks the reported CP endpoint by evaluating a direct binomial CDF with exact
integer combinations; it does not rerun the implementation's inverse-CDF code.
Receipts use logical relative artifact names and SHA256 identities only.
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


HORIZONS = [32, 64, 128, 256, 512, 1024, 2048]
LEARNED_ARMS = (["NATIVE_FP32"] + [f"UNIFORM_{b}" for b in range(2, 17)] +
                ["STOCHASTIC_4", "FULL_RESIDUAL_4_4", "FULL_RESIDUAL_4_8", "FULL_RESIDUAL_4_FP32",
                 "LOWRANK_4_8_R1", "LOWRANK_4_8_R2", "LOWRANK_4_8_R4", "UNTRANSPORTED_4_4",
                 "MIXED_4_8", "MIXED_6_8"])
HISTORICAL_ERRORS = {("s4_rotated", "uniform_2"), ("c31_moving_plane", "uniform_2")}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hash_id(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{label}: invalid SHA256 identifier")
    return value


def relative(name):
    path = PurePosixPath(name)
    require(bool(name) and not path.is_absolute() and ".." not in path.parts,
            "artifact name must be a safe relative path")
    return path.as_posix()


class Artifacts:
    def __init__(self, root, label):
        self.root, self.label = Path(root), relative(label)
        self.receipts = {}

    def read(self, name, expected=None):
        name = relative(name)
        path = self.root / name
        require(path.is_file(), f"{self.label}/{name}: required artifact missing")
        raw = path.read_bytes()
        digest = sha(raw)
        if expected is not None:
            require(digest == hash_id(expected, f"{self.label}/{name}"), f"{self.label}/{name}: SHA256 mismatch")
        self.receipts[name] = {"file": f"{self.label}/{name}", "sha256": digest, "bytes": len(raw)}
        return raw

    def json(self, name, expected=None):
        try:
            return json.loads(self.read(name, expected))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AuditError(f"{self.label}/{name}: invalid JSON") from exc

    def rows(self, name):
        try:
            return [json.loads(line) for line in self.read(name).splitlines() if line.strip()]
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AuditError(f"{self.label}/{name}: invalid JSONL") from exc

    def source_files(self, expected, *, flat=False):
        require(isinstance(expected, dict) and bool(expected), f"{self.label}: source identity map missing")
        for name, digest in expected.items():
            name = relative(name)
            candidates = ([f"frozen-source/{name}"] if flat else
                          [f"frozen-source/{name.replace('/', '--')}", f"frozen-source/{name}"])
            found = next((candidate for candidate in candidates if (self.root / candidate).is_file()), None)
            require(found is not None, f"{self.label}: frozen source missing for {name}")
            self.read(found, digest)


def close(actual, expected, label, *, absolute=2e-12):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual),
            f"{label}: expected finite scalar")
    require(math.isclose(float(actual), float(expected), rel_tol=2e-11, abs_tol=absolute),
            f"{label}: scalar mismatch")


def correctness(store, name, expected_shape=(512, 2048), expected_sha=None):
    import io
    raw = store.read(name, expected_sha)
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            require(set(saved.files) == {"shape", "packed"}, f"{name}: unexpected correctness fields")
            declared, packed = saved["shape"], saved["packed"]
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, AuditError):
            raise
        raise AuditError(f"{name}: invalid correctness archive") from exc
    require(declared.dtype.kind in "iu" and declared.shape == (2,), f"{name}: invalid shape metadata")
    shape = tuple(int(value) for value in declared)
    require(shape == tuple(expected_shape), f"{name}: wrong sequence count or horizon")
    require(packed.dtype == np.uint8 and packed.shape == (shape[0], (shape[1] + 7) // 8),
            f"{name}: invalid packed byte layout")
    if shape[1] % 8:
        require(not np.any(packed[:, -1] >> (shape[1] % 8)), f"{name}: nonzero padding")
    return np.unpackbits(packed, axis=1, count=shape[1], bitorder="little").astype(bool)


def check_cp_endpoint(bound, failures, count, alpha, label):
    require(0 < alpha < 1, f"{label}: invalid alpha")
    close(bound, bound, label)
    require(0 <= bound <= 1, f"{label}: bound outside [0,1]")
    if failures == count:
        close(bound, 1.0, label)
    elif failures == 0:
        close(bound, 1 - alpha ** (1 / count), label)
    else:
        # n=512: exact integer combinations fit safely when converted to double.
        cdf = math.fsum(math.comb(count, j) * bound**j * (1 - bound)**(count-j)
                        for j in range(failures + 1))
        require(math.isclose(cdf, alpha, rel_tol=2e-7, abs_tol=1e-14), f"{label}: binomial CDF does not equal alpha")


def check_summary(summary, correct, *, expected_n=512, expected_horizon=2048,
                  horizons=HORIZONS, family_size, alpha=0.05):
    """Reconstruct each statistic directly from retained correctness bits."""
    require(correct.shape == (expected_n, expected_horizon), "summary: wrong correctness denominator")
    require(summary["n_sequences"] == expected_n and summary["max_horizon"] == expected_horizon,
            "summary: wrong sequence denominator or horizon")
    tau = []
    for row in correct:
        failures = np.flatnonzero(~row)
        tau.append(int(failures[0] + 1) if failures.size else None)
    require(summary["tau"] == tau, "summary: tau disagrees with retained correctness")
    ids = summary["sequence_ids"]
    require(len(ids) == expected_n and len(set(ids)) == expected_n, "summary: invalid independent sequence IDs")
    require(summary["observed_failures"] == sum(value is not None for value in tau), "summary: observed failure count mismatch")
    require(summary["right_censored_sequences"] == tau.count(None), "summary: censor count mismatch")
    mean_length = math.fsum(expected_horizon if value is None else value - 1 for value in tau) / expected_n
    close(summary["restricted_mean_failure_free_length"], mean_length, "summary: RMST")
    confidence = summary["confidence"]
    require(confidence["family_size"] == family_size, "summary: wrong prespecified family denominator")
    close(confidence["family_alpha"], alpha, "summary: family alpha")
    close(confidence["per_comparison_alpha"], alpha / family_size, "summary: corrected alpha")
    require([row["horizon"] for row in summary["horizons"]] == list(horizons), "summary: horizon grid mismatch")
    counts = []
    for row in summary["horizons"]:
        horizon = row["horizon"]
        failures = sum(value is not None and value <= horizon for value in tau)
        require(row["first_failures"] == failures, "summary: horizon first-failure count mismatch")
        close(row["failure_probability"], failures / expected_n, "summary: F(T)")
        close(row["survival_probability"], 1 - failures / expected_n, "summary: survival")
        check_cp_endpoint(row["failure_upper_bound"], failures, expected_n, alpha / family_size, "summary: CP endpoint")
        close(row["survival_lower_bound"], 1 - row["failure_upper_bound"], "summary: survival lower bound")
        counts.append(failures)
    step = summary["step_accuracy"]
    if step is not None:
        close(step["overall"], int(correct.sum()) / correct.size, "summary: overall accuracy")
        require([item["horizon"] for item in step["at_horizon"]] == list(horizons), "summary: step grid mismatch")
        require([item["horizon"] for item in step["prefix"]] == list(horizons), "summary: prefix grid mismatch")
        for item in step["at_horizon"]:
            close(item["accuracy"], int(correct[:, item["horizon"]-1].sum()) / expected_n, "summary: step accuracy")
        for item in step["prefix"]:
            t = item["horizon"]
            close(item["accuracy"], int(correct[:, :t].sum()) / (expected_n*t), "summary: prefix accuracy")
    if "step_correct_counts" in summary:
        require(summary["step_correct_counts"] == correct.sum(axis=0).tolist(), "summary: step correct counts mismatch")
        require(len(summary["step_accuracy_by_position"]) == expected_horizon, "summary: step accuracy length mismatch")
        for reported, count in zip(summary["step_accuracy_by_position"], correct.sum(axis=0)):
            close(reported, int(count) / expected_n, "summary: per-position accuracy")
        require([row["horizon"] for row in summary["token_counts_at_horizons"]] == list(horizons), "summary: token grid mismatch")
        for row in summary["token_counts_at_horizons"]:
            t, q = row["horizon"], max(row["horizon"] // 4, 1)
            require(row["prefix_total"] == expected_n*t and row["final_quarter_total"] == expected_n*q,
                    "summary: token denominator mismatch")
            require(row["prefix_correct"] == int(correct[:, :t].sum()) and
                    row["final_quarter_correct"] == int(correct[:, t-q:t].sum()), "summary: token count mismatch")
    for field, statistic in [("empirical_T_epsilon", "failure_probability"),
                             ("confidence_supported_T_epsilon_lower_bound", "failure_upper_bound")]:
        for record in summary[field]:
            eligible = [row["horizon"] for row in summary["horizons"] if row[statistic] <= record["epsilon"]]
            require(record["horizon"] == (max(eligible) if eligible else None), f"summary: {field} mismatch")
    return {"n_sequences": expected_n, "horizon": expected_horizon, "first_failures": counts,
            "restricted_mean_failure_free_length": mean_length, "correctness_sha256": sha(np.packbits(correct, axis=1, bitorder="little").tobytes())}


def shape_product(shape):
    require(isinstance(shape, list) and shape and all(type(n) is int and n > 0 for n in shape), "ledger: invalid array shape")
    return math.prod(shape)


def codec_bytes(config):
    shape = config["shape"]
    count = shape_product(shape)
    require(len(shape) == 3, "ledger: codec shape must have three axes")
    if config["format"] == "ckda-native-v1":
        require(config["dtype"] in ("<f4", "<f8"), "ledger: unsupported native dtype")
        return count * int(config["dtype"][-1]), 0, 0
    require(config["format"] == "ckda-packed-v1", "ledger: unsupported packed format")
    heads, keys, values = shape
    widths = config["mixed_bits"]
    if widths is None:
        widths = [[config["bits"]] * keys for _ in range(heads)]
    require(len(widths) == heads and all(len(row) == keys for row in widths), "ledger: invalid mixed map shape")
    require(all(type(b) is int and 2 <= b <= 16 for row in widths for b in row), "ledger: invalid bit width")
    require(config["scale_dtype"] == "<f4" and config["bit_order"] == "lsb_first", "ledger: unsupported scale/packing metadata")
    rng = 16 if config["stochastic"] else 0
    require((config["rng"] is not None) == bool(rng), "ledger: RNG metadata mismatch")
    return (values * sum(map(sum, widths)) + 7) // 8, heads * 4, rng


def check_online_ledger(ledger, config_raw, basis_raw, *, count=512, table_bytes=0):
    config = json.loads(config_raw)
    require(config["format"] == "ckda-online-v1" and config["cursor"] == "<u8", "ledger: invalid online format")
    require(canonical(config) == config_raw, "ledger: noncanonical online configuration")
    require(config["gauge_bytes_per_stream"] == 0, "ledger: unsupported gauge bytes")
    basis_shape = config["basis_shape"]
    expected_basis = 0 if basis_shape is None else shape_product(basis_shape) * 4
    require(len(basis_raw) == expected_basis, "ledger: basis byte count mismatch")
    if basis_shape is not None:
        require(config["basis_dtype"] == "<f4", "ledger: wrong basis dtype")
        require(np.isfinite(np.frombuffer(basis_raw, dtype="<f4")).all(), "ledger: nonfinite basis")
    state_data, state_scales, state_rng = codec_bytes(config["state"])
    residual_data, residual_scales, residual_rng = (0, 0, 0) if config["residual"] is None else codec_bytes(config["residual"])
    stream = 8 + state_data + state_scales + state_rng + residual_data + residual_scales + residual_rng
    codec_shared = len(config_raw) + len(basis_raw)
    expected = dict(per_stream_persistent_bytes=stream, state_code_bytes=state_data, residual_code_bytes=residual_data,
                    scale_bytes=state_scales+residual_scales, rng_bytes=state_rng+residual_rng,
                    cursor_bytes=8, gauge_bytes=0, padding_bytes=0,
                    shared_bytes=codec_shared+table_bytes, shared_config_bytes=len(config_raw), shared_basis_bytes=len(basis_raw),
                    payload_tensor_storage_bytes=count*stream)
    for key, value in expected.items():
        require(ledger[key] == value, f"ledger: {key} mismatch")
    if table_bytes:
        require(ledger["shared_token_coefficient_bytes"] == table_bytes and ledger["shared_codec_bytes"] == codec_shared,
                "ledger: shared table/config decomposition mismatch")
    for n in (1, 16, 128):
        require(ledger["total_bytes"][str(n)] == codec_shared+table_bytes+n*stream, "ledger: amortization denominator mismatch")
    return {"per_stream_bytes": stream, "codec_shared_bytes": codec_shared, "shared_bytes": codec_shared+table_bytes}


def json_prefix(raw):
    require(raw.startswith(b"{"), "shared buffer: missing JSON object prefix")
    depth, quoted, escaped = 0, False, False
    for position, byte in enumerate(raw):
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte == 123:
            depth += 1
        elif byte == 125:
            depth -= 1
            if depth == 0:
                end = position+1
                return json.loads(raw[:end]), end
    raise AuditError("shared buffer: truncated JSON prefix")


def check_toy_shared(row, raw, *, count=512):
    config, end = json_prefix(raw)
    basis_count = 0 if config["basis_shape"] is None else shape_product(config["basis_shape"])*4
    layout = check_online_ledger(row["ledger"]["adapter"], raw[:end], raw[end:end+basis_count], count=count)
    program = raw[end+basis_count:]
    specification, header = json_prefix(program)
    require(program[header:header+1] == b"\n", "toy shared: program separator missing")
    require(specification["name"] == row["condition"], "toy shared: condition mismatch")
    if "operator_shape" in specification:
        expected = header+1+8*(shape_product(specification["operator_shape"])+shape_product(specification["prototype_shape"]))
    else:
        offset = header+1+8*shape_product(specification["prototype_shape"])
        coeff, coeff_header = json_prefix(program[offset:])
        require(coeff["format"] == "toy-coefficient-int8-v1", "toy shared: missing coefficient format")
        require(program[offset+coeff_header:offset+coeff_header+1] == b"\n", "toy shared: coefficient separator missing")
        expected = offset+coeff_header+1+shape_product(coeff["shape"])+4*shape_product(coeff["shape"][:-2])
    require(len(program) == expected, "toy shared: transition/readout byte count mismatch")
    ledger = row["ledger"]
    for key, value in {"stream_payload_bytes":layout["per_stream_bytes"], "batch_payload_nbytes":count*layout["per_stream_bytes"],
                       "adapter_shared_bytes":end+basis_count, "toy_program_shared_bytes":len(program), "total_shared_bytes":len(raw)}.items():
        require(ledger[key] == value, f"toy ledger: {key} mismatch")
    for n in (1,16,128):
        close(ledger["amortized_bytes_per_stream"][str(n)], layout["per_stream_bytes"]+len(raw)/n, "toy ledger: amortized bytes")
    return config


def check_capacity(row, raw):
    require(row["status"] == "ok" and row["n_sequences"] == 512 and row["max_horizon"] == 2048,
            "toy capacity: record dimensions/status mismatch")
    require(row["tau"] == [None]*512 and row["step_accuracy"] == 1, "toy capacity: claimed exact tracker record changed")
    metadata, end = json_prefix(raw)
    table = shape_product(metadata["table_shape"])
    prototypes = shape_product(metadata["prototype_shape"])*8
    require(raw[end:end+1] == b"\n" and len(raw) == end+1+table+prototypes, "toy capacity: shared byte layout mismatch")
    ledger = row["ledger"]
    for key, value in {"stream_payload_bytes":9,"total_shared_bytes":len(raw),"multiplication_table_bytes":table,
                       "decoder_config_bytes":end+1,"prototype_bytes":prototypes}.items():
        require(ledger[key] == value, f"toy capacity ledger: {key} mismatch")
    for n in (1,16,128):
        close(ledger["amortized_bytes_per_stream"][str(n)],9+len(raw)/n,"toy capacity ledger: amortized bytes")


def audit_toys(original_dir, repair_dir=None):
    original = Artifacts(original_dir, "toy_original")
    protocol = original.json("protocol-frozen.json")
    protocol_id = sha(canonical(protocol))
    require(original.read("protocol-frozen.sha256").decode().strip() == protocol_id, "toy: canonical protocol hash mismatch")
    original.source_files(protocol["code_sha256"], flat=True)
    require(protocol["splits"]["TEST"] == {"n":512,"horizon":2048,"seed":3001} and protocol["family_size"] == 875,
            "toy: primary sample/family declaration changed")
    rows = original.rows("results.jsonl")
    primary = [row for row in rows if row["arm"] != "exact_symbolic_id"]
    require(len(rows) == 130 and len(primary) == 125, "toy: expected 125 arms and five capacity controls")
    expected_pairs = {(condition["name"], arm) for condition in protocol["conditions"] for arm in protocol["arms"]}
    require(len(expected_pairs) == 125 and {(row["condition"],row["arm"]) for row in primary} == expected_pairs,
            "toy: frozen arm family mismatch")
    errors = {(row["condition"],row["arm"]) for row in primary if row["status"] == "error"}
    require(errors == HISTORICAL_ERRORS, "toy: historical error record set changed")
    capacities = [row for row in rows if row["arm"] == "exact_symbolic_id"]
    require({row["condition"] for row in capacities} == {condition["name"] for condition in protocol["conditions"]},
            "toy: capacity condition set mismatch")
    for row in capacities:
        check_capacity(row,original.read(f"shared/TEST--{row['condition']}--exact_symbolic_id.bin"))
    checked_original = 0
    for row in primary:
        require(row["protocol_sha256"] == protocol_id and row["split"] == "TEST", "toy: row protocol/split mismatch")
        if row["status"] == "error":
            continue
        key = f"TEST--{row['condition']}--{row['arm']}"
        correct = correctness(original, f"correctness/{key}.npz")
        check_summary(row["summary"], correct, family_size=875)
        config = check_toy_shared(row, original.read(f"shared/{key}.bin"))
        require(config == protocol["arm_configurations"][row["condition"]][row["arm"]], "toy: shared config differs from frozen arm")
        checked_original += 1
    result = {"canonical_protocol_sha256":protocol_id, "original_verified_success_records":checked_original,
              "historical_error_records":2, "historical_stochastic_v1_records":5,
              "capacity_controls":5, "capacity_control_note":"symbolic controls retained; no raw correctness artifact was recorded for independent replay"}
    stores = [original]
    if repair_dir is not None:
        repair = Artifacts(repair_dir, "toy_repair")
        identity = repair.json("erratum-identity.json")
        identity_id = sha(repair.read("erratum-identity.json"))
        require(repair.read("erratum-identity.sha256").decode().strip() == identity_id, "toy repair: identity hash mismatch")
        require(identity["canonical_protocol_sha256"] == protocol_id, "toy repair: original protocol mismatch")
        require(identity["protocol_file_sha256"] == sha(original.read("protocol-frozen.json")), "toy repair: exact protocol hash mismatch")
        require(identity["prior_results_file_sha256"] == sha(original.read("results.jsonl")), "toy repair: prior results changed")
        require(repair.read("original-protocol-frozen.json") == original.read("protocol-frozen.json"), "toy repair: original protocol copy differs")
        repair.source_files(identity["new_source_file_sha256"])
        require(identity["calibration_identity_file_sha256"] == sha(original.read("calibration-identity.json")), "toy repair: calibration identity mismatch")
        prior = repair.rows("prior-attempts.jsonl")
        selected = {pair for pair in expected_pairs if pair in errors or pair[1] == "stochastic_4"}
        require(len(prior) == 7 and {(row["result"]["condition"],row["result"]["arm"]) for row in prior} == selected,
                "toy repair: historical attempt retention mismatch")
        old_by_pair = {(row["condition"],row["arm"]):row for row in rows}
        for entry in prior:
            row = entry["result"]
            require(row == old_by_pair[(row["condition"],row["arm"])], "toy repair: historical record was altered")
            require(entry["prior_attempt_id"] == identity["prior_attempt_id"], "toy repair: historical attempt ID mismatch")
        reruns = repair.rows("rerun-results.jsonl")
        require(len(reruns) == 7 and {(row["condition"],row["arm"]) for row in reruns} == selected, "toy repair: rerun arm set mismatch")
        for row in reruns:
            pair = (row["condition"],row["arm"])
            key = f"TEST--{pair[0]}--{pair[1]}"
            require(row["previous_result_sha256"] == sha(canonical(old_by_pair[pair])), "toy repair: original row hash mismatch")
            require(row["erratum_identity_sha256"] == identity_id and row["protocol_sha256"] == protocol_id,
                    "toy repair: rerun identity mismatch")
            check_summary(row["summary"],correctness(repair,f"correctness/{key}.npz"),family_size=875)
            check_toy_shared(row,repair.read(f"shared/{key}.bin"))
        derived = repair.rows("derived-results.jsonl")
        current = [row for row in derived if row["arm"] != "exact_symbolic_id"]
        require(len(derived) == 130 and len(current) == 125 and {(r["condition"],r["arm"]) for r in current} == expected_pairs,
                "toy repair: current arm family mismatch")
        version_counts = {}
        for row in current:
            pair = (row["condition"],row["arm"])
            owner = repair if pair in selected else original
            key = f"TEST--{pair[0]}--{pair[1]}"
            require(row["status"] == "ok" and row["protocol_sha256"] == protocol_id, "toy repair: current record incomplete or wrong protocol")
            require(row["erratum_identity_sha256"] == identity_id, "toy repair: current erratum hash mismatch")
            require(not Path(row["raw_artifact_owner"]).is_absolute(), "toy repair: absolute artifact owner path")
            require((repair.root/row["raw_artifact_owner"]).resolve() == owner.root.resolve(), "toy repair: raw artifact owner disagrees with retained source")
            correct = correctness(owner, f"correctness/{key}.npz")
            check_summary(row["summary"], correct, family_size=875)
            config = check_toy_shared(row, owner.read(f"shared/{key}.bin"))
            require(config == identity["arm_configurations"][f"{pair[0]}/{pair[1]}"], "toy repair: codec config mismatch")
            version_counts[row["evaluation_version"]] = version_counts.get(row["evaluation_version"],0)+1
        require(sorted(version_counts.values()) == [2,5,118], "toy repair: current version composition mismatch")
        for row in derived:
            if row["arm"] == "exact_symbolic_id":
                check_capacity(row,original.read(f"shared/TEST--{row['condition']}--exact_symbolic_id.bin"))
        manifest = repair.json("manifest.json")
        require(manifest["status"] == "complete" and manifest["derived_primary_arms"] == 125 and manifest["reevaluated_arms"] == 7,
                "toy repair: incomplete manifest")
        result.update(current_verified_records=125, current_version_counts=version_counts, erratum_identity_sha256=identity_id)
        stores.append(repair)
    result["artifacts"] = [entry for store in stores for entry in store.receipts.values()]
    return result


def check_token_table(config_raw, raw):
    config = json.loads(config_raw)
    require(config["format"] in ("ckda-token-coefficients-v1","ckda-token-coefficients-v2-b1-t1") and
            config["token_ids"] == list(range(7)), "learned: token table metadata mismatch")
    offset = 0
    for _, entry in sorted(config["entries"].items()):
        count = shape_product(entry["shape"])
        require(entry["dtype"] == "<f4" and entry["offset"] == offset and entry["bytes"] == count*4, "learned: token table byte layout mismatch")
        offset += count*4
    require(offset == len(raw), "learned: token table byte length mismatch")
    require(np.isfinite(np.frombuffer(raw,dtype="<f4")).all(), "learned: nonfinite token table")
    return len(config_raw)+len(raw)


def check_learned_arm(name, config):
    state, residual = config["state"], config["residual"]
    require(state["shape"] == [12,16,16], "learned: state shape differs from declared architecture")
    expected_mode = "untransported" if name == "UNTRANSPORTED_4_4" else "transported"
    require(config["mode"] == expected_mode, "learned: arm transport mode mismatch")
    if name == "NATIVE_FP32":
        require(state["format"] == "ckda-native-v1" and state["dtype"] == "<f4" and residual is None,
                "learned: native arm configuration mismatch")
        return
    expected_bits = int(name.rsplit("_",1)[1]) if name.startswith("UNIFORM_") else (6 if name == "MIXED_6_8" else 4)
    require(state["format"] == "ckda-packed-v1" and state["bits"] == expected_bits and state["max_abs"] is None,
            "learned: state precision/range mismatch")
    require(state["stochastic"] == (name == "STOCHASTIC_4"), "learned: stochastic arm mismatch")
    if name == "STOCHASTIC_4":
        require(state["rng"] == "splitmix64_seedkey_v2_next_counter_le_u64", "learned: uncorrected stochastic generator")
    if name.startswith("MIXED_"):
        widths = state["mixed_bits"]
        promoted = 8 if expected_bits == 6 else 4
        require(isinstance(widths,list) and len(widths) == 12 and all(len(row) == 16 and row.count(8) == promoted and
                    all(value in (expected_bits,8) for value in row) for row in widths), "learned: mixed precision map mismatch")
    else:
        require(state["mixed_bits"] is None, "learned: unexpected mixed precision")
    if name.startswith("LOWRANK_"):
        rank = int(name[-1])
        require(config["basis_shape"] == [12,16,rank] and residual is not None and residual["shape"] == [12,rank,16]
                and residual["format"] == "ckda-packed-v1" and residual["bits"] == 8, "learned: lowrank residual mismatch")
    elif name.startswith("FULL_RESIDUAL_") or name == "UNTRANSPORTED_4_4":
        require(config["basis_shape"] is None and residual is not None and residual["shape"] == [12,16,16],
                "learned: full residual shape mismatch")
        if name.endswith("FP32"):
            require(residual["format"] == "ckda-native-v1" and residual["dtype"] == "<f4", "learned: floating residual dtype mismatch")
        else:
            require(residual["format"] == "ckda-packed-v1" and residual["bits"] == int(name[-1]), "learned: residual precision mismatch")
    else:
        require(residual is None and config["basis_shape"] is None, "learned: unexpected residual cache")


def audit_learned(directory, checkpoints=None):
    temporary = Artifacts(directory, "learned")
    manifest = temporary.json("manifest.json")
    seed = manifest["model_seed"]
    require(type(seed) is int and seed in (0,1,2), "learned: invalid model seed")
    store = Artifacts(directory, f"learned_seed{seed}")
    manifest = store.json("manifest.json")
    require(manifest["phase"] == "FROZEN_PRIMARY" and manifest["test_accessed"] is True, "learned: not a completed primary TEST run")
    require(manifest["checkpoint_training_status"] == "TRAINING_COMPLETE" and manifest["checkpoint_updates"] == 20000,
            "learned: wrong checkpoint completion state")
    protocol = store.json("protocol.json", manifest["protocol_sha256"])
    store.json("evaluation_runtime.json", manifest["runtime_sha256"])
    require(protocol["arms"] == LEARNED_ARMS and protocol["splits"]["TEST"] == {"seed":3001,"sequences":512,"length":2048},
            "learned: frozen arms or TEST sample changed")
    require(manifest["upstream_commit"] == protocol["upstream_commit"] and re.fullmatch(r"[0-9a-f]{40}",manifest["upstream_commit"]),
            "learned: upstream commit identity mismatch")
    required_sources = {"codec/learned.py","codec/online.py","codec/packed.py","codec/groups.py","codec/survival.py","scripts/evaluate_learned.py"}
    require(required_sources.issubset(manifest["source_sha256"]), "learned: required source identity missing")
    require(bool(manifest["required_upstream_source_sha256"]), "learned: upstream source identity missing")
    for name,digest in manifest["required_upstream_source_sha256"].items():
        hash_id(digest,f"upstream source {relative(name)}")
    store.source_files(manifest["source_sha256"])
    hash_id(manifest["checkpoint_sha256"], "learned checkpoint")
    hash_id(manifest["calibration_sha256"], "learned calibration")
    store.read("calibration.npz",manifest["calibration_sha256"])
    training = store.json("training-result.json")
    require(training["checkpoint_sha256"] == manifest["checkpoint_sha256"] and training["seed"] == seed and
            training["completed_updates"] == 20000 and training["status"] == "TRAINING_COMPLETE", "learned: training/checkpoint identity mismatch")
    checkpoint_verified = False
    checkpoint = None if checkpoints is None else checkpoints.get(seed)
    if checkpoint is None and (store.root/"checkpoint.pt").is_file():
        store.read("checkpoint.pt",manifest["checkpoint_sha256"])
        checkpoint_verified = True
    elif checkpoint is not None:
        require(Path(checkpoint).is_file(), f"checkpoint_seed{seed}: file missing")
        require(sha(Path(checkpoint).read_bytes()) == manifest["checkpoint_sha256"], f"checkpoint_seed{seed}: SHA256 mismatch")
        checkpoint_verified = True
    index = store.json("index.json")
    require(store.read("index.sha256").decode().strip() == sha(store.read("index.json")), "learned: index checksum mismatch")
    require(index["manifest"] == manifest, "learned: index manifest differs")
    require(set(index["splits"]["TEST"]) == set(LEARNED_ARMS), "learned: index arm set mismatch")
    require({path.stem for path in (store.root/"TEST").glob("*.json")} == set(LEARNED_ARMS), "learned: expected exactly 26 TEST arm records")
    family_size = 26*len(HORIZONS)*3
    require(index["family_size"] == family_size, "learned: wrong cross-seed confidence family")
    table_config = store.read("token_coefficients.json",manifest["token_table_config_sha256"])
    table_raw = store.read("token_coefficients.bin",manifest["token_table_data_sha256"])
    table_size = check_token_table(table_config,table_raw)
    if manifest.get("schema") == "case010-learned-evaluation-v2-canonical-token-torch":
        require(json.loads(table_config)["format"] == "ckda-token-coefficients-v2-b1-t1", "learned: canonical route/table identity mismatch")
    require(manifest["shared_token_table_bytes"] == table_size, "learned: shared token table ledger mismatch")
    inputs = store.json("TEST_inputs.json")
    require(inputs["sequences"] == 512 and inputs["group_length"] == 2048 and inputs["split"] == "TEST", "learned: TEST input declaration mismatch")
    for key in ("token_sha256","gold_sha256"):
        hash_id(inputs[key],f"learned: {key}")
    arm_results = {}
    for name in LEARNED_ARMS:
        indexed = index["splits"]["TEST"][name]
        require(indexed["file"] == f"TEST/{name}.json", "learned: index relative path mismatch")
        summary = store.json(f"TEST/{name}.json",indexed["sha256"])
        require(summary["model_seed"] == seed and summary["sequence_ids"] == inputs["sequence_ids"], "learned: model seed/input IDs mismatch")
        artifact = summary["correctness_artifact"]
        require(artifact["file"] == f"{name}.correctness.npz" and artifact["shape"] == [512,2048] and
                artifact["bitorder"] == "little" and artifact["bos_included"] is False, "learned: correctness metadata mismatch")
        correct = correctness(store,f"TEST/{artifact['file']}",expected_sha=artifact["sha256"])
        arm_results[name] = check_summary(summary,correct,family_size=family_size)
        config_raw = store.read(f"codecs/{name}.json",summary["config_sha256"])
        config = json.loads(config_raw)
        check_learned_arm(name,config)
        basis_raw = b"" if config["basis_shape"] is None else store.read(f"codecs/{name}.basis.bin",summary["basis_sha256"])
        require(sha(basis_raw) == summary["basis_sha256"], "learned: basis hash mismatch")
        check_online_ledger(summary["ledger"],config_raw,basis_raw,table_bytes=table_size)
        hash_id(summary["final_payload_sha256"], "learned final payload")
        close(indexed["rmst"],summary["restricted_mean_failure_free_length"],"learned index RMST")
        close(indexed["final_survival"],summary["horizons"][-1]["survival_probability"],"learned index survival")
        require(indexed["execution"] == summary["execution"]["status"], "learned: execution status mismatch")
    return {"model_seed":seed,"verified_test_arms":26,"sequences_per_arm":512,"protocol_sha256":manifest["protocol_sha256"],
            "checkpoint_sha256":manifest["checkpoint_sha256"],"checkpoint_file_bytes_verified":checkpoint_verified,
            "checkpoint_note":"actual bytes verified" if checkpoint_verified else "checkpoint bytes not included; identity cross-checked against retained training result",
            "final_payload_note":"final payload hash IDs retained; payload array bytes are formula-checked against reported nbytes, not independently read from an absent final-state file",
            "arms":arm_results,"artifacts":list(store.receipts.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy-original",type=Path,required=True)
    parser.add_argument("--toy-repair",type=Path)
    parser.add_argument("--learned",type=Path,action="append",default=[])
    parser.add_argument("--checkpoint",action="append",default=[],metavar="SEED=PATH")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "audit output already exists; raw/previous receipts are never overwritten")
    checkpoints = {}
    for item in args.checkpoint:
        seed, separator, path = item.partition("=")
        require(separator and seed in ("0","1","2") and path, "checkpoint must be SEED=PATH")
        require(int(seed) not in checkpoints, "duplicate checkpoint seed")
        checkpoints[int(seed)] = Path(path)
    receipt = {"schema":"case010-independent-retained-record-audit-v1","status":"PASS",
               "auditor_file":"scripts/audit_records.py","auditor_sha256":sha(Path(__file__).read_bytes()),
               "numpy_version":np.__version__,
               "scope":"raw correctness bits, independently reconstructed survival/counts, direct-binomial CP endpoint checks, actual shared buffers, payload formulas, and retained identity hashes; no model rerun",
               "toy":audit_toys(args.toy_original,args.toy_repair),
               "learned":[audit_learned(path,checkpoints) for path in args.learned]}
    seeds = [item["model_seed"] for item in receipt["learned"]]
    require(len(set(seeds)) == len(seeds), "duplicate learned seed directories")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as output:
        json.dump(receipt,output,indent=2,allow_nan=False)
        output.write("\n")
    print(json.dumps({"status":"PASS","toy_current_verified":receipt["toy"].get("current_verified_records",123),
                      "learned_seeds":seeds,"receipt_sha256":sha(args.output.read_bytes())}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"status":"FAIL","error":str(error)}),file=sys.stderr)
        raise SystemExit(2)
