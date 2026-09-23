"""Frozen, CPU-only Phase A toy protocol and causal finite-state evaluation.

All primary adapters see only the current packed state and the causal affine
operator.  Integer group gold and prototype labels are evaluator-only objects.
The separately named symbolic tracker is a storage-capacity control.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from .groups import frozen_sequences, make_group
from .packed import NativeFloatCodec, PackedCodec
from .survival import DEFAULT_HORIZONS, first_failure_times, summarize_first_failures


ARM_NAMES = ("native_fp32", "native_fp64") + tuple(f"uniform_{b}" for b in range(2, 17)) + (
    "stochastic_4", "fullres_4_4", "fullres_4_fp32", "lowrank_4_8",
    "oldframe_4_4", "mixed_4_8", "mixed_6_8", "coefficient_int8_fp32",
)
CONDITION_NAMES = ("s4_signed_lattice", "c31_rotation_2d", "s4_rotated",
                   "c31_moving_plane", "c31_switching_plane")
SPLITS = {"CAL": {"n": 64, "horizon": 32, "seed": 1001},
          "DEV": {"n": 128, "horizon": 128, "seed": 2001},
          "TEST": {"n": 512, "horizon": 2048, "seed": 3001}}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _rotation(theta):
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)


@dataclass(frozen=True)
class ToyCondition:
    name: str
    group_name: str
    base_operators: np.ndarray
    base_prototypes: np.ndarray
    schedule: str = "fixed"

    @property
    def dimension(self):
        return self.base_operators.shape[-1]

    @property
    def shape(self):
        return (1, self.dimension, 1)

    def frame(self, step):
        if step < 0 or not isinstance(step, (int, np.integer)):
            raise ValueError("frame step must be a nonnegative integer")
        if self.schedule == "fixed":
            return np.eye(self.dimension)
        if self.schedule == "moving":
            angle = step * math.pi / 97
        elif self.schedule == "switching":
            angle = (step // 32 % 2) * math.pi / 2
        else:
            raise ValueError("unknown coordinate schedule")
        frame = np.eye(4)
        for i, j in ((0, 2), (1, 3)):
            frame[np.ix_([i, j], [i, j])] = _rotation(angle)
        return frame

    def operators(self, tokens, step):
        if step < 1:
            raise ValueError("predicted steps start at 1")
        operators = self.base_operators[np.asarray(tokens)]
        if self.schedule != "fixed":
            operators = self.frame(step) @ operators @ self.frame(step - 1).T
        return operators

    def prototypes(self, step):
        return self.base_prototypes @ self.frame(step).T

    def initial(self, batch):
        return np.broadcast_to(self.base_prototypes[0], (batch, self.dimension)).copy()[:, None, :, None]

    def specification(self):
        return {"name": self.name, "group": self.group_name, "shape": list(self.shape),
                "operators": self.base_operators.tolist(), "prototypes": self.base_prototypes.tolist(),
                "schedule": self.schedule,
                "schedule_definition": {"fixed": "identity", "moving": "G02(t*pi/97) G13(t*pi/97)",
                                        "switching": "G02(a) G13(a), a=(floor(t/32) mod 2)*pi/2"}[self.schedule],
                "gold": "y_t = x_t * y_(t-1), exact integer group composition",
                "readout": "nearest squared-Euclidean scheduled prototype; smallest-ID argmin tie"}

    def shared_bytes(self):
        """The actual serialized shared toy program, transition table and decoder."""
        metadata = self.specification()
        del metadata["operators"], metadata["prototypes"]
        metadata.update(operator_shape=list(self.base_operators.shape),
                        prototype_shape=list(self.base_prototypes.shape), table_dtype="<f8")
        return (canonical_json(metadata) + b"\n" + self.base_operators.astype("<f8").tobytes()
                + self.base_prototypes.astype("<f8").tobytes())


def make_conditions():
    s4 = make_group("S4")
    permutation = s4.matrices.astype(np.float64)
    lattice = np.asarray([-7., -3., 1., 7.])
    sign = np.diag([-1., 1., -1., 1.])
    signed = sign @ permutation @ sign
    signed_prototypes = (sign @ permutation @ lattice[..., None])[..., 0]
    rng = np.random.Generator(np.random.PCG64(41017))
    q, r = np.linalg.qr(rng.normal(size=(4, 4)))
    q *= np.where(np.diag(r) < 0, -1., 1.)
    rotated = q @ permutation @ q.T
    normalized = lattice / np.linalg.norm(lattice)
    rotated_prototypes = (q @ permutation @ normalized[..., None])[..., 0]
    rotations = np.stack([_rotation(2 * math.pi * element / 31) for element in range(31)])
    rotation_prototypes = rotations[:, :, 0]
    embedded = np.broadcast_to(np.eye(4), (31, 4, 4)).copy()
    embedded[:, :2, :2] = rotations
    embedded_prototypes = embedded[:, :, 0]
    return (
        ToyCondition("s4_signed_lattice", "S4", signed, signed_prototypes),
        ToyCondition("c31_rotation_2d", "C31", rotations, rotation_prototypes),
        ToyCondition("s4_rotated", "S4", rotated, rotated_prototypes),
        ToyCondition("c31_moving_plane", "C31", embedded, embedded_prototypes, "moving"),
        ToyCondition("c31_switching_plane", "C31", embedded, embedded_prototypes, "switching"),
    )


def readout(state, prototypes):
    values = np.asarray(state)
    if values.ndim != 4 or values.shape[1] != 1 or values.shape[3] != 1:
        raise ValueError("toy readout requires [batch,1,key,1]")
    values = values[:, 0, :, 0]
    finite = np.all(np.isfinite(values), axis=1)
    safe = np.where(finite[:, None], values, 0)
    distance = np.sum((safe[:, None, :] - prototypes[None, :, :]) ** 2, axis=-1)
    labels = np.argmin(distance, axis=1).astype(np.int64)
    labels[~finite] = -1
    return labels


def affine_transition(operators):
    """Closure contains only A; no labels, prototypes, or reference trajectory."""
    operators = np.asarray(operators)
    def transition(state):
        dtype = state.dtype
        result = np.einsum("bij,bj->bi", operators.astype(dtype, copy=False), state[:, 0, :, 0], optimize=False)
        return result[:, None, :, None]
    return transition


def calibrate(condition):
    """Freeze a CAL-only basis and generic error/persistence mixed-bit ranking."""
    config = SPLITS["CAL"]
    batch = frozen_sequences(condition.group_name, "CAL", config["n"], config["horizon"], seed=config["seed"])
    codec = PackedCodec(condition.shape, bits=4)
    state = codec.encode(condition.initial(config["n"]))
    errors, weighted = [], []
    for step in range(1, config["horizon"] + 1):
        operators = condition.operators(batch.tokens[:, step - 1], step)
        working = affine_transition(operators)(codec.decode(state))
        state = codec.encode(working)
        error = (working - codec.decode(state))[:, 0, :, 0].astype(np.float64)
        errors.append(error)
        if step < config["horizon"]:
            following = condition.operators(batch.tokens[:, step], step + 1)
            column_norm_squared = np.sum(following ** 2, axis=1)
            weighted.append(error ** 2 * column_norm_squared)
    samples = np.concatenate(errors, axis=0)
    centered = samples - samples.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    rank = 1 if condition.dimension == 2 else 2
    basis = eigenvectors[:, np.argsort(-eigenvalues, kind="stable")[:rank]].copy()
    for column in range(rank):
        pivot = np.argmax(np.abs(basis[:, column]))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    score = np.concatenate(weighted, axis=0).mean(axis=0)
    ranking = np.argsort(-score, kind="stable")
    mixed4 = np.full(condition.dimension, 4, dtype=np.uint8)
    mixed6 = np.full(condition.dimension, 6, dtype=np.uint8)
    mixed4[ranking[:max(1, math.ceil(condition.dimension / 4))]] = 8
    mixed6[ranking[:max(1, math.ceil(condition.dimension / 2))]] = 8
    return {"basis": basis.tolist(), "rank": rank, "covariance": covariance.tolist(),
            "basis_origin": "fixed CAL covariance of local base4 causal-rounding errors; not learned",
            "calibration": dict(config), "error_persistence_scores": score.tolist(),
            "ranking": ranking.tolist(), "mixed_4_8": mixed4.tolist(), "mixed_6_8": mixed6.tolist(),
            "ranking_label": "generic CAL_error_persistence; not DAMP",
            "score_definition": "mean(local_error^2 * next_transition_column_norm^2), final CAL step excluded",
            "persistence_note": "orthogonal toy operators have unit column norms; no fitted persistence predictor"}


def make_adapter(condition, arm, calibration):
    from .online import OnlineAdapter
    shape = condition.shape
    if arm in ("native_fp32", "coefficient_int8_fp32"):
        return OnlineAdapter(NativeFloatCodec(shape, "float32"))
    if arm == "native_fp64":
        return OnlineAdapter(NativeFloatCodec(shape, "float64"))
    if arm.startswith("uniform_"):
        return OnlineAdapter(PackedCodec(shape, bits=int(arm.rsplit("_", 1)[1])))
    if arm == "stochastic_4":
        return OnlineAdapter(PackedCodec(shape, bits=4, stochastic=True))
    if arm in ("mixed_4_8", "mixed_6_8"):
        return OnlineAdapter(PackedCodec(shape, bits=int(arm.split("_")[1]), mixed_bits=calibration[arm]))
    base = PackedCodec(shape, bits=4)
    if arm == "fullres_4_4":
        return OnlineAdapter(base, PackedCodec(shape, bits=4))
    if arm == "fullres_4_fp32":
        return OnlineAdapter(base, NativeFloatCodec(shape, "float32"))
    if arm == "oldframe_4_4":
        return OnlineAdapter(base, PackedCodec(shape, bits=4), mode="untransported")
    if arm == "lowrank_4_8":
        basis = np.asarray(calibration["basis"], dtype=np.float32)[None, :, :]
        return OnlineAdapter(base, PackedCodec((1, calibration["rank"], 1), bits=8), basis=basis)
    raise ValueError(f"unknown toy arm {arm}")


@dataclass(frozen=True)
class Int8Operators:
    codes: np.ndarray
    scales: np.ndarray
    dynamic: bool

    def at(self, tokens, step):
        codes = self.codes[step - 1] if self.dynamic else self.codes
        scales = self.scales[step - 1] if self.dynamic else self.scales
        return codes[tokens].astype(np.float32) * scales[tokens, None, None]

    def shared_bytes(self):
        header = canonical_json({"format": "toy-coefficient-int8-v1", "shape": list(self.codes.shape),
                                 "dynamic": self.dynamic, "scale_dtype": "<f4", "code_dtype": "int8"})
        return header + b"\n" + self.codes.tobytes() + self.scales.astype("<f4").tobytes()


def quantized_operators(condition, horizon):
    labels = np.arange(len(condition.base_operators))
    matrices = condition.base_operators
    dynamic = condition.schedule != "fixed"
    if dynamic:
        matrices = np.stack([condition.operators(labels, step) for step in range(1, horizon + 1)])
    scales = (np.max(np.abs(matrices), axis=(-2, -1)) / 127).astype(np.float32)
    codes = np.rint(matrices / scales[..., None, None]).clip(-127, 127).astype(np.int8)
    return Int8Operators(codes, scales, dynamic)


def protocol(smoke=False, include_dev=False):
    splits = {name: dict(values) for name, values in SPLITS.items()}
    if smoke:
        splits["TEST"].update(n=8, horizon=64)
        splits["DEV"].update(n=8, horizon=64)
    horizons = [t for t in DEFAULT_HORIZONS if t <= splits["TEST"]["horizon"]]
    return {"schema": "case010-phase-a-toys-v1", "primary": not smoke, "smoke": smoke,
            "conditions": [condition.specification() for condition in make_conditions()],
            "arms": list(ARM_NAMES), "splits": splits, "evaluate_dev": include_dev,
            "dev_policy": "optional ranking-only, no protocol changes after freeze",
            "horizons": horizons, "family_size": len(CONDITION_NAMES) * len(ARM_NAMES) * len(horizons),
            "family_scope": "all five conditions x all 25 arms x all evaluated TEST horizons",
            "epsilons": [0.05, 0.01], "alpha": 0.05,
            "stochastic_replicates_per_sequence": 1, "stochastic_seed_offset": 740000,
            "shared_amortization_N": [1, 16, 128], "capacity_control": "exact_symbolic_id",
            "coefficient_control": "secondary FP32 state, separately INT8-quantized entire transition table",
            "selection": "no TEST-based arm selection; report every frozen arm including failures",
            "codecs": "actual byte-backed payload including dynamic scales, RNG, residuals, and uint64 cursor",
            "correctness_storage": "per-arm compressed packbits [sequence,time], little bitorder",
            "readout_tie_rule": "smallest integer label on exact squared-distance tie",
            "update_gold_access": False,
            "runtime_scope": "CPU reference only; no GPU kernel or deployment throughput claim"}


def _shared_adapter_bytes(adapter):
    shared = adapter.config_bytes + adapter.basis_bytes
    if len(shared) != adapter.shared_bytes:
        raise ValueError("adapter shared byte count does not match serialized buffers")
    return shared


def _step_finite_rows(adapter, state, operators, active):
    """Retry healthy rows unchanged; numerical failures are evaluator terminals.

    A rejected batch has not committed any new packed state or RNG counter.
    Only the explicitly identified failing rows leave the active set.  Their
    last valid payload stays allocated for conservative unchanged byte counts;
    it is never used as a substitute continuation or scored correct output.
    """
    from .online import NonfiniteStateError
    pending = np.flatnonzero(active)
    output = np.full((len(active),) + tuple(adapter.shape), np.nan, dtype=np.float64)
    failed, diagnostics = [], {}
    while len(pending):
        subset = state if len(pending) == len(active) else type(state)(state.payload[pending].copy(order="C"))
        try:
            new_state, represented, diagnostics = adapter.step(subset, affine_transition(operators[pending]))
        except NonfiniteStateError as error:
            bad = np.asarray(error.bad_rows)
            if bad.dtype.kind != "b" or bad.shape != (len(pending),) or not bad.any():
                raise ValueError("invalid row-local numerical failure metadata") from error
            for row in pending[bad]:
                failed.append((int(row), str(error.stage)))
            active[pending[bad]] = False
            pending = pending[~bad]
            continue
        output[pending] = represented
        if len(pending) == len(active):
            state = new_state
        else:
            payload = state.payload.copy(order="C")
            payload[pending] = new_state.payload
            state = type(state)(payload)
        break
    return state, output, diagnostics, failed


def evaluate_arm(condition, arm, calibration, batch, family_size, horizons, alpha=0.05):
    adapter = make_adapter(condition, arm, calibration)
    n, horizon = batch.tokens.shape
    seeds = np.arange(n, dtype=np.uint64) + np.uint64(batch.seed + 740000) if arm == "stochastic_4" else None
    state = adapter.initial(condition.initial(n), seeds=seeds)
    coefficient = quantized_operators(condition, horizon) if arm == "coefficient_int8_fp32" else None
    correct = np.empty((n, horizon), dtype=bool)
    active = np.ones(n, dtype=bool)
    numerical_failure_times, numerical_failure_stages = [None] * n, [None] * n
    squared_error = norm_error = phase_error = 0.0
    projection_leakage_sum = 0.0
    projection_leakage_count = diagnostic_observations = 0
    started = time.monotonic()
    for step in range(1, horizon + 1):
        tokens = batch.tokens[:, step - 1]
        operators = coefficient.at(tokens, step) if coefficient is not None else condition.operators(tokens, step)
        state, output, diagnostics, failed_rows = _step_finite_rows(adapter, state, operators, active)
        for row, stage in failed_rows:
            numerical_failure_times[row], numerical_failure_stages[row] = step, stage
        prototypes = condition.prototypes(step)
        predicted = readout(output, prototypes)
        correct[:, step - 1] = predicted == batch.gold[:, step - 1]
        values = output[active, 0, :, 0].astype(np.float64)
        reference = prototypes[batch.gold[active, step - 1]]
        diagnostic_observations += len(values)
        squared_error += float(np.sum((values - reference) ** 2))
        norm_error += float(np.sum((np.linalg.norm(values, axis=1) - np.linalg.norm(reference, axis=1)) ** 2))
        if condition.group_name == "C31":
            canonical = values @ condition.frame(step)
            angle = np.arctan2(canonical[:, 1], canonical[:, 0])
            reference_angle = batch.gold[active, step - 1] * (2 * math.pi / 31)
            difference = (angle - reference_angle + math.pi) % (2 * math.pi) - math.pi
            phase_error += float(np.sum(difference ** 2))
        if "projection_leakage_mse" in diagnostics:
            projection_leakage_sum += float(diagnostics["projection_leakage_mse"]) * len(values)
            projection_leakage_count += len(values)
    runtime = time.monotonic() - started
    taus = first_failure_times(correct)
    summary = summarize_first_failures(taus, horizon, horizons=horizons, sequence_ids=batch.sequence_ids,
                                      family_size=family_size, alpha=alpha, correct=correct)
    # The shared toy program is actually serialized JSON.  The secondary
    # coefficient table replaces its floating operators in the stored bundle.
    shared_model = condition.shared_bytes()
    if coefficient is not None:
        decoder_spec = condition.specification()
        del decoder_spec["operators"], decoder_spec["prototypes"]
        decoder_spec.update(prototype_shape=list(condition.base_prototypes.shape), prototype_dtype="<f8")
        shared_model = (canonical_json(decoder_spec) + b"\n" + condition.base_prototypes.astype("<f8").tobytes()
                        + coefficient.shared_bytes())
    shared_adapter = _shared_adapter_bytes(adapter)
    stream_bytes = state.payload.shape[1]
    shared_bytes = len(shared_adapter) + len(shared_model)
    ledger = {"stream_payload_bytes": stream_bytes, "batch_payload_nbytes": state.payload.nbytes,
              "adapter_shared_bytes": len(shared_adapter), "toy_program_shared_bytes": len(shared_model),
              "total_shared_bytes": shared_bytes,
              "amortized_bytes_per_stream": {str(count): stream_bytes + shared_bytes / count for count in (1, 16, 128)},
              "definition": "actual serialized stream plus shared adapter and toy program; no equalizing padding",
              "adapter": adapter.ledger(state)}
    diagnostics = {"state_mse_per_coordinate": squared_error / (diagnostic_observations * condition.dimension) if diagnostic_observations else None,
                   "norm_mse": norm_error / diagnostic_observations if diagnostic_observations else None,
                   "phase_mse_radians2": phase_error / diagnostic_observations if condition.group_name == "C31" and diagnostic_observations else None,
                   "mean_local_projection_leakage": projection_leakage_sum / projection_leakage_count if projection_leakage_count else None,
                   "finite_observation_count": diagnostic_observations,
                   "diagnostic_scope": "descriptive FP64 output-side diagnostics on finite observations only; no fitted failure prediction"}
    return {"condition": condition.name, "arm": arm, "split": batch.split, "status": "ok",
            "runtime_seconds": runtime,
            "runtime_scope": "CPU diagnostic-loop timing includes gold comparison and descriptive diagnostics; production complete-call not measured",
            "summary": summary, "ledger": ledger, "diagnostics": diagnostics,
            "numerical_status": {"terminal_failure_times": numerical_failure_times,
                                  "terminal_failure_stages": numerical_failure_stages,
                                  "n_terminal_sequences": sum(time is not None for time in numerical_failure_times),
                                  "policy": "row-local numerical nonrepresentability is absorbing incorrect from that step; earlier first failures remain earlier",
                                  "row_masks_scope": "evaluator termination bookkeeping, never model input or substitute cache"}}, correct, shared_adapter + shared_model


def symbolic_capacity_control(condition, batch):
    """Explicit symbolic algorithm; not an experimental adapter or infinity claim."""
    group = make_group(condition.group_name)
    table = group.multiplication.astype(np.uint8)
    n, horizon = batch.tokens.shape
    payload = np.zeros((n, 9), dtype=np.uint8)  # ID + exact uint64 cursor.
    correct = np.empty((n, horizon), dtype=bool)
    for step in range(1, horizon + 1):
        payload[:, 0] = table[batch.tokens[:, step - 1], payload[:, 0]]
        payload[:, 1:] = np.full((n, 1), step, dtype="<u8").view(np.uint8)
        correct[:, step - 1] = payload[:, 0] == batch.gold[:, step - 1]
    decoder = {"prototype_shape": list(condition.base_prototypes.shape), "prototype_dtype": "<f8",
               "schedule": condition.specification()["schedule_definition"],
               "group": group.name, "table_shape": list(table.shape), "table_dtype": "uint8",
               "stream_layout": "uint8 ID + little-endian uint64 cursor",
               "scope": "storage capacity control, exact symbolic update, finite tested horizon only"}
    config_bytes = canonical_json(decoder) + b"\n"
    prototype_bytes = condition.base_prototypes.astype("<f8").tobytes()
    shared = config_bytes + table.tobytes() + prototype_bytes
    return {"condition": condition.name, "arm": "exact_symbolic_id", "split": batch.split,
            "status": "ok", "n_sequences": n, "max_horizon": horizon,
            "tau": first_failure_times(correct), "step_accuracy": float(correct.mean()),
            "interpretation": f"no failure through {horizon}; storage capacity control, no infinite horizon claim",
            "ledger": {"stream_payload_bytes": 9, "total_shared_bytes": len(shared),
                       "multiplication_table_bytes": table.nbytes, "decoder_config_bytes": len(config_bytes),
                       "prototype_bytes": len(prototype_bytes),
                       "amortized_bytes_per_stream": {str(count): 9 + len(shared) / count for count in (1, 16, 128)}}}, shared


def run_protocol(output, smoke=False, include_dev=False, progress=None):
    """Freeze complete protocol and CAL parameters before evaluating any DEV/TEST."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frozen = protocol(smoke=smoke, include_dev=include_dev)
    (output / "protocol-declared.json").write_bytes(canonical_json(frozen) + b"\n")
    calibrations = {condition.name: calibrate(condition) for condition in make_conditions()}
    calibration_identity = {}
    for name, calibration in calibrations.items():
        basis_bytes = np.asarray(calibration["basis"], dtype="<f4")[None, :, :].tobytes()
        calibration["basis_bytes_sha256"] = hashlib.sha256(basis_bytes).hexdigest()
        calibration_identity[name] = {"basis_bytes_sha256": calibration["basis_bytes_sha256"],
                                      "calibration_sha256": hashlib.sha256(canonical_json(calibration)).hexdigest(),
                                      "split": "CAL", **SPLITS["CAL"]}
    frozen["calibrations"] = calibrations
    frozen["arm_configurations"] = {
        condition.name: {arm: json.loads(make_adapter(condition, arm, calibrations[condition.name]).config_bytes)
                         for arm in ARM_NAMES} for condition in make_conditions()}
    frozen["code_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in ("toy.py", "online.py", "packed.py", "groups.py", "survival.py")}
    protocol_bytes = canonical_json(frozen)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    (output / "protocol-frozen.json").write_bytes(protocol_bytes + b"\n")
    (output / "protocol-frozen.sha256").write_text(protocol_hash + "\n")
    (output / "calibration-identity.json").write_bytes(canonical_json(calibration_identity) + b"\n")
    (output / "shared").mkdir()
    (output / "correctness").mkdir()
    completed, failures = [], []
    evaluated_splits = ["DEV", "TEST"] if include_dev else ["TEST"]
    with (output / "results.jsonl").open("x") as results, (output / "tau.jsonl").open("x") as tau_log:
        for split in evaluated_splits:
            config = frozen["splits"][split]
            horizons = [t for t in DEFAULT_HORIZONS if t <= config["horizon"]]
            for condition in make_conditions():
                batch = frozen_sequences(condition.group_name, split, config["n"], config["horizon"], seed=config["seed"])
                for arm in ARM_NAMES:
                    key = f"{split}--{condition.name}--{arm}"
                    try:
                        result, correct, shared = evaluate_arm(condition, arm, calibrations[condition.name], batch,
                                                              frozen["family_size"], horizons)
                        result["protocol_sha256"] = protocol_hash
                        np.savez_compressed(output / "correctness" / f"{key}.npz",
                                            packed=np.packbits(correct, axis=1, bitorder="little"),
                                            shape=np.asarray(correct.shape, dtype=np.int64))
                        (output / "shared" / f"{key}.bin").write_bytes(shared)
                        tau_log.write(json.dumps({"condition": condition.name, "arm": arm, "split": split,
                                                  "sequence_ids": batch.sequence_ids, "tau": result["summary"]["tau"],
                                                  "max_horizon": config["horizon"], "protocol_sha256": protocol_hash},
                                                 separators=(",", ":")) + "\n")
                        tau_log.flush()
                        completed.append(key)
                    except Exception as error:
                        result = {"condition": condition.name, "arm": arm, "split": split, "status": "error",
                                  "error": repr(error), "protocol_sha256": protocol_hash}
                        (output / f"failure--{key}.txt").write_text(traceback.format_exc())
                        failures.append(key)
                    results.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
                    results.flush()
                    if progress:
                        progress(key, result)
                capacity, shared = symbolic_capacity_control(condition, batch)
                capacity["protocol_sha256"] = protocol_hash
                results.write(json.dumps(capacity, separators=(",", ":"), allow_nan=False) + "\n")
                results.flush()
                (output / "shared" / f"{split}--{condition.name}--exact_symbolic_id.bin").write_bytes(shared)
    manifest = {"protocol_sha256": protocol_hash, "completed_arms": len(completed), "failed_arms": failures,
                "runtime_seconds": time.monotonic() - started, "primary": not smoke,
                "output_directory_name": output.name, "status": "complete" if not failures else "completed_with_failures"}
    (output / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    return manifest
