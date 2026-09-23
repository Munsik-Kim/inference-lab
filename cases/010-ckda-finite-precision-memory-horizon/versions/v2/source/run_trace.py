"""Five frozen codec arms with scalar-only failure-aware diagnostic trajectories."""
from __future__ import annotations

import time

import numpy as np
import torch

from . import common
from .arms import load_arms
from .online_v2 import PackedCodec
from .reference import learned, v1
from .run_precision import (ScalarHistory, deadline_expired, masked, parser,
                            save_cell, setup, stable_l2, stable_mse)


def _candidate_write(adapter, state, coeff):
    """Only candidate reconstruction and current coefficients reach transition."""
    updated = None

    def transition(x):
        nonlocal updated
        updated = v1.torch_transition(x, coeff)
        return updated

    next_state, represented, _ = adapter.step(state, transition, diagnostics=False)
    return next_state, represented, updated


def _scale_values(codec, packed, n):
    if not isinstance(codec, PackedCodec):
        return {name: np.full(n, np.nan) for name in ("minimum", "maximum", "mean")}
    scales, _ = codec._parts(packed)
    return {"minimum": scales.min(1), "maximum": scales.max(1),
            "mean": scales.astype(np.float64).mean(1)}


def trace_values(adapter, state, represented, updated, *, native, shadow,
                 native_active, shadow_active, logits, target):
    """Temporary state views are reduced immediately to one scalar per stream."""
    n = len(represented)
    _, code, _, z, r = adapter._parts(state)
    active = code == 0
    decoded = adapter.state_codec.decode(z)
    residual = np.zeros_like(decoded)
    if r is not None:
        residual = adapter.expand(adapter.residual_codec.decode(r))
    state64 = represented.astype(np.float64)
    native64, shadow64 = np.asarray(native, dtype=np.float64), np.asarray(shadow, dtype=np.float64)
    native_valid = active & native_active
    shadow_valid = active & shadow_active
    logits = np.asarray(logits, dtype=np.float64)
    read_valid = active & np.isfinite(logits).all(1)
    alternatives = logits.copy()
    alternatives[np.arange(n), target] = -np.inf
    ordered = np.sort(logits, axis=1)
    values = {
        "active": active, "logits_finite": read_valid,
        "native_shadow_active": np.asarray(native_active, dtype=bool),
        "fp64_shadow_active": np.asarray(shadow_active, dtype=bool),
        "state_norm": masked(stable_l2(state64), active),
        "base_state_norm": masked(stable_l2(decoded), active),
        "residual_norm": masked(stable_l2(residual), active),
        "native_state_norm": masked(stable_l2(native64), native_active),
        "fp64_shadow_state_norm": masked(stable_l2(shadow64), shadow_active),
        "native_state_error_mse": masked(stable_mse(state64-native64), native_valid),
        "native_state_error_l2": masked(stable_l2(state64-native64), native_valid),
        "fp64_shadow_error_mse": masked(stable_mse(state64-shadow64), shadow_valid),
        "fp64_shadow_error_l2": masked(stable_l2(state64-shadow64), shadow_valid),
        "native_vs_fp64_shadow_mse": masked(stable_mse(native64-shadow64), native_active & shadow_active),
        "gold_margin": masked(logits[np.arange(n), target]-alternatives.max(1), read_valid),
        "top_two_margin": masked(ordered[:, -1]-ordered[:, -2], read_valid),
    }
    if updated is None:
        values["self_writeback_mse"] = np.full(n, np.nan)
        values["projection_leakage_mse"] = np.full(n, np.nan)
    else:
        values["self_writeback_mse"] = masked(stable_mse(updated.astype(np.float64)-state64), active)
        leakage = np.zeros_like(state64)
        if adapter.basis is not None:
            # Reproduce the committed candidate's FP32 residual/projection first;
            # promote before diagnostic subtraction and squaring.
            error32 = updated-decoded
            projected32 = adapter.project(error32)
            leakage = error32.astype(np.float64)-adapter.expand(projected32.astype(np.float64))
        values["projection_leakage_mse"] = masked(stable_mse(leakage), active)
    for name, value in _scale_values(adapter.state_codec, z, n).items():
        values["state_scale_"+name] = masked(value, active)
    residual_scales = (_scale_values(adapter.residual_codec, r, n) if r is not None else
                       {name: np.full(n, np.nan) for name in ("minimum", "maximum", "mean")})
    for name, value in residual_scales.items():
        values["residual_scale_"+name] = masked(value, active)
    return values


@torch.inference_mode()
def execute_trace(model, table, tokens, gold, adapters, *, deadline=None):
    """Evaluator shadows never enter `_candidate_write` or candidate state."""
    n, horizon = tokens.shape
    if gold.shape != tokens.shape or horizon < 1:
        raise ValueError("Expected nonempty matching token/gold arrays")
    if not adapters or any(adapter.mode != "transported" for adapter in adapters.values()):
        raise ValueError("This frozen trace supports transported diagnostic arms only")
    shape = next(iter(adapters.values())).shape
    states = {name: adapter.initial(np.zeros((n,)+shape, dtype=np.float32))
              for name, adapter in adapters.items()}
    predictions = {name: np.full((n, horizon+1), -1, dtype=np.int8) for name in adapters}
    histories = {name: ScalarHistory(n) for name in adapters}
    invalid_readout = {name: np.zeros(n, dtype=np.int64) for name in adapters}
    active_updates = {name: 0 for name in adapters}
    terminal_noops = {name: 0 for name in adapters}
    native = np.zeros((n,)+shape, dtype=np.float32)
    shadow = torch.zeros((n,)+shape, dtype=torch.float64)
    native_active, shadow_active = np.ones(n, bool), np.ones(n, bool)
    native_first, shadow_first = [None]*n, [None]*n
    completed = 0
    start = time.perf_counter()
    for position in range(horizon+1):
        if deadline_expired(deadline):
            break
        ids = np.full(n, 6, dtype=np.int64) if position == 0 else tokens[:, position-1]
        target = np.zeros(n, dtype=np.int64) if position == 0 else gold[:, position-1].astype(np.int64)
        coeff = v1.gather(table, ids)
        # These independent, unquantized shadow recurrences receive coefficients
        # and their own prior state, never the candidate reconstruction.
        native = v1.torch_transition(native, coeff)
        coeff64 = {name: torch.from_numpy(value).double() for name, value in coeff.items()}
        shadow = learned.transition(shadow, coeff64)
        new_native = native_active & ~np.isfinite(native).all(axis=(1, 2, 3))
        new_shadow = shadow_active & ~torch.isfinite(shadow).flatten(1).all(1).numpy()
        for row in np.flatnonzero(new_native):
            native_first[row] = position
        for row in np.flatnonzero(new_shadow):
            shadow_first[row] = position
        native_active &= ~new_native
        shadow_active &= ~new_shadow
        for name, adapter in adapters.items():
            before = adapter.terminal_info(states[name])["active"]
            active_updates[name] += int(before.sum())
            terminal_noops[name] += int((~before).sum())
            states[name], represented, updated = _candidate_write(adapter, states[name], coeff)
            active = adapter.terminal_info(states[name])["active"]
            output = v1.logits_from_numpy(model, represented, coeff)
            if tuple(output.shape) != (n, 6):
                raise ValueError("Original readout must return the entire six-class vector")
            logits = output.numpy()
            finite = np.isfinite(logits).all(1)
            invalid_readout[name] += active & ~finite
            pred = logits.argmax(1).astype(np.int8)
            pred[~active | ~finite] = -1
            predictions[name][:, position] = pred
            histories[name].append(trace_values(adapter, states[name], represented, updated,
                native=native, shadow=shadow.numpy(), native_active=native_active,
                shadow_active=shadow_active, logits=logits, target=target))
        native[~native_active] = 0
        shadow[torch.from_numpy(~shadow_active)] = 0
        completed += 1
    elapsed = time.perf_counter()-start
    records = {}
    for name, adapter in adapters.items():
        info = adapter.terminal_info(states[name])
        records[name] = dict(predictions=predictions[name][:, :completed],
            scalars=histories[name].arrays(), final_state=states[name],
            first_terminal=[None if active else int(t) for active, t in zip(info["active"], info["first_terminal_write"])],
            terminal_codes=info["terminal_code"].tolist(),
            execution=dict(completed_writes=completed, expected_writes=horizon+1,
                status="COMPLETE" if completed == horizon+1 else "NOT_COMPLETE",
                elapsed_seconds=elapsed, active_update_attempts=active_updates[name],
                elapsed_scope="whole five-arm diagnostic loop including shadows and scalar summaries; not arm latency",
                terminal_noop_steps=terminal_noops[name], invalid_readout_counts=invalid_readout[name].tolist(),
                finite_state_readout_failure_is_terminal=False,
                execution_reference="source.evaluation.sequence_call; diagnostic side effects checked against prediction/payload parity"))
    shadows = dict(native_first_terminal=native_first, fp64_first_terminal=shadow_first,
                   native_terminal_count=int((~native_active).sum()), fp64_terminal_count=int((~shadow_active).sum()),
                   native_dtype="float32", fp64_dtype="float64", coefficient_source="unchanged retained FP32 table promoted for FP64",
                   passed_to_candidate=False, full_tensor_history_retained=False)
    return records, shadows


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    model, table, tokens, gold, identity, deadline, _ = setup(args)
    adapters, selection = load_arms(args.seed)
    if set(adapters) != set(common.ARMS):
        raise ValueError("Expected the exact five frozen trace arms")
    records, shadows = execute_trace(model, table, tokens, gold, adapters, deadline=deadline)
    cells = {}
    for name, record in records.items():
        ledger = common.ledger_with_table(adapters[name], record.pop("final_state"), args.seed)
        save_cell(args.output/name, record, gold, dict(identity, arm=name), ledger=ledger,
                  extra={"shadows": shadows, "diagnostic_scope": "own write-back/projection errors are distinct from independent native/FP64-shadow state errors; no causal attribution from averages"})
        cells[name] = common.file_sha(args.output/name/"summary.json")
    complete = all(r["execution"]["status"] == "COMPLETE" for r in records.values())
    common.write_json(args.output/"index.json", dict(schema="case010-v2-scalar-trace-v1",
        status="COMPLETE" if complete else "NOT_COMPLETE", identity=identity, cpu_threads=2,
        cells=cells, shadows=shadows, codec_selection_sha256=common.file_sha(
            common.ROOT/"inputs/codecs"/f"seed{args.seed}"/"selection.json"),
        runner_sha256=common.file_sha(__file__)))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
