"""Bounded fixed-coefficient precision cohort; run only after protocol freeze."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import warnings

import numpy as np
import torch

from . import common, metrics
from .precision import MODES, PrecisionDiagnostic, summarize_step


def stable_l2(values):
    """Per-row FP64 norm, scaled before squaring; invalid values stay invalid."""
    x = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    scale = np.max(np.abs(x), axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = np.divide(x, scale[:, None], out=np.zeros_like(x), where=scale[:, None] != 0)
        result = scale * np.sqrt(np.sum(normalized * normalized, axis=1))
    result[~np.isfinite(x).all(1)] = np.nan
    return result


def stable_mse(values):
    x = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    scale = np.max(np.abs(x), axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = np.divide(x, scale[:, None], out=np.zeros_like(x), where=scale[:, None] != 0)
        result = scale * (scale * np.mean(normalized * normalized, axis=1))
    result[~np.isfinite(x).all(1)] = np.nan
    return result


def masked(values, valid):
    result = np.asarray(values, dtype=np.float64).copy()
    result[~np.asarray(valid, dtype=bool)] = np.nan
    result[~np.isfinite(result)] = np.nan
    return result


class ScalarHistory:
    """Only per-sequence scalars cross time; never recurrent state or logits."""
    def __init__(self, batch):
        self.batch = batch
        self.columns = {}

    def append(self, values):
        if self.columns and set(values) != set(self.columns):
            raise ValueError("Diagnostic scalar fields changed during the cohort")
        for key, value in values.items():
            array = np.asarray(value)
            if array.shape != (self.batch,):
                raise ValueError("History accepts one scalar per input, not state/logit tensors")
            self.columns.setdefault(key, []).append(array.copy())

    def arrays(self):
        return {key: np.stack(value, axis=1) for key, value in self.columns.items()}


def aggregate_scalars(arrays, predictions, gold):
    """Per-time statistics with varying, explicitly counted validity populations."""
    n, writes = predictions.shape
    correct = predictions[:, 1:] == gold[:, :max(0, writes-1)]
    failed = np.zeros((n, writes), dtype=bool)
    if writes > 1:
        failed[:, 1:] = np.maximum.accumulate(~correct, axis=1)
    before = ~failed
    before[:, :1] = False  # BOS is a separate, unscored point.
    masks = {"all": np.ones_like(failed), "before_first_failure": before,
             "at_or_after_first_failure": failed}
    out = {"before_first_failure": before, "at_or_after_first_failure": failed}
    for key, value in arrays.items():
        if value.dtype.kind == "b":
            out[key + "__count"] = value.sum(0).astype(np.int32)
            continue
        if value.dtype.kind not in "fiu":
            continue
        for label, population in masks.items():
            valid = population & np.isfinite(value)
            count = valid.sum(0)
            selected = np.where(valid, value, np.nan).astype(np.float64)
            with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                warnings.simplefilter("ignore", category=RuntimeWarning)
                scale = np.nanmax(np.abs(selected), axis=0)
                normalized = np.divide(selected, scale, out=np.zeros_like(selected), where=scale != 0)
                mean = scale * np.nansum(normalized, axis=0) / count
                mean[(count > 0) & (scale == 0)] = 0
                for stat, result in (("mean", mean), ("median", np.nanmedian(selected, axis=0)),
                                     ("maximum", np.nanmax(selected, axis=0))):
                    out[f"{key}__{label}__{stat}"] = result
            out[f"{key}__{label}__valid_count"] = count.astype(np.int32)
    return out


def deadline_expired(deadline):
    return deadline is not None and time.monotonic() >= deadline


@torch.inference_mode()
def execute_precision(diagnostic, tokens, gold, *, deadline=None):
    """Run a supplied bounded cohort; callers retain only returned scalar history."""
    n, horizon = tokens.shape
    if gold.shape != tokens.shape or horizon < 1:
        raise ValueError("Expected nonempty matching token/gold arrays")
    states = diagnostic.initial_states(n)
    predictions = {mode: np.full((n, horizon+1), -1, dtype=np.int8) for mode in MODES}
    histories = {mode: ScalarHistory(n) for mode in MODES}
    terminal = {mode: np.zeros(n, dtype=bool) for mode in MODES}
    first_terminal = {mode: [None]*n for mode in MODES}
    invalid_readout = {mode: np.zeros(n, dtype=np.int64) for mode in MODES}
    completed, traces = 0, None
    start = time.perf_counter()
    for position in range(horizon+1):
        if deadline_expired(deadline):
            break
        ids = np.full(n, 6, dtype=np.int64) if position == 0 else tokens[:, position-1]
        target = np.zeros(n, dtype=np.int64) if position == 0 else gold[:, position-1]
        states, logits, traces = diagnostic.step(states, ids)
        detail = summarize_step(states, logits, target, position)
        for mode in MODES:
            bad = ~torch.isfinite(states[mode]).flatten(1).all(1).numpy()
            new = bad & ~terminal[mode]
            for row in np.flatnonzero(new):
                first_terminal[mode][row] = position
            terminal[mode] |= bad
        for mode in MODES:
            row = detail["modes"][mode]
            active = ~terminal[mode]
            logit_finite = np.asarray(row["logits_finite"], dtype=bool)
            invalid_readout[mode] += active & ~logit_finite
            pred = np.asarray(row["prediction"], dtype=np.int8)
            pred[~active] = -1
            predictions[mode][:, position] = pred
            state = states[mode].double().numpy()
            reference = states["D00"].double().numpy()
            values = {"active": active, "logits_finite": active & logit_finite,
                      "state_norm": masked(stable_l2(state), active),
                      "state_difference_l2_from_D00": masked(stable_l2(state-reference), active & ~terminal["D00"]),
                      "state_difference_mse_from_D00": masked(stable_mse(state-reference), active & ~terminal["D00"])}
            for name in ("gold_margin", "top_two_margin", "logit_max_difference_from_D00"):
                valid = active & logit_finite
                if name.endswith("from_D00"):
                    valid &= ~terminal["D00"] & np.asarray(detail["modes"]["D00"]["logits_finite"])
                values[name] = masked(np.asarray(row[name], dtype=np.float64), valid)
            histories[mode].append(values)
        # Terminal precision rows are computation placeholders only. They never
        # become valid again or contribute their zeros to diagnostic aggregates.
        for mode in MODES:
            states[mode][torch.from_numpy(terminal[mode])] = 0
        completed += 1
    elapsed = time.perf_counter()-start
    records = {}
    for mode in MODES:
        records[mode] = dict(predictions=predictions[mode][:, :completed],
            scalars=histories[mode].arrays(), first_terminal=first_terminal[mode],
            terminal_codes=terminal[mode].astype(np.uint8).tolist(),
            execution=dict(completed_writes=completed, expected_writes=horizon+1,
                status="COMPLETE" if completed == horizon+1 else "NOT_COMPLETE",
                elapsed_seconds=elapsed, invalid_readout_counts=invalid_readout[mode].tolist(),
                elapsed_scope="whole four-mode diagnostic loop including scalar summaries; not mode latency",
                terminal_definition="first nonfinite recurrent state; persistent absorbing mask",
                finite_state_readout_failure_is_terminal=False))
    return records, traces


def save_cell(folder, record, gold, identity, *, ledger=None, extra=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    pred = record["predictions"]
    arrays = record["scalars"]
    np.savez_compressed(folder/"scalar_trajectories.npz", **arrays)
    np.savez_compressed(folder/"time_aggregates.npz", **aggregate_scalars(arrays, pred, gold))
    diagnostic_files = {name: common.file_sha(folder/name) for name in
                        ("scalar_trajectories.npz", "time_aggregates.npz")}
    details = {"diagnostic_files_sha256": diagnostic_files,
               "completion_status": record["execution"]["status"],
               "observed_group_tokens": max(0, pred.shape[1]-1),
               "expected_group_tokens": record["execution"]["expected_writes"]-1,
               "trajectory_axes": ["sample", "BOS_then_group_position"],
               "diagnostic_nonfinite_encoding": "NaN scalar plus explicit active/logit masks and per-statistic valid counts",
               "aggregate_populations": "all; before first label failure excluding BOS; at/after first label failure; each intersects that metric's finite valid values",
               "full_state_or_logit_history_retained": False}
    if extra:
        details.update(extra)
    if pred.shape[1] < 2:
        np.savez_compressed(folder/"predictions_partial.npz", predictions=pred)
        summary = dict(identity, execution=record["execution"], **details,
                       status="NOT_COMPLETE", metrics=None, reason="fewer than one scored group token")
        common.write_json(folder/"summary.json", summary)
        return summary
    return metrics.save_result(folder, predictions=pred, gold=gold[:, :pred.shape[1]-1],
        identity=dict(identity, family_size=None), execution=record["execution"],
        ledger={} if ledger is None else ledger, first_terminal=record["first_terminal"],
        terminal_codes=record["terminal_codes"], extra=details)


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--upstream", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--deadline-seconds", type=float, default=None,
                        help="Cooperative cap from setup through loop; artifact writes finish afterward")
    result.add_argument("--deadline-epoch", type=float, default=None,
                        help="Parent UTC epoch deadline; earlier of this and relative cap is enforced")
    return result


def setup(args):
    if args.deadline_seconds is not None and args.deadline_seconds <= 0:
        raise ValueError("Deadline must be positive")
    deadline = None if args.deadline_seconds is None else time.monotonic()+args.deadline_seconds
    if args.deadline_epoch is not None:
        parent_deadline = time.monotonic()+(args.deadline_epoch-time.time())
        deadline = parent_deadline if deadline is None else min(deadline, parent_deadline)
    protocol = common.verify_protocol()
    if args.output.exists():
        raise ValueError("Output directory must be new")
    torch.set_num_threads(2)
    tokens, gold, sample_ids, cohort = common.load_cohort("diagnostic")
    if tokens.shape != (32, 2048) or cohort["seed"] != 40101:
        raise ValueError("Diagnostic cohort differs from frozen 32x2048 seed40101")
    model, table = common.load_model(args.upstream, args.checkpoint, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(model_seed=args.seed, checkpoint_sha256=common.CHECKPOINTS[args.seed],
        cohort="diagnostic", cohort_definition=cohort, sample_ids_sha256=cohort["sample_ids_sha256"],
        input_manifest_sha256=common.file_sha(common.ROOT/"inputs/manifest.json"),
        protocol_sha256=common.file_sha(common.ROOT/"protocol_v2.json"),
        parent_deadline_epoch=args.deadline_epoch, relative_deadline_seconds=args.deadline_seconds,
        sample_ids=sample_ids, family_size=None, scientific_scope="diagnostic only; not confirmatory or added to fresh sample N")
    return model, table, tokens, gold, identity, deadline, protocol


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    model, table, tokens, gold, identity, deadline, _ = setup(args)
    from .reference import v1
    diagnostic = PrecisionDiagnostic(model, table, {
        "model_seed": args.seed, "checkpoint_sha256": common.CHECKPOINTS[args.seed],
        "verified_checkpoint_sha256": common.CHECKPOINTS,
        "token_table_sha256": common.sha(v1.table_bytes(table)[1])})
    records, dtype_trace = execute_precision(diagnostic, tokens, gold, deadline=deadline)
    cells = {}
    for mode, record in records.items():
        save_cell(args.output/mode, record, gold, dict(identity, arm=mode),
                  extra={"precision": diagnostic.metadata()})
        cells[mode] = common.file_sha(args.output/mode/"summary.json")
    complete = all(r["execution"]["status"] == "COMPLETE" for r in records.values())
    common.write_json(args.output/"index.json", dict(schema="case010-v2-precision-cohort-v1",
        status="COMPLETE" if complete else "NOT_COMPLETE", identity=identity, cpu_threads=2,
        cells=cells, dtype_trace=dtype_trace, precision=diagnostic.metadata(),
        runner_sha256=common.file_sha(__file__)))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
