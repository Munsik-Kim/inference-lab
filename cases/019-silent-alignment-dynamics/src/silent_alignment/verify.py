"""CPU replay: hash checks, algebra, saved-factor metrics and raw reductions.

No ODE is re-solved here. All-grid checks use independently recombined raw
observables; complete factor checks are limited to the eight saved checkpoints.
"""

from pathlib import Path
import numpy as np

from . import math as model
from .analysis import regenerate
from .engine import load_assets
from .io import (Meter, assert_bundle, canonical, check_files, digest, fresh_output,
                 read_json, seal_files, utc, write_json)


def verify_run(root, run, output):
    output = fresh_output(output)
    meter = Meter()
    run = Path(run)
    try:
        manifest = read_json(run / "run_manifest.json")
        if manifest["manifest_rows_sha256"] != digest(canonical(manifest["files"])):
            raise ValueError("MANIFEST_IDENTITY")
        check_files(run, manifest["files"])
        freeze = read_json(run / "execution_freeze.json")
        assert_bundle(root, freeze)
        check_files(run / "freeze", freeze["files"])
        if digest(canonical(freeze["files"])) != freeze["bundle_sha256"]:
            raise ValueError("BUNDLE_HASH")
        config = read_json(run / "config.json")
        if (run / "config.json").read_bytes() != (run / "freeze/configs/p0.json").read_bytes():
            raise ValueError("CONFIG_BYTES")
        seal = read_json(run / "input_seal.json")
        if seal["input_manifest_sha256"] != digest((run / "input_manifest.json").read_bytes()):
            raise ValueError("INPUT_MANIFEST_CHANGED")
        plan = read_json(run / "input_manifest.json")
        check_files(run / "assets", plan["assets"])
        if len(plan["trajectories"]) != 17 or len({m["id"] for m in plan["trajectories"]}) != 17:
            raise ValueError("KEY_SET")
        checks = []
        tol = config["tolerances"]
        for meta in plan["trajectories"]:
            C, left, right, sigma, R = load_assets(run, meta)
            with np.load(run / "raw" / (meta["id"] + ".npz"), allow_pickle=False) as raw:
                arrays = {k: raw[k] for k in raw.files}
            if arrays["left_rank"][0] != 0 or not np.isnan(arrays["left_distance"][0]):
                raise ValueError("ZERO_LEFT_DIRECTION_NOT_NA")
            if not np.all(arrays["right_rank"] == meta["m"]):
                raise ValueError("RIGHT_FACTOR_RANK")
            if arrays["right_distance"][0] < 0:
                raise ValueError("PROJECTOR_DISTANCE_NEGATIVE")
            gram_norm_z2 = np.einsum("tij,tji->t", arrays["gram_u"], arrays["gram_v"])
            z2 = np.sum(arrays["z_singular_sorted"]**2, axis=1)
            algebra_loss = .5 * float(sigma @ sigma) + .5 * z2 - arrays["target_mode_signed"] @ sigma
            lmin = .5 * float(sigma[meta["m"]:] @ sigma[meta["m"]:])
            errors = {"z2_gram_vs_singular": float(np.max(np.abs(gram_norm_z2-z2))),
                      "norm_z": float(np.max(np.abs(arrays["z_frobenius"]**2-z2))),
                      "loss_recombined": float(np.max(np.abs(algebra_loss-arrays["loss"]))),
                      "loss_min": float(np.max(np.abs(arrays["loss_min"]-lmin))),
                      "residual": float(np.max(np.abs((algebra_loss-lmin)/(.5*float(sigma@sigma)-lmin)-arrays["learnable_residual"])))}
            drift = np.linalg.norm(arrays["gram_v"] - arrays["gram_u"] - 2*meta["rho"]*np.eye(meta["m"]), axis=(1, 2))
            errors["imbalance"] = float(np.max(np.abs(drift-arrays["imbalance_absolute"])))
            for side in ("left", "right"):
                valid = np.isfinite(arrays[side+"_distance"])
                angle_d = np.sum(np.sin(arrays[side+"_angles_radians"][valid])**2, axis=1)/meta["m"]
                errors[side+"_projector_vs_angles"] = float(np.max(np.abs(angle_d-arrays[side+"_distance"][valid])))
            if max(errors.values()) > tol["raw_algebra_absolute"]:
                raise ValueError("RAW_ALGEBRA_MISMATCH " + meta["id"] + str(errors))
            checkpoint_max = 0.0
            with np.load(run / "checkpoints" / (meta["id"] + ".npz"), allow_pickle=False) as ck:
                if not np.array_equal(ck["C"], C) or not np.array_equal(ck["R"], R):
                    raise ValueError("CHECKPOINT_INPUT_IDENTITY")
                for j, t in enumerate(ck["times"]):
                    index = np.flatnonzero(np.isclose(arrays["times"], t, rtol=0, atol=1e-13))[0]
                    a, b = ck["primary_solver_scaled_A"][j], ck["primary_solver_scaled_B"][j]
                    if not np.array_equal(ck["U"][j], np.sqrt(2*meta["rho"])*a) or not np.array_equal(ck["V"][j], np.sqrt(2*meta["rho"])*b):
                        raise ValueError("PHYSICAL_SCALED_CHECKPOINT")
                    recomputed = model.metrics(C, left, sigma, right, a, b, meta["rho"], tol)
                    for k, value in recomputed.items():
                        actual, expected = np.asarray(value), np.asarray(arrays[k][index])
                        if not np.array_equal(np.isnan(actual), np.isnan(expected)):
                            raise ValueError("CHECKPOINT_NA_MISMATCH " + k)
                        finite = np.isfinite(actual)
                        error = float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.0
                        checkpoint_max = max(checkpoint_max, error)
                        if error > tol["raw_algebra_absolute"]:
                            raise ValueError("CHECKPOINT_METRICS " + k)
            checks.append({"id": meta["id"], "all_grid_algebra_max_errors": errors,
                           "saved_factor_checkpoints": 8, "checkpoint_max_error": checkpoint_max})
        actual_summary = regenerate(run, output)
        if canonical(actual_summary) != canonical(read_json(run / "derived/summary.json")):
            raise ValueError("SUMMARY_REGENERATION_MISMATCH")
        if (output / "summary.csv").read_bytes() != (run / "derived/summary.csv").read_bytes():
            raise ValueError("CSV_REGENERATION_MISMATCH")
        from .plots import figures
        figures(run, output)
        mismatches = []
        for file in sorted((output / "figures").iterdir()):
            if file.read_bytes() != (run / "derived/figures" / file.name).read_bytes():
                mismatches.append(file.name)
        if mismatches:
            raise ValueError("FIGURE_REGENERATION_MISMATCH " + str(mismatches))
        ledger = read_json(run / "attempt_ledger.json")
        # Reapply the predeclared acceptance thresholds to the recorded numerical
        # differences rather than merely trusting a stored PASS boolean.
        for trial in ledger:
            for ref in trial["references"]:
                for key in ("comparison_to_primary", "step_halving_comparison"):
                    if key not in ref:
                        continue
                    values = ref[key]
                    if (values["max_z_relative_frobenius"] > tol["z_relative"] or
                        values["max_z_absolute_near_zero"] > tol["z_absolute"] or
                        values["max_learnable_normalized_loss_absolute"] > tol["normalized_loss_absolute"] or
                        values["max_projector_distance_absolute"] > tol["projector_absolute"]):
                        raise ValueError("RECORDED_NUMERICAL_REFERENCE_FAILURE")
            for g in trial["gram_closed_form_checks"]:
                if not g["finite"] or g["relative_gram_error"] > tol["closed_form_relative"]:
                    raise ValueError("GRAM_REFERENCE_FAILURE")
            if trial["maximum_normalized_loss_increase"] > tol["monotonic_loss_increase_normalized"]:
                raise ValueError("LOSS_INCREASE")
            if trial["maximum_imbalance_relative_gram"] > tol["imbalance_relative_gram"]:
                raise ValueError("IMBALANCE_DRIFT")
        receipt = {"status": "PASS", "checked_utc": utc(), "source_bundle_sha256": freeze["bundle_sha256"],
                   "trajectories": 17, "all_grid_algebra_checks": checks, "factor_checkpoint_checks": 136,
                   "summary_and_csv": "exact bytes/equivalent canonical JSON", "figures": "six PNG/SVG files exact bytes",
                   "numerical_reference_receipts": "thresholds reapplied, no integration rerun",
                   "model_or_ode_reruns": 0, "cost": meter.receipt(),
                   "limits": "all-grid algebra plus saved-checkpoint factor metrics; not full factor reconstruction between checkpoints; same-machine CPU replay"}
        write_json(output / "verification.json", receipt)
        write_json(output / "output_manifest.json", seal_files(output, exclude=("output_manifest.json",)))
        return receipt
    except Exception as exc:
        write_json(output / "verification_failure.json", {"status": "FAIL", "error_type": type(exc).__name__,
                   "error": str(exc), "cost": meter.receipt()})
        raise
