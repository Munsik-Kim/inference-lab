"""One frozen 17-trajectory pilot; all numerical re-solves have separate records."""

import shutil
from pathlib import Path
import numpy as np

from . import math as model
from .analysis import regenerate
from .io import (Budget, Meter, bundle, canonical, digest, entry, environment, fresh_output,
                 git_identity, read_json, seal_files, utc, write_json)
from .solvers import dop853, rk4, scalar_reference


def make_inputs(config, output):
    assets = output / "assets"
    assets.mkdir()
    trajectories = []
    costs = []
    for block in ("A", "B", "C"):
        b = config["blocks"][block]
        if block == "A":
            for i, sigma in enumerate(b["sigmas"]):
                C = np.array([[sigma]], dtype=np.float64)
                np.savez_compressed(assets / f"A_target{i}.npz", C=C, left=np.eye(1), right=np.eye(1), sigma=np.array([sigma]))
                for j, rho in enumerate(b["rhos"]):
                    np.savez_compressed(assets / f"A_R{i}.npz", R=np.eye(1)) if j == 0 else None
                    trajectories.append({"id": f"A_sigma{i}_rho{j}", "block": block, "rho": rho,
                                         "initialization": "aligned", "replicate": 0,
                                         "target_asset": f"A_target{i}.npz", "R_asset": f"A_R{i}.npz",
                                         "sigma_scalar": sigma})
            continue
        p, d, m = b["p"], b["d"], b["m"]
        left = model.stiefel(p, p, config["streams"][block + "_left"])
        right = model.stiefel(d, d, config["streams"][block + "_right"])
        sigma = np.array(b["spectrum"] if block == "B" else
                         np.r_[np.linspace(**{k: b["top_spectrum"][k] for k in ("start", "stop")}, num=b["top_spectrum"]["count"]),
                               np.linspace(**{k: b["bottom_spectrum"][k] for k in ("start", "stop")}, num=b["bottom_spectrum"]["count"])]).astype(np.float64)
        C = (left * sigma) @ right.T
        np.savez_compressed(assets / f"{block}_target.npz", C=C, left=left, right=right, sigma=sigma)
        meter = Meter()
        # Obtain oracle initialization through the actual SVD rather than using
        # a future learned direction; its CPU cost is explicitly measured.
        _, measured_sigma, rt = np.linalg.svd(C, full_matrices=False)
        oracle_R = rt.T[:, :m]
        costs.append({"block": block, "operation": "target SVD for oracle aligned initialization", **meter.receipt()})
        np.savez_compressed(assets / f"{block}_R_aligned.npz", R=oracle_R)
        rhos = b["rhos"] if block == "B" else [b["rho"]]
        for rep in range(b["random_repetitions"]):
            R = model.stiefel(d, m, config["streams"][f"{block}_random{rep}"])
            np.savez_compressed(assets / f"{block}_R_random{rep}.npz", R=R)
            for j, rho in enumerate(rhos):
                trajectories.append({"id": f"{block}_random{rep}_rho{j}", "block": block, "rho": rho,
                                     "initialization": "random", "replicate": rep,
                                     "target_asset": f"{block}_target.npz", "R_asset": f"{block}_R_random{rep}.npz"})
        for j, rho in enumerate(rhos):
            trajectories.append({"id": f"{block}_aligned_rho{j}", "block": block, "rho": rho,
                                 "initialization": "aligned", "replicate": 0,
                                 "target_asset": f"{block}_target.npz", "R_asset": f"{block}_R_aligned.npz"})
    for meta in trajectories:
        with np.load(assets / meta["target_asset"], allow_pickle=False) as a:
            p, d = a["C"].shape
            sigma = a["sigma"]
        with np.load(assets / meta["R_asset"], allow_pickle=False) as a:
            m = a["R"].shape[1]
        meta.update(p=p, d=d, m=m, amplitude=float(np.sqrt(2 * meta["rho"])),
                    **model.target_diagnostics(sigma, m, config["tolerances"]),
                    target_singular_directions_identifiable=bool(len(set(sigma.tolist())) == len(sigma)))
        meta["target_sha256"] = digest((assets / meta["target_asset"]).read_bytes())
        meta["R_sha256"] = digest((assets / meta["R_asset"]).read_bytes())
    if len(trajectories) != config["planned_primary_trajectories"] or len({t["id"] for t in trajectories}) != 17:
        raise ValueError("PLANNED_TRAJECTORY_SET")
    write_json(output / "oracle_initialization_cost.json", costs)
    return {"generated_utc": utc(), "outputs_observed": 0, "trajectories": trajectories,
            "assets": seal_files(assets), "random_sample_reuse": "same saved R across rho and every solver"}


def load_assets(run, meta):
    with np.load(run / "assets" / meta["target_asset"], allow_pickle=False) as a:
        C, left, right, sigma = (a[k] for k in ("C", "left", "right", "sigma"))
    with np.load(run / "assets" / meta["R_asset"], allow_pickle=False) as a:
        R = a["R"]
    return C, left, right, sigma, R


def observations(C, left, right, sigma, A, B, rho, tol, budget):
    columns = {}
    for index, (a, b) in enumerate(zip(A, B)):
        if index % 20 == 0:
            budget.check()
        row = model.metrics(C, left, sigma, right, a, b, rho, tol)
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return {k: np.asarray(v) for k, v in columns.items()}


def compare_paths(C, left, right, sigma, rho, A, B, refA, refB, tol, budget):
    max_rel, max_abs, max_loss, max_projection = 0.0, 0.0, 0.0, 0.0
    near_zero_max_abs = 0.0
    normalized = 0.5 * float(sigma[:B.shape[-1]] @ sigma[:B.shape[-1]])
    for i, (a, b, ra, rb) in enumerate(zip(A, B, refA, refB)):
        if i % 20 == 0:
            budget.check()
        z, zr = 2 * rho * a @ b.T, 2 * rho * ra @ rb.T
        absolute = float(np.linalg.norm(z - zr))
        scale = float(np.linalg.norm(zr))
        max_abs = max(max_abs, absolute)
        if scale > tol["z_near_zero_norm"]:
            max_rel = max(max_rel, absolute / scale)
        else:
            near_zero_max_abs = max(near_zero_max_abs, absolute)
        dl = abs(0.5 * np.sum((z - C)**2) - 0.5 * np.sum((zr - C)**2)) / normalized
        max_loss = max(max_loss, float(dl))
        for f, rf, target in ((a, ra, left[:, :a.shape[1]]), (b, rb, right[:, :b.shape[1]])):
            d1 = model.subspace(f, target, tol)[0]
            d2 = model.subspace(rf, target, tol)[0]
            if np.isfinite(d1) != np.isfinite(d2):
                raise ValueError("REFERENCE_RANK_DISAGREEMENT")
            if np.isfinite(d1):
                max_projection = max(max_projection, abs(d1 - d2))
    passed = (max_rel <= tol["z_relative"] and near_zero_max_abs <= tol["z_absolute"] and
              max_loss <= tol["normalized_loss_absolute"] and max_projection <= tol["projector_absolute"])
    return {"max_z_relative_frobenius": max_rel, "max_z_absolute_frobenius": max_abs,
            "max_z_absolute_near_zero": near_zero_max_abs,
            "max_learnable_normalized_loss_absolute": max_loss,
            "max_projector_distance_absolute": max_projection, "pass": bool(passed)}


def run_pilot(root, config_path, output):
    if Path(config_path).resolve() != (root / "configs/p0.json").resolve():
        raise ValueError("USE_THE_FROZEN_CASE_CONFIG")
    output = fresh_output(output)
    meter = Meter()
    config = read_json(config_path)
    budget = Budget(config["limits"]["pilot_reference_worker_wall_seconds"])
    for name in ("raw", "checkpoints", "metadata", "freeze", "derived"):
        (output / name).mkdir()
    freeze = bundle(root)
    freeze.update(frozen_utc=utc(), primary_outputs_observed=0, git=git_identity(root),
                  identity_scope="Python source, tests, specs and configs; docs/results excluded")
    write_json(output / "execution_freeze.json", freeze)
    for item in freeze["files"]:
        target = output / "freeze" / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / item["path"], target)
    shutil.copyfile(config_path, output / "config.json")
    write_json(output / "environment.json", environment())
    plan = make_inputs(config, output)
    write_json(output / "input_manifest.json", plan)
    write_json(output / "input_seal.json", {"input_manifest_sha256": digest((output / "input_manifest.json").read_bytes()),
                                           "sealed_utc": utc(), "primary_outputs_observed": 0,
                                           "source_bundle_sha256": freeze["bundle_sha256"]})
    time_spec = config["time"]
    times = np.unique(np.r_[np.round(np.arange(round(time_spec["end"] / time_spec["display_step"]) + 1) *
                                    time_spec["display_step"], 12), time_spec["extra_initial_times"]])
    checkpoint_idx = np.array([np.flatnonzero(np.isclose(times, t, atol=1e-13, rtol=0))[0]
                               for t in time_spec["checkpoints"]])
    ledger = []
    status = {"execution_status": "RUNNING", "implementation_validation": "PENDING",
              "primary_planned": 17, "primary_completed": 0, "numerical_reference_solves": 0,
              "phenomenon": "NOT_AN_IMPLEMENTATION_GATE"}
    write_json(output / "status.json", status)
    try:
        for meta in plan["trajectories"]:
            budget.check()
            task_meter = Meter()
            C, left, right, sigma, R = load_assets(output, meta)
            rho = meta["rho"]
            write_json(output / "metadata" / (meta["id"] + ".json"), meta)
            attempt = {"id": meta["id"], "submitted_utc": utc(), "status": "STARTED"}
            ledger.append(attempt)
            write_json(output / "attempt_ledger.json", ledger)
            A, B, primary = dop853(C, R, rho, times, config["solver"], budget)
            obs = observations(C, left, right, sigma, A, B, rho, config["tolerances"], budget)
            np.savez_compressed(output / "raw" / (meta["id"] + ".npz"), times=times, **obs)
            U, V = np.sqrt(2*rho) * A[checkpoint_idx], np.sqrt(2*rho) * B[checkpoint_idx]
            np.savez_compressed(output / "checkpoints" / (meta["id"] + ".npz"),
                                times=times[checkpoint_idx], U=U, V=V, C=C, R=R,
                                primary_solver_scaled_A=A[checkpoint_idx], primary_solver_scaled_B=B[checkpoint_idx])
            references = []
            previous = None
            for h in config["solver"]["reference_steps"]:
                reference_meter = Meter()
                ra, rb, detail = rk4(C, R, rho, times, h, budget)
                checks = compare_paths(C, left, right, sigma, rho, A, B, ra, rb, config["tolerances"], budget)
                detail.update(comparison_to_primary=checks, **reference_meter.receipt())
                if previous is not None:
                    detail["step_halving_comparison"] = compare_paths(C, left, right, sigma, rho,
                                                                      previous[0], previous[1], ra, rb,
                                                                      config["tolerances"], budget)
                previous = (ra, rb)
                references.append(detail)
                status["numerical_reference_solves"] += 1
            scalar = None
            if meta["block"] == "A":
                zref, scalar = scalar_reference(meta["sigma_scalar"], rho, times, budget)
                scalar["max_absolute_z_difference"] = float(np.max(np.abs(obs["target_mode_signed"][:, 0] - zref)))
                scalar["pass"] = scalar["max_absolute_z_difference"] <= config["tolerances"]["z_absolute"] + config["tolerances"]["z_relative"] * meta["sigma_scalar"]
                status["numerical_reference_solves"] += 1
                np.savez_compressed(output / "raw" / (meta["id"] + "_scalar_reference.npz"),
                                    times=times, z=zref, early_approx=rho*np.sinh(2*meta["sigma_scalar"]*times),
                                    early_approx_valid_domain=(zref <= .01*meta["sigma_scalar"]))
            gram_checks, linear_diagnostics = [], []
            if meta["block"] == "B":
                for t in config["references"]["gram_closed_form"]["times"]:
                    index = np.flatnonzero(np.isclose(times, t, rtol=0, atol=1e-13))[0]
                    u, v = np.sqrt(2*rho)*A[index], np.sqrt(2*rho)*B[index]
                    X = np.vstack([u, v])
                    exact, detail = model.closed_gram(C, R, rho, t)
                    error = float(np.linalg.norm(X @ X.T - exact) / np.linalg.norm(exact))
                    gram_checks.append({"time": t, "relative_gram_error": error,
                                        "pass": bool(error <= config["tolerances"]["closed_form_relative"]), **detail})
                    lu, lv = model.linearized_factors(C, R, rho, t)
                    linear_diagnostics.append({"time": t, "relative_z_vs_small_rho_limit":
                                               float(np.linalg.norm(u@v.T-lu@lv.T)/np.linalg.norm(lu@lv.T)),
                                               "scope": "finite-rho scientific approximation error, not solver acceptance"})
            tol = config["tolerances"]
            maximum_loss_increase = float(np.max(np.diff(obs["loss"])) / obs["loss"][0])
            maximum_imbalance = float(np.max(obs["imbalance_relative_gram"]))
            numeric_pass = all(r["comparison_to_primary"]["pass"] for r in references)
            numeric_pass &= all(r.get("step_halving_comparison", {"pass": True})["pass"] for r in references)
            numeric_pass &= all(g["pass"] and g["finite"] for g in gram_checks)
            numeric_pass &= scalar is None or scalar["pass"]
            numeric_pass &= maximum_loss_increase <= tol["monotonic_loss_increase_normalized"]
            numeric_pass &= maximum_imbalance <= tol["imbalance_relative_gram"]
            attempt.update(status="COMPLETED", ended_utc=utc(), primary=primary, references=references,
                           scalar_reference=scalar, gram_closed_form_checks=gram_checks,
                           small_rho_scientific_diagnostic=linear_diagnostics,
                           maximum_normalized_loss_increase=maximum_loss_increase,
                           maximum_imbalance_relative_gram=maximum_imbalance,
                           numerical_validation_pass=bool(numeric_pass), cost=task_meter.receipt())
            status["primary_completed"] += 1
            write_json(output / "attempt_ledger.json", ledger)
            write_json(output / "status.json", status)
            print(meta["id"], "numeric", "PASS" if numeric_pass else "FAIL", flush=True)
            if sum(p.stat().st_size for p in output.rglob("*.npz")) > config["limits"]["raw_bytes_target"]:
                raise RuntimeError("RAW_BYTES_TARGET_EXCEEDED")
        budget.check()
        regenerate(output, output / "derived")
        from .plots import figures
        figures(output, output / "derived")
        budget.check()
        status.update(execution_status="COMPLETE", implementation_validation=("PASS" if all(
            a["numerical_validation_pass"] for a in ledger) else "FAIL"))
    except Exception as exc:
        status.update(execution_status="PARTIAL", implementation_validation="INCOMPLETE",
                      error_type=type(exc).__name__, error=str(exc))
        write_json(output / "failure.json", {"error_type": type(exc).__name__, "error": str(exc), "utc": utc(),
                                             "automatic_retry": False})
        raise
    finally:
        status["worker_cost"] = meter.receipt()
        status["budget_limit_seconds"] = config["limits"]["pilot_reference_worker_wall_seconds"]
        status["raw_npz_bytes"] = sum(p.stat().st_size for p in output.rglob("*.npz"))
        write_json(output / "status.json", status)
        write_json(output / "attempt_ledger.json", ledger)
        rows = seal_files(output, exclude=("run_manifest.json",))
        write_json(output / "run_manifest.json", {"files": rows, "manifest_rows_sha256": digest(canonical(rows)),
                                                   "scope": "self excluded; no self-hash claim"})
    return status
