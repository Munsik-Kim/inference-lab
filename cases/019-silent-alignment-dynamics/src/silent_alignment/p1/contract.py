"""P1-specific source freeze, pairing, target construction and resource debt."""

from pathlib import Path
import shutil
import time
import numpy as np

from ..io import canonical, digest, entry, read_json, write_json, utc, check_files
from ..math import stiefel

P0_BUNDLE = "4a7927504de5ccbe1928a951df0e42298d2ca8cbc798a25bf3ff320e28cea8d0"


def source_bundle(root):
    root = Path(root)
    p0 = read_json(root / "results/p0_20261023-095100/execution_freeze.json")
    if p0["bundle_sha256"] != P0_BUNDLE or digest(canonical(p0["files"])) != P0_BUNDLE:
        raise ValueError("P0_SOURCE_IDENTITY")
    check_files(root, p0["files"])
    own = sorted([*root.glob("src/silent_alignment/p1/*.py"), *root.glob("p1_tests/*.py"),
                  root / "configs/p1.json", root / "specs/p1_protocol.json",
                  root / "P1_AMENDMENT_01.md", root / "P1_SOURCE_MAP.md"])
    rows = [entry(p, root) for p in own]
    combined = sorted(rows + p0["files"], key=lambda x: x["path"])
    return {"own_files": rows, "read_only_p0_files": p0["files"], "files": combined,
            "p0_bundle_sha256": P0_BUNDLE, "bundle_sha256": digest(canonical(combined)),
            "own_bundle_sha256": digest(canonical(rows))}


def assert_source(root, frozen):
    actual = source_bundle(root)
    if actual["bundle_sha256"] != frozen["bundle_sha256"] or actual["files"] != frozen["files"]:
        raise ValueError("P1_SOURCE_CONFIG_OR_DEPENDENCY_CHANGED")
    return actual


def grid(config):
    t = config["time"]
    fine = np.arange(t["fine_count"], dtype=np.int64) * t["fine_step"]
    late = 20.0 + np.arange(1, t["late_count"] + 1, dtype=np.int64) * t["late_step"]
    times = np.unique(np.r_[np.round(fine, 12), t["extra_times"], np.round(late, 12)])
    if len(times) != 1702 or times[0] != 0 or times[-1] != 160 or not np.all(np.diff(times) > 0):
        raise ValueError("P1_GRID_CONTRACT")
    for t in config["time"]["checkpoints"]:
        if np.count_nonzero(times == t) != 1:
            raise ValueError("CHECKPOINT_NOT_ON_GRID")
    return times


def key(init, gi, ri, rep):
    return f"g{gi}_rho{ri}_" + (f"R{rep}" if init == "random" else "aligned")


def planned(config):
    rows = []
    refs = {tuple(r) for r in config["reference_selection"]}
    for gi, gap in enumerate(config["gaps"]):
        for ri, rho in enumerate(config["rhos"]):
            for init, reps in (("random", range(8)), ("aligned", [None])):
                for rep in reps:
                    rows.append({"key": key(init, gi, ri, rep), "initialization": init, "g": gap,
                                 "rho": rho, "g_index": gi, "rho_index": ri, "R_index": rep,
                                 "R_stream": config["R_streams"][rep] if rep is not None else None,
                                 "matrix_reference": (init, gap, rho, rep) in refs,
                                 "scalar_reference": init == "aligned"})
    if len(rows) != 81 or len({x["key"] for x in rows}) != 81 or sum(x["matrix_reference"] for x in rows) != 12:
        raise ValueError("P1_PLANNED_KEYS_OR_REFERENCES")
    order = np.random.default_rng(config["order_seed"]).permutation(len(rows))
    return [{**rows[int(i)], "order_index": n, "attempt_id": f"primary-{n:03d}"} for n, i in enumerate(order)]


def make_targets(C0, gaps, tolerance):
    """One SVD fixes the C0 basis; other SVDs only validate each changed target."""
    start = time.process_time()
    left, base_s, rt = np.linalg.svd(C0, full_matrices=False)
    cost = time.process_time() - start
    right = rt.T
    targets = []
    unchanged = np.ones(len(base_s), dtype=bool)
    unchanged[[31, 32]] = False
    for gi, g in enumerate(gaps):
        design = base_s.copy()
        design[31:33] = [.6 + g/2, .6 - g/2]
        if g == .4:
            C = C0.copy()  # preserve array bytes exactly
            measured_s, ml, mr = base_s.copy(), left, right
        else:
            delta = design[31:33] - base_s[31:33]
            C = C0 + (left[:, 31:33] * delta) @ right[:, 31:33].T
            ml, measured_s, mrt = np.linalg.svd(C, full_matrices=False)
            mr = mrt.T
        value_error = float(np.max(np.abs(measured_s-design)))
        unchanged_error = float(np.max(np.abs(measured_s[unchanged]-base_s[unchanged])))
        left_overlap = np.abs(np.diag(left.T @ ml))
        right_overlap = np.abs(np.diag(right.T @ mr))
        overlap_error = float(max(np.max(np.abs(left_overlap-1)), np.max(np.abs(right_overlap-1))))
        representation_error = float(np.linalg.norm(C-(left*measured_s)@right.T))
        if value_error > tolerance["spectrum_absolute"] or unchanged_error > tolerance["spectrum_absolute"]:
            raise ValueError("BOUNDARY_PAIR_SPECTRUM_MISMATCH")
        if overlap_error > tolerance["basis_overlap_absolute"]:
            raise ValueError("FIXED_BASIS_MISMATCH")
        targets.append({"C": C, "sigma": measured_s, "design_sigma": design,
                        "receipt": {"g_index": gi, "requested_gap": g,
                                    "actual_gap": float(measured_s[31]-measured_s[32]),
                                    "sigma32": float(measured_s[31]), "sigma33": float(measured_s[32]),
                                    "max_spectrum_error": value_error, "unchanged_62_max_error": unchanged_error,
                                    "basis_overlap_max_error": overlap_error,
                                    "fixed_basis_representation_frobenius_error": representation_error,
                                    "norm_C": float(np.linalg.norm(C)),
                                    "condition_C": float(measured_s[0]/measured_s[-1]),
                                    "loss_0": .5*float(np.sum(C*C)),
                                    "loss_min": .5*float(measured_s[32:]@measured_s[32:]),
                                    "C_array_bytes_sha256": digest(C.tobytes(order="C")),
                                    "C0_array_bytes_exact": bool(C.tobytes() == C0.tobytes()),
                                    "left_abs_overlap": left_overlap.tolist(), "right_abs_overlap": right_overlap.tolist()}})
    return targets, left, right, {"C0_svd_calls": 1, "C0_basis_svd_process_cpu_seconds": cost,
                                 "changed_target_validation_svd_calls": len(gaps)-1,
                                 "oracle_aligned_basis": "C0 SVD right top32; no extra SVD for each initialization"}


def make_inputs(root, config, output):
    assets = output / "assets"
    assets.mkdir()
    parent = root / config["parent_target"]
    shutil.copyfile(parent, assets / "parent_C0_exact.npz")
    with np.load(parent, allow_pickle=False) as a:
        C0 = a["C"]
    targets, left, right, svd_cost = make_targets(C0, config["gaps"], config["tolerances"])
    np.savez_compressed(assets / "fixed_basis.npz", left=left, right=right)
    Rall = []
    for rep in range(8):
        R = stiefel(64, 32, config["R_streams"][rep])
        if np.max(np.abs(R.T@R-np.eye(32))) > config["tolerances"]["input_orthogonality_absolute"]:
            raise ValueError("R_ORTHOGONALITY")
        Rall.append(R)
        np.savez_compressed(assets / f"R{rep}.npz", R=R)
    np.savez_compressed(assets / "R_aligned.npz", R=right[:, :32])
    for gi, target in enumerate(targets):
        np.savez_compressed(assets / f"target_g{gi}.npz", C=target["C"], sigma=target["sigma"], design_sigma=target["design_sigma"])
    records = planned(config)
    for record in records:
        gi, rep, rho = record["g_index"], record["R_index"], record["rho"]
        R = Rall[rep] if rep is not None else right[:, :32]
        V = np.sqrt(2*rho)*R
        gu, gv = np.zeros((32, 32)), V.T@V
        if np.linalg.norm(gv-2*rho*np.eye(32))/(2*rho*np.sqrt(32)) > config["tolerances"]["initial_gram_relative"]:
            raise ValueError("INITIAL_IMBALANCE")
        asset = f"R{rep}.npz" if rep is not None else "R_aligned.npz"
        record.update(target_asset=f"target_g{gi}.npz", R_asset=asset,
                      target_asset_sha256=digest((assets/f"target_g{gi}.npz").read_bytes()),
                      R_asset_sha256=digest((assets/asset).read_bytes()),
                      amplitude=float(np.sqrt(2*rho)), initial_U_norm=0.0,
                      initial_V_norm=float(np.linalg.norm(V)), initial_gram_u=gu.tolist(), initial_gram_v=gv.tolist(),
                      loss_0=targets[gi]["receipt"]["loss_0"], loss_min=targets[gi]["receipt"]["loss_min"],
                      loss_denominator=targets[gi]["receipt"]["loss_0"]-targets[gi]["receipt"]["loss_min"])
    manifest = {"created_utc": utc(), "p1_primary_outputs_observed": 0, "p0_outputs_previously_observed": 17,
                "parent_target_file_sha256": digest(parent.read_bytes()),
                "parent_C0_array_sha256": digest(C0.tobytes()), "target_receipts": [x["receipt"] for x in targets],
                "basis_cost": svd_cost, "planned_trajectories": records,
                "assets": [entry(p, assets) for p in sorted(assets.iterdir())]}
    return manifest


def load_inputs(run, row):
    with np.load(run / "assets/fixed_basis.npz", allow_pickle=False) as a:
        left, right = a["left"], a["right"]
    with np.load(run / "assets" / row["target_asset"], allow_pickle=False) as a:
        C, sigma = a["C"], a["sigma"]
    with np.load(run / "assets" / row["R_asset"], allow_pickle=False) as a:
        R = a["R"]
    return C, left, right, sigma, R


class ControlBudget:
    """Append-only-attempt debt; an unfinished attempt cannot be silently reset."""
    def __init__(self, path, limit, run):
        self.path = Path(path)
        self.limit = limit
        self.start = time.monotonic()
        if self.path.exists():
            record = read_json(self.path)
            if record["limit_seconds"] != limit or any(a["status"] == "RUNNING" for a in record["attempts"]):
                raise ValueError("UNRESOLVED_CONTROL_BUDGET_ATTEMPT")
        else:
            record = {"limit_seconds": limit, "attempts": []}
        self.record = record
        self.debt = sum(a["wall_seconds"] for a in record["attempts"])
        if self.debt >= limit:
            raise TimeoutError("P1_CUMULATIVE_BUDGET_EXHAUSTED")
        self.record["attempts"].append({"run": str(run), "started_utc": utc(), "status": "RUNNING", "prior_debt_seconds": self.debt})
        write_json(self.path, self.record)

    def check(self):
        if self.debt + time.monotonic()-self.start >= self.limit:
            raise TimeoutError("P1_CUMULATIVE_CONTROL_WORKER_WALL_LIMIT")

    def finish(self, status):
        self.record["attempts"][-1].update(status=status, ended_utc=utc(), wall_seconds=time.monotonic()-self.start)
        self.record["consumed_seconds"] = sum(a["wall_seconds"] for a in self.record["attempts"])
        write_json(self.path, self.record)
