"""Saved P1 CPU replay: no ODE solves, with explicit checkpoint coverage."""

from pathlib import Path
import numpy as np

from ..io import (Meter,canonical,check_files,digest,fresh_output,read_json,seal_files,utc,write_json)
from .contract import assert_source,grid,load_inputs,planned
from .metrics import measure,comparison_status,direction
from .analysis import regenerate


def validate_saved_arrays(raw,meta,C,sigma,tol):
    singular=raw["z_singular_sorted"];coeff=raw["target_mode_signed"]
    z2=np.sum(singular**2,axis=1)
    algebra_loss=.5*np.sum(C*C)+.5*z2-coeff@sigma
    errors={"loss":float(np.max(np.abs(algebra_loss-raw["loss"]))),
            "norm":float(np.max(np.abs(z2-raw["z_frobenius"]**2))),
            "z_gram_cross":float(np.max(np.abs(z2-raw["gram_cross_trace"]))),
            "residual":float(np.max(np.abs((algebra_loss-meta["loss_min"])/meta["loss_denominator"]-raw["learnable_residual"]))),
            "imbalance_eigen_norm":float(np.max(np.abs(np.linalg.norm(raw["imbalance_error_eigenvalues"],axis=1)-raw["imbalance_absolute"]))),
            "imbalance_ratio":float(np.max(np.abs(raw["imbalance_absolute"]/(raw["gram_u_frobenius"]+raw["gram_v_frobenius"])-raw["imbalance_relative_gram"])))}
    for side in ["u","v"]:
        eig=raw[f"gram_{side}_eigenvalues"]
        errors[f"gram_{side}_trace"]=float(np.max(np.abs(np.sum(eig,axis=1)-raw[f"gram_{side}_trace"])))
        errors[f"gram_{side}_norm"]=float(np.max(np.abs(np.linalg.norm(eig,axis=1)-raw[f"gram_{side}_frobenius"])))
    for side in ["left","right"]:
        cos=raw[side+"_cosines"]
        valid=np.isfinite(cos).all(axis=1)
        d=np.sum(1-cos[valid]**2,axis=1)/32
        worst=1-np.min(cos[valid],axis=1)**2
        errors[side+"_projector_cosine"]=float(np.max(np.abs(d-raw[side+"_distance"][valid])))
        errors[side+"_worst"]=float(np.max(np.abs(worst-raw[side+"_worst"][valid])))
        if np.max(cos[valid])>1+tol["cosine_roundoff_limit"] or np.min(cos[valid])<0:
            raise ValueError("COSINE_RANGE")
    if raw["left_rank"][0]!=0 or not np.isnan(raw["left_distance"][0]) or not np.isnan(raw["left_cosines"][0]).all():
        raise ValueError("ZERO_LEFT_NOT_NA")
    if not np.all(raw["right_rank"]==32) or not np.all(raw["left_rank"][1:]==32):raise ValueError("RANK_LOSS")
    if max(errors.values())>tol["raw_algebra_absolute"]:raise ValueError("ALL_GRID_RAW_ALGEBRA "+str(errors))
    return errors


def verify(root,run,output):
    output=fresh_output(output);meter=Meter();run=Path(run)
    try:
        manifest=read_json(run/"run_manifest.json")
        if digest(canonical(manifest["files"]))!=manifest["manifest_rows_sha256"]:raise ValueError("RUN_MANIFEST_HASH")
        check_files(run,manifest["files"])
        frozen=read_json(run/"execution_freeze.json");assert_source(root,frozen);check_files(run/"freeze",frozen["files"])
        if digest(canonical(frozen["files"]))!=frozen["bundle_sha256"]:raise ValueError("BUNDLE_HASH")
        config=read_json(run/"config.json");tol=config["tolerances"]
        if (run/"config.json").read_bytes()!=(run/"freeze/configs/p1.json").read_bytes():raise ValueError("CONFIG_IDENTITY")
        inputs=read_json(run/"input_manifest.json");check_files(run/"assets",inputs["assets"])
        if digest((run/"input_manifest.json").read_bytes())!=frozen["input_manifest_sha256"]:raise ValueError("INPUT_FREEZE")
        if [r["key"] for r in planned(config)]!=[r["key"] for r in inputs["planned_trajectories"]]:raise ValueError("ORDER_SEED_KEYS")
        records=inputs["planned_trajectories"];times=grid(config)
        if len({r["key"] for r in records})!=81:raise ValueError("KEY_UNIQUENESS")
        with np.load(run/"assets/parent_C0_exact.npz",allow_pickle=False) as a:C0=a["C"]
        with np.load(run/"assets/target_g0.npz",allow_pickle=False) as a:
            if C0.tobytes()!=a["C"].tobytes():raise ValueError("C0_BASELINE_BYTES")
        if digest((run/"assets/parent_C0_exact.npz").read_bytes())!=inputs["parent_target_file_sha256"]:raise ValueError("PARENT_BYTES")
        checks=[]
        for meta in records:
            C,left,right,sigma,R=load_inputs(run,meta)
            with np.load(run/"raw"/(meta["key"]+".npz"),allow_pickle=False) as a:raw={k:a[k] for k in a.files}
            if not np.array_equal(raw["times"],times):raise ValueError("GRID_BYTES")
            errors=validate_saved_arrays(raw,meta,C,sigma,tol)
            maximum=0.
            with np.load(run/"checkpoints"/(meta["key"]+".npz"),allow_pickle=False) as ck:
                if len(ck["times"])!=11 or not np.array_equal(ck["C"],C) or not np.array_equal(ck["R"],R):raise ValueError("CHECKPOINT_INPUT")
                for j,t in enumerate(ck["times"]):
                    i=int(np.flatnonzero(times==t)[0]);a,b=ck["scaled_A"][j],ck["scaled_B"][j]
                    if not np.array_equal(ck["U"][j],np.sqrt(2*meta["rho"])*a) or not np.array_equal(ck["V"][j],np.sqrt(2*meta["rho"])*b):raise ValueError("PHYSICAL_SCALING")
                    recomputed=measure(C,left,right,sigma,a,b,meta["rho"],tol)
                    for k,v in recomputed.items():
                        v,expected=np.asarray(v),np.asarray(raw[k][i])
                        if not np.array_equal(np.isnan(v),np.isnan(expected)):raise ValueError("CHECKPOINT_NA "+k)
                        finite=np.isfinite(v)
                        err=float(np.max(np.abs(v[finite]-expected[finite]))) if finite.any() else 0.
                        maximum=max(maximum,err)
                        if err>tol["raw_algebra_absolute"]:raise ValueError("CHECKPOINT_METRIC "+k)
            checks.append({"key":meta["key"],"grid_points":len(times),"all_grid_errors":errors,"checkpoints":11,"checkpoint_metric_max_error":maximum})
        reference_checks=[]
        for meta in records:
            if not meta["matrix_reference"]:continue
            C,left,right,sigma,R=load_inputs(run,meta);rho=meta["rho"]
            for suffix in ["h001","h0005","step_halving"]:
                with np.load(run/"references"/(meta["key"]+"_"+suffix+".npz"),allow_pickle=False) as a:
                    arrays={k:a[k] for k in a.files if k!="times"}
                result=comparison_status(arrays,tol)
                if not result["pass"]:raise ValueError("RECORDED_REFERENCE_NUMERICAL_FAILURE "+meta["key"])
                checkpoint_error=None
                if suffix!="step_halving":
                    checkpoint_error=0.
                    with (np.load(run/"references"/(meta["key"]+"_"+suffix+"_checkpoints.npz"),allow_pickle=False) as ck,
                          np.load(run/"checkpoints"/(meta["key"]+".npz"),allow_pickle=False) as main):
                        for j,t in enumerate(ck["times"]):
                            i=int(np.flatnonzero(times==t)[0]);u,v=ck["U"][j],ck["V"][j]
                            absdiff=np.linalg.norm(main["U"][j]@main["V"][j].T-u@v.T)
                            checkpoint_error=max(checkpoint_error,abs(float(absdiff-arrays["z_absolute"][i])))
                            for side,f,basis in [("right",v,right[:,:32]),("left",u,left[:,:32])]:
                                m=direction(f/np.sqrt(2*rho),basis,tol)
                                for field in ["distance","worst"]:
                                    if np.isfinite(m[field]):
                                        checkpoint_error=max(checkpoint_error,abs(float(m[field]-arrays[side+"_"+field+"_reference"][i])))
                    if checkpoint_error>tol["raw_algebra_absolute"]:raise ValueError("REFERENCE_CHECKPOINT_MISMATCH")
                reference_checks.append({"key":meta["key"],"kind":suffix,"comparison":result,"checkpoint_max_error":checkpoint_error})
        scalar_checks=[]
        for meta in records:
            if not meta["scalar_reference"]:continue
            C,left,right,sigma,R=load_inputs(run,meta)
            with np.load(run/"references"/(meta["key"]+"_scalar_vector.npz"),allow_pickle=False) as a:
                z=a["z"];absolute=a["z_absolute"];norm=a["z_reference_norm"];ld=a["normalized_loss_difference"]
            near=norm<=tol["z_near_zero_norm"]
            rel=float(np.max(absolute[~near]/norm[~near]));small=float(np.max(absolute[near]))
            if rel>tol["z_relative"] or small>tol["z_absolute"] or np.max(ld)>tol["normalized_loss_absolute"]:raise ValueError("SCALAR_REFERENCE_FAILURE")
            if z.shape!=(1702,32) or not np.array_equal(z[0],np.zeros(32)):raise ValueError("SCALAR_COMPONENT_SCOPE")
            scalar_checks.append({"key":meta["key"],"components":32,"max_z_relative":rel,"max_loss_difference":float(np.max(ld))})
        summary=regenerate(run,output)
        if canonical(summary)!=canonical(read_json(run/"derived/summary.json")):raise ValueError("SUMMARY_REGENERATION")
        for name in ["trajectory_ledger.csv","groups.csv","paired_individual.csv","mode_time_fits.csv"]:
            if (output/name).read_bytes()!=(run/"derived"/name).read_bytes():raise ValueError("CSV_REGENERATION "+name)
        from .plots import figures
        figures(run,output)
        for file in (output/"figures").iterdir():
            if file.read_bytes()!=(run/"derived/figures"/file.name).read_bytes():raise ValueError("FIGURE_REGENERATION "+file.name)
        journal=read_json(run/"integration_journal.json")
        actual={k:sum(j["kind"]==k and j["status"]=="COMPLETED" for j in journal) for k in ["primary","matrix_reference","scalar_vector"]}
        if actual!={"primary":81,"matrix_reference":24,"scalar_vector":9} or len(journal)!=114:raise ValueError("INTEGRATION_COUNTS")
        if summary["completed"]!=81:raise ValueError("NOT_ALL_TRAJECTORIES_COMPLETED")
        for row in summary["rows"]:
            if row["max_imbalance_relative_gram"]>tol["imbalance_relative_gram"] or row["max_normalized_loss_increase"]>tol["monotonic_loss_increase_normalized"]:
                raise ValueError("RAW_INVARIANT_GATE")
        receipt={"status":"PASS","utc":utc(),"source_bundle_sha256":frozen["bundle_sha256"],"trajectories":81,
                 "grid_observations":81*1702,"physical_factor_checkpoints":81*11,"reference_matrix_integrations_covered":24,
                 "reference_full_grid_comparison_receipts":reference_checks,"scalar_vector_references":scalar_checks,
                 "primary_raw_checks":checks,"summary_csv_and_figures":"exact regeneration",
                 "ode_reruns":0,"cost":meter.receipt(),
                 "limits":"all-grid algebra and 11 factor checkpoints; selected 12 matrix references only; no full factor reconstruction between checkpoints; same-machine CPU replay"}
        write_json(output/"verification.json",receipt)
        write_json(output/"output_manifest.json",seal_files(output,exclude=("output_manifest.json",)))
        return receipt
    except Exception as exc:
        write_json(output/"verification_failure.json",{"status":"FAIL","error_type":type(exc).__name__,"error":str(exc),"cost":meter.receipt()})
        raise
