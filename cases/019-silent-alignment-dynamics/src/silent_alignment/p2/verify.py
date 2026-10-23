"""Zero-ODE stored-data reconstruction in a fresh external directory."""
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from ..io import (fresh_output,read_json,write_json,canonical,digest,entry,check_files,Meter)
from ..p1.metrics import measure
from .contract import assert_source,load_branch
from .diagnostics import clean
from .analysis import summarize
from .plots import render


def verify(root,run,output):
    root,run=Path(root),Path(run);out=fresh_output(output);meter=Meter()
    frozen=read_json(run/"execution_freeze.json");assert_source(root,frozen)
    manifest=read_json(run/"run_manifest.json")
    if digest(canonical(manifest["files"]))!=manifest["bundle_sha256"]:raise ValueError("RUN_MANIFEST_HASH")
    check_files(run,manifest["files"]);check_files(run/"freeze",frozen["files"])
    inputs=read_json(run/"input_manifest.json");seal=read_json(run/"input_seal.json")
    check_files(run,[seal["input_manifest"]]);check_files(run/"assets",seal["assets"])
    cfg=read_json(run/"config.json");max_loss=max_norm=max_ck=0.;checkpoints=0;grids=0
    for row in inputs["planned_branches"]:
        C,l,r,s,U0,V0=load_branch(run,row);den=row["loss_denominator"]
        with np.load(run/"raw"/(row["key"]+".npz"),allow_pickle=False) as a:
            ss=np.sum(a["z_singular_sorted"]**2,axis=1)
            loss=.5*np.sum(C*C)+.5*ss-a["target_mode_signed"]@s
            le=float(np.max(np.abs(loss-a["loss"])));ne=float(np.max(np.abs(np.sqrt(ss)-a["z_frobenius"])))
            if le>1e-8 or ne>1e-8:raise ValueError("RAW_LOSS_NORM_RECOMBINATION")
            if np.max(np.abs((a["loss"]-row["loss_min"])/den-a["learnable_residual"]))>1e-10:raise ValueError("COMMON_NORMALIZATION_CHANGED")
            max_loss=max(max_loss,le);max_norm=max(max_norm,ne);grids+=len(a["times"])
            initial_imb=V0.T@V0-U0.T@U0
            with np.load(run/"checkpoints"/(row["key"]+".npz"),allow_pickle=False) as b:
                np.testing.assert_array_equal(b["U"][0],U0);np.testing.assert_array_equal(b["V"][0],V0)
                for i,t in enumerate(b["times"]):
                    j=int(np.flatnonzero(a["times"]==t)[0]);u,v=b["U"][i],b["V"][i]
                    rec=measure(C,l,r,s,u/np.sqrt(2*row["rho"]),v/np.sqrt(2*row["rho"]),row["rho"],cfg["tolerances"])
                    for name in ["loss","z_frobenius","right_distance","left_distance","right_worst","left_worst","gram_u_trace","gram_v_trace","imbalance_absolute"]:
                        diff=float(abs(rec[name]-a[name][j]));max_ck=max(max_ck,diff)
                        if diff>1e-9:raise ValueError("PHYSICAL_CHECKPOINT_MISMATCH: "+name)
                    drift=np.linalg.norm(v.T@v-u.T@u-initial_imb)
                    if abs(drift-a["observed_imbalance_drift_absolute"][j])>1e-12:raise ValueError("OBSERVED_INITIAL_DRIFT_REPLAY")
                    checkpoints+=1
    # Recompute all J/G/A, first crossings and persistence from raw, then
    # reproduce deterministic graphics. No stored endpoint/event bool is used
    # as the answer for these calculations.
    summary=summarize(run,out/"derived",cfg);render(root,run,out/"derived/figures",cfg,summary)
    generated=[]
    for p in sorted((out/"derived").rglob("*")):
        if not p.is_file():continue
        rel=p.relative_to(out);original=run/rel
        if p.read_bytes()!=original.read_bytes():raise ValueError("DERIVED_REGENERATION_DIFFERENCE: "+str(rel))
        if p.suffix==".svg":ET.parse(p)
        generated.append(entry(p,out))
    for row in inputs["parents"]:
        # Recompute transforms/Gram matching independently from preserved parent
        # CONTINUE factors and fixed rotation matrices, rather than reading PASS.
        from .core import transform,matching
        with np.load(run/"assets"/(row["parent_key"]+"_CONTINUE.npz"),allow_pickle=False) as a:u,v=a["U"],a["V"]
        with np.load(run/"assets/rotations.npz",allow_pickle=False) as a:rot={k:a[k] for k in a.files}
        rr=next(x for x in inputs["planned_branches"] if x["parent_key"]==row["parent_key"])
        C,l,r,s,_,_=load_branch(run,rr)
        for arm in ["CONTINUE","GAUGE","ALIGN","ROTATE"]:
            up,vp=transform(u,v,l,r,arm,rot)
            with np.load(run/"assets"/(row["parent_key"]+"_"+arm+".npz"),allow_pickle=False) as a:
                if not np.array_equal(up,a["U"]) or not np.array_equal(vp,a["V"]):raise ValueError("TRANSFORM_REGENERATION")
            m=matching(C,l,r,s,u,v,up,vp,arm,rot,row["rho"],cfg["tolerances"])
            if not m["pass"]:raise ValueError("REPLAY_MATCHING_CONTRACT")
    # Full Z/reference paths not saved: gate series are re-evaluated with their
    # frozen thresholds; physical reference checkpoints remain inspectable.
    from ..p1.metrics import comparison_status
    series=0
    for folder in [run/"matching",run/"references"]:
        for p in folder.glob("*.npz"):
            if p.name.endswith("_checkpoints.npz"):continue
            with np.load(p,allow_pickle=False) as a:
                d={k:a[k] for k in a.files}
                if "z_reference_norm" in d:
                    if not comparison_status(d,cfg["tolerances"])["pass"]:raise ValueError("SAVED_COMPARISON_GATE")
                    series+=1
    receipt={"status":"PASS","primary_count":len(inputs["planned_branches"]),"parent_count":len(inputs["parents"]),
             "all_grid_loss_norm_recombinations":grids,"physical_checkpoints":checkpoints,
             "max_loss_recombination_absolute":max_loss,"max_norm_recombination_absolute":max_norm,
             "max_checkpoint_metric_difference":max_ck,"comparison_series_rechecked":series,
             "generated_files":generated,"ODE_reruns":0,"cost":meter.receipt(),
             "scope":"same-machine CPU saved-data replay; no unsaved full factor reconstruction; reference gate arrays plus stored physical checkpoints",
             "shared":"hash/serialization, metric definitions, plotting; independent algebraic loss/norm recombination and physical checkpoint input checks"}
    write_json(out/"verification.json",clean(receipt));return receipt
