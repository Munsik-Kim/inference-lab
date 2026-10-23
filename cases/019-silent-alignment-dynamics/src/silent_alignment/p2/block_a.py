"""Read-only P1 graph/mode audit and the four specified physical RK4 paths."""
import numpy as np
from ..io import read_json,write_json,entry
from ..p1.contract import load_inputs
from ..p1.analysis import event
from ..p1.metrics import direction
from .core import graph_setup,graph_metrics,reference
from .diagnostics import persistence,compare,csv_write,clean


def execute(root,config,output,budget,journal):
    p1=root/config["parent_run"];dest=output/"block_a";dest.mkdir()
    (dest/"graph").mkdir();(dest/"references").mkdir()
    manifest=read_json(p1/"input_manifest.json");rows=manifest["planned_trajectories"]
    graph_rows=[];mode_rows=[];overlaps=[];cross=[];svds={};setups={};references=[]
    with np.load(p1/"assets/fixed_basis.npz",allow_pickle=False) as a:fixedleft,fixedright=a["left"],a["right"]
    for gi in range(3):
        with np.load(p1/f"assets/target_g{gi}.npz",allow_pickle=False) as a:C=a["C"]
        l,s,rt=np.linalg.svd(C,full_matrices=False);q=rt.T
        svds[gi]=(l,s,q)
        overlaps.append({"g_index":gi,"actual_C_reconstruction_error":float(np.linalg.norm((l*s)@q.T-C)),
                         "left_abs_diagonal_overlap_error":float(np.max(np.abs(np.abs(np.diag(fixedleft.T@l))-1))),
                         "right_abs_diagonal_overlap_error":float(np.max(np.abs(np.abs(np.diag(fixedright.T@q))-1))),
                         "top_right_projector_error":float(np.linalg.norm(q[:,:32]@q[:,:32].T-fixedright[:,:32]@fixedright[:,:32].T))})
    evaluations=0
    for row in rows:
        budget.check();key=row["key"];C,left,right,sigma,R=load_inputs(p1,row)
        q=svds[row["g_index"]][2];sig=svds[row["g_index"]][1]
        M,setup=graph_setup(q,R,config["graph_max_top_condition"])
        setups[key]=setup
        with np.load(p1/"raw"/(key+".npz"),allow_pickle=False) as a:
            t=a["times"];mean,worst=graph_metrics(sig,row["rho"],t,M,budget);evaluations+=len(t)
            dm=np.abs(mean-a["right_distance"]);dw=np.abs(worst-a["right_worst"])
            graph_event=event(t,worst,worst[0]*.5,already=row["initialization"]=="aligned")
            raw_event=event(t,a["right_worst"],a["right_worst"][0]*.5,already=row["initialization"]=="aligned")
            np.savez_compressed(dest/"graph"/(key+".npz"),times=t,right_distance=mean,right_worst=worst,
                                distance_difference=dm,worst_difference=dw)
            item={"key":key,"g":row["g"],"rho":row["rho"],"R_index":row["R_index"],"initialization":row["initialization"],
                  "max_mean_error":float(np.max(dm)),"max_worst_error":float(np.max(dw)),
                  "graph_worst_half":graph_event,"p1_worst_half_recalculated":raw_event,
                  "graph_setup":setup,"pass":bool(np.max(dm)<=config["graph_absolute"] and np.max(dw)<=config["graph_absolute"])}
            graph_rows.append(item)
            if row["initialization"]=="random":
                for mode in [32,33]:
                    pr=persistence(t,a["target_mode_signed"][:,mode-1]/sigma[mode-1])
                    mode_rows.append({"key":key,"g":row["g"],"rho":row["rho"],"R_index":row["R_index"],"mode":mode,
                                      "P1_first_crossing_unchanged":event(t,-a["target_mode_signed"][:,mode-1]/sigma[mode-1],-.1),
                                      **pr})
        with np.load(p1/"checkpoints"/(key+".npz"),allow_pickle=False) as ck:
            for ix,tt in enumerate(ck["times"]):
                d=direction(ck["V"][ix]/np.sqrt(2*row["rho"]),q[:,:32],config["tolerances"])
                jj=int(np.flatnonzero(t==tt)[0])
                if abs(d["distance"]-mean[jj])>config["graph_absolute"] or abs(d["worst"]-worst[jj])>config["graph_absolute"]:
                    raise ValueError("GRAPH_CHECKPOINT_PROJECTOR_MISMATCH")
                if row["R_index"]==2 and row["g_index"]==2:
                    z=ck["U"][ix]@ck["V"][ix].T
                    cross.append({"key":key,"time":float(tt),"l32_Z_r33":float(left[:,31]@z@right[:,32]),
                                  "l33_Z_r32":float(left[:,32]@z@right[:,31]),
                                  "scope":"physical saved checkpoint only"})
        if not item["pass"]:raise ValueError("BLOCK_A_GRAPH_AGREEMENT: "+key)
    # Same eight initializations, not 24 independent initializations.
    initial=[]
    for rep in range(8):
        selected=[r for r in graph_rows if r["R_index"]==rep and r["rho"]==1e-6]
        r=next(x for x in selected if x["g"]==.4)
        initial.append({"R_label":f"R{rep}","R_index":rep,"R_stream":[61023020,1,rep],
                        **r["graph_setup"],"worst_half_by_gap":{str(x["g"]):x["p1_worst_half_recalculated"]["time"] for x in selected}})
    write_json(dest/"graph_comparison.json",clean(graph_rows));csv_write(dest/"graph_comparison.csv",graph_rows)
    write_json(dest/"mode_persistence.json",clean(mode_rows));csv_write(dest/"mode_persistence.csv",mode_rows)
    write_json(dest/"R8_overlap.json",initial);csv_write(dest/"R8_overlap.csv",initial)
    csv_write(dest/"R2_checkpoint_cross_modes.csv",cross);write_json(dest/"basis_comparison.json",overlaps)
    # Additional posthoc R2 reference: physical initial U=0,V=sqrt(2rho)R,
    # independently integrated from zero through the complete original horizon.
    for key in ["g2_rho0_R2","g2_rho2_R2"]:
        row=next(x for x in rows if x["key"]==key);C,l,q,sig,R=load_inputs(p1,row)
        with np.load(p1/"raw"/(key+".npz"),allow_pickle=False) as a:times=a["times"]
        U0=np.zeros((64,32));V0=np.sqrt(2*row["rho"])*R;previous=None
        for h in config["solver"]["reference_steps"]:
            attempt={"kind":"block_a_RK4","key":key,"h":h,"status":"STARTED"};journal(attempt)
            u,v,details=reference(C,U0,V0,times,h,budget)
            # Compare to all-grid P1 loss/directions and exact saved physical
            # Z at checkpoints; P1 never stored full all-grid factors.
            with np.load(p1/"raw"/(key+".npz"),allow_pickle=False) as a:
                dm=[];dw=[];ld=[];m32=[];m33=[];rloss=[]
                for i,(uu,vv) in enumerate(zip(u,v)):
                    if i%32==0:budget.check()
                    z=uu@vv.T;dr=direction(vv,q[:,:32],config["tolerances"])
                    dm.append(abs(dr["distance"]-a["right_distance"][i]));dw.append(abs(dr["worst"]-a["right_worst"][i]))
                    ll=.5*np.sum((z-C)**2);ld.append(abs(ll-a["loss"][i])/row["loss_denominator"]);rloss.append(ll)
                    m32.append(l[:,31]@z@q[:,31]);m33.append(l[:,32]@z@q[:,32])
            with np.load(p1/"checkpoints"/(key+".npz"),allow_pickle=False) as a:
                ix=[int(np.flatnonzero(times==t)[0]) for t in a["times"]]
                za=[];zn=[]
                for j,i in enumerate(ix):
                    zr=a["U"][j]@a["V"][j].T;za.append(np.linalg.norm(u[i]@v[i].T-zr));zn.append(np.linalg.norm(zr))
                np.savez_compressed(dest/"references"/(key+f"_h{h}_checkpoints.npz"),times=a["times"],U=u[ix],V=v[ix])
            near=np.asarray(zn)<=config["tolerances"]["z_near_zero_norm"]
            relative=float(np.max(np.asarray(za)[~near]/np.asarray(zn)[~near]));absolute=float(np.max(np.asarray(za)[near])) if near.any() else 0.
            result={"max_z_relative_frobenius_at_saved_checkpoints":relative,"max_z_absolute_near_zero":absolute,
                    "max_normalized_loss_difference_all_grid":float(np.max(ld)),"max_mean_difference_all_grid":float(np.max(dm)),
                    "max_worst_difference_all_grid":float(np.max(dw))}
            tol=config["tolerances"]
            passed=relative<=tol["z_relative"] and absolute<=tol["z_absolute"] and max(dm)<=tol["projector_absolute"] and max(dw)<=tol["worst_absolute"] and max(ld)<=tol["normalized_loss_absolute"]
            arrays=dict(times=times,reference_loss=rloss,reference_mode32=m32,reference_mode33=m33,
                        right_distance_difference=dm,right_worst_difference=dw,normalized_loss_difference=ld)
            np.savez_compressed(dest/"references"/(key+f"_h{h}.npz"),**arrays)
            halving=None
            if previous is not None:
                hh,halving=compare(C,l,q,sig,u,v,*previous,tol,budget)
                np.savez_compressed(dest/"references"/(key+"_step_halving.npz"),times=times,**hh)
                if not halving["pass"]:passed=False
            references.append({"key":key,"h":h,"pass":bool(passed),"comparison":result,"step_halving":halving,"solver":details})
            journal(dict(attempt,status="COMPLETED" if passed else "FAILED",details=details))
            if not passed:raise ValueError("BLOCK_A_R2_RK4_COMPARISON_FAILED")
            previous=(u,v)
    receipt={"status":"PASS","algebraic_time_evaluations":evaluations,"P1_trajectories":len(graph_rows),
             "physical_checkpoint_direction_comparisons":81*11,"random_mode_series":len(mode_rows),
             "max_graph_mean_error":max(r["max_mean_error"] for r in graph_rows),
             "max_graph_worst_error":max(r["max_worst_error"] for r in graph_rows),
             "new_RK4_integrations":len(references),"step_halving_pairs":2,"references":references,
             "separate_path":"finite-rho graph uses actual C SVD, stable log filter and top-block solve; no ODE output in formula",
             "shared":"NumPy/SciPy, target/rho/R; checkpoint projector helper shared with P1; new physical RK4 shares P0 physical_rhs only",
             "P1_full_factor_path_not_stored":"Z comparison to P1 at saved checkpoints only; all-grid loss and right directions checked"}
    write_json(dest/"receipt.json",clean(receipt));return receipt
