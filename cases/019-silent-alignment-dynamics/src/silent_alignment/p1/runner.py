"""Sequential, bounded P1 orchestration using unchanged P0 integrators."""

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import resource
import shutil
import numpy as np
from scipy.integrate import solve_ivp

from ..io import (Meter,canonical,digest,entry,environment,fresh_output,git_identity,
                  read_json,seal_files,utc,write_json)
from ..solvers import dop853,rk4
from .contract import (ControlBudget,assert_source,grid,load_inputs,make_inputs,source_bundle)
from .metrics import compact,compare
from .analysis import regenerate


def scalar_vector(sigma,rho,times,budget):
    def f(t,z):
        budget.check()
        return 2*np.hypot(z,rho)*(sigma-z)
    sol=solve_ivp(f,(0,float(times[-1])),np.zeros(32),method="DOP853",t_eval=times,rtol=1e-12,atol=1e-14)
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError("SCALAR_VECTOR_REFERENCE_FAILED: "+sol.message)
    return sol.y.T,{"solver":"independent_32_scalar_z_DOP853","nfev":sol.nfev,"components":32,
                    "accepted_steps":None,"rejected_steps":None,
                    "step_count_reason":"solve_ivp t_eval result does not expose internal accepted/rejected counts"}


def scalar_comparison(C,left,right,sigma,rho,A,B,z,tol,budget):
    absolute=[];norm=[];lossdiff=[]
    den=.5*np.sum(C*C)-.5*(sigma[32:]@sigma[32:])
    for i,(a,b,zi) in enumerate(zip(A,B,z)):
        if i%16==0:budget.check()
        matrix=2*rho*a@b.T
        scalar=(left[:,:32]*zi)@right[:,:32].T
        absolute.append(float(np.linalg.norm(matrix-scalar)));norm.append(float(np.linalg.norm(scalar)))
        lossdiff.append(float(abs(.5*np.sum((matrix-C)**2)-.5*np.sum((scalar-C)**2))/den))
    absolute,norm,lossdiff=map(np.asarray,(absolute,norm,lossdiff))
    near=norm<=tol["z_near_zero_norm"]
    relative=float(np.max(absolute[~near]/norm[~near])) if (~near).any() else 0.
    small=float(np.max(absolute[near])) if near.any() else 0.
    maximum=float(np.max(lossdiff))
    return {"z_absolute":absolute,"z_reference_norm":norm,"normalized_loss_difference":lossdiff}, {
        "max_z_relative_frobenius":relative,"max_z_absolute_near_zero":small,
        "max_normalized_loss_difference":maximum,
        "pass":bool(relative<=tol["z_relative"] and small<=tol["z_absolute"] and maximum<=tol["normalized_loss_absolute"])}


def resource_check(root,output,external,config,enforce=True):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    raw=sum(p.stat().st_size for p in output.rglob("*.npz"))
    # All P1-only files plus this task's external evidence; P0 original files
    # are counted only in the new external preservation copy.
    evidence_files=sum(p.stat().st_size for p in external.rglob("*") if p.is_file())
    p1_other=sum(p.stat().st_size for p in root.glob("P1_*.md"))
    disk=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())+evidence_files+p1_other
    if enforce:
        if rss>config["limits"]["peak_rss_target_kib"]:raise MemoryError("P1_PEAK_RSS_TARGET_EXCEEDED")
        if raw>config["limits"]["raw_checkpoint_target_bytes"]:raise RuntimeError("P1_RAW_TARGET_EXCEEDED")
        if disk>config["limits"]["hard_additional_disk_bytes"]:raise RuntimeError("P1_ADDITIONAL_DISK_HARD_LIMIT")
    return {"peak_rss_kib":rss,"npz_bytes":raw,"additional_disk_accounted_bytes":disk,
            "disk_scope":"run, external evidence/preservation copy and P1 reports; small source/config extras inventoried finally"}


def run(root,config_path,output,budget_ledger):
    root=Path(root)
    if Path(config_path).resolve()!=(root/"configs/p1.json").resolve():raise ValueError("USE_FROZEN_P1_CONFIG")
    output=fresh_output(output);meter=Meter();config=read_json(config_path)
    budget=ControlBudget(budget_ledger,config["limits"]["cumulative_control_worker_wall_seconds"],output)
    external=Path(budget_ledger).parent
    for d in ["assets_placeholder","raw","checkpoints","references","freeze","derived"]:
        if d!="assets_placeholder":(output/d).mkdir()
    attempts=[];journal=[];status={"execution_status":"RUNNING","implementation_status":"PENDING",
                                 "primary_planned":81,"primary_started":0,"primary_completed":0,
                                 "primary_failed":0,"primary_missing":81,
                                 "planned_integrations":{"primary":81,"matrix_reference":24,"scalar_vector":9},
                                 "phenomenon":"NOT_AN_IMPLEMENTATION_GATE"}
    write_json(output/"attempt_ledger.json",attempts);write_json(output/"integration_journal.json",journal)

    def invoke(kind,label,fn):
        record={"kind":kind,"label":label,"status":"STARTED","submitted_utc":utc()}
        journal.append(record);write_json(output/"integration_journal.json",journal)
        clock=Meter()
        try:
            result=fn();record.update(status="COMPLETED",ended_utc=utc(),solver=result[-1],cost=clock.receipt())
            return result
        except Exception as exc:
            record.update(status="FAILED",ended_utc=utc(),error_type=type(exc).__name__,error=str(exc),cost=clock.receipt())
            raise
        finally:write_json(output/"integration_journal.json",journal)

    try:
        if shutil.disk_usage(output).free<config["limits"]["hard_additional_disk_bytes"]:
            raise RuntimeError("INSUFFICIENT_FREE_DISK_FOR_P1_LIMIT")
        source=source_bundle(root)
        for row in source["files"]:
            p=output/"freeze"/row["path"];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/row["path"],p)
        shutil.copyfile(config_path,output/"config.json")
        write_json(output/"environment.json",environment())
        inputs=make_inputs(root,config,output)
        write_json(output/"input_manifest.json",inputs)
        freeze={**source,"frozen_utc":utc(),"frozen_kst":datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "version":config["version"],"git":git_identity(root),"p1_primary_outputs_observed":0,
                "p0_primary_outputs_previously_observed":17,
                "input_manifest_sha256":digest((output/"input_manifest.json").read_bytes()),
                "planned_order":[r["key"] for r in inputs["planned_trajectories"]]}
        write_json(output/"execution_freeze.json",freeze)
        write_json(output/"input_seal.json",{"sealed_utc":utc(),"p1_primary_outputs_observed":0,
                   "source_bundle_sha256":source["bundle_sha256"],"input_manifest_sha256":freeze["input_manifest_sha256"]})
        times=grid(config);ck=np.array([np.flatnonzero(times==t)[0] for t in config["time"]["checkpoints"]])
        tol=config["tolerances"]
        for meta in inputs["planned_trajectories"]:
            budget.check();assert_source(root,freeze);resource_check(root,output,external,config)
            unit=Meter();attempt={"key":meta["key"],"attempt_id":meta["attempt_id"],"order_index":meta["order_index"],
                                "status":"STARTED","started_utc":utc(),"matrix_references":[],"scalar_reference":None}
            attempts.append(attempt);status["primary_started"]+=1;status["primary_missing"]-=1
            write_json(output/"attempt_ledger.json",attempts)
            C,left,right,sigma,R=load_inputs(output,meta);rho=meta["rho"]
            primary=Meter()
            A,B,detail=invoke("primary",meta["key"],lambda:dop853(C,R,rho,times,config["solver"],budget))
            observations=compact(C,left,right,sigma,A,B,rho,tol,budget)
            primary_cost=primary.receipt()
            if not np.all(observations["right_rank"]==32) or not np.all(observations["left_rank"][1:]==32):
                raise ValueError("PRIMARY_FACTOR_RANK_LOSS")
            if meta["initialization"]=="aligned" and (abs(observations["right_distance"][0])>tol["aligned_distance"] or abs(observations["right_worst"][0])>tol["aligned_worst_absolute"]):
                raise ValueError("ALIGNED_INPUT_GATE")
            np.savez_compressed(output/"raw"/(meta["key"]+".npz"),times=times,
                                loss_min=np.asarray(meta["loss_min"]),loss_denominator=np.asarray(meta["loss_denominator"]),**observations)
            scale=np.sqrt(2*rho)
            np.savez_compressed(output/"checkpoints"/(meta["key"]+".npz"),times=times[ck],U=scale*A[ck],V=scale*B[ck],
                                scaled_A=A[ck],scaled_B=B[ck],C=C,R=R)
            attempt.update(primary_solver=detail,primary_and_metrics_cost=primary_cost)
            numerical=[]
            if meta["matrix_reference"]:
                previous=None
                for h in config["solver"]["reference_steps"]:
                    ref_meter=Meter();suffix="h001" if h==.01 else "h0005"
                    ra,rb,rd=invoke("matrix_reference",meta["key"]+"_"+suffix,lambda:rk4(C,R,rho,times,h,budget))
                    arrays,comparison=compare(C,left,right,sigma,rho,A,B,ra,rb,tol,budget)
                    np.savez_compressed(output/"references"/(meta["key"]+"_"+suffix+".npz"),times=times,**arrays)
                    np.savez_compressed(output/"references"/(meta["key"]+"_"+suffix+"_checkpoints.npz"),
                                        times=times[ck],U=scale*ra[ck],V=scale*rb[ck])
                    record={"h":h,"solver":rd,"comparison":comparison,"cost":ref_meter.receipt(),"scope":"full 0 to160, every saved time"}
                    numerical.append(comparison["pass"])
                    if previous is not None:
                        step_arrays,step_result=compare(C,left,right,sigma,rho,previous[0],previous[1],ra,rb,tol,budget)
                        np.savez_compressed(output/"references"/(meta["key"]+"_step_halving.npz"),times=times,**step_arrays)
                        record["step_halving_comparison"]=step_result;numerical.append(step_result["pass"])
                    attempt["matrix_references"].append(record);previous=(ra,rb)
                del previous,ra,rb
            if meta["scalar_reference"]:
                z,sd=invoke("scalar_vector",meta["key"]+"_32_modes",lambda:scalar_vector(sigma[:32],rho,times,budget))
                arrays,comparison=scalar_comparison(C,left,right,sigma,rho,A,B,z,tol,budget)
                np.savez_compressed(output/"references"/(meta["key"]+"_scalar_vector.npz"),times=times,z=z,**arrays)
                attempt["scalar_reference"]={"solver":sd,"comparison":comparison};numerical.append(comparison["pass"])
            max_increase=float(np.max(np.diff(observations["loss"]))/meta["loss_0"])
            max_imb=float(np.max(observations["imbalance_relative_gram"]))
            numerical.extend([max_increase<=tol["monotonic_loss_increase_normalized"],max_imb<=tol["imbalance_relative_gram"]])
            attempt.update(status="COMPLETED",numerical_status="PASS" if all(numerical) else "FAIL",
                           ended_utc=utc(),unit_cost=unit.receipt(),maximum_normalized_loss_increase=max_increase,
                           maximum_imbalance_relative_gram=max_imb)
            status["primary_completed"]+=1
            write_json(output/"attempt_ledger.json",attempts);write_json(output/"status.json",status)
            print(f"{status['primary_completed']:02d}/81 {meta['key']} numeric {attempt['numerical_status']}",flush=True)
            del A,B,observations
        budget.check();assert_source(root,freeze)
        regenerate(output,output/"derived")
        from .plots import figures
        figures(output,output/"derived")
        budget.check()
        expected={"primary":81,"matrix_reference":24,"scalar_vector":9}
        actual={k:sum(j["kind"]==k and j["status"]=="COMPLETED" for j in journal) for k in expected}
        if actual!=expected:raise ValueError("ACTUAL_INTEGRATION_COUNTS")
        status.update(execution_status="COMPLETE",implementation_status="PASS" if all(a["numerical_status"]=="PASS" for a in attempts) else "FAIL")
    except Exception as exc:
        status.update(execution_status="PARTIAL",implementation_status="INCOMPLETE",error_type=type(exc).__name__,error=str(exc))
        if attempts and attempts[-1]["status"]=="STARTED":
            attempts[-1].update(status="FAILED",error_type=type(exc).__name__,error=str(exc),ended_utc=utc())
            status["primary_failed"]+=1
        write_json(output/"failure.json",{"utc":utc(),"error_type":type(exc).__name__,"error":str(exc),"automatic_retry":False})
        raise
    finally:
        budget.finish(status["execution_status"])
        status["worker_cost"]=meter.receipt()
        status["cumulative_worker_budget"]=read_json(budget_ledger)
        status["integration_counts"]={kind:{s:sum(r["kind"]==kind and r["status"]==s for r in journal) for s in ["STARTED","COMPLETED","FAILED"]} for kind in ["primary","matrix_reference","scalar_vector"]}
        status["resource_usage"]=resource_check(root,output,external,config,enforce=False)
        write_json(output/"attempt_ledger.json",attempts);write_json(output/"status.json",status)
        files=seal_files(output,exclude=("run_manifest.json",))
        write_json(output/"run_manifest.json",{"files":files,"manifest_rows_sha256":digest(canonical(files)),"self_excluded":True})
    return status
