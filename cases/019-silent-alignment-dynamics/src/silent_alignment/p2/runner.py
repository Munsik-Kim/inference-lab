"""One sequential, budgeted P2 worker; prior scientific bytes are read-only."""
from pathlib import Path
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone,timedelta
import numpy as np
from ..io import (read_json,write_json,canonical,digest,entry,check_files,fresh_output,
                  seal_files,Meter,environment,git_identity,utc)
from ..p1.metrics import comparison_status
from .contract import (source_bundle,assert_source,prepare,load_branch,grid,time_join,WorkerBudget)
from .core import integrate,reference
from .diagnostics import compact,compare,clean
from .analysis import summarize


def regressions(root,external,budget):
    dest=external/"frozen_regressions";dest.mkdir()
    results=[]
    for phase,runname,folder in [("P0","p0_20261023-095100","tests"),("P1","p1_20261023-104046","p1_tests")]:
        target=dest/phase;target.mkdir()
        freeze=read_json(root/"results"/runname/"execution_freeze.json")
        for row in freeze["files"]:
            q=target/row["path"];q.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/"results"/runname/"freeze"/row["path"],q)
        if phase=="P1":
            for p in [Path("results/p0_20261023-095100/execution_freeze.json"),Path("results/p0_20261023-095100/assets/C_target.npz")]:
                q=target/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/p,q)
        results.append(test_process(target,folder,external/(phase+"_regression"),budget))
    results.append(test_process(root,"p2_tests",external/"P2_new_tests",budget))
    if any(x["returncode"]!=0 for x in results):raise ValueError("P2_CPU_REGRESSION_FAILURE")
    return results


def test_process(root,folder,prefix,budget):
    # Fresh process/loader per suite: avoid the historical stateful loader error.
    code='''import sys,unittest,json,time,resource
from pathlib import Path
started=time.monotonic();cpu=time.process_time()
suite=unittest.TestLoader().discover(sys.argv[1],pattern="test*.py")
result=unittest.TextTestRunner(verbosity=2).run(suite)
r={"unique_tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skips":len(result.skipped),"wall_seconds":time.monotonic()-started,"process_cpu_seconds":time.process_time()-cpu,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"pass":result.wasSuccessful()}
Path(sys.argv[2]).write_text(json.dumps(r,indent=2)+"\\n")
sys.exit(0 if r["pass"] else 1)
'''
    env=dict(os.environ,PYTHONPATH=str(root/"src"),PYTHONDONTWRITEBYTECODE="1")
    command=[sys.executable,"-B","-c",code,folder,str(prefix)+".json"]
    start=time.monotonic()
    with open(str(prefix)+".log","w") as log:
        p=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
                         timeout=max(1,budget.limit-budget.debt-(time.monotonic()-budget.start)))
    budget.check()
    receipt=read_json(str(prefix)+".json") if Path(str(prefix)+".json").exists() else {"pass":False,"unique_tests":None}
    return {"suite":folder,"source_root":str(root),"returncode":p.returncode,"receipt":receipt,
            "subprocess_wall_seconds":time.monotonic()-start,"command":command}


def run(root,config_path,output,budget_path,external,launch_time=None):
    # Fresh path must fail before budget mutation or numerical work.
    output=fresh_output(output);root=Path(root);external=Path(external)
    config=read_json(config_path);meter=Meter();launch_time=launch_time or time.monotonic()
    prior=config["budget"]["prior_P1_outer_worker_seconds"]
    parent_debit=read_json(root/"provenance/p1/budget_reconciliation.json")["conservative_total_worker_debit_seconds"]
    if parent_debit!=prior:raise ValueError("P1_BUDGET_DEBT_IDENTITY")
    budget=WorkerBudget(budget_path,min(config["budget"]["P2_worker_seconds"],1800-prior),prior,launch_time)
    journal_rows=[];completed=[];ref_receipts=[];gauge_receipts=[];continue_receipts=[]
    for name in ["raw","checkpoints","completion","references","matching","freeze"]:(output/name).mkdir()
    def journal(row):
        value={"utc":utc(),**row};journal_rows.append(value)
        with open(output/"integration_journal.jsonl","a",encoding="utf-8") as f:f.write(json.dumps(clean(value),ensure_ascii=False,allow_nan=False)+"\n")
    try:
        env=environment();write_json(output/"environment.json",env);write_json(output/"git_start.json",git_identity(root))
        frozen=source_bundle(root)
        for row in frozen["files"]:
            q=output/"freeze"/row["path"];q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/row["path"],q)
        write_json(output/"config.json",config)
        inputs=prepare(root,config,output)
        frozen.update(created_utc=utc(),created_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                      base_git_commit=git_identity(root)["base_git_commit"],working_source_uncommitted=True,
                      p2_primary_outputs_observed=0,initial_transform_metrics_observed=True,
                      prior_p1_primary_outputs_observed=81,input_manifest=entry(output/"input_manifest.json",output),
                      input_seal=entry(output/"input_seal.json",output),checkpoint_lists={str(x):__import__("silent_alignment.p2.contract",fromlist=["checkpoints"]).checkpoints(x) for x in config["t_stars"]})
        write_json(output/"execution_freeze.json",frozen)
        tests=regressions(root,external,budget);write_json(output/"test_receipts.json",tests)
        assert_source(root,frozen)
        from .block_a import execute
        print("P2 frozen; regression complete; starting graph and R2 references",flush=True)
        ba=execute(root,config,output,budget,journal)
        print("Block A PASS; starting128 checkpoint continuations",flush=True)
        times=grid();gauge_cache={};refs_set={(x["g_index"],x["R_index"],x["t_star"],x["arm"]) for x in config["reference_selection"]}
        p1=root/config["parent_run"]
        for row in inputs["planned_branches"]:
            budget.check();assert_source(root,frozen);started=Meter();key=row["key"]
            journal({"kind":"primary","key":key,"attempt_id":row["attempt_id"],"status":"STARTED","order_index":row["order_index"]})
            C,left,right,sigma,U0,V0=load_branch(output,row)
            if digest((output/"assets"/row["input_asset"]).read_bytes())!=row["input_sha256"]:raise ValueError("BRANCH_INPUT_CHANGED")
            U,V,detail=integrate(C,U0,V0,row["rho"],times,config["solver"],budget)
            data=compact(C,left,right,sigma,U,V,row["rho"],config["tolerances"],budget)
            data["times"]=times;np.savez_compressed(output/"raw"/(key+".npz"),**data)
            ix=[int(np.flatnonzero(times==t)[0]) for t in row["checkpoints"]]
            np.savez_compressed(output/"checkpoints"/(key+".npz"),times=times[ix],U=U[ix],V=V[ix])
            tol=config["tolerances"]
            if np.max(data["observed_imbalance_drift_absolute"])>config["drift_contract"]["new_observed_drift_absolute"] or np.max(data["observed_imbalance_drift_relative_gram"])>tol["imbalance_relative_gram"]:
                raise ValueError("NEW_OBSERVED_IMBALANCE_DRIFT")
            if np.max(data["imbalance_relative_gram"])>tol["imbalance_relative_gram"]:raise ValueError("NOMINAL_IMBALANCE_DRIFT")
            if np.max(np.diff(data["loss"]))/row["loss_denominator"]>tol["monotonic_loss_increase_normalized"]:raise ValueError("LOSS_NONMONOTONIC_NUMERICAL")
            if row["arm"] in ["CONTINUE","GAUGE"]:
                parent=row["parent_key"]
                if parent not in gauge_cache:gauge_cache[parent]=(U,V,row["arm"])
                else:
                    ru,rv,other=gauge_cache.pop(parent)
                    arrays,result=compare(C,left,right,sigma,U,V,ru,rv,tol,budget)
                    np.savez_compressed(output/"matching"/(parent+"_gauge_path.npz"),times=times,**arrays)
                    gauge_receipts.append({"parent_key":parent,"comparison":result,"scope":"full physical Z and loss at all1677 grid times, two separately integrated branches"})
                    if not result["pass"]:raise ValueError("GAUGE_PATH_MISMATCH")
                    del ru,rv
            if row["arm"]=="CONTINUE":
                with np.load(p1/"raw"/(row["p1_key"]+".npz"),allow_pickle=False) as old:
                    ni,oi=time_join(times,old["times"],row["t_star"])
                    differences={k:float(np.max(np.abs(data[k][ni]-old[k][oi]))) for k in ["loss","right_distance","right_worst","left_distance","left_worst","target_mode_signed"]}
                    differences["z_singular_sorted"]=float(np.max(np.abs(data["z_singular_sorted"][ni]-old["z_singular_sorted"][oi,:32])))
                with np.load(p1/"checkpoints"/(row["p1_key"]+".npz"),allow_pickle=False) as old:
                    ci,oi=time_join(times[ix],old["times"],row["t_star"])
                    errors=[]
                    for j,k in zip(ci,oi):
                        zz=old["U"][k]@old["V"][k].T
                        errors.append(float(np.linalg.norm(U[ix[j]]@V[ix[j]].T-zz)/max(np.linalg.norm(zz),1e-300)))
                passed=differences["loss"]/row["loss_denominator"]<=tol["normalized_loss_absolute"] and max(differences[k] for k in ["right_distance","left_distance","right_worst","left_worst"])<=tol["projector_absolute"] and max(errors)<=tol["z_relative"]
                cc={"parent_key":row["parent_key"],"exact_common_grid_count":len(ni),"physical_Z_checkpoint_count":len(errors),
                    "max_checkpoint_Z_relative":max(errors),"all_grid_raw_differences":differences,"pass":bool(passed),
                    "join":"exact integer hundredth-tick intersection, no nearest/interpolation"}
                continue_receipts.append(cc)
                if not passed:raise ValueError("CONTINUE_P1_OVERLAP_MISMATCH")
            if (row["g_index"],row["R_index"],row["t_star"],row["arm"]) in refs_set:
                previous=None
                for h in config["solver"]["reference_steps"]:
                    journal({"kind":"branch_RK4","key":key,"h":h,"status":"STARTED"})
                    ru,rv,rd=reference(C,U0,V0,times,h,budget)
                    arrays,result=compare(C,left,right,sigma,U,V,ru,rv,tol,budget)
                    np.savez_compressed(output/"references"/(key+f"_h{h}.npz"),times=times,**arrays)
                    np.savez_compressed(output/"references"/(key+f"_h{h}_checkpoints.npz"),times=times[ix],U=ru[ix],V=rv[ix])
                    halving=None
                    if previous is not None:
                        hh,halving=compare(C,left,right,sigma,ru,rv,*previous,tol,budget)
                        np.savez_compressed(output/"references"/(key+"_step_halving.npz"),times=times,**hh)
                    ref_receipts.append({"key":key,"h":h,"solver":rd,"comparison":result,"step_halving":halving})
                    journal({"kind":"branch_RK4","key":key,"h":h,"status":"COMPLETED" if result["pass"] else "FAILED","details":rd})
                    if not result["pass"] or (halving and not halving["pass"]):raise ValueError("BRANCH_REFERENCE_MISMATCH")
                    previous=(ru,rv)
                del previous,ru,rv
            completion={"key":key,"status":"COMPLETED","solver":detail,"cost":started.receipt(),
                        "raw":entry(output/"raw"/(key+".npz"),output),"checkpoint":entry(output/"checkpoints"/(key+".npz"),output)}
            write_json(output/"completion"/(key+".json"),completion);completed.append(completion)
            journal({"kind":"primary","key":key,"status":"COMPLETED","details":detail})
            if len(completed)%4==0:print(f"Primary {len(completed)}/128; elapsed {time.monotonic()-launch_time:.2f}s",flush=True)
            # Remove current primary arrays except one required temporary gauge partner.
            del U,V,data
            if sum(p.stat().st_size for p in output.rglob("*") if p.is_file())>config["budget"]["hard_new_disk_bytes"]:
                raise RuntimeError("P2_DISK_HARD_CAP")
        if gauge_cache:raise ValueError("MISSING_GAUGE_PAIR")
        if len(ref_receipts)!=12 or len(gauge_receipts)!=32 or len(continue_receipts)!=32:raise ValueError("P2_COUNTS_INCOMPLETE")
        write_json(output/"reference_receipts.json",clean(ref_receipts));write_json(output/"gauge_receipts.json",gauge_receipts);write_json(output/"continue_receipts.json",continue_receipts)
        summary=summarize(output,output/"derived",config)
        from .plots import render
        figures=render(root,output,output/"derived/figures",config,summary);budget.check();assert_source(root,frozen)
        npz_bytes=sum(p.stat().st_size for p in output.rglob("*.npz"))
        if npz_bytes>config["budget"]["raw_checkpoint_target_bytes"]:raise RuntimeError("P2_RAW_TARGET_EXCEEDED")
        counts={"primary_planned":128,"primary_started":128,"primary_completed":len(completed),"primary_failed":0,"primary_missing":0,
                "new_RK4_planned":16,"new_RK4_completed":ba["new_RK4_integrations"]+len(ref_receipts),
                "new_RK4_failed":0,"new_RK4_missing":0,"algebraic_time_evaluations":ba["algebraic_time_evaluations"],
                "P0_regression":tests[0]["receipt"]["unique_tests"],"P1_regression":tests[1]["receipt"]["unique_tests"],"P2_unique_tests":tests[2]["receipt"]["unique_tests"],
                "new_RK4_step_halving_pairs":8,"new_primary_rhs":sum(x["solver"]["rhs_calls"] for x in completed),
                "new_primary_accepted_steps":sum(x["solver"]["accepted_steps"] for x in completed),"new_primary_rejected_steps":None,
                "new_RK4_rhs":sum(x["solver"]["nfev"] for x in ref_receipts)+sum(x["solver"]["nfev"] for x in ba["references"])}
        status={"execution_status":"COMPLETE","implementation_status":"PASS","empirical_findings":"see derived summary and P2_REPORT; no direction-effect requirement", "counts":counts,
                "npz_bytes":npz_bytes,"figures":figures,"meter":meter.receipt(),
                "children_resource":{"user_seconds":resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,"system_seconds":resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
                                     "peak_rss_kib":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss},
                "source_bundle_sha256":frozen["bundle_sha256"],"own_source_bundle_sha256":frozen["own_bundle_sha256"],"GPU":0,"model_requests":0,"servers":0,"installs":0,"git_writes":0,"P3":"NOT_RUN"}
        write_json(output/"status.json",status)
        files=seal_files(output,exclude=("run_manifest.json",));write_json(output/"run_manifest.json",{"files":files,"bundle_sha256":digest(canonical(files))})
        budget.finish("COMPLETE");print("P2 execution COMPLETE / implementation PASS",flush=True)
        return status
    except BaseException as exc:
        status={"execution_status":"PARTIAL","implementation_status":"BLOCKED","exception":type(exc).__name__,"message":str(exc),
                "completed_primary_keys":[x["key"] for x in completed],"planned_primary":128,
                "primary_started":len([x for x in journal_rows if x["kind"]=="primary" and x["status"]=="STARTED"]),
                "completed_branch_RK4":len(ref_receipts),"meter":meter.receipt(),"utc":utc()}
        write_json(output/"failure.json",status);budget.finish("PARTIAL")
        raise
