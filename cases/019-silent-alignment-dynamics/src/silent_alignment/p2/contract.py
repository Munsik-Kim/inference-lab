"""Pinned parents, explicit new source and paired intervention inputs."""
from pathlib import Path
import shutil
import time
import numpy as np
from ..io import canonical,digest,entry,read_json,write_json,utc,check_files
from ..p1.contract import source_bundle as p1_bundle,load_inputs
from .core import rotations,transform,matching

P1_COMBINED="69e91d0bb8b0cff7dd6ad6ca9738e2f6a5d79fd122ec5c06c2de6e71211017e9"
P1_OWN="9cb1c834cb026ddb14f08f71e811a0e84eeafe021fd68a2a1c24c8f6c3175338"
P1_INPUT="d582da45a1a1c4753b850e24642e6f0e8bd2a47a4de88abdfa0b6679cba77883"
ARMS=("CONTINUE","GAUGE","ALIGN","ROTATE")


def source_bundle(root):
    root=Path(root);old=p1_bundle(root)
    if old["bundle_sha256"]!=P1_COMBINED or old["own_bundle_sha256"]!=P1_OWN:
        raise ValueError("P1_PINNED_SOURCE_CHANGED")
    paths=sorted([*root.glob("src/silent_alignment/p2/*.py"),*root.glob("p2_tests/*.py"),
                  root/"configs/p2.json",root/"P2_PROTOCOL.md",root/"P2_SOURCE_MAP.md"])
    own=[entry(p,root) for p in paths]
    rows=sorted(own+old["files"],key=lambda x:x["path"])
    return {"own_files":own,"read_only_p0_p1_files":old["files"],"files":rows,
            "own_bundle_sha256":digest(canonical(own)),"bundle_sha256":digest(canonical(rows)),
            "p1_combined_bundle_sha256":P1_COMBINED,"p1_own_bundle_sha256":P1_OWN}


def assert_source(root,frozen):
    actual=source_bundle(root)
    if actual["files"]!=frozen["files"] or actual["bundle_sha256"]!=frozen["bundle_sha256"]:
        raise ValueError("P2_SOURCE_OR_CONFIG_CHANGED")


def grid():
    t=np.unique(np.r_[np.arange(1001,dtype=np.int64)*.02,.25,
                      np.round(20+np.arange(1,676,dtype=np.int64)*.2,12)])
    t=np.round(t,12)
    if len(t)!=1677 or t[0]!=0 or t[-1]!=155 or not np.all(np.diff(t)>0):
        raise ValueError("P2_GRID_CONTRACT")
    return t


def checkpoints(tstar):
    common=[0,.25,.5,1,2,5,10,20,40,80,155]
    old=[0,.25,.5,1,2,5,10,20,40,80,160]
    return sorted(set(common+[round(t-tstar,12) for t in old if 0<=t-tstar<=155]))


def time_join(tau,old_times,tstar):
    # Contract grids have exact hundredth ticks. This is an exact rational-grid
    # join, not a nearest-neighbour tolerance or an interpolated old value.
    new=np.rint((tau+tstar)*100).astype(np.int64)
    old=np.rint(old_times*100).astype(np.int64)
    if np.max(np.abs(new/100-(tau+tstar)))>1e-10 or np.max(np.abs(old/100-old_times))>1e-10:
        raise ValueError("NONCONTRACT_TIME_GRID")
    ix={int(v):i for i,v in enumerate(old)}
    pairs=[(i,ix[int(v)]) for i,v in enumerate(new) if int(v) in ix]
    return np.array([x[0] for x in pairs],dtype=int),np.array([x[1] for x in pairs],dtype=int)


def planned(config):
    parents=[{"parent_key":f"g{gi}_R{rep}_t{t}","p1_key":f"g{gi}_rho2_R{rep}",
              "g_index":gi,"g":g,"rho":config["rho"],"R_index":rep,"t_star":t,
              "R_stream":[61023020,1,rep],"checkpoints":checkpoints(t)}
             for gi,g in zip([0,2],config["gaps"]) for rep in range(8) for t in config["t_stars"]]
    # Parent-block randomization permits a transient all-grid factor comparison
    # of CONTINUE/GAUGE without saving ~1.7GiB of full paths. Every branch order
    # is fixed before output; no result-dependent reordering.
    rng=np.random.default_rng(config["order_seed"]);result=[]
    for pi in rng.permutation(len(parents)):
        p=parents[int(pi)]
        for ai in rng.permutation(4):
            arm=ARMS[int(ai)];row=dict(p,arm=arm,key=p["parent_key"]+"_"+arm)
            row["order_index"]=len(result);row["attempt_id"]=f"p2-primary-{len(result):03d}";result.append(row)
    if len(result)!=128 or len({x["key"] for x in result})!=128:raise ValueError("P2_KEYS")
    return parents,result


def prepare(root,config,output):
    p1=root/config["parent_run"]
    if digest((p1/"input_manifest.json").read_bytes())!=P1_INPUT:raise ValueError("P1_INPUT_IDENTITY")
    inputs=read_json(p1/"input_manifest.json");rows={r["key"]:r for r in inputs["planned_trajectories"]}
    parents,order=planned(config);assets=output/"assets";assets.mkdir()
    rot=rotations(config["rotation_seed"])
    np.savez_compressed(assets/"rotations.npz",**rot)
    shutil.copyfile(p1/"assets/fixed_basis.npz",assets/"fixed_basis.npz")
    for gi in [0,2]:shutil.copyfile(p1/f"assets/target_g{gi}.npz",assets/f"target_g{gi}.npz")
    for rep in range(8):shutil.copyfile(p1/f"assets/R{rep}.npz",assets/f"R{rep}.npz")
    receipt=[]
    for parent in parents:
        key=parent["p1_key"];r=rows[key];C,left,right,sigma,R=load_inputs(p1,r)
        ck=p1/"checkpoints"/(key+".npz")
        with np.load(ck,allow_pickle=False) as a:
            ix=np.flatnonzero(a["times"]==parent["t_star"])
            if len(ix)!=1:raise ValueError("MISSING_EXACT_PARENT_CHECKPOINT")
            U,V=a["U"][ix[0]],a["V"][ix[0]]
        branch={}
        for arm in ARMS:
            up,vp=transform(U,V,left,right,arm,rot)
            m=matching(C,left,right,sigma,U,V,up,vp,arm,rot,parent["rho"],config["tolerances"])
            if not m["pass"]:raise ValueError("INTERVENTION_MATCHING_FAILED: "+parent["parent_key"]+arm)
            branch[arm]=m
            np.savez_compressed(assets/(parent["parent_key"]+"_"+arm+".npz"),U=up,V=vp)
        parent.update(parent_checkpoint=entry(ck,p1),loss_orig0=r["loss_0"],loss_min=r["loss_min"],
                      loss_denominator=r["loss_denominator"],matching=branch,
                      parent_U_sha256=digest(U.tobytes()),parent_V_sha256=digest(V.tobytes()))
        receipt.append({"parent_key":parent["parent_key"],"arms":branch})
    for row in order:
        p=next(p for p in parents if p["parent_key"]==row["parent_key"])
        row.update(loss_orig0=p["loss_orig0"],loss_min=p["loss_min"],loss_denominator=p["loss_denominator"],
                   input_asset=row["key"]+".npz",input_sha256=digest((assets/(row["key"]+".npz")).read_bytes()))
    manifest={"created_utc":utc(),"initial_transform_metrics_already_computed":True,
              "p2_primary_ode_outputs_observed":0,"p1_primary_outputs_previously_observed":81,
              "selection":"R2 delay83.061701 and re-exit seen in P1; fixed t*=1,5, gaps .4,.025; posthoc selection disclosed",
              "p1_input_manifest_sha256":P1_INPUT,"parents":parents,"planned_branches":order,
              "rotation_streams":{"O":[61023022,1],"O_L":[61023022,2],"O_R":[61023022,3]},
              "order_rule":"seed61023023 parent permutation then four-arm permutation within each parent block",
              "assets":[entry(p,assets) for p in sorted(assets.iterdir())]}
    write_json(output/"input_manifest.json",manifest)
    write_json(output/"input_seal.json",{"created_utc":utc(),"p2_primary_outputs_observed":0,
               "input_manifest":entry(output/"input_manifest.json",output),"assets":manifest["assets"]})
    return manifest


def load_branch(output,row):
    with np.load(output/"assets/fixed_basis.npz",allow_pickle=False) as a:left,right=a["left"],a["right"]
    with np.load(output/f"assets/target_g{row['g_index']}.npz",allow_pickle=False) as a:C,sigma=a["C"],a["sigma"]
    with np.load(output/"assets"/row["input_asset"],allow_pickle=False) as a:U,V=a["U"],a["V"]
    return C,left,right,sigma,U,V


class WorkerBudget:
    def __init__(self,path,limit,prior_campaign_debit,launch_time):
        self.path=Path(path);self.start=launch_time;self.limit=limit;self.prior=prior_campaign_debit
        if self.path.exists():
            record=read_json(self.path)
            if any(x["status"]=="RUNNING" for x in record["attempts"]):raise ValueError("UNRESOLVED_P2_ATTEMPT")
        else:record={"p1_campaign_debit_seconds":prior_campaign_debit,"campaign_limit_seconds":1800,"p2_limit_seconds":limit,"attempts":[]}
        if record["p1_campaign_debit_seconds"]!=prior_campaign_debit:raise ValueError("CAMPAIGN_DEBT_CHANGED")
        self.debt=sum(x["wall_seconds"] for x in record["attempts"])
        self.record=record;record["attempts"].append({"started_utc":utc(),"status":"RUNNING","prior_p2_debt_seconds":self.debt})
        write_json(self.path,record);self.check()
    def check(self):
        if time.monotonic()-self.start+self.debt>=self.limit:raise TimeoutError("P2_WORKER_BUDGET_EXHAUSTED")
    def finish(self,status):
        self.record["attempts"][-1].update(status=status,ended_utc=utc(),wall_seconds=time.monotonic()-self.start)
        consumed=sum(x["wall_seconds"] for x in self.record["attempts"])
        self.record.update(p2_consumed_seconds=consumed,campaign_consumed_seconds=self.prior+consumed,
                           campaign_remaining_seconds=1800-self.prior-consumed)
        write_json(self.path,self.record)
