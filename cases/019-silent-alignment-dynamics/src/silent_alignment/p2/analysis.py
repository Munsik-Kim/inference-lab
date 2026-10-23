"""Same-parent effects with a common target loss denominator."""
from pathlib import Path
import numpy as np
from ..io import read_json,write_json
from ..p1.analysis import event
from .diagnostics import persistence,csv_write,clean,describe


def effect(lc0,la0,lct,lat,denominator):
    j=lc0-la0;g=(la0-lat)-(lc0-lct);a=lct-lat
    if abs((j+g)-a)>1e-12*(1+abs(lct)+abs(lat)):raise ValueError("J_G_A_IDENTITY")
    return {"J":float(j),"G":float(g),"A":float(a),
            "J_normalized":float(j/denominator),"G_normalized":float(g/denominator),"A_normalized":float(a/denominator)}


def summarize(run,output,config):
    output=Path(output);output.mkdir(exist_ok=False,parents=True)
    manifest=read_json(run/"input_manifest.json");ledger=[];fixed=[];groups=[];events=[]
    for row in manifest["planned_branches"]:
        key=row["key"];meta=read_json(run/"completion"/(key+".json"))
        with np.load(run/"raw"/(key+".npz"),allow_pickle=False) as a:
            t=a["times"];aligned=row["arm"]=="ALIGN"
            ev={"right_half":event(t,a["right_distance"],.5*a["right_distance"][0],already=aligned),
                "worst_half":event(t,a["right_worst"],.5*a["right_worst"][0],already=aligned)}
            with np.load(run/f"assets/target_g{row['g_index']}.npz",allow_pickle=False) as b:sigma=b["sigma"]
            for mode in [1,32,33]:
                ev[f"mode{mode}"]=persistence(t,a["target_mode_signed"][:,mode-1]/sigma[mode-1])
            record={"key":key,"parent_key":row["parent_key"],"g":row["g"],"rho":row["rho"],"R_index":row["R_index"],
                    "t_star":row["t_star"],"arm":row["arm"],"order_index":row["order_index"],"status":meta["status"],
                    "solver":meta["solver"],"cost":meta["cost"],"events":ev,
                    "initial_loss":float(a["loss"][0]),"final_loss":float(a["loss"][-1]),
                    "final_residual":float(a["learnable_residual"][-1]),
                    "max_nominal_imbalance_absolute":float(np.max(a["imbalance_absolute"])),
                    "max_nominal_imbalance_relative_gram":float(np.max(a["imbalance_relative_gram"])),
                    "max_observed_imbalance_drift_absolute":float(np.max(a["observed_imbalance_drift_absolute"])),
                    "max_observed_imbalance_drift_relative_gram":float(np.max(a["observed_imbalance_drift_relative_gram"]))}
            ledger.append(record);events.append({"key":key,"events":ev})
            with np.load(run/"raw"/(row["parent_key"]+"_CONTINUE.npz"),allow_pickle=False) as c:
                if not np.array_equal(t,c["times"]):raise ValueError("PAIRED_GRID_DIFFERENCE")
                for tau in config["time"]["fixed_effect_times"]:
                    ix=np.flatnonzero(t==tau)
                    if len(ix)!=1:raise ValueError("FIXED_TIME_MISSING")
                    i=ix[0];value=effect(float(c["loss"][0]),float(a["loss"][0]),float(c["loss"][i]),float(a["loss"][i]),row["loss_denominator"])
                    if tau==0 and value["G"]!=0:raise ValueError("G_ZERO_AT_ORIGIN")
                    fixed.append({"key":key,"parent_key":row["parent_key"],"g":row["g"],"t_star":row["t_star"],
                                  "R_index":row["R_index"],"arm":row["arm"],"tau":tau,"absolute_time":tau+row["t_star"],
                                  "loss":float(a["loss"][i]),"loss_denominator":row["loss_denominator"],**value,
                                  "right_distance":float(a["right_distance"][i]),"right_worst":float(a["right_worst"][i]),
                                  "right_distance_minus_continue":float(a["right_distance"][i]-c["right_distance"][i]),
                                  "right_worst_minus_continue":float(a["right_worst"][i]-c["right_worst"][i]),
                                  "loss_rate":float(a["loss_rate"][i]),
                                  **{f"mode{j}_fraction":float(a["target_mode_signed"][i,j-1]/sigma[j-1]) for j in [1,32,33]}})
    for g in config["gaps"]:
        for star in config["t_stars"]:
            for arm in ["CONTINUE","GAUGE","ALIGN","ROTATE"]:
                rr=[r for r in ledger if (r["g"],r["t_star"],r["arm"])==(g,star,arm)]
                if len(rr)!=8 or {r["R_index"] for r in rr}!=set(range(8)):raise ValueError("GROUP_PAIRING_DENOMINATOR")
                for tau in config["time"]["fixed_effect_times"]:
                    xx=[r for r in fixed if (r["g"],r["t_star"],r["arm"],r["tau"])==(g,star,arm,tau)]
                    metrics=["J","G","A","J_normalized","G_normalized","A_normalized","right_distance","right_worst",
                             "right_distance_minus_continue","right_worst_minus_continue","loss_rate","mode1_fraction","mode32_fraction","mode33_fraction"]
                    groups.append({"g":g,"t_star":star,"arm":arm,"tau":tau,"R_repetitions":8,"failed":0,"missing":0,
                                   "statistics":{k:describe([r[k] for r in xx]) for k in metrics}})
    event_groups=[]
    for g in config["gaps"]:
        for star in config["t_stars"]:
            for arm in ["CONTINUE","GAUGE","ALIGN","ROTATE"]:
                rr=[r for r in ledger if (r["g"],r["t_star"],r["arm"])==(g,star,arm)]
                for name in ["right_half","worst_half","mode1","mode32","mode33"]:
                    values=[r["events"][name]["first_entry"]["time"] if name.startswith("mode") else r["events"][name]["time"] for r in rr]
                    event_groups.append({"g":g,"t_star":star,"arm":arm,"event":name,**describe(values),
                                         "interpretation":"reached-only descriptive times; non-reached null, not filled with155"})
    summary={"trajectory_count":len(ledger),"parent_count":len(manifest["parents"]),"random_R_count":8,
             "ledger":ledger,"fixed_time_paired":fixed,"groups":groups,"event_groups":event_groups,
             "counting":"128 branches from32 parents of8 shared R, not128 independent datasets; differences are paired before grouping",
             "effect_definition":"A=J+G; common original-target L0-Lmin denominator; J is instantaneous construction benefit"}
    write_json(output/"summary.json",clean(summary));write_json(output/"events.json",clean(events))
    csv_write(output/"trajectory_ledger.csv",ledger);csv_write(output/"fixed_time_paired.csv",fixed)
    csv_write(output/"groups.csv",groups);csv_write(output/"event_groups.csv",event_groups)
    return summary
