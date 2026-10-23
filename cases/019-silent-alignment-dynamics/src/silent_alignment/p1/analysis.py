"""P1 raw reductions, explicit censoring, paired contrasts and mode fits."""

import csv
from itertools import combinations
from pathlib import Path
import numpy as np

from ..analysis import crossing
from ..io import read_json,write_json

EVENTS = ["T_align_half","T_visible10","T_worst_half","T_mode10_1","T_mode10_32"]
VALUES = EVENTS+["S"]


def event(times,values,threshold,already=False):
    result=crossing(times,values,threshold,already)
    result["observed_end"]=float(times[-1])
    result["bracket_width"]=(result["bracket"][1]-result["bracket"][0]) if "bracket" in result else None
    return result


def summarize_raw(meta,raw,sigma,config,attempt):
    t=raw["times"]
    aligned=meta["initialization"]=="aligned"
    if aligned and (abs(raw["right_distance"][0])>config["tolerances"]["aligned_distance"] or
                    abs(raw["right_worst"][0])>config["tolerances"]["aligned_worst_absolute"]):
        raise ValueError("ALIGNED_START_NOT_ALIGNED")
    results={"T_align_half":event(t,raw["right_distance"],.5*raw["right_distance"][0],aligned),
             "T_visible10":event(t,raw["learnable_residual"],.9),
             "T_worst_half":event(t,raw["right_worst"],.5*raw["right_worst"][0],aligned)}
    for mode in (1,32):
        results[f"T_mode10_{mode}"]=event(t,-raw["target_mode_signed"][:,mode-1]/sigma[mode-1],-.1)
    separation=(results["T_visible10"]["time"]-results["T_align_half"]["time"] if
                results["T_visible10"]["time"] is not None and results["T_align_half"]["time"] is not None else None)
    return {**{k:meta[k] for k in ["key","attempt_id","order_index","initialization","g","rho","g_index","rho_index","R_index","R_stream","amplitude","loss_0","loss_min","loss_denominator"]},
            "execution_status":attempt.get("status","MISSING"),"numerical_status":attempt.get("numerical_status","PENDING"),
            "sigma1":float(sigma[0]),"sigma32":float(sigma[31]),"S":separation,"events":results,
            "primary_and_metrics_wall_seconds":attempt.get("primary_and_metrics_cost",{}).get("wall_seconds"),
            "unit_including_references_wall_seconds":attempt.get("unit_cost",{}).get("wall_seconds"),
            "loss_end":float(raw["loss"][-1]),"residual_end":float(raw["learnable_residual"][-1]),
            "right_distance_initial":float(raw["right_distance"][0]),"right_distance_end":float(raw["right_distance"][-1]),
            "right_worst_initial":float(raw["right_worst"][0]),"right_worst_end":float(raw["right_worst"][-1]),
            "max_imbalance_absolute":float(np.max(raw["imbalance_absolute"])),
            "max_imbalance_relative_gram":float(np.max(raw["imbalance_relative_gram"])),
            "max_normalized_loss_increase":float(np.max(np.diff(raw["loss"]))/meta["loss_0"]),
            "max_cosine_overshoot":float(max(np.nanmax(raw["left_cosine_max_overshoot"]),np.max(raw["right_cosine_max_overshoot"])))}


def value(row,metric):
    if row.get("execution_status")!="COMPLETED":return None
    return row.get("S") if metric=="S" else row.get("events",{}).get(metric,{}).get("time")


def descriptives(values,total):
    finite=[x for x in values if x is not None]
    arr=np.asarray(finite,dtype=float)
    return {"reached":len(finite),"total":total,"not_reached_or_unavailable":total-len(finite),
            "mean":float(np.mean(arr)) if len(arr) else None,
            "median":float(np.median(arr)) if len(arr) else None,
            "sample_sd":float(np.std(arr,ddof=1)) if len(arr)>1 else None,
            "min":float(np.min(arr)) if len(arr) else None,"max":float(np.max(arr)) if len(arr) else None,
            "scope":"conditional on available/reached events; no censored event-time substitution"}


def grouping(rows,plan,config):
    groups=[]
    lookup={r["key"]:r for r in rows}
    for g in config["gaps"]:
        for rho in config["rhos"]:
            for init,den in [("random",8),("aligned",1)]:
                keys=[r["key"] for r in plan if r["g"]==g and r["rho"]==rho and r["initialization"]==init]
                present=[lookup[k] for k in keys if k in lookup]
                stats={metric:descriptives([value(r,metric) for r in present],den) for metric in VALUES}
                for metric in EVENTS:
                    stats[metric].update(not_reached=sum(r.get("events",{}).get(metric,{}).get("status")=="NOT_REACHED" for r in present),
                                         execution_failed=sum(r["execution_status"]=="FAILED" for r in present),
                                         missing=sum(r["execution_status"] in ["MISSING","STARTED"] for r in present))
                groups.append({"g":g,"rho":rho,"initialization":init,"denominator":den,"statistics":stats})
    return groups


def paired(rows,config):
    lookup={(r["g"],r["rho"],r["initialization"],r["R_index"]):r for r in rows}
    output=[]
    for axis in ("rho","gap","random_minus_aligned"):
        contrasts=[]
        if axis=="rho":
            for g in config["gaps"]:
                for before,after in combinations(config["rhos"],2):
                    contrasts.append((f"g={g}; rho {after:g} minus {before:g}",[(g,before,"random",r) for r in range(8)],[(g,after,"random",r) for r in range(8)]))
        elif axis=="gap":
            for rho in config["rhos"]:
                for before,after in combinations(config["gaps"],2):
                    contrasts.append((f"rho={rho:g}; g {after:g} minus {before:g}",[(before,rho,"random",r) for r in range(8)],[(after,rho,"random",r) for r in range(8)]))
        else:
            for g in config["gaps"]:
                for rho in config["rhos"]:
                    contrasts.append((f"g={g}; rho={rho:g}; random minus aligned",[(g,rho,"aligned",None)]*8,[(g,rho,"random",r) for r in range(8)]))
        for label,before_keys,after_keys in contrasts:
            individual=[]
            for rep,(bk,ak) in enumerate(zip(before_keys,after_keys)):
                b,a=lookup[bk],lookup[ak]
                delta={m:(value(a,m)-value(b,m) if value(a,m) is not None and value(b,m) is not None else None) for m in VALUES}
                individual.append({"R_index":rep,"before_key":b["key"],"after_key":a["key"],"difference":delta,
                                   "unavailable":[m for m in VALUES if delta[m] is None]})
            output.append({"axis":axis,"contrast":label,"denominator_paired_R":8,"individual":individual,
                           "statistics":{m:descriptives([r["difference"][m] for r in individual],8) for m in VALUES},
                           "aligned_scope":"one shared deterministic control, not eight measurements" if axis=="random_minus_aligned" else None})
    return output


def mode_fits(rows,config):
    lookup={(r["g"],r["rho"],r["initialization"],r["R_index"]):r for r in rows}
    fits=[]
    for g in config["gaps"]:
        for init,reps in [("aligned",[None]),("random",range(8))]:
            for rep in reps:
                selected=[lookup[g,rho,init,rep] for rho in config["rhos"]]
                for mode in (1,32):
                    x=np.log(1/np.asarray(config["rhos"]))
                    values=[value(r,f"T_mode10_{mode}") for r in selected]
                    row={"g":g,"initialization":init,"R_index":rep,"mode":mode,
                         "keys":[r["key"] for r in selected],"log_inverse_rho":x.tolist(),"times":values,
                         "sigma_actual":selected[0][f"sigma{mode}"],
                         "predicted_leading_coefficient":1/(2*selected[0][f"sigma{mode}"]),
                         "fit_scope":"three finite rho points; not an asymptotic-law certification"}
                    if any(v is None for v in values):
                        row.update(status="INCOMPLETE_OR_CENSORED_NO_FIT",slope=None,intercept=None,residuals=None)
                    else:
                        slope,intercept=np.polyfit(x,np.asarray(values),1)
                        row.update(status="FIT_THREE_REACHED_POINTS",slope=float(slope),intercept=float(intercept),
                                   residuals=(np.asarray(values)-(slope*x+intercept)).tolist())
                    fits.append(row)
    return fits


def csv_write(path,columns,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def regenerate(run,output):
    run,output=Path(run),Path(output)
    config=read_json(run/"config.json")
    inputs=read_json(run/"input_manifest.json");plan=inputs["planned_trajectories"]
    ledger=read_json(run/"attempt_ledger.json");attempts={r["key"]:r for r in ledger}
    rows=[]
    for meta in plan:
        path=run/"raw"/(meta["key"]+".npz")
        attempt=attempts.get(meta["key"],{"status":"MISSING"})
        with np.load(run/"assets"/meta["target_asset"],allow_pickle=False) as a:sigma=a["sigma"]
        if not path.exists():
            rows.append({**meta,"execution_status":attempt["status"],"S":None,"events":{},
                         "sigma1":float(sigma[0]),"sigma32":float(sigma[31]),
                         "numerical_status":attempt.get("numerical_status","UNAVAILABLE")})
            continue
        with np.load(path,allow_pickle=False) as raw:rows.append(summarize_raw(meta,raw,sigma,config,attempt))
    groups=grouping(rows,plan,config);pairs=paired(rows,config);fits=mode_fits(rows,config)
    summary={"schema":"case019-p1-summary-v1","source_bundle_sha256":read_json(run/"execution_freeze.json")["bundle_sha256"],
             "planned":81,"completed":sum(r["execution_status"]=="COMPLETED" for r in rows),
             "rows":rows,"groups":groups,"paired_contrasts":pairs,"mode_time_fits":fits,
             "inference_scope":"conditional-on-fixed-P0-target, repeated initialization; no p-values or population certification"}
    write_json(output/"summary.json",summary)
    columns=["key","attempt_id","order_index","initialization","g","rho","R_index","execution_status","numerical_status",
             "primary_and_metrics_wall_seconds","unit_including_references_wall_seconds","loss_0","loss_min","loss_denominator",
             "S","right_distance_end","right_worst_end"]
    for event_name in EVENTS:
        columns.extend([event_name+"_status",event_name+"_time",event_name+"_bracket_low",event_name+"_bracket_high",event_name+"_bracket_width"])
    flat=[]
    for row in rows:
        r={k:row.get(k) for k in columns}
        for ev in EVENTS:
            er=row.get("events",{}).get(ev,{})
            r.update({ev+"_status":er.get("status","UNAVAILABLE"),ev+"_time":er.get("time"),
                      ev+"_bracket_low":er.get("bracket",[None,None])[0],ev+"_bracket_high":er.get("bracket",[None,None])[1],
                      ev+"_bracket_width":er.get("bracket_width")})
        flat.append(r)
    csv_write(output/"trajectory_ledger.csv",columns,flat)
    gr=[]
    for group in groups:
        for metric,stats in group["statistics"].items():
            gr.append({"g":group["g"],"rho":group["rho"],"initialization":group["initialization"],"metric":metric,
                       **{k:stats.get(k) for k in ["reached","total","not_reached","execution_failed","missing","mean","median","sample_sd","min","max"]}})
    csv_write(output/"groups.csv",list(gr[0]),gr)
    pr=[]
    for p in pairs:
        for individual in p["individual"]:
            pr.append({"axis":p["axis"],"contrast":p["contrast"],"R_index":individual["R_index"],
                       "before_key":individual["before_key"],"after_key":individual["after_key"],**individual["difference"]})
    csv_write(output/"paired_individual.csv",list(pr[0]),pr)
    fit_rows=[{k:f.get(k) for k in ["g","initialization","R_index","mode","sigma_actual","predicted_leading_coefficient","status","slope","intercept"]}|
              {"residuals":str(f["residuals"])} for f in fits]
    csv_write(output/"mode_time_fits.csv",list(fit_rows[0]),fit_rows)
    return summary
