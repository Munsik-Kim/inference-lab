"""Grid-defined persistence and physically evaluated compact branch quantities."""
import csv
import json
import numpy as np
from ..p1.metrics import measure,direction
from ..math import physical_rhs
from ..p1.analysis import event


def persistence(times,values,threshold=.1,duration=10.):
    t=np.asarray(times);v=np.asarray(values)
    if len(t)!=len(v) or len(t)<2 or not np.all(np.diff(t)>0) or not np.isfinite(v).all():
        raise ValueError("INVALID_PERSISTENCE_SERIES")
    above=v>=threshold;ups=[];downs=[];intervals=[]
    start=float(t[0]) if above[0] else None
    for i in range(1,len(t)):
        if above[i]==above[i-1]:continue
        crossing=float(t[i-1]+(threshold-v[i-1])*(t[i]-t[i-1])/(v[i]-v[i-1]))
        r={"time":crossing,"bracket":[float(t[i-1]),float(t[i])],"bracket_width":float(t[i]-t[i-1])}
        if above[i]:ups.append(r);start=crossing
        else:
            downs.append(r);intervals.append({"start":start,"end":crossing,"right_censored":False});start=None
    if start is not None:intervals.append({"start":start,"end":float(t[-1]),"right_censored":True})
    last=intervals[-1] if intervals and intervals[-1]["right_censored"] else None
    sustained=next((x for x in intervals if x["end"]-x["start"]>=duration),None)
    first={"status":"ALREADY_ABOVE" if above[0] else ("REACHED" if ups else "NOT_REACHED"),
           "time":float(t[0]) if above[0] else (ups[0]["time"] if ups else None),
           "bracket":[float(t[0]),float(t[0])] if above[0] else (ups[0]["bracket"] if ups else None)}
    return {"first_entry":first,"up_entries":ups,"down_exits":downs,"up_entry_count":len(ups),
            "down_exit_count":len(downs),"intervals":intervals,"observed_end":float(t[-1]),
            "last_retained_entry":None if last is None else last["start"],
            "last_observed_duration":None if last is None else last["end"]-last["start"],
            "first_sustained10":None if sustained is None else {"start":sustained["start"],"end":sustained["start"]+duration,
               "containing_observed_interval_end":sustained["end"]},
            "end_only_entry":bool(last and last["start"]==float(t[-1])),
            "insufficient_remaining_for10":bool(last and last["end"]-last["start"]<duration),
            "definition":"piecewise-linear saved grid >=.1, observed duration only; no assertion between actual continuous solver samples or after horizon"}


def compact(C,left,right,sigma,U,V,rho,tol,budget):
    scale=np.sqrt(2*rho);initial_imb=V[0].T@V[0]-U[0].T@U[0];columns={}
    # This schema drops duplicated full eigenvalue arrays; traces, Gram norms,
    # cosine vectors, ranks and both absolute/relative drift remain explicit.
    keep=["loss","learnable_residual","z_frobenius","target_mode_signed",
          "gram_u_trace","gram_v_trace","gram_u_frobenius","gram_v_frobenius","gram_cross_trace",
          "imbalance_absolute","imbalance_relative_gram"]
    for side in ["left","right"]:
        keep += [side+"_"+name for name in ["distance","cosines","theta_max","worst","rank","condition","cosine_max_overshoot","cosine_out_of_range_count"]]
    for i,(u,v) in enumerate(zip(U,V)):
        if i%16==0:budget.check()
        row=measure(C,left,right,sigma,u/scale,v/scale,rho,tol)
        for k in keep:columns.setdefault(k,[]).append(row[k])
        columns.setdefault("z_singular_sorted",[]).append(row["z_singular_sorted"][:U.shape[2]])
        gu,gv=u.T@u,v.T@v;delta=gv-gu-initial_imb
        columns.setdefault("observed_imbalance_drift_absolute",[]).append(float(np.linalg.norm(delta)))
        columns.setdefault("observed_imbalance_drift_relative_gram",[]).append(float(np.linalg.norm(delta)/(np.linalg.norm(gu)+np.linalg.norm(gv))))
        du,dv=physical_rhs(C,u,v)
        columns.setdefault("loss_rate",[]).append(-float(np.sum(du*du)+np.sum(dv*dv)))
    return {k:np.asarray(v) for k,v in columns.items()}


def compare(C,left,right,sigma,U,V,RU,RV,tol,budget):
    den=.5*np.sum(C*C)-.5*np.sum(sigma[U.shape[2]:]**2)
    columns={};scale=1. # All inputs physical, reference uses original U/V ODE.
    for i,(u,v,ru,rv) in enumerate(zip(U,V,RU,RV)):
        if i%16==0:budget.check()
        z,zr=u@v.T,ru@rv.T
        row={"z_absolute":np.linalg.norm(z-zr),"z_reference_norm":np.linalg.norm(zr),
             "normalized_loss_difference":abs(.5*np.sum((z-C)**2)-.5*np.sum((zr-C)**2))/den,
             "reference_loss":.5*np.sum((zr-C)**2),"reference_mode32":left[:,31]@zr@right[:,31],
             "reference_mode33":left[:,32]@zr@right[:,32]}
        for side,f,rf,target in (("right",v,rv,right[:,:v.shape[1]]),("left",u,ru,left[:,:u.shape[1]])):
            a,b=direction(f,target,tol),direction(rf,target,tol)
            for metric in ["distance","worst"]:
                row[side+"_"+metric+"_reference"]=b[metric]
                row[side+"_"+metric+"_difference"]=abs(a[metric]-b[metric])
        for k,vv in row.items():columns.setdefault(k,[]).append(vv)
    arrays={k:np.asarray(v) for k,v in columns.items()}
    from ..p1.metrics import comparison_status
    return arrays,comparison_status(arrays,tol)


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False,separators=(",",":")) if isinstance(v,(dict,list,np.ndarray)) else v for k,v in r.items()})


def describe(values,total=8):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    return {"total":total,"valid":len(x),"missing_or_not_reached":total-len(x),
            "mean":float(np.mean(x)) if len(x) else None,"median":float(np.median(x)) if len(x) else None,
            "sample_sd":float(np.std(x,ddof=1)) if len(x)>1 else None,
            "min":float(np.min(x)) if len(x) else None,"max":float(np.max(x)) if len(x) else None}
