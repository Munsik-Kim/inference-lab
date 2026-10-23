"""Four compact figure families from saved P1 observations only."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..io import read_json


def figures(run,output):
    run,output=Path(run),Path(output)
    out=output/"figures";out.mkdir(exist_ok=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"svg.hashsalt":"case019-p1-v1",
                         "axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":140})
    summary=read_json(output/"summary.json");config=read_json(run/"config.json")
    plan=read_json(run/"input_manifest.json")["planned_trajectories"]
    lookup={(r["g"],r["rho"],r["initialization"],r["R_index"]):r for r in summary["rows"]}
    colors=["#175d9e","#be5118","#217c50"]

    def raw(g,rho,init,rep,fields):
        r=lookup[g,rho,init,rep]
        with np.load(run/"raw"/(r["key"]+".npz"),allow_pickle=False) as a:
            return {k:a[k] for k in fields}

    def save(fig,name):
        fig.savefig(out/(name+".png"),metadata={"Software":"Case019 P1"})
        fig.savefig(out/(name+".svg"),metadata={"Date":None,"Creator":"Case019 P1"})
        plt.close(fig)

    fig,axes=plt.subplots(3,2,figsize=(12,10),layout="constrained")
    for ri,rho in enumerate(config["rhos"]):
        data=[raw(.4,rho,"random",rep,["times","right_distance","learnable_residual","target_mode_signed"]) for rep in range(8)]
        t=data[0]["times"];keep=t<=20
        aligned=raw(.4,rho,"aligned",None,["learnable_residual","target_mode_signed"])
        target=next(p for p in plan if p["g"]==.4 and p["rho"]==rho)
        with np.load(run/"assets"/target["target_asset"],allow_pickle=False) as a:sigma=a["sigma"]
        dr=np.median([x["right_distance"] for x in data],axis=0)
        loss=np.median([x["learnable_residual"] for x in data],axis=0)
        for col,x in enumerate((t,t/np.log(1/rho))):
            axes[0,col].plot(x[keep],dr[keep],color=colors[ri],label=f"rho={rho:g}, random median n=8")
            for mode,style in [(0,"-"),(31,":")]:
                y=np.median([d["target_mode_signed"][:,mode]/sigma[mode] for d in data],axis=0)
                axes[1,col].plot(x[keep],y[keep],color=colors[ri],ls=style,label=f"rho={rho:g}, mode {mode+1}")
            axes[2,col].plot(x[keep],loss[keep],color=colors[ri],label=f"rho={rho:g}, random n=8")
            axes[2,col].plot(x[keep],aligned["learnable_residual"][keep],color=colors[ri],ls="--",label=f"rho={rho:g}, aligned n=1")
    for col in (0,1):
        axes[0,col].set(ylabel="Mean-direction distance D_R",title="Boundary gap g=0.4")
        axes[1,col].set(ylabel="Signed target coefficient / sigma_i")
        axes[2,col].set(ylabel="Learnable loss residual")
        for ax in axes[:,col]:
            ax.set_xlabel("Gradient-flow time t" if col==0 else "t / log(1/rho)")
            ax.legend(fontsize=7,ncol=1 if col==0 else 2)
    fig.suptitle("Paired initialization-size comparison | 8 random R samples per cell; unique aligned control\n"
                 "Pointwise medians are summaries, not additional trajectories; top modes 1 and 32")
    save(fig,"01_rho_direction_modes_loss")

    fig,axes=plt.subplots(2,2,figsize=(12,7),layout="constrained")
    for gi,g in enumerate(config["gaps"]):
        data=[raw(g,1e-6,"random",rep,["times","right_distance","right_worst"]) for rep in range(8)]
        t=data[0]["times"]
        for row,metric in enumerate(["right_distance","right_worst"]):
            values=np.asarray([d[metric] for d in data]);mid=np.median(values,axis=0)
            low,high=np.quantile(values,[.25,.75],axis=0)
            for col,end in enumerate([20,160]):
                mask=t<=end
                axes[row,col].plot(t[mask],mid[mask],color=colors[gi],label=f"g={g:g}, median n=8")
                axes[row,col].fill_between(t[mask],low[mask],high[mask],color=colors[gi],alpha=.12)
    for col in (0,1):
        axes[0,col].set(ylabel="Average projector distance D_R",title="Near time (0 to 20)" if col==0 else "Full horizon (0 to 160)")
        axes[1,col].set(ylabel="Worst direction 1 - min(cos)^2")
        for ax in axes[:,col]:ax.set_xlabel("Gradient-flow time t");ax.legend(fontsize=8)
    fig.suptitle("Boundary-pair comparison at rho=1e-6 | same 8 R across gaps\n"
                 "Shading: initialization interquartile range; other 62 target singular values fixed")
    save(fig,"02_gap_average_and_worst")

    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout="constrained",sharey=True)
    for gi,g in enumerate(config["gaps"]):
        ax=axes[gi];nr=0
        for rep in range(8):
            rows=[lookup[g,rho,"random",rep] for rho in config["rhos"]]
            x=np.log10(np.asarray(config["rhos"]))
            y=np.asarray([r["S"] if r["S"] is not None else np.nan for r in rows])
            nr+=int(np.isnan(y).sum())
            ax.plot(x,y,"o-",lw=.8,markersize=4,alpha=.8,label=f"R{rep}")
        ax.axhline(0,color="black",lw=1,ls="--",label="Same event time")
        ax.set(xlabel="log10(rho)",title=f"g={g:g}; available S {24-nr}/24")
        if nr:ax.text(.03,.03,f"Unreached/unavailable S: {nr}\nNo event-time substitution",transform=ax.transAxes)
        ax.set_xticks([-6,-4,-2])
    axes[0].set_ylabel("S = T_visible10 - T_align_half (flow time)")
    axes[-1].legend(fontsize=7,ncol=2)
    fig.suptitle("Event ordering for every paired R | S > 0: mean-direction half-crossing first\n"
                 "Lines connect the same R; censored events remain missing")
    save(fig,"03_separation_by_seed")

    fig,axes=plt.subplots(3,2,figsize=(11,10),layout="constrained")
    for gi,g in enumerate(config["gaps"]):
        for col,mode in enumerate([1,32]):
            ax=axes[gi,col]
            for rep in range(8):
                fit=next(f for f in summary["mode_time_fits"] if f["g"]==g and f["mode"]==mode and f["initialization"]=="random" and f["R_index"]==rep)
                y=[v if v is not None else np.nan for v in fit["times"]]
                ax.plot(fit["log_inverse_rho"],y,"o-",color="0.6",alpha=.5,lw=.7,ms=3,label="Random R, n=8" if rep==0 else None)
            fit=next(f for f in summary["mode_time_fits"] if f["g"]==g and f["mode"]==mode and f["initialization"]=="aligned")
            x=np.asarray(fit["log_inverse_rho"]);y=np.asarray([v if v is not None else np.nan for v in fit["times"]])
            ax.plot(x,y,"o-",color=colors[gi],lw=2,label="Aligned, unique n=1 per rho")
            if np.isfinite(y).all():
                slope=fit["predicted_leading_coefficient"];intercept=float(np.mean(y-slope*x))
                ax.plot(x,slope*x+intercept,"--",color="black",label=f"Leading slope {slope:.3f}; fitted intercept")
            ax.set(xlabel="log(1/rho)",ylabel=f"Time to signed mode {mode} / sigma = 0.1",title=f"g={g:g}, sigma_{mode}={fit['sigma_actual']:.4f}")
            ax.legend(fontsize=7)
    fig.suptitle("Mode activation: observed finite-rho times versus a leading-coefficient prediction\n"
                 "Three rho values; missing times are not fitted or replaced")
    save(fig,"04_mode_activation_times")
