"""Four deterministic scientific figures from stored P1/P2 arrays only."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..io import read_json

COLORS={"CONTINUE":"#245b89","GAUGE":"#8b5fa8","ALIGN":"#18845d","ROTATE":"#c75c29"}


def render(root,run,dest,config,summary):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({"font.size":9,"svg.hashsalt":"case019-p2-v1","axes.grid":True,"grid.alpha":.22})
    def save(fig,name):
        fig.tight_layout();fig.savefig(dest/(name+".png"),dpi=140,metadata={"Software":"Case019 P2 stored-data renderer"})
        fig.savefig(dest/(name+".svg"),metadata={"Date":None,"Creator":"Case019 P2 stored-data renderer"});plt.close(fig)
    p1=root/config["parent_run"]
    fig,axes=plt.subplots(2,2,figsize=(11.5,7))
    for ri,label,color in [(0,"rho=.01","#b54f2a"),(2,"rho=1e-6","#245b89")]:
        key=f"g2_rho{ri}_R2"
        with np.load(p1/"raw"/(key+".npz"),allow_pickle=False) as a:
            t=a["times"]
            with np.load(p1/"assets/target_g2.npz",allow_pickle=False) as b:s=b["sigma"]
            axes[0,0].plot(t,a["target_mode_signed"][:,31]/s[31],label=label,color=color)
            axes[0,1].plot(t,a["target_mode_signed"][:,32]/s[32],label=label,color=color)
            axes[1,0].plot(t,a["right_worst"],label=label,color=color)
            for mode,ax in [(32,axes[0,0]),(33,axes[0,1])]:
                rr=next(r for r in read_json(run/"block_a/mode_persistence.json") if r["key"]==key and r["mode"]==mode)
                if rr["first_sustained10"]:
                    a0=rr["first_sustained10"]["start"];ax.plot([a0,a0+10],[.115,.115],color=color,lw=3,alpha=.65)
        with np.load(run/"block_a/graph"/(key+".npz"),allow_pickle=False) as a:
            axes[1,0].plot(a["times"][::20],a["right_worst"][::20],".",color=color,ms=2,label=label+" graph")
        with np.load(run/"block_a/references"/(key+"_h0.005.npz"),allow_pickle=False) as a:
            axes[1,1].plot(a["times"],a["right_worst_difference"],label=label,color=color)
    for ax in axes[0]:ax.axhline(.1,color="black",ls="--",lw=.8);ax.set_ylim(-.03,1.05)
    axes[0,0].set(xlim=(0,50),ylabel="signed mode32 / sigma32",title="R2: first entry, re-exit and 10-time grid retention")
    axes[0,1].set(xlim=(0,50),ylabel="signed mode33 / sigma33",title="Crossing is not permanent acquisition")
    axes[1,0].set(xlim=(0,160),ylabel="right worst sin² angle",title="R2 long tail: stored ODE and finite-rho graph")
    axes[1,1].set(xlim=(0,160),ylabel="absolute worst-error difference",title="Additional original U/V RK4 h=.005 vs P1")
    for ax in axes.flat:ax.set_xlabel("original gradient-flow time t");ax.legend(fontsize=7)
    save(fig,"01_R2_tail_and_persistence")
    m=read_json(run/"input_manifest.json");p=next(x for x in m["parents"] if x["parent_key"]=="g2_R2_t5")
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for arm,col in COLORS.items():
        v=p["matching"][arm]["metrics"]
        axes[0].plot(np.arange(1,33),v["z_singular_sorted"],color=col,label=arm,ls="--" if arm=="GAUGE" else "-")
    axes[0].set(xlabel="sorted prediction singular-value index",ylabel="singular value",title="Same size at intervention: R2,g=.025,t*=5")
    names=list(COLORS);axes[1].bar(names,[p["matching"][a]["metrics"]["right"]["distance"] for a in names],color=list(COLORS.values()))
    axes[1].set(ylabel="absolute right mean alignment error",title="Directions change; GAUGE is a sham")
    axes[2].bar(names,[p["matching"][a]["metrics"]["loss"] for a in names],color=list(COLORS.values()))
    axes[2].set(ylabel="raw loss at tau=0+",title="Starting loss is allowed to jump")
    for ax in axes[1:]:ax.tick_params(axis="x",rotation=30)
    axes[0].legend(fontsize=7);save(fig,"02_intervention_matching")
    fig,axes=plt.subplots(4,4,figsize=(14,11),sharex=True)
    for col,(g,gi,star) in enumerate([(.4,0,1),(.4,0,5),(.025,2,1),(.025,2,5)]):
        for arm,color in COLORS.items():
            series=[]
            for rep in range(8):
                with np.load(run/f"raw/g{gi}_R{rep}_t{star}_{arm}.npz",allow_pickle=False) as a:
                    with np.load(run/f"assets/target_g{gi}.npz",allow_pickle=False) as b:sigma=b["sigma"]
                    series.append(np.array([a["learnable_residual"],a["right_distance"],a["right_worst"],a["target_mode_signed"][:,31]/sigma[31]]))
                    t=a["times"]
            median=np.median(series,axis=0)
            for i in range(4):axes[i,col].plot(t,median[i],label=arm,color=color,ls="--" if arm=="GAUGE" else "-")
        axes[0,col].set_title(f"g={g}, t*={star}; pointwise median R8")
        for i in range(4):axes[i,col].set_xlim(0,155);axes[i,col].set_xscale("symlog",linthresh=1)
        axes[-1,col].set_xlabel("post-intervention time tau (symlog)")
    for i,name in enumerate(["loss residual (common original denominator)","right mean alignment error","right worst sin² angle","signed mode32 / sigma32"]):axes[i,0].set_ylabel(name)
    axes[0,0].legend(fontsize=7);save(fig,"03_paired_arm_paths")
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,(g,star) in enumerate([(.025,1),(.025,5)]):
        for col,name in enumerate(["J_normalized","G_normalized","A_normalized"]):
            ax=axes[row,col]
            for x,arm in enumerate(["GAUGE","ALIGN","ROTATE"]):
                vals=[r[name] for r in summary["fixed_time_paired"] if r["g"]==g and r["t_star"]==star and r["arm"]==arm and r["tau"]==20]
                ax.scatter(x+np.linspace(-.09,.09,len(vals)),vals,color=COLORS[arm],s=18)
                ax.plot([x-.14,x+.14],[np.median(vals)]*2,color="black",lw=2)
            ax.axhline(0,color="black",lw=.7);ax.set_xticks(range(3),["GAUGE","ALIGN","ROTATE"])
            ax.set_title(f"{name[0]}, g={g}, t*={star}, tau=20; paired R8")
            ax.set_ylabel("difference / original (L0-Lmin)")
    fig.suptitle("A=J+G: instantaneous construction gain and additional subsequent loss decrease",fontsize=11)
    save(fig,"04_J_G_A_paired_points")
    return [str(p.name) for p in sorted(dest.iterdir())]
