"""Supplementary Fig. 3 | Insurance by income group: the NFIP shortfall decomposition and take-up by census-tract LMI quartile
(formerly Extended Data Fig. 4d,e; moved to the Supplementary Information on 2026-11-07). Data from one run of fig5_gap_nsi.py."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
exec(open("fig5_gap_nsi.py").read().split("# ---------------- figure")[0])
from nc_style import *
import nc_style as _ns, importlib; importlib.reload(_ns); from nc_style import *
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
TK_LMI=dict(tk); C_PAID,C_NOP,C_CAP=PROT,"#C9C9C9","white"
def stack(ax,x,p,n,c,u,w=0.62):
    ax.bar(x,p,w,color=C_PAID); ax.bar(x,n,w,bottom=p,color=C_NOP); ax.bar(x,c,w,bottom=np.array(p)+np.array(n),color=C_CAP,edgecolor=PROT,hatch="//////",lw=0.4)
    ax.scatter(x,u,marker="_",s=60,color="black",lw=1.0,zorder=5)
LEG_STACK=[Patch(fc=C_PAID,label="Paid (expected)"),Patch(fc=C_NOP,label="Gap: no policy"),Patch(fc=C_CAP,ec=PROT,hatch="//////",lw=0.4,label="Gap: cover < loss"),Line2D([],[],marker="_",ls="",color="black",ms=6,mew=1.0,label="Paid (upper bound)")]
xg=np.arange(4); qlab=["<35","35–51","51–70","≥70"]
fig=plt.figure(figsize=(FULL,56*MM))
a=axes_mm(fig,14,12,46,38)
stack(a,xg,[pct(AAIg[g],"neutral") for g in labs],[pct(AAIg[g],"nopolicy") for g in labs],[pct(AAIg[g],"cap") for g in labs],[pct(AAIg[g],"upper") for g in labs])
a.set_xticks(xg); a.set_xticklabels(qlab); a.set_ylim(0,100); a.set_xlabel("LMI share of tract (%)"); a.set_ylabel("Share of annual surge loss (%)")
a.legend(handles=LEG_STACK,loc="upper left",bbox_to_anchor=(1.02,1.0),fontsize=5.5); panel(a,"a",dx=-12)
b=axes_mm(fig,112,12,64,38)
for vals,lab,mk,col,mf in (([100*TK_LMI[g] for g in labs],"Take-up in 1-in-100-yr footprint","o",PROT,PROT),
                            ([pct(AAIg[g],"upper") for g in labs],"Insurable share, upper bound","s",PROT,"white"),
                            ([pct(AAIg[g],"full_stat") for g in labs],"Ceiling: full take-up at statutory cap","^",SURGE10,"white")):
    b.plot(xg,vals,"-",color=col,lw=0.6,alpha=0.6); b.plot(xg,vals,mk,ms=3.2,mfc=mf,mec=col,mew=0.8,ls="",label=lab)
b.set_xticks(xg); b.set_xticklabels(qlab); b.set_ylim(0,100); b.set_xlabel("Low- and moderate-income (LMI) share of tract (%)"); b.set_ylabel("%"); b.grid(axis="y"); b.legend(loc="lower right",fontsize=5.5)
panel(b,"b",dx=-12)
save(fig,"SI_Fig3",outdirs=("figures",))
