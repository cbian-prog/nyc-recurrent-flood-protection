"""Supplementary Fig. 3: a, New York vs Miami-Dade insurance-gap decomposition on the common storm-category basis;
b, why the comparison is on a category basis only (category-basis vs water-level-basis event loss for New York)."""
import json, numpy as np
from nc_style import *
from matplotlib.patches import Patch; from matplotlib.lines import Line2D
OUT=("figures",)
# ---------------- Supplementary Fig. 3
two={c:json.load(open(f"insurance_city_{c}.json"))["0.477"] for c in ("nyc","miami")}
fig=plt.figure(figsize=(FULL,54*MM)); a=axes_mm(fig,13,8,40,40); keys=["nyc","miami"]; xx=np.arange(2)
p=[two[k]["neutral_pct"] for k in keys]; n=[two[k]["gap_nopolicy_pct"] for k in keys]; cp=[two[k]["gap_cap_pct"] for k in keys]
a.bar(xx,p,0.6,color=PROT); a.bar(xx,n,0.6,bottom=p,color="#C9C9C9"); a.bar(xx,cp,0.6,bottom=np.array(p)+np.array(n),color="white",ec=PROT,hatch="//////",lw=0.4)
a.scatter(xx,[two[k]["upper_pct"] for k in keys],marker="_",s=60,color="black",lw=1.0,zorder=5); a.scatter(xx,[two[k]["full_takeup_pct"] for k in keys],marker="^",s=12,color=SURGE10,zorder=5)
[a.text(i,101,f"AAI {two[k]['aai_m']:.0f} M",ha="center",va="bottom",fontsize=5.5) for i,k in enumerate(keys)]
a.set_xticks(xx); a.set_xticklabels(["New York\nCity","Miami-Dade"]); a.set_ylim(0,110); a.set_yticks([0,20,40,60,80,100]); a.set_ylabel("Share of annual loss (%)")
a.legend(handles=[Patch(fc=PROT,label="Paid (expected)"),Patch(fc="#C9C9C9",label="Gap: no policy"),Patch(fc="white",ec=PROT,hatch="//////",lw=0.4,label="Gap: cover < loss"),
                  Line2D([],[],marker="_",ls="",color="black",ms=6,mew=1.0,label="Paid (upper bound)"),Line2D([],[],marker="^",ls="",color=SURGE10,ms=3,label="Full take-up, statutory cap")],loc="upper left",bbox_to_anchor=(1.02,1.0))
panel(a,"a",dx=-10)
# b: why the comparison is on a category basis only (formerly Extended Data Fig. 2b)
H=json.load(open("figED3_hazard_basis.json")); cats=["1","2","3","4"]; Lcat=[H["cat_loss_bn"][k] for k in cats]; Lwl=H["wl_loss_at_cat_rp_bn"]; rpc=[45,143,400,1000]; xc=np.arange(4)
b=axes_mm(fig,101,8,75,40)
b.bar(xc-0.2,Lcat,0.4,color=NEUTRAL,label="Category basis (SLOSH MOM × STORM)"); b.bar(xc+0.2,Lwl,0.4,color=SURGE10,label="Water-level basis (NACCS), New York")
for i in range(4): b.text(i-0.2,Lcat[i]+2,f"{Lcat[i]/Lwl[i]:.1f}×",ha="center",va="bottom",fontsize=5.5)
b.set_xticks(xc); b.set_xticklabels([f"Cat {c_}\n{r:,} yr" for c_,r in zip(cats,rpc)],fontsize=5.5); b.set_ylabel("Event structure loss (USD bn)"); b.set_ylim(0,max(Lcat)*1.5); b.legend(loc="upper left",fontsize=5.5)
b.text(0.02,0.78,f"Annual loss {H['aai_cat_m']:,.0f} vs {H['aai_wl_m']:.0f} M yr$^{{-1}}$\n({H['aai_cat_m']/H['aai_wl_m']:.1f}× overstatement)",transform=b.transAxes,ha="left",va="top",fontsize=5.5)
panel(b,"b",dx=-11)
save(fig,"SI_Fig3",outdirs=OUT)   # Supplementary Fig. 3 since 2026-11-04 (two-city comparison and the category basis)
