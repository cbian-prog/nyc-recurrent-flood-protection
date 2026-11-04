"""Extended Data Fig. 3 | Basement homes: allocation of informal units and geography by borough.
Usage: python nc_ed456.py   (the former branches for the old Extended Data Figs 6 and 7 were removed on 2026-11-04)"""
import sys, numpy as np, pandas as pd
exec(open("figED6_basements.py").read().split("# ---- figure")[0])
from nc_style import *
lots_b=pd.DataFrame({"1-in-100-yr surge":p[FP["Surge 100-yr"]>0].groupby("boro").size(),"Extreme rain":p[FP["DEP extreme"]>0].groupby("boro").size()}).reindex(["Queens","Brooklyn","Staten Is.","Bronx","Manhattan"]).fillna(0)
print(lots_b)
fig=plt.figure(figsize=(FULL,58*MM)); fps=list(FP); short=["Surge\n10 yr","Surge\n100 yr","Surge\n1,000 yr","Rain\nmoderate","Rain\nextreme"]
a=axes_mm(fig,12,12,72,40); x=np.arange(len(fps)); cols=[PROT,PROT_LIGHT,"#DDDDDD"]
for j,k in enumerate(W): a.bar(x+(j-1)*0.27,[alloc[f][k]/1e3 for f in fps],0.27,color=cols[j],ec=PROT,lw=0.3,label=k)
a.set_xticks(x); a.set_xticklabels(short); a.set_ylabel("Informal units in footprint (thousand)"); a.legend(loc="upper left"); a.grid(axis="y")
panel(a,"a",dx=-10)
b=axes_mm(fig,104,12,72,40); xb=np.arange(5)
b.bar(xb-0.2,lots_b["1-in-100-yr surge"]/1e3,0.4,color=SURGE100,label="1-in-100-yr surge"); b.bar(xb+0.2,lots_b["Extreme rain"]/1e3,0.4,color=RAIN_EXT,label="Extreme rain")
b.set_xticks(xb); b.set_xticklabels(lots_b.index); b.set_ylabel("Basement lots in footprint (thousand)"); b.legend(loc="upper right"); b.grid(axis="y")
panel(b,"b",dx=-10)
save(fig,"ED_Fig3",outdirs=("figures",))
