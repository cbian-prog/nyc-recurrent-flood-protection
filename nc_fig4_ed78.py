"""Fig. 4 (insurance, R4) and Extended Data Fig. 4 (policy file, Sandy claims, block-group map; the income split is Supplementary Fig. 3, nc_si_income.py).
All from one run of the gap decomposition in fig5_gap_nsi.py."""
import json, warnings, numpy as np, pandas as pd, geopandas as gpd, rasterio
warnings.filterwarnings("ignore")
exec(open("fig5_gap_nsi.py").read().split("# ---------------- figure")[0])
from nc_style import *
TK_LMI=dict(tk)
from matplotlib.colors import BoundaryNorm, ListedColormap; from matplotlib.lines import Line2D; from matplotlib.patches import Patch
C_PAID,C_NOP,C_CAP=PROT,"#C9C9C9","white"
def stack(ax,x,p,n,c,u,w=0.62):
    ax.bar(x,p,w,color=C_PAID); ax.bar(x,n,w,bottom=p,color=C_NOP); ax.bar(x,c,w,bottom=np.array(p)+np.array(n),color=C_CAP,edgecolor=PROT,hatch="//////",lw=0.4)
    ax.scatter(x,u,marker="_",s=60,color="black",lw=1.0,zorder=5)
LEG_STACK=[Patch(fc=C_PAID,label="Paid (expected)"),Patch(fc=C_NOP,label="Gap: no policy"),Patch(fc=C_CAP,ec=PROT,hatch="//////",lw=0.4,label="Gap: cover < loss"),Line2D([],[],marker="_",ls="",color="black",ms=6,mew=1.0,label="Paid (upper bound)")]
# =========================================================== Fig. 4
fig=plt.figure(figsize=(FULL,78*MM))
b=axes_mm(fig,12,14,44,46); x=np.arange(len(RPS)); L=[EV[r]["loss"]/1e9 for r in RPS]; R=[EV[r]["upper"]/1e9 for r in RPS]
b.bar(x,L,0.62,color=SURGE1000,ec=SURGE100,lw=0.5,label="Modelled loss"); b.bar(x,R,0.62,color=PROT,label="NFIP cover (upper bound)")
for i in range(len(RPS)): b.text(i,L[i]+0.6,f"{100*R[i]/L[i]:.0f}%",ha="center",va="bottom",fontsize=6)
b.set_xticks(x); b.set_xticklabels([f"{r:,}" for r in RPS]); b.set_ylim(0,max(L)*1.45); b.set_xlabel("Surge return period (yr)"); b.set_ylabel("Event structure loss (USD bn)"); b.legend(loc="upper left")
panel(b,"a",dx=-10,dy=10)
c=axes_mm(fig,72,14,44,46); rp_grid=1/aeps; m=(rp_grid>=5)&(rp_grid<=2000); Lg=np.maximum(interp(tot["loss"]),1)
for k,lab,col,ls in [("upper","Current, upper bound","black","-"),("neutral","Current, expected",PROT,"-"),("full_local","Full take-up, local cover",PROT_LIGHT,"--"),("full_stat","Full take-up, statutory cap",PROT_LIGHT,":")]:
    c.semilogx(rp_grid[m],(100*interp(tot[k])/Lg)[m],ls,color=col,lw=1.0 if col=="black" else 0.9,label=lab)
c.set_ylim(0,100); c.set_xlim(5,2000); c.set_xlabel("Surge return period (yr)"); c.set_ylabel("Insurable share of loss (%)"); c.grid(which="major"); c.legend(loc="upper right",borderaxespad=0.1)
panel(c,"b",dx=-10,dy=10)
d=axes_mm(fig,132,14,44,46); rows=[EV[rp] for rp in RPS]+[AAI]; xd=np.arange(len(rows))
stack(d,xd,[pct(r,"neutral") for r in rows],[pct(r,"nopolicy") for r in rows],[pct(r,"cap") for r in rows],[pct(r,"upper") for r in rows])
d.set_xticks(xd); d.set_xticklabels([f"{r:,}" for r in RPS]+["AAI"]); d.set_ylim(0,100); d.set_ylabel("Share of loss (%)"); d.set_xlabel("Surge return period (yr)")
d.legend(handles=LEG_STACK,loc="upper left",bbox_to_anchor=(0.0,1.18),ncol=2,columnspacing=0.8)
panel(d,"c",dx=-10,dy=10)
save(fig,"Fig4")
# =========================================================== Extended Data Fig. 7
E7=json.load(open("figED7_nfip.json")); order=["Queens","Brooklyn","Staten Is.","Manhattan","Bronx"]
Lbg=np.array([pd.Series(loss_usace(depths[i])).groupby(bg).sum().reindex(bgi).fillna(0).values for i in range(len(levels))])
LL=np.array([np.interp(wl,levels,Lbg[:,j],left=0) for j in range(len(bgi))]).T; recb=np.minimum(LL,cov_bg[None,:])
aai_bg=-np.trapz(LL,aeps,axis=0); rec_bg=-np.trapz(recb,aeps,axis=0)
gdf=gpd.read_file("data/nfip/tl_2020_36_bg.shp"); gdf=gdf[gdf.COUNTYFP.isin(["061","047","081","005","085"])].to_crs("EPSG:4326").set_index("GEOID")
gdf["aai"]=pd.Series(aai_bg,index=bgi).reindex(gdf.index).fillna(0); gdf["pct"]=pd.Series(100*rec_bg/np.where(aai_bg>0,aai_bg,np.nan),index=bgi).reindex(gdf.index); hit=gdf[gdf.aai>1e4]
with rasterio.open("data/nyc_dem_55m.tif") as r: Dm,Dt,Dn=r.read(1),r.transform,r.nodata
dext=[Dt.c,Dt.c+Dm.shape[1]*Dt.a,Dt.f+Dm.shape[0]*Dt.e,Dt.f]; base=np.where((Dm==Dn)|(Dm<0),0.0,1.0); lonc=Dt.c+(np.arange(Dm.shape[1])+.5)*Dt.a; base[((Dm==Dn)|(Dm<-500))&(lonc[None,:]<-74.24)]=1.0
# Extended Data Fig. 4: policies and Sandy claims; borough take-up panel dropped 2026-11-04, income panels moved to the SI 2026-11-07, block-group map dropped 2026-11-08 (not discussed in the text)
fig=plt.figure(figsize=(FULL,56*MM))
a=axes_mm(fig,13,10,66,40); labs_=list(E7["cover_classes"]); n=np.array([E7["cover_classes"][k] for k in labs_])/1e3; cv=[E7["cover_bn"][k] for k in labs_]
a.bar(range(5),n,0.62,color=PROT); [a.text(i,n[i]+0.4,f"{cv[i]:.1f} bn",ha="center",va="bottom",fontsize=6) for i in range(5)]
a.set_xticks(range(5)); a.set_xticklabels(labs_); a.set_xlabel("Building cover per contract (USD)"); a.set_ylabel("Contracts (thousand)"); a.set_ylim(0,n.max()*1.25); panel(a,"a",dx=-10)
c=axes_mm(fig,104,10,73,40); mb=[E7["sandy_modelled_bn"][k] for k in order]; pb=[E7["sandy_claims_building_bn"][k] for k in order]; pc=[E7["sandy_claims_contents_bn"][k] for k in order]
xx=np.arange(5); c.bar(xx-0.2,mb,0.4,color=SURGE1000,ec=SURGE100,lw=0.5,label="Modelled structure loss"); c.bar(xx+0.2,pb,0.4,color=PROT,label="NFIP building claims"); c.bar(xx+0.2,pc,0.4,bottom=pb,color=PROT_LIGHT,label="NFIP contents claims")
c.set_xticks(xx); c.set_xticklabels(order,rotation=30,ha="right",rotation_mode="anchor"); c.set_ylabel("Hurricane Sandy (USD bn)"); c.legend(loc="upper right",fontsize=5.5); panel(c,"b",dx=-10)
save(fig,"ED_Fig4",outdirs=("figures",))
