"""Fig. 2 — impact vs return period by asset, on one x-axis (NACCS water-level basis, connected bathtub).
(a) building structure loss (NSI x USACE); (b) basement lots and informal units; (c) underground stations and entrances;
(d) rainfall: observed street-flood probability by rain intensity at sensors (frequency, not return period)."""
import json, warnings, numpy as np, pandas as pd, rasterio
from scipy import ndimage
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12,"axes.titlesize":12})
exec(open("paper_a_core.py").read().split("# ---------------- Sandy validation")[0])   # NSI, depths, levels, loss_usace, dem, tr
curve=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in curve)); WLm=np.array([curve[str(t)] for t in T])
RPS=np.array([2,5,10,20,50,100,200,500,1000,2000,5000,10000]); WL=np.interp(np.log(RPS),np.log(T),WLm)-NAVD_MINUS_MSL
# (a) building loss
Lus=np.array([loss_usace(depths[i]).sum() for i in range(len(levels))]); loss_rp=np.interp(WL,levels,Lus,left=0)/1e9
# masks per RP
def wet_mask(w):
    below=(dem_v<w)&valid; lab,_=ndimage.label(below); return np.isin(lab,np.unique(lab[seed&(lab>0)]))&below&(dem_v>=0)
def at(lon,lat,m):
    c=np.floor((np.asarray(lon)-tr.c)/tr.a).astype(int); r_=np.floor((np.asarray(lat)-tr.f)/tr.e).astype(int); k=(r_>=0)&(r_<dem.shape[0])&(c>=0)&(c<dem.shape[1])
    o=np.zeros(len(c),bool); o[k]=m[r_[k],c[k]]; return o
# (b) basements
b=pd.read_csv("data/pluto_bsmt.csv",low_memory=False).dropna(subset=["latitude","longitude"])
b=b[b.bldgclass.str[0].isin(["A","B"])|b.bldgclass.isin(["C0","C1","C2","C3"])].reset_index(drop=True); inf_per=114000/len(b)
# (c) subway
S=pd.DataFrame(json.load(open("data/subway/stations.json"))); S["lat"]=S.gtfs_latitude.astype(float); S["lon"]=S.gtfs_longitude.astype(float); S=S[S.structure=="Subway"]
E=pd.DataFrame(json.load(open("data/subway/entrances.json"))); E["lat"]=E.entrance_latitude.astype(float); E["lon"]=E.entrance_longitude.astype(float); E=E[E.station_id.isin(S.station_id)]
bl,st,en=[],[],[]
for w in WL:
    m=wet_mask(w); bl.append(int(at(b.longitude,b.latitude,m).sum())); st.append(int(at(S.lon,S.lat,m).sum())); en.append(int(at(E.lon,E.lat,m).sum()))
bl,st,en=np.array(bl),np.array(st),np.array(en)
out=dict(rp=RPS.tolist(),wl_navd=WL.round(2).tolist(),loss_bn=loss_rp.round(2).tolist(),basement_lots=bl.tolist(),informal_units=(bl*inf_per).round().tolist(),underground_stations=st.tolist(),entrances=en.tolist())
# share of each annual-average impact contributed by events more frequent than T years (independent of any normalising event)
RPD=np.logspace(0,4,61); WLD=np.interp(np.log(RPD),np.log(T),WLm)-NAVD_MINUS_MSL; AE=1/RPD
dens={"loss":np.interp(WLD,levels,Lus,left=0)}; dens["basement_lots"],dens["underground_stations"],dens["entrances"]=[],[],[]
for w in WLD:
    m=wet_mask(w); dens["basement_lots"].append(at(b.longitude,b.latitude,m).sum()); dens["underground_stations"].append(at(S.lon,S.lat,m).sum()); dens["entrances"].append(at(E.lon,E.lat,m).sum())
ann=lambda y,lo: float(-np.trapz(np.asarray(y,float)[AE>=lo],AE[AE>=lo]))
out["annual_share_frequent"]={k:{str(t):ann(y,1/t)/ann(y,0) for t in (10,20,50,100)} for k,y in dens.items()}
out["annual_average"]={k:ann(y,0) for k,y in dens.items()}
out["annual_cum"]=dict(rp=RPD.tolist(),**{k:[ann(y,1/t)/ann(y,0) for t in RPD] for k,y in dens.items()})   # Fig. 2b
print("share of annual average from events more frequent than 10/20/50/100 yr:",{k:[round(100*v,1) for v in d.values()] for k,d in out["annual_share_frequent"].items()})
json.dump(out,open("fig2_impact_rp.json","w"),indent=1); print(pd.DataFrame({k:v for k,v in out.items() if k not in ("annual_share_frequent","annual_average","annual_cum")}).to_string(index=False))
# growth 100 -> 1000 yr
i100,i1000=list(RPS).index(100),list(RPS).index(1000)
print(f"growth 100->1000 yr: loss x{loss_rp[i1000]/loss_rp[i100]:.1f} | basement lots x{bl[i1000]/max(bl[i100],1):.1f} | stations x{st[i1000]/max(st[i100],1):.1f} | entrances x{en[i1000]/max(en[i100],1):.1f}")
# (d) rainfall
bins=[("<0.5",0.61),("0.5–1.0",7.6),("1.0–1.75",19.2),("1.75–2.5",36.4)]
fig,ax=plt.subplots(2,2,figsize=(7.5,7.6)); ax=ax.ravel(); C="#1B4965"
ax[0].semilogx(RPS,loss_rp,"-o",color=C); ax[0].set(ylabel="Structure loss (USD bn)"); ax[0].set_title("a  Buildings",loc="left",fontweight="bold"); 
ax[0].axvline(139,color="#B33A30",ls=":"); ax[0].text(150,loss_rp.max()*0.88,"Sandy",color="#B33A30",fontsize=12)
ax[1].semilogx(RPS,bl/1e3,"-o",color="#C9772E"); ax[1].set(ylabel="Basement lots (thousand)"); ax[1].set_title("b  Basement dwellings",loc="left",fontweight="bold")
ax1b=ax[1].twinx(); ax1b.set_ylim(np.array(ax[1].get_ylim())*1e3*inf_per); ax1b.set_ylabel("Informal units (allocated)")
ax[2].semilogx(RPS,st,"-o",color="#2E7D5B",label="Underground stations"); ax[2].set(ylabel="Underground stations"); ax[2].set_title("c  Subway",loc="left",fontweight="bold")
ax2b=ax[2].twinx(); ax2b.semilogx(RPS,en,"--s",color="#7A5EA8",label="Entrances"); ax2b.set_ylabel("Entrances"); h1,l1=ax[2].get_legend_handles_labels(); h2,l2=ax2b.get_legend_handles_labels(); ax[2].legend(h1+h2,l1+l2,loc="upper left",frameon=False,fontsize=11)
for a in ax[:3]: a.set_xlabel("Return period (yr)"); a.grid(alpha=.3); a.set_xlim(2,1e4)
ax[3].bar(range(4),[v for _,v in bins],color=["#BDD7E7","#6BAED6","#2171B5","#08306B"]); ax[3].set_xticks(range(4)); ax[3].set_xticklabels(["<0.5","0.5–1","1–1.75","1.75–2.5"],rotation=25,ha="right")
for i,(_,v) in enumerate(bins): ax[3].text(i,v+0.8,f"{v:.1f}%",ha="center")
ax[3].axvline(2.5,color="#B33A30",ls=":"); ax[3].text(2.45,42,"design\nstandard",color="#B33A30",ha="right",va="top")
ax[3].set(ylabel="Flood probability per rainy day (%)",xlabel="Max hourly rain at sensor (in/hr)",ylim=(0,45)); ax[3].set_title("d  Rainfall",loc="left",fontweight="bold"); ax[3].grid(axis="y",alpha=.3)
plt.tight_layout(); plt.savefig("fig2_impact_rp_en.png",dpi=300,bbox_inches="tight"); plt.savefig("fig2_impact_rp_en.pdf",bbox_inches="tight"); print("saved fig2_impact_rp_en.png")
