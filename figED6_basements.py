"""Data for Extended Data Fig. 3a and Supplementary Table 1 — basement dwellings: allocation sensitivity, borough split, below-grade loss share, depth classes.
a  informal basement units inside each footprint under three allocation rules (uniform; LMI-weighted; LMI above 30% only)
b  informal units by borough, 1-in-100-yr surge vs DEP extreme rainfall (uniform allocation)
c  share of modelled surge structure loss that occurs below the first floor (basement buildings, depth relative to first floor < 0), by return period
d  basement lots by depth class inside the footprints
"""
import glob, json, warnings, numpy as np, pandas as pd, geopandas as gpd
from rasterio import features; from scipy import ndimage
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("paper_a_core.py").read().split("# ---------------- Sandy validation")[0])
matplotlib.rcParams.update({"font.size":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":10,"axes.titlesize":12})
INFORMAL=114_000
p=pd.read_csv("data/pluto_bsmt.csv",low_memory=False).dropna(subset=["latitude","longitude"])
p=p[p.bldgclass.str[0].isin(["A","B"])|p.bldgclass.isin(["C0","C1","C2","C3"])].reset_index(drop=True)
B2C={"1":"061","2":"005","3":"047","4":"081","5":"085"}; p["boro"]=p.bbl.astype(str).str[0].map({"1":"Manhattan","2":"Bronx","3":"Brooklyn","4":"Queens","5":"Staten Is."})
p["tract"]="36"+p.bbl.astype(str).str[0].map(B2C)+(p.ct2010*100).round().astype(int).astype(str).str.zfill(6)
import urllib.request; cd=pd.DataFrame(json.load(urllib.request.urlopen("https://data.cityofnewyork.us/resource/qmcw-ur37.json?$limit=3000",timeout=60)))
cd["tract"]="36"+cd.borocode.map(B2C)+cd.boroct.str[1:].str.zfill(6); eq=pd.to_numeric(cd.set_index("tract").lomod_pct,errors="coerce"); p["lmi"]=p.tract.map(eq)
print(f"lots {len(p):,} | with LMI share {p.lmi.notna().sum():,}")
W={"Uniform":np.ones(len(p)),"LMI-weighted":p.lmi.fillna(p.lmi.median()).values,"LMI above 30% only":np.maximum(p.lmi.fillna(p.lmi.median()).values-30,0)}
for k in W: p[k]=INFORMAL*W[k]/W[k].sum()
# footprints on the water-level basis
curve=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in curve)); WLm=np.array([curve[str(t)] for t in T])
wl=lambda r: float(np.interp(np.log(r),np.log(T),WLm))-NAVD_MINUS_MSL
def surge_depth(w):
    below=(dem_v<w)&valid; lab,_=ndimage.label(below); conn=np.isin(lab,np.unique(lab[seed&(lab>0)]))&below&(dem_v>=0); return np.where(conn,w-dem_v,0.0)
import nc_citygrid as CG   # citywide cells; surge depths exist only inside the surge window
def at(m): return CG.at(p.longitude.values,p.latitude.values,m).astype(float)
FP={}
def at_w(m):   # surge depths live on the window grid: same lookup as fig2_impact_rp.py, so the counts agree with Fig. 2
    c=np.floor((p.longitude.values-tr.c)/tr.a).astype(int); r_=np.floor((p.latitude.values-tr.f)/tr.e).astype(int); k=(r_>=0)&(r_<m.shape[0])&(c>=0)&(c<m.shape[1])
    o=np.zeros(len(p)); o[k]=m[r_[k],c[k]]; return o
for r in (10,100,1000): FP[f"Surge {r}-yr"]=at_w(surge_depth(wl(r)))
for name,pat in (("DEP moderate","*Moderate*Current*"),("DEP extreme","*Extreme*")):
    g=gpd.read_file(glob.glob(f"data/pluvial/NYCFloodStormwaterFloodMaps/{pat}/*.gdb")[0]).to_crs("EPSG:4326"); d=np.zeros(CG.SHAPE_C)
    for cat,dep in ((1,0.20),(2,0.46)):
        m=CG.ras(g[g.Flooding_Category==cat].geometry); d=np.maximum(d,np.where(m,dep,0))
    FP[name]=at(d)
alloc={fp:{k:float(p[k][FP[fp]>0].sum()) for k in W} for fp in FP}; lots={fp:int((FP[fp]>0).sum()) for fp in FP}
print(pd.DataFrame(alloc).round(0)); print("lots:",lots)
# b: by borough
bb=pd.DataFrame({fp:p[FP[fp]>0].groupby("boro")["Uniform"].sum() for fp in ("Surge 100-yr","DEP extreme")}).reindex(["Queens","Brooklyn","Staten Is.","Bronx","Manhattan"]).fillna(0)
# c: below-grade share of surge loss by return period
aeps=np.logspace(0,-4,120); wl0=np.interp(np.log(1/aeps),np.log(T),WLm)-NAVD_MINUS_MSL; rp=1/aeps
def loss_split(d_m):
    d_ff=d_m/FT-found_ft; f=np.zeros(len(s))
    for k in CURVES: m=key==k; f[m]=usace_frac(d_ff[m],k)
    tot=val*f*share3*(d_m>0); below=np.where(wb&(d_ff<0),tot,0.0)  # buildings with basement and water still below first floor
    return tot.sum(),below.sum()
TOT=[];BEL=[]
for i in range(len(levels)): t_,b_=loss_split(depths[i]); TOT.append(t_); BEL.append(b_)
TOT=np.interp(wl0,levels,TOT,left=0); BEL=np.interp(wl0,levels,BEL,left=0); share_below=np.where(TOT>0,BEL/np.maximum(TOT,1),np.nan)
aai_tot=-np.trapz(TOT,aeps)/1e6; aai_bel=-np.trapz(BEL,aeps)/1e6; print(f"below-first-floor loss share of AAI {100*aai_bel/aai_tot:.0f}% ({aai_bel:.0f} of {aai_tot:.0f} M)")
# d: depth classes
dc={}
for fp in ("Surge 10-yr","Surge 100-yr","Surge 1000-yr"): d=FP[fp]; dc[fp]=[int(((d>0)&(d<=.3)).sum()),int(((d>.3)&(d<=1)).sum()),int((d>1).sum())]
for fp in ("DEP moderate","DEP extreme"): d=FP[fp]; dc[fp]=[int((abs(d-.2)<.01).sum()),int((abs(d-.46)<.01).sum()),0]
json.dump(dict(informal_by_allocation=alloc,lots=lots,by_boro_uniform=bb.to_dict(),below_first_floor_share_aai=aai_bel/aai_tot,depth_classes=dc),open("figED6_basements.json","w"),indent=1)
# ---- figure
fig,ax=plt.subplots(2,2,figsize=(7.5,7.2),gridspec_kw=dict(hspace=.62,wspace=.35,left=.09,right=.985,top=.97,bottom=.12))
a=ax[0,0]; fps=list(FP); x=np.arange(len(fps)); cols=["#1B4965","#2E7D5B","#9CC5A1"]
for j,k in enumerate(W): a.bar(x+(j-1)*.27,[alloc[f][k]/1e3 for f in fps],.27,color=cols[j],label=k)
a.set_xticks(x); a.set_xticklabels(["Surge 10 yr","Surge 100 yr","Surge 1,000 yr","Rain moderate","Rain extreme"],fontsize=9.5,rotation=30,ha="right",rotation_mode="anchor"); a.set_ylabel("Informal units (thousand)"); a.legend(loc="upper left",frameon=False,fontsize=9); a.set_title("a",loc="left",fontweight="bold"); a.grid(axis="y",alpha=.3)
b=ax[0,1]; xb=np.arange(5); b.bar(xb-.2,bb["Surge 100-yr"]/1e3,.4,color="#E39A6A",label="Surge 100-yr"); b.bar(xb+.2,bb["DEP extreme"]/1e3,.4,color="#4292C6",label="Rain extreme")
b.set_xticks(xb); b.set_xticklabels(bb.index,rotation=30,ha="right",rotation_mode="anchor"); b.set_ylabel("Informal units (thousand)"); b.legend(frameon=False,fontsize=9); b.set_title("b",loc="left",fontweight="bold"); b.grid(axis="y",alpha=.3)
c=ax[1,0]; m=(rp>=2)&(rp<=1e4); c.semilogx(rp[m],100*share_below[m],"-",color="#8E1B1B",lw=1.8); c.set_xlim(2,1e4); c.set_ylim(0,100)
c.set_xlabel("Return period (yr)"); c.set_ylabel("Loss below first floor (% of event loss)"); c.grid(alpha=.3); c.set_title("c",loc="left",fontweight="bold"); c.text(.97,.95,f"AAI share {100*aai_bel/aai_tot:.0f}%",transform=c.transAxes,ha="right",va="top",fontsize=10)
d=ax[1,1]; names=list(dc); xd=np.arange(len(names)); arr=np.array([dc[n] for n in names])/1e3
d.bar(xd,arr[:,0],color="#C9D3DA",label="≤0.3 m / nuisance"); d.bar(xd,arr[:,1],bottom=arr[:,0],color="#6F97AE",label="0.3–1 m / deep ≥1 ft"); d.bar(xd,arr[:,2],bottom=arr[:,0]+arr[:,1],color="#1B4965",label=">1 m")
d.set_xticks(xd); d.set_xticklabels(["Surge 10 yr","Surge 100 yr","Surge 1,000 yr","Rain moderate","Rain extreme"],fontsize=9.5,rotation=30,ha="right",rotation_mode="anchor"); d.set_ylabel("Basement lots (thousand)"); d.legend(loc="upper left",frameon=False,fontsize=9); d.set_title("d",loc="left",fontweight="bold"); d.grid(axis="y",alpha=.3)
plt.savefig("figED6_basements_en.png",dpi=300,bbox_inches="tight"); plt.savefig("figED6_basements_en.pdf",bbox_inches="tight"); print("saved figED6_basements_en.png")
