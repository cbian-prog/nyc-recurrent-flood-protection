"""Supplementary Fig. 2 | Sensitivity of the expected NFIP payment to where policies sit inside a block group.

The expected estimate in Fig. 4 and Fig. 3b places a block group's contracts at random among its structures.
Insured buildings are unlikely to be random: purchase is required inside the map and buyers know their own risk.
Three placements, each conserving the number of contracts and the mean cover per contract in every block group:
  random    : take-up_i = contracts / structures                                       (used in the paper)
  map       : contracts rated in A/V zones go to structures inside the effective SFHA, the rest to structures
              outside it (overflow moves to the other set)
  ranked    : contracts go to the structures with the largest expected loss first (adverse-selection bound);
              surge: ranked by structure AAI; rain: ranked by the structure's loss in the extreme rainfall scenario
Expected payment_i = take-up_i x min(loss_i, mean cover per contract).  Output nc_si_placement.json, SI_Fig3.
"""
import glob, json, warnings, numpy as np, pandas as pd, geopandas as gpd
from rasterio import features
warnings.filterwarnings("ignore")
exec(open("paper_a_core.py").read().split("# ---------------- Sandy validation")[0])
bg=s.cbfips.astype(str).str[:12].values
nf=pd.read_parquet("data/nfip/all_current.parquet"); nf["av"]=nf.ratedFloodZone.astype(str).str[0].isin(["A","V"])
ncon=nf.groupby("censusGeoid").size(); nav=nf[nf.av].groupby("censusGeoid").size(); bcov=nf.groupby("censusGeoid").totalBuildingInsuranceCoverage.sum()
avgcov=(bcov/ncon).reindex(bg).fillna(0).values
# structures inside the effective SFHA (point in polygon)
Z=gpd.read_file("data/fema_nfhl/nfhl_zones_nyc.geojson"); Z=Z[Z.SFHA_TF=="T"][["geometry"]].reset_index(drop=True)
gS=gpd.GeoDataFrame(geometry=gpd.points_from_xy(s.x,s.y),crs="EPSG:4326"); j=gpd.sjoin(gS,Z,how="left",predicate="within")
sf=~j.index_right.isna().groupby(level=0).all().values
D=pd.DataFrame({"bg":bg,"sf":sf})
nS=D.groupby("bg").sf.sum(); nT=D.groupby("bg").size(); nX=nT-nS
c_all=ncon.reindex(nT.index).fillna(0); c_av=nav.reindex(nT.index).fillna(0); c_x=c_all-c_av
# --- random
take_rand=(c_all/nT).clip(upper=1).reindex(bg).values
# --- map-informed
tS=np.where(nS>0,np.minimum(1,c_av/np.maximum(nS,1)),0); over=c_av-tS*nS
tX=np.where(nX>0,np.minimum(1,(c_x+over)/np.maximum(nX,1)),0); overX=np.maximum(c_x+over-tX*nX,0)
tS=np.where(nS>0,np.minimum(1,tS+overX/np.maximum(nS,1)),0)
take_map=np.where(sf,pd.Series(tS,index=nT.index).reindex(bg).values,pd.Series(tX,index=nT.index).reindex(bg).values)
# --- per-level structure loss and AAI weights
curve=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in curve)); WLm=np.array([curve[str(t)] for t in T])
aeps=np.logspace(0,-4,120); wl=np.interp(np.log(1/aeps),np.log(T),WLm)-NAVD_MINUS_MSL
W=np.array([-np.trapz(np.interp(wl,levels,np.eye(len(levels))[k],left=0),aeps) for k in range(len(levels))])   # AAI = W @ level totals
Lv=np.stack([loss_usace(depths[i]).astype("float32") for i in range(len(levels))])            # levels x structures
aai_i=W@Lv
def ranked_take(score):
    r=pd.DataFrame({"bg":bg,"sc":score}).groupby("bg").sc.rank(ascending=False,method="first").values
    return (r<=c_all.reindex(bg).fillna(0).values).astype(float)
take_rank=ranked_take(aai_i)
def ev_at(rp,tot): return float(np.interp(np.interp(np.log(rp),np.log(T),WLm)-NAVD_MINUS_MSL,levels,tot))
out={"surge":{}}
for name,tk in (("random",take_rand),("map",take_map),("ranked",take_rank)):
    paid=np.array([(tk*np.minimum(Lv[i],avgcov)).sum() for i in range(len(levels))]); nop=np.array([((1-tk)*Lv[i]).sum() for i in range(len(levels))]); tot=Lv.sum(1)
    out["surge"][name]=dict(aai_paid_pct=float(100*(W@paid)/(W@tot)),aai_nopolicy_pct=float(100*(W@nop)/(W@tot)),
                            event_paid_pct={str(rp):100*ev_at(rp,paid)/ev_at(rp,tot) for rp in (10,100,1000)})
    print("surge",name,{k:(round(v,1) if not isinstance(v,dict) else {a:round(b,1) for a,b in v.items()}) for k,v in out["surge"][name].items()})
# --- rain (extreme DEP scenario), rain-side vs surge-side block groups as in pluvial_insurance_nsi.py
wl1000=float(np.interp(np.log(1000),np.log(T),WLm))-NAVD_MINUS_MSL; jj=int(np.argmin(abs(levels-wl1000)))
v_bg=pd.Series(val).groupby(bg).sum(); vs_bg=pd.Series(val*(depths[jj]>0)).groupby(bg).sum(); surge_side=((vs_bg/v_bg)>=0.5).reindex(bg).values
import nc_citygrid as CG   # citywide rain depths; the split applies where surge is modelled (block groups >= 50% of value in the window)
dom=((pd.Series(val*CG.in_window(s.x.values,s.y.values)).groupby(bg).sum()/v_bg)>=0.5).reindex(bg).values
g=gpd.read_file(glob.glob("data/pluvial/NYCFloodStormwaterFloodMaps/*Extreme*/*.gdb")[0]).to_crs("EPSG:4326"); d=CG.dep_depth(g,s.x.values,s.y.values)
li=loss_usace(d)
out["rain"]={}
for name,tk in (("random",take_rand),("map",take_map),("ranked_surge",take_rank),("ranked_rain",ranked_take(li))):
    out["rain"][name]={}
    for side,m in (("rain_side",~surge_side&dom),("surge_side",surge_side)):
        out["rain"][name][side]=float(100*(tk*np.minimum(li,avgcov))[m].sum()/li[m].sum())
    print("rain",name,{k:round(v,2) for k,v in out["rain"][name].items()})
json.dump(out,open("nc_si_placement.json","w"),indent=1)
# ---------------- figure
from nc_style import *
fig=plt.figure(figsize=(FULL,56*MM)); cols={"random":PROT,"map":"#7F7F7F","ranked":"#C9C9C9"}
a=axes_mm(fig,13,7,76,42); x=np.arange(4); labs=["1-in-10","1-in-100","1-in-1,000","AAI"]
for k_,(name,lab) in enumerate((("random","Random (used)"),("map","Map-informed"),("ranked","Highest loss first"))):
    o=out["surge"][name]; v=[o["event_paid_pct"]["10"],o["event_paid_pct"]["100"],o["event_paid_pct"]["1000"],o["aai_paid_pct"]]
    a.bar(x+(k_-1)*0.26,v,0.26,color=cols[name],ec=INK,lw=0.3,label=lab); [a.text(x[i]+(k_-1)*0.26,v[i]+1,f"{v[i]:.0f}",ha="center",va="bottom",fontsize=5.5) for i in range(4)]
a.set_xticks(x); a.set_xticklabels(labs); a.set_ylabel("Expected NFIP payment (% of surge loss)"); a.set_ylim(0,max(60,a.get_ylim()[1])); a.legend(loc="upper right"); a.grid(axis="y")
panel(a,"a",dx=-11)
b=axes_mm(fig,106,7,70,42); x=np.arange(2)
for k_,(name,lab,col) in enumerate((("random","Random (used)",PROT),("map","Map-informed","#7F7F7F"),("ranked_surge","Highest surge loss first","#C9C9C9"),("ranked_rain","Highest rain loss first","white"))):
    v=[out["rain"][name]["rain_side"],out["rain"][name]["surge_side"]]
    b.bar(x+(k_-1.5)*0.2,v,0.2,color=col,ec=INK,lw=0.3,label=lab); [b.text(x[i]+(k_-1.5)*0.2,v[i]+0.4,f"{v[i]:.1f}",ha="center",va="bottom",fontsize=5) for i in range(2)]
b.set_xticks(x); b.set_xticklabels(["Rain-side block groups","Surge-side block groups"]); b.set_ylabel("Expected payment (% of rain loss)"); b.legend(loc="upper left"); b.grid(axis="y")
b.set_ylim(0,max(out["rain"][n]["surge_side"] for n in out["rain"])*1.45)
panel(b,"b",dx=-11)
save(fig,"SI_Fig2",outdirs=("figures",))   # Supplementary Fig. 2 since 2026-11-04
