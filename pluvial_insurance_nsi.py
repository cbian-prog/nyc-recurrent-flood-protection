"""NFIP overlay on pluvial (DEP stormwater map) loss on the publication basis (NSI x USACE generic curves),
split into surge-side and pluvial-only block groups on the water-level basis.

surge-side block group : >= 50% of the block group's structure value lies inside the 1-in-1,000-yr still-water reach
pluvial-only block group: the rest, among block groups with >= 50% of their value inside the surge model domain
outside surge domain    : block groups mostly outside the surge DEM window, where the surge reach is not modelled
Rain depths are taken on the citywide cells (nc_citygrid.py), so rainfall loss covers the five boroughs.
upper bound            : min(block-group pluvial loss, block-group NFIP building cover)
expected (neutral)     : take-up x min(structure loss, mean cover per contract), policies placed at random in the block group
Output: pluvial_insurance_nsi.json
"""
import glob, json, warnings, numpy as np, pandas as pd, geopandas as gpd
from rasterio import features
warnings.filterwarnings("ignore")
exec(open("paper_a_core.py").read().split("# ---------------- Sandy validation")[0])
bg=s.cbfips.astype(str).str[:12].values
nfip=pd.read_parquet("data/nfip/all_current.parquet")
bcov=nfip.groupby("censusGeoid").totalBuildingInsuranceCoverage.sum(); ncon=nfip.groupby("censusGeoid").size()
n_bg=pd.Series(bg).value_counts(); take=(ncon.reindex(n_bg.index).fillna(0)/n_bg).clip(upper=1); avgcov=(bcov/ncon).reindex(n_bg.index).fillna(0)
# surge-side definition on the water-level basis: value share inside the 1-in-1,000-yr reach
curve=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in curve)); WLm=np.array([curve[str(t)] for t in T])
wl1000=float(np.interp(np.log(1000),np.log(T),WLm))-NAVD_MINUS_MSL; j=int(np.argmin(abs(levels-wl1000))); wet1000=depths[j]>0
v_bg=pd.Series(val).groupby(bg).sum(); vs_bg=pd.Series(val*wet1000).groupby(bg).sum(); surge_side=(vs_bg/v_bg)>=0.5
import nc_citygrid as CG
dom_bg=(pd.Series(val*CG.in_window(s.x.values,s.y.values)).groupby(bg).sum()/v_bg)>=0.5
print(f"1-in-1,000-yr level {wl1000:.2f} m NAVD88 (ladder {levels[j]:.2f}); surge-side block groups {int(surge_side.sum()):,} of {len(surge_side):,}")
DEPTH_CAT={1:0.20,2:0.46}; SCEN=[("Moderate, current SL",2.13,"*Moderate*Current*"),("Extreme, 2080 SL",3.66,"*Extreme*")]
ARI=np.array([1,2,5,10,25,50,100,200,500,1000]); D60=np.array([1.07,1.28,1.62,1.90,2.28,2.57,2.88,3.22,3.71,4.11]); rp_of=lambda i: float(np.exp(np.interp(np.log(i),np.log(D60),np.log(ARI))))
res={}
for name,inten,pat in SCEN:
    g=gpd.read_file(glob.glob(f"data/pluvial/NYCFloodStormwaterFloodMaps/{pat}/*.gdb")[0]).to_crs("EPSG:4326")
    d=CG.dep_depth(g,s.x.values,s.y.values,tuple(DEPTH_CAT.items()))
    li=loss_usace(d); lb=pd.Series(li).groupby(bg).sum(); cov=bcov.reindex(lb.index).fillna(0); rec=np.minimum(lb,cov); hitbg=lb>0
    exp_i=take.reindex(bg).values*np.minimum(li,avgcov.reindex(bg).values); eb=pd.Series(exp_i).groupby(bg).sum()
    print(f"\n{name} ({inten} in/hr, ~1-in-{rp_of(inten):.0f} yr): flooded structures {int((d>0).sum()):,}, loss {li.sum()/1e9:.2f} bn")
    out={}
    ss_=surge_side.reindex(lb.index).fillna(False); dm_=dom_bg.reindex(lb.index).fillna(False)
    for lab,sel in [("all",hitbg),("surge-side BGs",hitbg&ss_),("pluvial-only BGs",hitbg&~ss_&dm_),("outside surge domain BGs",hitbg&~ss_&~dm_)]:
        L,R,E=lb[sel].sum(),rec[sel].sum(),eb[sel].sum(); nopol=int((cov[sel]==0).sum()); con=int(ncon.reindex(lb.index).fillna(0)[sel].sum())
        out[lab]=dict(loss_bn=L/1e9,upper_bn=R/1e9,upper_pct=100*R/L,expected_bn=E/1e9,expected_pct=100*E/L,bgs=int(sel.sum()),bgs_nopolicy=nopol,contracts_in_bgs=con,
                      loss_share_pct=100*L/lb[hitbg].sum(),contract_share_pct=100*con/len(nfip))
        print(f"  {lab:18} loss {L/1e9:5.2f} bn ({100*L/lb[hitbg].sum():3.0f}% of scenario) | upper {100*R/L:4.1f}% | expected {100*E/L:4.1f}% | BGs {int(sel.sum()):5,} no-policy {nopol:5,} ({100*nopol/max(sel.sum(),1):.0f}%) | contracts {con:6,} ({100*con/len(nfip):.1f}% of city)")
    res[name]=out
json.dump(res,open("pluvial_insurance_nsi.json","w"),indent=1,default=float); print("saved pluvial_insurance_nsi.json")
