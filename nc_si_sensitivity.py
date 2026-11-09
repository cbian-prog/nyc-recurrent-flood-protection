"""Supplementary Table 2 | Sensitivity of the headline numbers.

1 flood map basis   : effective FIRM (used), 2015 preliminary FIRM (PFIRM), either map (New York City's building code applies the
                      more restrictive of the two), modelled 1-in-100-yr reach
2 rainfall scenario : DEP extreme (2080 sea level, used) and DEP moderate (current sea level), so that the 2080 tidal boundary
                      of the extreme scenario cannot drive the result
3 design intensity  : storm-day bootstrap of the share of flood days below 1.75 in/h (a storm floods many sensors at once)
4 rain/surge split  : block groups classed as surge-side at 25/50/75% of structure value inside the 1-in-100 or 1-in-1,000-yr reach
5 take-up by map and tidal sensors inside the map, on the PFIRM
Output nc_si_sensitivity.json, tables/Table_S2.csv (and a sheet in Supplementary_Tables.xlsx)."""
import glob, json, warnings, numpy as np, pandas as pd, geopandas as gpd
import nc_events
from rasterio import features
warnings.filterwarnings("ignore")
exec(open("paper_a_core.py").read().split("# ---------------- Sandy validation")[0])
M=np.load("data/paper_a/masks_20m.npz"); sfha,s100,s1000,city=M["sfha"],M["s100"],M["s1000"],M["city"]
land=city&valid&(dem>=0)
ras=lambda geoms: features.rasterize(((g_,1) for g_ in geoms if g_ is not None),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool)
PF=gpd.read_file("data/fema_nfhl/pfirm2015_zones_nyc.geojson"); PF=PF[PF.SFHA_TF=="T"]; pfirm=ras(PF.geometry)
EF=gpd.read_file("data/fema_nfhl/nfhl_zones_nyc.geojson"); EF=EF[EF.SFHA_TF=="T"]
MAPS={"Effective FIRM (used)":sfha,"2015 preliminary FIRM":pfirm,"Either map":sfha|pfirm,"Modelled 1-in-100-yr reach":s100}
def dep_mask(pat):
    g=gpd.read_file(glob.glob(f"data/pluvial/NYCFloodStormwaterFloodMaps/{pat}/*.gdb")[0]).to_crs("EPSG:4326"); return g,ras(g[g.Flooding_Category.isin([1,2])].geometry)
SCEN={"extreme (2080 sea level, used)":"*Extreme*","moderate (current sea level)":"*Moderate*Current*"}
P=pd.read_csv("data/pluto_bsmt.csv",low_memory=False).dropna(subset=["latitude","longitude"])
P=P[P.bldgclass.str[0].isin(["A","B"])|P.bldgclass.isin(["C0","C1","C2","C3"])].reset_index(drop=True)
St=pd.DataFrame(json.load(open("data/subway/stations.json"))); U=St[St.structure=="Subway"]; U_lon=U.gtfs_longitude.astype(float).values; U_lat=U.gtfs_latitude.astype(float).values
def at(lon,lat,m):
    c=((np.asarray(lon)-tr.c)/tr.a).astype(int); r_=((np.asarray(lat)-tr.f)/tr.e).astype(int); k=(r_>=0)&(r_<m.shape[0])&(c>=0)&(c<m.shape[1]); o=np.zeros(len(c),bool); o[k]=m[r_[k],c[k]]; return o
rows=[]; OUT={"coverage":{}}
# footprint shares and lots against the FEMA maps are citywide (cells extended beyond the surge DEM, true cell area);
# against the modelled 1-in-100-yr reach they are taken inside the surge model domain, where that reach exists
import nc_citygrid as CG
MC=np.load("data/paper_a/masks_city.npz"); cityC=MC["city"]
_lat=CG.TR_C.f+(np.arange(CG.SHAPE_C[0])+.5)*CG.TR_C.e; CELLC=np.repeat(((abs(CG.TR_C.a)*111320*np.cos(np.radians(_lat)))*(abs(CG.TR_C.e)*110950))[:,None],CG.SHAPE_C[1],1)
MAPSC={"Effective FIRM (used)":MC["sfha"],"2015 preliminary FIRM":CG.ras(PF.geometry)}; MAPSC["Either map"]=MAPSC["Effective FIRM (used)"]|MAPSC["2015 preliminary FIRM"]
inwP=CG.in_window(P.longitude,P.latitude); inwU=CG.in_window(U_lon,U_lat)
dep_geo={}
for sname,pat in SCEN.items():
    g,R=dep_mask(pat); dep_geo[sname]=g; R=R&land
    RC0=CG.ras(g[g.Flooding_Category.isin([1,2])].geometry); RC=RC0&cityC; lots=CG.at(P.longitude,P.latitude,RC0); stn=CG.at(U_lon,U_lat,RC0)
    s1000_out=float((R&~s1000).sum()/R.sum()); OUT["coverage"][sname]={"area_km2":float(CELLC[RC].sum()/1e6),"outside_1000yr_reach":s1000_out}
    rows.append(("Rain footprint outside the 1-in-1,000-yr surge reach",f"rainfall {sname}; surge model domain",f"{100*s1000_out:.0f}%"))
    for mname in list(MAPSC)+["Modelled 1-in-100-yr reach"]:
        if mname in MAPSC:
            Mk=MAPSC[mname]; a=float(CELLC[RC&~Mk].sum()/CELLC[RC].sum()); L=lots; S_=stn
            l=int((L&~CG.at(P.longitude,P.latitude,Mk)).sum()); s_=int((S_&~CG.at(U_lon,U_lat,Mk)).sum()); basis=f"{mname}; rainfall {sname}"
        else:
            a=float((R&~s100).sum()/R.sum()); L=lots&inwP; S_=stn&inwU
            l=int((L&~CG.at(P.longitude,P.latitude,MC["s100"])).sum()); s_=int((S_&~CG.at(U_lon,U_lat,MC["s100"])).sum()); basis=f"{mname}; rainfall {sname}; surge model domain"
        OUT["coverage"][sname][mname]=dict(area_out=a,lots_out=l,lots=int(L.sum()),stations_out=s_,stations=int(S_.sum()))
        rows.append((f"Rain footprint outside the flood zone: area / basement lots / underground stations",basis,f"{100*a:.0f}% / {100*l/max(L.sum(),1):.0f}% ({l:,} of {int(L.sum()):,}) / {s_} of {int(S_.sum())}"))
print(json.dumps(OUT["coverage"],indent=0)[:1500])
# ---- 3. bootstrap of the share below design (storm days resampled: a storm floods many sensors at once)
SD=pd.read_parquet("data/paper_a/sensor_days_model.parquet").reset_index(drop=True); F=SD[SD.y==1]
rng=np.random.default_rng(1); days=F.date.unique()
g_all=F.groupby("date").apply(lambda d: pd.Series({"n":len(d),"b":(d.pmax<=1.75).sum(),"n12":(d.dmax>=12).sum(),"b12":((d.dmax>=12)&(d.pmax<=1.75)).sum()}))
bs=[];bs12=[]
for _ in range(2000):
    k=g_all.loc[rng.choice(days,len(days))]; bs.append(k.b.sum()/k.n.sum()); bs12.append(k.b12.sum()/max(k.n12.sum(),1))
_SS=json.load(open("nc_storm_stats.json"))   # one source for these intervals: the storm-day bootstrap in nc_storm_stats.py
OUT["below_design_ci"]=dict(any=_SS["below_design_ci95"],ge12in=_SS["below_design_1ft_ci95"])
rows.append(("Flood days below 1.75 in/h (hourly)","storm-day bootstrap, 95% interval",f"{100*(F.pmax<=1.75).mean():.0f}% ({100*OUT['below_design_ci']['any'][0]:.0f}–{100*OUT['below_design_ci']['any'][1]:.0f}%); at 1 ft {100*(F[F.dmax>=12].pmax<=1.75).mean():.0f}% ({100*OUT['below_design_ci']['ge12in'][0]:.0f}–{100*OUT['below_design_ci']['ge12in'][1]:.0f}%)"))
print("bootstrap",OUT["below_design_ci"])
# ---- 4. rain-side / surge-side split
bg=s.cbfips.astype(str).str[:12].values
nf=pd.read_parquet("data/nfip/all_current.parquet"); bcov=nf.groupby("censusGeoid").totalBuildingInsuranceCoverage.sum(); ncon=nf.groupby("censusGeoid").size()
n_bg=pd.Series(bg).value_counts(); take=(ncon.reindex(n_bg.index).fillna(0)/n_bg).clip(upper=1); avgcov=(bcov/ncon).reindex(n_bg.index).fillna(0)
curve=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in curve)); WLm=np.array([curve[str(t)] for t in T])
def struct_loss(g): return loss_usace(CG.dep_depth(g,s.x.values,s.y.values))   # citywide rain depths
v_bg=pd.Series(val).groupby(bg).sum(); OUT["split"]={}
dom=((pd.Series(val*CG.in_window(s.x.values,s.y.values)).groupby(bg).sum()/v_bg)>=0.5)   # block groups where surge is modelled
for sname,g in dep_geo.items():
    li=struct_loss(g); lb=pd.Series(li).groupby(bg).sum(); hitbg=lb>0; eb=pd.Series(take.reindex(bg).values*np.minimum(li,avgcov.reindex(bg).values)).groupby(bg).sum()
    cov=bcov.reindex(lb.index).fillna(0); rec=np.minimum(lb,cov)
    for rp in (100,1000):
        j=int(np.argmin(abs(levels-(float(np.interp(np.log(rp),np.log(T),WLm))-NAVD_MINUS_MSL)))); share=(pd.Series(val*(depths[j]>0)).groupby(bg).sum()/v_bg)
        for thr in (0.25,0.5,0.75):
            ss=(share>=thr).reindex(lb.index).fillna(False); rs=hitbg&~ss&dom.reindex(lb.index).fillna(False); su=hitbg&ss
            con=ncon.reindex(lb.index).fillna(0)
            o=dict(loss_rain_pct=float(100*lb[rs].sum()/lb[hitbg].sum()),contracts_rain_pct=float(100*con[rs].sum()/len(nf)),contracts_surge_pct=float(100*con[su].sum()/len(nf)),
                   exp_rain_pct=float(100*eb[rs].sum()/lb[rs].sum()),exp_surge_pct=float(100*eb[su].sum()/lb[su].sum()),
                   up_rain_pct=float(100*rec[rs].sum()/lb[rs].sum()),up_surge_pct=float(100*rec[su].sum()/lb[su].sum()),nopolicy_rain_bgs_pct=float(100*(cov[rs]==0).mean()))
            OUT["split"][f"{sname}|{rp}|{thr}"]=o
            rows.append(("Rain-side share of rain loss / of NFIP contracts; expected insurable share rain side vs surge side",
                         f"rainfall {sname}; surge-side if ≥{int(100*thr)}% of value inside the 1-in-{rp:,}-yr reach"+(" (used)" if (rp==1000 and thr==0.5 and sname.startswith('extreme')) else ""),
                         f"{o['loss_rain_pct']:.0f}% / {o['contracts_rain_pct']:.0f}%; {o['exp_rain_pct']:.1f}% vs {o['exp_surge_pct']:.1f}%"))
print("split",{k:{a:round(b,1) for a,b in v.items()} for k,v in OUT["split"].items()})
# ---- 5. take-up and tidal sensors on the PFIRM (point in polygon)
def inpoly(lon,lat,polys):
    gg=gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon,lat),crs="EPSG:4326"); j_=gpd.sjoin(gg,polys[["geometry"]].reset_index(drop=True),how="left",predicate="within")
    return ~j_.index_right.isna().groupby(level=0).all().values
av=nf.ratedFloodZone.astype(str).str[0].isin(["A","V"])
for mname,polys in (("Effective FIRM (used)",EF),("2015 preliminary FIRM",PF)):
    ins=inpoly(s.x.values,s.y.values,polys)
    if mname.startswith("2015"):
        tu=dict(in_map=float(100*av.sum()/ins.sum()),outside=float(100*(~av).sum()/(~ins).sum()),structures_in=int(ins.sum()))
        OUT["takeup_pfirm"]=tu; rows.append(("NFIP contracts per 100 structures, inside / outside the flood zone","2015 preliminary FIRM (A/V-rated contracts vs structures inside)",f"{tu['in_map']:.0f} / {tu['outside']:.1f} ({tu['structures_in']:,} structures inside)"))
Sx=nc_events.sensors(); Tt=Sx[Sx.tidally_influenced=="Yes"]
tin=inpoly(Tt.longitude.astype(float).values,Tt.latitude.astype(float).values,PF); OUT["tidal_in_pfirm"]=int(tin.sum())
tef=inpoly(Tt.longitude.astype(float).values,Tt.latitude.astype(float).values,EF); OUT["tidal_in_effective"]=int(tef.sum())
rows.append(("Tidal sensors inside the flood zone","2015 preliminary FIRM",f"{int(tin.sum())} of {len(Tt)} (effective FIRM: {int(tef.sum())} of {len(Tt)})"))
OUT["pfirm_land_km2"]=float((pfirm&land).sum()*400/1e6); OUT["sfha_land_km2"]=float((sfha&land).sum()*400/1e6)
json.dump(OUT,open("nc_si_sensitivity.json","w"),indent=1)
T7=pd.DataFrame(rows,columns=["Quantity","Basis","Value"]); T7.to_csv("tables/Table_S2.csv",index=False)
# the workbook is written by make_tables.py (run it after this script)
print(T7.to_string())
