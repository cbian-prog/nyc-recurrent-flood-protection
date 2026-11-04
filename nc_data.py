"""Numbers for the restructured paper (Fig. 1c, Fig. 3, and text), computed once and saved to nc_numbers.json.

1. Observed flood frequency by source: FloodNet events per sensor-year (tidal / non-tidal sensors), from deployment dates.
2. Rain at sensors (MRMS): flood rate by max-hourly-rain bin on rainy sensor-days; share of flood days below the 1.75 in/h design intensity.
3. Expected street-ponding days per year at lots inside the DEP extreme footprint: rainy days per year by max-hourly bin at the
   lot's MRMS cell x the flood rate in that bin. Rates are conditional on a rainy day (daily total >= 0.1 in), so only rainy days are counted.
   Between-sensor differences in the rain-to-flood response are estimated with a random-intercept logistic model (sensor intercept,
   rain-bin fixed effects) and carried to the lots by giving each lot a sensor-type deviation drawn from that distribution.
All FloodNet event times (published in UTC) and all radar hours are converted to New York local time before days are formed.
4. Return period of the 1.75 in/h design intensity from NOAA Atlas 14 (Central Park, 60-min).
5. Coverage of the protection instruments: FEMA effective flood map (NFHL Special Flood Hazard Area) and the modelled
   1-in-100-yr still-water reach, against the rainfall footprint, rain-exposed basement lots and underground stations.
"""
import glob, json, warnings, numpy as np, pandas as pd, geopandas as gpd, rasterio
import nc_events   # sensor table with the paper's tidal flag (nc_events.sensors)
from rasterio import features; from scipy import ndimage
warnings.filterwarnings("ignore")
OUT={}
# ---------------------------------------------------------------- 1. events per sensor-year
S=nc_events.sensors(); E=pd.DataFrame(json.load(open("data/floodnet/events.json")))
import nc_events; E=nc_events.load(); MIN_DRAIN=nc_events.MIN_DRAIN   # a flood is an event that takes >= 30 min to drain (nc_events.py)
print(f"events counted as floods (drain >= {MIN_DRAIN:.0f} min): {len(E):,}")
E["t"]=pd.to_datetime(E.flood_start_time).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None); END=E.t.max().normalize()  # local time
S["inst"]=pd.to_datetime(S.date_installed,errors="coerce"); S["rem"]=pd.to_datetime(S.get("date_removed"),errors="coerce").fillna(END)
S["yrs"]=((S.rem.clip(upper=END)-S.inst).dt.days/365.25).clip(lower=0)
S["n"]=S.sensor_id.map(E.groupby("sensor_id").size()).fillna(0); S["tidal"]=S.tidally_influenced.eq("Yes")
S=S[S.yrs>=0.5]  # at least six months online
for lab,m in (("tidal",S.tidal),("nontidal",~S.tidal)):
    r=S.n[m]/S.yrs[m]
    OUT[f"events_per_sensor_year_{lab}"]=dict(n_sensors=int(m.sum()),pooled=float(S.n[m].sum()/S.yrs[m].sum()),median=float(r.median()),q25=float(r.quantile(.25)),q75=float(r.quantile(.75)),
                                              share_with_event=float((S.n[m]>0).mean()))
OUT["record_end"]=str(END.date()); OUT["events_total"]=int(len(E)); OUT["events_tidal_share"]=float(E.sensor_id.isin(nc_events.sensors().query("tidally_influenced == 'Yes'").sensor_id).mean())   # all tidal sensors, not only those online six months
# flood days (distinct days with an event) per sensor-year, by maximum event depth: same unit for tidal and rain sensors
E["depth"]=pd.to_numeric(E.max_depth_inches,errors="coerce"); E["day"]=E.t.dt.normalize()
DEPTHS={"any":0.0,"4in":4.0,"12in":12.0}
for dk,thr in DEPTHS.items():
    nd=S.sensor_id.map(E[E.depth>=thr].groupby("sensor_id").day.nunique()).fillna(0)
    for lab,m in (("tidal",S.tidal),("nontidal",~S.tidal)):
        r=nd[m]/S.yrs[m]
        OUT[f"flood_days_per_sensor_year_{lab}_{dk}"]=dict(n_sensors=int(m.sum()),median=float(r.median()),q25=float(r.quantile(.25)),q75=float(r.quantile(.75)),
                                                            q10=float(r.quantile(.1)),q90=float(r.quantile(.9)),pooled=float(nd[m].sum()/S.yrs[m].sum()))
Sall=nc_events.sensors(); Sall["tidal"]=Sall.tidally_influenced.eq("Yes")
for lab,m in (("tidal",Sall.tidal),("nontidal",~Sall.tidal)):
    d=E.depth[E.sensor_id.isin(Sall.sensor_id[m])]
    OUT[f"event_depth_{lab}"]=dict(n_events=int(len(d)),median_in=float(d.median()),share_ge4=float((d>=4).mean()),share_ge12=float((d>=12).mean()))
print({k:v for k,v in OUT.items() if k.startswith("events")})
# ---------------------------------------------------------------- 2. sensor-day table (as zhao_test2_mrms.py)
MM=25.4; DESIGN=1.75
Sn=nc_events.sensors(); Sn["lat"]=pd.to_numeric(Sn.latitude,errors="coerce"); Sn["lon"]=pd.to_numeric(Sn.longitude,errors="coerce")
Sn=Sn.dropna(subset=["lat","lon"]); Sn=Sn[Sn.tidally_influenced!="Yes"].reset_index(drop=True); Sn["installed"]=pd.to_datetime(Sn.date_installed); Sn["removed"]=pd.to_datetime(Sn.get("date_removed"),errors="coerce").fillna(pd.Timestamp("2100-01-01"))   # days after removal are not observed
files=sorted(glob.glob("data/mrms/*.npz")); z0=np.load(files[0]); a,b_,c0,d_,e,f0=z0["transform"]; shape=z0["h00"].shape
mc=((Sn.lon.values-c0)/a).astype(int); mr=((Sn.lat.values-f0)/e).astype(int); ok=(mr>=0)&(mr<shape[0])&(mc>=0)&(mc<shape[1]); Sn=Sn[ok].reset_index(drop=True); mc,mr=mc[ok],mr[ok]
hours,vals=[],[]; LD={}   # local day -> [max, total, n hours] for every radar cell
for f in files:
    z=np.load(f); day=pd.Timestamp(f.split("/")[-1][:8])
    for k in z.files:
        if not k.startswith("h"): continue
        v=z[k].astype("float32"); v[v<0]=0; t_utc=day+pd.Timedelta(hours=int(k[1:])); hours.append(t_utc); vals.append(v[mr,mc])
        ld=pd.Timestamp(t_utc,tz="UTC").tz_convert("America/New_York").tz_localize(None).normalize()
        if ld not in LD: LD[ld]=[v/MM,v/MM,1]
        else: LD[ld][0]=np.maximum(LD[ld][0],v/MM); LD[ld][1]=LD[ld][1]+v/MM; LD[ld][2]+=1
ldays=sorted(d for d,x in LD.items() if x[2]>=20)
daymax=[LD[d][0] for d in ldays]; daytot=[LD[d][1] for d in ldays]
H=pd.DataFrame(np.array(vals)/MM,index=pd.DatetimeIndex(hours,tz="UTC"),columns=Sn.sensor_id).tz_convert("America/New_York"); H.index=H.index.tz_localize(None)
cnt=H.groupby(H.index.normalize()).size(); days=cnt[cnt>=20].index
pmax=H.groupby(H.index.normalize()).max().loc[days]; ptot=H.groupby(H.index.normalize()).sum().loc[days]
E["date"]=E.t.dt.normalize(); rows=[]
for i,s in Sn.iterrows():
    sid=s.sensor_id; dd=pmax.index[(pmax.index>=s.installed+pd.Timedelta(days=7))&(pmax.index<=s.removed)&(ptot[sid]>=0.1)]
    if len(dd)==0: continue
    dmax=E[E.sensor_id==sid].groupby("date").depth.max()
    rows.append(pd.DataFrame({"sid":sid,"date":dd,"y":[int(x in dmax.index) for x in dd],"dmax":[float(dmax.get(x,0)) for x in dd],"pmax":pmax.loc[dd,sid].values}))
SD=pd.concat(rows); fl=SD[SD.y==1]
# covariates for the storm-and-place model (nc_storm_model.py): Battery water level above MHHW within 2 h of the day's peak rain hour
# at the sensor, and whether the sensor lies within 1 km of open water (55 m DEM water mask)
from scipy import ndimage as _ndi
_T=pd.read_csv("data/coops/battery_hourly_mhhw_utc.csv"); _T["t"]=pd.to_datetime(_T.t).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
_T=_T.set_index("t").v.astype(float).rolling(5,center=True).max(); _T=_T[~_T.index.duplicated()]
_peak=H.groupby(H.index.normalize()).idxmax()
SD["peak_hour"]=[_peak.at[d,s_] if d in _peak.index else pd.NaT for d,s_ in zip(SD.date,SD.sid)]
SD["tide_mhhw"]=_T.reindex(SD.peak_hour).values; SD["hi"]=(SD.tide_mhhw>0).astype(int)
with rasterio.open("data/nyc_dem_55m.tif") as _r: _D,_Dt,_Dn=_r.read(1),_r.transform,_r.nodata
WATER_DIST=_ndi.distance_transform_edt(~((_D==_Dn)|(_D<0.5)))*55.0
def to_water(lon,lat): return WATER_DIST[((np.asarray(lat)-_Dt.f)/_Dt.e).astype(int),((np.asarray(lon)-_Dt.c)/_Dt.a).astype(int)]
_sl=Sn.set_index("sensor_id"); SD["to_water"]=to_water(_sl.loc[SD.sid,"lon"].values,_sl.loc[SD.sid,"lat"].values); SD["near"]=(SD.to_water<1000).astype(int)
SD.drop(columns=["peak_hour"]).to_parquet("data/paper_a/sensor_days_model.parquet")
BINS=((0,0.5),(0.5,1.0),(1.0,1.75),(1.75,2.5))
rate={f"{lo}-{hi}":float(SD.y[(SD.pmax>lo)&(SD.pmax<=hi)].mean()) for lo,hi in BINS}
rate_d={dk:{f"{lo}-{hi}":float(((SD.y==1)&(SD.dmax>=thr))[(SD.pmax>lo)&(SD.pmax<=hi)].mean()) for lo,hi in BINS} for dk,thr in DEPTHS.items()}
fl12=SD[(SD.y==1)&(SD.dmax>=12)]
OUT["flood_rate_by_bin_by_depth"]=rate_d
OUT["flood_days_ge12in"]=int(len(fl12)); OUT["flood_days_ge12in_below_design"]=float((fl12.pmax<=DESIGN).mean())
nbin={f"{lo}-{hi}":int(((SD.pmax>lo)&(SD.pmax<=hi)).sum()) for lo,hi in BINS}
OUT.update(sensor_days=int(len(SD)),flood_days=int(SD.y.sum()),n_sensors_rain=int(SD.sid.nunique()),flood_rate_by_bin=rate,sensor_days_by_bin=nbin,
           flood_days_below_design=float((fl.pmax<=DESIGN).mean()),flood_days_below_design_n=int((fl.pmax<=DESIGN).sum()))
print("sensor-days",len(SD),"flood days",int(SD.y.sum()),"below design",OUT["flood_days_below_design"],rate)
# between-sensor variation of the rain-to-flood response: random-intercept logistic model, any depth
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
SD["bin"]=pd.cut(SD.pmax,[0,0.5,1.0,1.75,99],labels=["b0","b1","b2","b3"])
mm=BinomialBayesMixedGLM.from_formula("y ~ C(bin)",{"sensor":"0+C(sid)"},SD).fit_vb()
SD_SENSOR=float(np.exp(mm.vcp_mean[0]))
lg=lambda p: np.log(p/(1-p)); ex=lambda x: 1/(1+np.exp(-x))
def band(r,q): return float(ex(lg(r)+q*SD_SENSOR)) if 0<r<1 else float(r)
OUT["between_sensor_sd_logodds"]=SD_SENSOR
OUT["flood_rate_band_by_bin"]={dk:{k:dict(p10=band(v,-1.2816),p90=band(v,1.2816)) for k,v in rd.items()} for dk,rd in [("any",rate)]+[(k,rate_d[k]) for k in ("12in",)]}
print("between-sensor SD (log-odds)",round(SD_SENSOR,2),OUT["flood_rate_band_by_bin"]["any"])
# ---------------------------------------------------------------- 3. ponding days per year at DEP-extreme lots (rainy days only)
daymax=np.stack(daymax); daytot=np.stack(daytot)
span_yr=(pd.Timestamp(files[-1].split("/")[-1][:8])-pd.Timestamp(files[0].split("/")[-1][:8])).days/365.25
rainy=daytot>=0.1
def ponding(rt):
    out=np.zeros(shape)
    for lo,hi,r_ in [(lo,hi,rt[f"{lo}-{hi}"]) for lo,hi in BINS]+[(2.5,99,rt["1.75-2.5"])]: out+=r_*(rainy&(daymax>lo)&(daymax<=hi)).sum(0)/span_yr
    return out
pond=ponding(rate); pond_d={dk:ponding(rate_d[dk]) for dk in DEPTHS}
rainy_days_yr=rainy.sum(0)/span_yr
with rasterio.open("data/manhattan_dem_20m.tif") as r: dem,tr,nod=r.read(1),r.transform,r.nodata
gd=gpd.read_file(glob.glob("data/pluvial/NYCFloodStormwaterFloodMaps/*Extreme*/*.gdb")[0]).to_crs("EPSG:4326")
dep=features.rasterize(((x,1) for x in gd[gd.Flooding_Category.isin([1,2])].geometry),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool)
P=pd.read_csv("data/pluto_bsmt.csv",low_memory=False).dropna(subset=["latitude","longitude"])
P=P[P.bldgclass.str[0].isin(["A","B"])|P.bldgclass.isin(["C0","C1","C2","C3"])].reset_index(drop=True)
def at(lon,lat,m,t=tr):
    c=((np.asarray(lon)-t.c)/t.a).astype(int); r_=((np.asarray(lat)-t.f)/t.e).astype(int); k=(r_>=0)&(r_<m.shape[0])&(c>=0)&(c<m.shape[1]); o=np.zeros(len(c),m.dtype); o[k]=m[r_[k],c[k]]; return o
import nc_citygrid as CG   # lots are matched on the same cells extended to the five boroughs (the surge DEM stops at 74.12 W)
depC=CG.ras(gd[gd.Flooding_Category.isin([1,2])].geometry); inrain=CG.at(P.longitude,P.latitude,depC)
pm=((P.latitude.values-f0)/e).astype(int); pc=((P.longitude.values-c0)/a).astype(int); okp=(pm>=0)&(pm<shape[0])&(pc>=0)&(pc<shape[1])
pd_lot=np.full(len(P),np.nan); pd_lot[okp]=pond[pm[okp],pc[okp]]; v=pd_lot[inrain&okp]
# rainy days per year in each bin at each lot's radar cell, and a sensor-type deviation per lot (between-sensor distribution)
EDG=[(0,0.5),(0.5,1.0),(1.0,1.75),(1.75,99)]
cnt_cell=np.stack([(rainy&(daymax>lo)&(daymax<=hi)).sum(0)/span_yr for lo,hi in EDG])
sel=inrain&okp; nl=cnt_cell[:,pm[sel],pc[sel]]
np.savez("data/paper_a/lot_rain_bins.npz",nl=nl,lon=P.longitude.values[sel],lat=P.latitude.values[sel],to_water=to_water(P.longitude.values[sel],P.latitude.values[sel]),
         rainy_hi_share=np.array([float(SD.hi[(SD.near==1)&(SD.pmax>lo)&(SD.pmax<=hi)].mean()) for lo,hi in EDG]),span_yr=span_yr)   # for nc_storm_model.py
u=np.random.default_rng(0).normal(0,SD_SENSOR,sel.sum())
def lot_freq(rt,shift):
    rr=np.array([rt[f"{lo}-{hi}"] for lo,hi in BINS])[:,None]
    pr=np.where((rr>0)&(rr<1),ex(lg(np.clip(rr,1e-9,1-1e-9))+shift[None,:]),rr)
    return (pr*nl).sum(0)
OUT["ponding_days_per_year_rain_lots"]=dict(median=float(np.median(v)),q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75)),n_lots=int(len(v)),
                                             rainy_days_per_year_median=float(np.median(rainy_days_yr[pm[sel],pc[sel]])),span_years=span_yr)
for dk in DEPTHS:
    vv=pond_d[dk][pm[sel],pc[sel]]; vs=lot_freq(rate_d[dk],u)
    OUT[f"ponding_days_per_year_rain_lots_{dk}"]=dict(median=float(np.median(vv)),q25=float(np.quantile(vv,.25)),q75=float(np.quantile(vv,.75)),
        q10_rain_only=float(np.quantile(vv,.1)),q90_rain_only=float(np.quantile(vv,.9)),
        q10=float(np.quantile(vs,.1)),q90=float(np.quantile(vs,.9)),median_with_site=float(np.median(vs)))
print("ponding",OUT["ponding_days_per_year_rain_lots"])
# ---------------------------------------------------------------- 4. design intensity return period (Atlas 14, Central Park 60-min)
ARI=np.array([1,2,5,10,25,50,100,200,500,1000]); D60=np.array([1.07,1.28,1.62,1.90,2.28,2.57,2.88,3.22,3.71,4.11])
rp=lambda i: float(np.exp(np.interp(np.log(i),np.log(D60),np.log(ARI))))
OUT["design_rp_yr"]=rp(DESIGN); OUT["dep_moderate_rp_yr"]=rp(2.13); OUT["dep_extreme_rp_yr"]=rp(3.66)
# ---------------------------------------------------------------- 5. protection coverage
bg=gpd.read_file("data/nfip/tl_2020_36_bg.shp"); bg=bg[bg.COUNTYFP.isin(["061","047","081","005","085"])].to_crs("EPSG:4326"); boro=bg.dissolve(by="COUNTYFP"); boro["geometry"]=boro.geometry.buffer(0)
city=features.rasterize(((x,1) for x in boro.geometry),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool)
Z=gpd.read_file("data/fema_nfhl/nfhl_zones_nyc.geojson"); Z=Z[Z.SFHA_TF=="T"]
sfha=features.rasterize(((g,1) for g in Z.geometry if g is not None),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool)
valid=dem!=nod; dem_v=np.where(valid,dem,1e4); seed=np.zeros(dem.shape,bool); seed[0,:]=seed[-1,:]=seed[:,0]=seed[:,-1]=True; seed&=dem_v<-1
curve=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in curve)); WLm=np.array([curve[str(t)] for t in T])
wl=lambda r: float(np.interp(np.log(r),np.log(T),WLm))-0.063
def surge(w):
    below=(dem_v<w)&valid; lab,_=ndimage.label(below); return np.isin(lab,np.unique(lab[seed&(lab>0)]))&below&(dem_v>=0)
s100=surge(wl(100)); s1000=surge(wl(1000))
rain=dep&city&valid&(dem>=0)
# true cell area: the grid is 0.00018 deg (about 15 x 20 m), not 20 x 20 m; city-wide footprint statistics are in nc_areas.py
_lat=tr.f+(np.arange(dem.shape[0])+.5)*tr.e; CELL=np.repeat(((abs(tr.a)*111320*np.cos(np.radians(_lat)))*(abs(tr.e)*110950))[:,None],dem.shape[1],1)
OUT["rain_footprint_km2"]=float(CELL[rain].sum()/1e6)
OUT["rain_area_outside_sfha"]=float((rain&~sfha).sum()/rain.sum()); OUT["rain_area_outside_s100"]=float((rain&~s100).sum()/rain.sum()); OUT["rain_area_outside_s1000"]=float((rain&~s1000).sum()/rain.sum())
OUT["sfha_land_km2"]=float(CELL[sfha&city&valid&(dem>=0)].sum()/1e6); OUT["s100_land_km2"]=float(CELL[s100&city].sum()/1e6)
OUT["sfha_vs_s100_overlap"]=dict(sfha_in_s100=float((sfha&s100&city).sum()/(sfha&city&valid&(dem>=0)).sum()),s100_in_sfha=float((sfha&s100&city).sum()/(s100&city).sum()))
sfhaC=CG.ras(Z.geometry); s100C=CG.embed(s100); s1000C=CG.embed(s1000)
lots_rain=inrain.astype(bool); lots_sfha=CG.at(P.longitude,P.latitude,sfhaC).astype(bool); lots_s100=CG.at(P.longitude,P.latitude,s100C).astype(bool)
inw=CG.in_window(P.longitude,P.latitude)   # comparisons with the modelled surge reach are made where it is modelled
OUT["basement_lots_rain_outside_surge_window"]=int((lots_rain&~inw).sum()); OUT["basement_lots_rain_in_domain"]=int((lots_rain&inw).sum())
OUT["basement_lots_rain"]=int(lots_rain.sum()); OUT["basement_lots_rain_outside_sfha"]=int((lots_rain&~lots_sfha).sum()); OUT["basement_lots_rain_outside_s100"]=int((lots_rain&inw&~lots_s100).sum())
St=pd.DataFrame(json.load(open("data/subway/stations.json"))); St["lat"]=St.gtfs_latitude.astype(float); St["lon"]=St.gtfs_longitude.astype(float); U=St[St.structure=="Subway"]
us_rain=CG.at(U.lon,U.lat,depC).astype(bool); us_sfha=CG.at(U.lon,U.lat,sfhaC).astype(bool); us_s100=CG.at(U.lon,U.lat,s100C).astype(bool)
OUT["stations_underground"]=int(len(U)); OUT["stations_rain"]=int(us_rain.sum()); OUT["stations_rain_outside_sfha"]=int((us_rain&~us_sfha).sum()); OUT["stations_rain_outside_s100"]=int((us_rain&~us_s100).sum())
OUT["stations_s100"]=int(us_s100.sum()); OUT["stations_rain_and_s100"]=int((us_rain&us_s100).sum())
nf=pd.read_parquet("data/nfip/all_current.parquet"); z=nf.ratedFloodZone.astype(str).str[0]
OUT["policies_in_AV_share"]=float(z.isin(["A","V"]).mean()); OUT["policies_mandatory_share"]=float(nf.mandatoryPurchaseFlag.astype(bool).mean())
# ---------------------------------------------------------------- 6. tide: where the most frequent floods are, relative to the map
Zs=Z[["geometry"]].reset_index(drop=True)
def in_sfha(lon,lat):
    g=gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon,lat),crs="EPSG:4326"); j=gpd.sjoin(g,Zs,how="left",predicate="within")
    return ~j.index_right.isna().groupby(level=0).all().values
Ts=Sall[Sall.tidal].copy(); Ts["lon"]=Ts.longitude.astype(float); Ts["lat"]=Ts.latitude.astype(float)
Ts["sfha"]=in_sfha(Ts.lon.values,Ts.lat.values); Ts["s100"]=at(Ts.lon,Ts.lat,s100).astype(bool); Ts["s1000"]=at(Ts.lon,Ts.lat,s1000).astype(bool)
tdays=E[E.sensor_id.isin(Ts.sensor_id)].groupby("sensor_id").day.nunique()
OUT["tidal_sensors_geography"]=dict(n=int(len(Ts)),in_sfha=int(Ts.sfha.sum()),in_s100=int(Ts.s100.sum()),in_s1000=int(Ts.s1000.sum()),
    flood_days_at_sensors_in_sfha=float(tdays[tdays.index.isin(Ts.sensor_id[Ts.sfha])].sum()/tdays.sum()),
    flood_days_at_sensors_in_s100=float(tdays[tdays.index.isin(Ts.sensor_id[Ts.s100])].sum()/tdays.sum()))
yrs_b,wl_b=np.load("data/battery_annual_max.npy"); mx=wl_b[yrs_b>=2020].max()
OUT["battery_max_since_2020"]=dict(m_navd88=float(mx),year=int(yrs_b[yrs_b>=2020][np.argmax(wl_b[yrs_b>=2020])]),rp_yr=float(np.exp(np.interp(mx+0.063,WLm,np.log(T)))))
# ---------------------------------------------------------------- 7. surge frequency at a place: first return period at which a basement lot's cell floods to a given depth
RPL=np.logspace(0,4,81); first={dk:np.full(dem.shape,np.inf) for dk in DEPTHS}
for r_ in RPL:
    w=wl(r_); W=surge(w); dd=w-dem_v
    for dk,thr in DEPTHS.items():
        hit=W&(dd>=max(thr*0.0254,1e-6))&np.isinf(first[dk]); first[dk][hit]=r_
lots100=lots_s100&at(P.longitude,P.latitude,city).astype(bool)
for dk in DEPTHS:
    rp_l=at(P.longitude,P.latitude,first[dk])[lots100]; f=1/rp_l   # floods per year (0 if never within 1e4 yr)
    OUT[f"surge_floods_per_year_s100_lots_{dk}"]=dict(n_lots=int(lots100.sum()),median=float(np.median(f)),q25=float(np.quantile(f,.25)),q75=float(np.quantile(f,.75)),
                                                       q10=float(np.quantile(f,.1)),q90=float(np.quantile(f,.9)))
# ---------------------------------------------------------------- 7b. model against model (Fig. 2c): basement lots reached by each modelled footprint at the same
# depth thresholds (4 in, 1 ft), both counted at the lot's cell where surge is modelled; surge by return period, rainfall at the two DEP scenarios
RPC=[2,5,10,20,50,100,200,500,1000]; rpl={dk:at(P.longitude,P.latitude,first[dk]) for dk in ("4in","12in")}
gmod=gpd.read_file(glob.glob("data/pluvial/NYCFloodStormwaterFloodMaps/*Moderate*Current*/*.gdb")[0]).to_crs("EPSG:4326")
dmod=CG.dep_depth(gmod,P.longitude.values,P.latitude.values); dext=CG.dep_depth(gd,P.longitude.values,P.latitude.values)
OUT["model_exposure"]=dict(surge=dict(rp=RPC,**{dk:[int(((rpl[dk]<=R)&np.isfinite(rpl[dk])&inw).sum()) for R in RPC] for dk in ("4in","12in")}),
    rain=dict(moderate=dict(rp=OUT["dep_moderate_rp_yr"],**{"4in":int(((dmod>0)&inw).sum()),"12in":int(((dmod>=0.46)&inw).sum())}),
              extreme=dict(rp=OUT["dep_extreme_rp_yr"],**{"4in":int(((dext>0)&inw).sum()),"12in":int(((dext>=0.46)&inw).sum())})))
print("model exposure",OUT["model_exposure"])
# ---------------------------------------------------------------- 8. does insurance follow the map? NFIP take-up inside and outside the SFHA, within Sandy-flooded and other block groups
NSI=pd.read_parquet("data/nsi_nyc/nsi.parquet"); NSI["bg"]=NSI.cbfips.astype(str).str[:12]
NSI["sfha"]=in_sfha(NSI.x.values,NSI.y.values)
sz=gpd.read_file("data/sandy_zone.geojson")[["geometry"]]; gN=gpd.GeoDataFrame(geometry=gpd.points_from_xy(NSI.x,NSI.y),crs="EPSG:4326")
js=gpd.sjoin(gN,sz,how="left",predicate="within"); NSI["sandy"]=~js.index_right.isna().groupby(level=0).all().values
nf["bg"]=nf.censusGeoid.astype(str).str[:12]; nf["av"]=z.isin(["A","V"]).values
sandy_bg=NSI.groupby("bg").sandy.mean()>=0.5
def takeup(bgs=None):
    n_=NSI if bgs is None else NSI[NSI.bg.isin(bgs)]; p_=nf if bgs is None else nf[nf.bg.isin(bgs)]
    return dict(in_sfha=float(100*p_.av.sum()/max(n_.sfha.sum(),1)),outside=float(100*(~p_.av).sum()/max((~n_.sfha).sum(),1)),
                structures_in=int(n_.sfha.sum()),structures_out=int((~n_.sfha).sum()),contracts_av=int(p_.av.sum()),contracts_other=int((~p_.av).sum()))
OUT["takeup_per100_by_map"]=dict(all=takeup(),sandy_flooded_bgs=takeup(sandy_bg.index[sandy_bg]),other_bgs=takeup(sandy_bg.index[~sandy_bg]),
                                  structures_sfha_and_sandy=int((NSI.sfha&NSI.sandy).sum()))
OUT["min_drain_min"]=MIN_DRAIN; json.dump(OUT,open(f"nc_numbers{nc_events.suffix()}.json","w"),indent=1); print(json.dumps({k:v for k,v in OUT.items() if not isinstance(v,dict)},indent=0))
if nc_events.suffix()=="": np.savez_compressed("data/paper_a/masks_20m.npz",sfha=sfha,s100=s100,s1000=s1000,rain=dep,city=city)
if nc_events.suffix()=="": np.savez_compressed("data/paper_a/masks_city.npz",sfha=sfhaC,s100=s100C,s1000=s1000C,rain=depC,city=CG.ras(boro.geometry))
