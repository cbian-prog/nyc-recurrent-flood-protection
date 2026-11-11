"""Fig. 1 | Hazard: two geographies, three frequencies (R1).
a  source map: surge reach at 1-in-10/100/1,000-yr still-water levels, DEP rainfall footprints, tidal reach at +3 ft SLR, FloodNet sensors, subway routes
b  observed flood probability per rainy day at non-tidal sensors, by maximum hourly radar rain (MRMS) at the sensor; design intensity marked
c  frequency ladder: how often a place floods, by source (filled = observed, open = modelled)"""
import glob, json, warnings, numpy as np, pandas as pd, rasterio, geopandas as gpd
import nc_events   # sensor table with the paper's tidal flag (nc_events.sensors)
from rasterio import features; from scipy import ndimage; from shapely.geometry import LineString
warnings.filterwarnings("ignore")
from nc_style import *
from matplotlib.colors import ListedColormap; from matplotlib.patches import Patch; from matplotlib.lines import Line2D
N=json.load(open("nc_numbers.json")); SH=json.load(open("nc_si_subhourly.json"))
# ---------------- map layers (as fig1_source_map.py)
curve=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in curve)); WLm=np.array([curve[str(t)] for t in T])
wl=lambda rp: float(np.interp(np.log(rp),np.log(T),WLm))-0.063
with rasterio.open("data/manhattan_dem_20m.tif") as r: dem,tr,nod=r.read(1),r.transform,r.nodata
valid=dem!=nod; dem_v=np.where(valid,dem,1e4); seed=np.zeros(dem.shape,bool); seed[0,:]=seed[-1,:]=seed[:,0]=seed[:,-1]=True; seed&=dem_v<-1
def wet(w):
    below=(dem_v<w)&valid; lab,_=ndimage.label(below); return np.isin(lab,np.unique(lab[seed&(lab>0)]))&below&(dem_v>=0)
bgs=gpd.read_file("data/nfip/tl_2020_36_bg.shp"); bgs=bgs[bgs.COUNTYFP.isin(["061","047","081","005","085"])].to_crs("EPSG:4326")
nyc=features.rasterize(((g,1) for g in bgs.geometry.buffer(0.001)),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool)
surge=np.zeros(dem.shape,"int8")
for rp,code in ((1000,1),(100,2),(10,3)): surge[wet(wl(rp))&nyc]=code
dext=[tr.c,tr.c+dem.shape[1]*tr.a,tr.f+dem.shape[0]*tr.e,tr.f]
gd=gpd.read_file(glob.glob("data/pluvial/NYCFloodStormwaterFloodMaps/*Extreme*/*.gdb")[0]).to_crs("EPSG:4326")
gm=gpd.read_file(glob.glob("data/pluvial/NYCFloodStormwaterFloodMaps/*Moderate*Current*/*.gdb")[0]).to_crs("EPSG:4326")
pl_ext=features.rasterize(((x,1) for x in gd[gd.Flooding_Category.isin([1,2])].geometry),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool)
pl_mod=features.rasterize(((x,1) for x in gm[gm.Flooding_Category.isin([1,2])].geometry),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool)
with rasterio.open("data/slr/nyc55_3ft.tif") as r: f3=r.read(1); s3=(f3>=.5); s3ext=[r.transform.c,r.transform.c+r.width*r.transform.a,r.transform.f+r.height*r.transform.e,r.transform.f]
with rasterio.open("data/slr/nyc55_0ft.tif") as r: f0=r.read(1)
s3new=s3&~(f0>=.5)
S=nc_events.sensors(); S["lat"]=pd.to_numeric(S.latitude,errors="coerce"); S["lon"]=pd.to_numeric(S.longitude,errors="coerce")
import nc_events; E=nc_events.load(); S["n"]=S.sensor_id.map(E.groupby("sensor_id").size()).fillna(0)
# observed flood days per year at each sensor (as in nc_data.py: distinct local days with a counted flood, over the years online)
E["t"]=pd.to_datetime(E.flood_start_time).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None); END=E.t.max().normalize()
S["inst"]=pd.to_datetime(S.date_installed,errors="coerce"); S["rem"]=pd.to_datetime(S.get("date_removed"),errors="coerce").fillna(END)
S["yrs"]=((S.rem.clip(upper=END)-S.inst).dt.days/365.25).clip(lower=0); S["nd"]=S.sensor_id.map(E.assign(d=E.t.dt.normalize()).groupby("sensor_id").d.nunique()).fillna(0)
S["fdpy"]=np.where((S.nd>0)&(S.yrs>0),S.nd/S.yrs.clip(lower=1/12),np.nan)   # flood days per year; sensors online < 1 month are floored at 1 month
Sall=S.dropna(subset=["lat","lon"]); Tt=Sall[(Sall.tidally_influenced=="Yes")&(Sall.n>0)]; Sf=Sall[Sall.fdpy.notna()]; S0=Sall[Sall.fdpy.isna()]
with rasterio.open("data/nyc_dem_55m.tif") as r: D,Dt,Dn=r.read(1),r.transform,r.nodata
bext=[Dt.c,Dt.c+D.shape[1]*Dt.a,Dt.f+D.shape[0]*Dt.e,Dt.f]; base=np.where((D==Dn)|(D<0),0.0,1.0); lonc=Dt.c+(np.arange(D.shape[1])+.5)*Dt.a; base[((D==Dn)|(D<-500))&(lonc[None,:]<-74.24)]=1.0
sh=pd.read_csv("data/subway/gtfs/shapes.txt"); trp=pd.read_csv("data/subway/gtfs/trips.txt",usecols=["route_id","shape_id"]).drop_duplicates(); sh=sh.merge(trp,on="shape_id",how="left")
L=gpd.GeoDataFrame([(sid,g.route_id.iloc[0],LineString(zip(g.sort_values("shape_pt_sequence").shape_pt_lon,g.sort_values("shape_pt_sequence").shape_pt_lat))) for sid,g in sh.groupby("shape_id")],columns=["shape_id","route","geometry"],crs="EPSG:4326")
L["len"]=L.geometry.length; Lr=L.sort_values("len",ascending=False).drop_duplicates("route")
# ---------------- figure
fig=plt.figure(figsize=(FULL,118*MM))
a=axes_mm(fig,10,11,92,101)
a.imshow(base,extent=bext,cmap=ListedColormap([WATER,LAND]),interpolation="nearest",rasterized=True)
a.imshow(np.where(surge>0,surge,np.nan),extent=dext,cmap=ListedColormap([SURGE1000,SURGE100,SURGE10]),vmin=1,vmax=3,interpolation="nearest",rasterized=True)
a.imshow(np.where(pl_ext,1.0,np.nan),extent=dext,cmap=ListedColormap([RAIN_EXT]),interpolation="nearest",rasterized=True)
a.imshow(np.where(pl_mod,1.0,np.nan),extent=dext,cmap=ListedColormap([RAIN_MOD]),interpolation="nearest",rasterized=True)
a.imshow(np.where(s3new,1.0,np.nan),extent=s3ext,cmap=ListedColormap([TIDE]),interpolation="nearest",rasterized=True)
Lr.plot(ax=a,color="#8C8C8C",lw=0.4,zorder=4)
a.plot([-74.12,-74.12],[40.53,40.655],color=INK,ls=(0,(4,2)),lw=0.6,zorder=4.5)   # west edge of the surge model domain, drawn only across Staten Island (the city land west of it)
a.text(-74.115,40.535,"surge modelled\neast of this line",fontsize=6,color=INK,ha="left",va="bottom",zorder=7)
from matplotlib.colors import LogNorm
FD_CMAP="YlOrRd"; FD_NORM=LogNorm(vmin=0.1,vmax=30)
a.scatter(S0.lon,S0.lat,s=1.2,c="#9A9A9A",lw=0,zorder=5)
a.scatter(Sf.lon,Sf.lat,s=5,c=Sf.fdpy.clip(0.1,30),cmap=FD_CMAP,norm=FD_NORM,lw=0.2,ec=INK,zorder=6)
a.scatter(Tt.lon,Tt.lat,s=14,facecolors="none",ec=TIDE,lw=0.6,zorder=7)   # tidal sensors with floods: green ring, as in panel c
cax=axes_mm(fig,16,69,24,2.0); cb=plt.colorbar(plt.cm.ScalarMappable(cmap=FD_CMAP,norm=FD_NORM),cax=cax,orientation="horizontal",ticks=[0.1,1,10,30]); cb.ax.set_xticklabels(["0.1","1","10","30"]); cb.set_label("Flood days per year at sensor",fontsize=6.5,labelpad=1); cb.ax.tick_params(labelsize=6,length=2,pad=1); cb.outline.set_linewidth(0.4)
a.set_xlim(-74.19,-73.72); a.set_ylim(40.53,40.92); a.set_aspect(1/np.cos(np.radians(40.7)))
a.set_xticks([-74.1,-74.0,-73.9,-73.8]); a.set_yticks([40.6,40.7,40.8,40.9]); a.set_xlabel("Longitude (°E)"); a.set_ylabel("Latitude (°N)")
for s_ in ("top","right"): a.spines[s_].set_visible(True)
h=[Patch(fc=SURGE10,label="Surge, 1-in-10 yr (2.1 m)"),Patch(fc=SURGE100,label="Surge, 1-in-100 yr (3.2 m)"),Patch(fc=SURGE1000,label="Surge, 1-in-1,000 yr (4.6 m)"),
   Patch(fc=RAIN_MOD,label="Rain, moderate scenario (2.13 in h⁻¹)"),Patch(fc=RAIN_EXT,label="Rain, extreme scenario (3.66 in h⁻¹)"),Patch(fc=TIDE,label="Tide at +3 ft sea level"),
   Line2D([],[],marker="o",ls="",mfc="#9A9A9A",mec="#9A9A9A",ms=1.8,label=f"FloodNet sensor, no flood (n = {len(S0)})"),Line2D([],[],marker="o",ls="",mfc="#FC8D59",mec=INK,mew=0.2,ms=2.8,label=f"Sensor with floods (n = {len(Sf)})"),Line2D([],[],marker="o",ls="",mfc="none",mec=TIDE,ms=4.2,mew=0.6,label=f"Tidal sensor with floods (n = {len(Tt)})"),
   Line2D([],[],color="#8C8C8C",lw=0.6,label="Subway route"),Line2D([],[],color=INK,ls=(0,(4,2)),lw=0.6,label="Surge modelled east of line")]
a.legend(handles=h,loc="upper left",handlelength=1.0,labelspacing=0.25,prop={"family":["Arial","DejaVu Sans"],"size":plt.rcParams["legend.fontsize"]})   # DejaVu Sans supplies the superscript minus, which Arial lacks
panel(a,"a",dx=-9)
# ---------------- b: frequency ladder (one row per layer of panel a, in its colours; Sandy as a reference event)
b=axes_mm(fig,137.6,11,38.4,92)   # right column, full height of the map; the flood probability by rain intensity is Extended Data Fig. 1c
# every row is a frequency at a place; circles = any depth, triangles = at least 1 ft (0.3 m)
# fill: "full" observed, "left" (half) estimated from observed rates, "none" modelled or statistical
fd=lambda lab,dk: N[f"flood_days_per_sensor_year_{lab}_{dk}"]
SM=json.load(open("nc_storm_model.json")); pdn=SM["any"]["point"]; p12=SM["12in"]["point"]   # storm-and-place model: typical lot, spread across places
sl=N["surge_floods_per_year_s100_lots_any"]; s12=N["surge_floods_per_year_s100_lots_12in"]
rows=[("Tidal sensors",fd("tidal","any")["pooled"],fd("tidal","12in")["pooled"],(fd("tidal","any")["q10"],fd("tidal","any")["q90"]),TIDE,"o","full"),
      ("Non-tidal sensors",fd("nontidal","any")["pooled"],fd("nontidal","12in")["pooled"],(fd("nontidal","any")["q10"],fd("nontidal","any")["q90"]),INK,"o","full"),   # colours follow panel a
      ("Rain, basement lots",pdn["lots_q50"],p12["lots_q50"],(pdn["lots_q10"],pdn["lots_q90"]),RAIN_EXT,"o","left"),
      ("Rain, moderate scenario",1/N["dep_moderate_rp_yr"],None,None,RAIN_MOD,"o","none"),
      ("Surge, basement lots",sl["median"],s12["median"],(sl["q10"],sl["q90"]),SURGE100,"o","none"),   # lots inside the 1-in-100-yr reach
      ("Surge, 1-in-10-yr level",0.1,None,None,SURGE10,"o","none"),
      ("Surge, 1-in-100-yr level",0.01,None,None,SURGE100,"o","none"),
      ("Sandy (2012)",1/139,None,None,SANDY,"D","none"),
      ("Rain, extreme scenario",1/N["dep_extreme_rp_yr"],None,None,RAIN_EXT,"o","none"),
      ("Surge, 1-in-1,000-yr level",0.001,None,None,SURGE1000,"o","none")]
rows=sorted(rows,key=lambda r:-r[1])
XMIN=7e-4
# flood days per year at each sensor (as nc_data.py section 1): the rates are strongly skewed, so the sensor rows show every sensor as a tick
Ev=nc_events.load(); Ev["t"]=pd.to_datetime(Ev.flood_start_time).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None); END=Ev.t.max().normalize()
Sr=nc_events.sensors(); Sr["yrs"]=((pd.to_datetime(Sr.get("date_removed"),errors="coerce").fillna(END).clip(upper=END)-pd.to_datetime(Sr.date_installed,errors="coerce")).dt.days/365.25).clip(lower=0)
Sr=Sr[Sr.yrs>=0.5]; Sr["nd"]=Sr.sensor_id.map(Ev.groupby("sensor_id").t.apply(lambda t:t.dt.normalize().nunique())).fillna(0); Sr["tidal"]=Sr.tidally_influenced.eq("Yes")
SENS={}
for lab,key,m in (("Tidal sensors","tidal",Sr.tidal),("Non-tidal sensors","nontidal",~Sr.tidal)):
    assert abs(Sr.nd[m].sum()/Sr.yrs[m].sum()-fd(key,"any")["pooled"])<1e-9 and int(m.sum())==fd(key,"any")["n_sensors"]   # same sensors and days as nc_numbers.json
    r=(Sr.nd[m]/Sr.yrs[m]).to_numpy(); SENS[lab]=(r[r>0],float((r==0).mean())); print(f"{lab}: {len(r)} sensors, {100*(r==0).mean():.0f}% no flood, max {r.max():.1f} flood days per year")
for i,(lab,x,x12,iqr,col,mk,fill) in enumerate(rows):
    y=len(rows)-1-i
    if iqr: lo,hi=max(iqr[0],XMIN*1.05),iqr[1]; b.plot([lo,hi],[y,y],color=col,lw=0.8,solid_capstyle="butt",alpha=0.55)
    if lab in SENS:
        b.vlines(SENS[lab][0],y-0.24,y+0.24,color=col,lw=0.3,alpha=0.7)
        b.text(XMIN*1.25,y+0.1,f"{100*SENS[lab][1]:.0f}% with no flood",color=col,fontsize=6,ha="left",va="bottom")
    kw=dict(mfc=col if fill=="full" else "white",mfcalt="white",fillstyle="full" if fill!="left" else "left",mec=col,mew=1.3 if col==SURGE1000 else 0.8,ls="")
    if fill=="left": kw.update(mfc=col)
    b.plot([x],[y],marker=mk,ms=3.5 if mk=="o" else 3,**kw)
    if x12: b.plot([x12],[y],marker="^",ms=3.5,**kw)
b.set_xscale("log"); b.set_xlim(XMIN,120); b.set_ylim(-0.7,len(rows)-0.3)   # the most flooded tidal sensor records about 90 flood days per year
b.set_yticks(range(len(rows))); b.set_yticklabels([r[0] for r in rows][::-1]); b.tick_params(axis="y",length=0)
b.set_xlabel("Floods per year at a place"); b.xaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10,numticks=12)); b.grid(axis="x",which="major")   # a label at every decade
top=b.secondary_xaxis("top",functions=(lambda x:1/np.maximum(x,1e-9),lambda x:1/np.maximum(x,1e-9)))
top.set_xticks([1000,100,10,1]); top.set_xticklabels(["1,000","100","10","1"]); top.set_xlabel("Return period (yr)",labelpad=2); top.tick_params(labelsize=7)
ys=[len(rows)-1-i for i,r in enumerate(rows) if r[0].startswith("Surge") or r[0].startswith("Sandy")]
bm=N["battery_max_since_2020"]
b.legend(handles=[Line2D([],[],marker="o",ls="",mfc=INK,mec=INK,ms=3.8,label="any depth"),Line2D([],[],marker="^",ls="",mfc=INK,mec=INK,ms=3.8,label="≥1 ft"),
                  Line2D([],[],marker="s",ls="",mfc=INK,mec=INK,ms=3.5,label="observed"),Line2D([],[],marker="s",ls="",mfc=INK,mfcalt="white",fillstyle="left",mec=INK,ms=3.5,mew=0.7,label="estimated"),
                  Line2D([],[],marker="s",ls="",mfc="white",mec=INK,ms=3.5,mew=0.7,label="modelled")],   # fill keys drawn as squares so they are not read as the circle (any depth) key
         loc="lower right",ncol=1,handletextpad=0.3,borderaxespad=0.3,labelspacing=0.3,fontsize=8)
panel(b,"b",dx=-30.6,dy=11)   # level with the a letter, above the return-period axis label
import os; save(fig,"Fig1",outdirs=tuple(os.environ["FIG_OUTDIR"].split(",")) if os.environ.get("FIG_OUTDIR") else ("figures",))
