"""Fig. 2 | Impact: who the flooding reaches (R2). Laid out as Fig. 1.
a  map at the size and extent of Fig. 1a: small-residential basement lots inside the 1-in-100-yr surge reach only, the
   DEP extreme rainfall footprint only, both or neither (overlap counted where surge is modelled), and underground
   subway stations by the same footprints; subway routes; west edge of the surge model domain
b  share of the annual-average structure loss, basement lots and underground-station entrances contributed by surge events
   more frequent than each return period (independent of any normalising event; absolute values in Supplementary Table 1)
c  model against model: basement lots reached by the modelled surge footprint and by the two DEP rainfall scenarios at the
   same depth thresholds (4 in, 1 ft), against return period, where surge is modelled (nc_numbers.json "model_exposure")
"""
import json, warnings, numpy as np, pandas as pd, rasterio, geopandas as gpd
from shapely.geometry import LineString
warnings.filterwarnings("ignore")
from nc_style import *
from matplotlib.colors import ListedColormap; from matplotlib.lines import Line2D
import nc_citygrid as CG
F=json.load(open("fig2_impact_rp.json")); rp=np.array(F["rp"]); loss=np.array(F["loss_bn"]); lots=np.array(F["basement_lots"])
st=np.array(F["underground_stations"]); en=np.array(F["entrances"])
N=json.load(open("nc_numbers.json")); ME=N["model_exposure"]
# ---- map layers
P=pd.read_csv("data/pluto_bsmt.csv",low_memory=False).dropna(subset=["latitude","longitude"])
P=P[P.bldgclass.str[0].isin(["A","B"])|P.bldgclass.isin(["C0","C1","C2","C3"])].reset_index(drop=True)
MC=np.load("data/paper_a/masks_city.npz"); inw=CG.in_window(P.longitude,P.latitude)   # the overlap is counted where surge is modelled
bs=CG.at(P.longitude,P.latitude,MC["s100"])&inw; bp=CG.at(P.longitude,P.latitude,MC["rain"]); bpw=bp&inw
n_s,n_r,n_b=int((bs&~bpw).sum()),int((bpw&~bs).sum()),int((bs&bpw).sum()); print("surge only",n_s,"rain only",n_r,"both",n_b)
Stn=pd.DataFrame(json.load(open("data/subway/stations.json"))); U=Stn[Stn.structure=="Subway"].copy(); U["lon"]=U.gtfs_longitude.astype(float); U["lat"]=U.gtfs_latitude.astype(float)
us=CG.at(U.lon,U.lat,MC["s100"]); up=CG.at(U.lon,U.lat,MC["rain"])
with rasterio.open("data/nyc_dem_55m.tif") as r: D,Dt,Dn=r.read(1),r.transform,r.nodata
bext=[Dt.c,Dt.c+D.shape[1]*Dt.a,Dt.f+D.shape[0]*Dt.e,Dt.f]; base=np.where((D==Dn)|(D<0),0.0,1.0); lonc=Dt.c+(np.arange(D.shape[1])+.5)*Dt.a; base[((D==Dn)|(D<-500))&(lonc[None,:]<-74.24)]=1.0
sh=pd.read_csv("data/subway/gtfs/shapes.txt"); trp=pd.read_csv("data/subway/gtfs/trips.txt",usecols=["route_id","shape_id"]).drop_duplicates(); sh=sh.merge(trp,on="shape_id",how="left")
L=gpd.GeoDataFrame([(sid,g.route_id.iloc[0],LineString(zip(g.sort_values("shape_pt_sequence").shape_pt_lon,g.sort_values("shape_pt_sequence").shape_pt_lat))) for sid,g in sh.groupby("shape_id")],columns=["shape_id","route","geometry"],crs="EPSG:4326")
L["len"]=L.geometry.length; Lr=L.sort_values("len",ascending=False).drop_duplicates("route")
# ---- figure
fig=plt.figure(figsize=(FULL,114*MM))
a=axes_mm(fig,10,8,92,101)
a.imshow(base,extent=bext,cmap=ListedColormap([WATER,LAND]),interpolation="nearest",rasterized=True)
o=~bs&~bp; a.scatter(P.longitude[o],P.latitude[o],s=0.04,c="#D2D2D2",lw=0,rasterized=True,zorder=2)
a.scatter(P.longitude[bp&~bs],P.latitude[bp&~bs],s=0.2,c=RAIN_EXT,lw=0,rasterized=True,zorder=3)     # west of the domain line: rainfall footprint only
a.scatter(P.longitude[bs&~bpw],P.latitude[bs&~bpw],s=0.2,c=SURGE100,lw=0,rasterized=True,zorder=3)
a.scatter(P.longitude[bs&bpw],P.latitude[bs&bpw],s=0.2,c=SANDY,lw=0,rasterized=True,zorder=3)
Lr.plot(ax=a,color="#8C8C8C",lw=0.4,zorder=4)
a.scatter(U.lon[~us&~up],U.lat[~us&~up],s=5,marker="s",c="white",ec="#7F7F7F",lw=0.35,zorder=5)
a.scatter(U.lon[up&~us],U.lat[up&~us],s=9,marker="s",c=RAIN_EXT,ec=INK,lw=0.4,zorder=6)
a.scatter(U.lon[us&~up],U.lat[us&~up],s=9,marker="s",c=SURGE100,ec=INK,lw=0.4,zorder=6)
a.scatter(U.lon[us&up],U.lat[us&up],s=9,marker="s",c=SANDY,ec=INK,lw=0.4,zorder=7)
a.plot([-74.12,-74.12],[40.53,40.655],color=INK,ls=(0,(4,2)),lw=0.6,zorder=4.5)
a.text(-74.115,40.535,"surge modelled\neast of this line",fontsize=6,color=INK,ha="left",va="bottom",zorder=8)
a.set_xlim(-74.19,-73.72); a.set_ylim(40.53,40.92); a.set_aspect(1/np.cos(np.radians(40.7)))
a.set_xticks([-74.1,-74.0,-73.9,-73.8]); a.set_yticks([40.6,40.7,40.8,40.9]); a.set_xlabel("Longitude (°E)"); a.set_ylabel("Latitude (°N)")
for s_ in ("top","right"): a.spines[s_].set_visible(True)
dot=lambda c,l: Line2D([],[],marker="o",ls="",mfc=c,mec="none",ms=3,label=l)
sq=lambda c,e,l: Line2D([],[],marker="s",ls="",mfc=c,mec=e,mew=0.4,ms=3.2,label=l)
a.legend(handles=[dot(SURGE100,f"1-in-100-yr surge only ({n_s:,})"),dot(RAIN_EXT,f"Extreme rain only ({n_r:,})"),dot(SANDY,f"Both ({n_b:,})"),dot("#D2D2D2","Neither"),
                  sq(SURGE100,INK,f"Surge only ({int((us&~up).sum())})"),sq(RAIN_EXT,INK,f"Rain only ({int((up&~us).sum())})"),sq(SANDY,INK,f"Both ({int((us&up).sum())})"),
                  sq("white","#7F7F7F",f"Neither ({int((~us&~up).sum())})"),Line2D([],[],color="#8C8C8C",lw=0.6,label="Subway route")],
         loc="upper left",title="Basement lots (dots), underground stations (squares)",alignment="left",handlelength=1.0,labelspacing=0.25)
panel(a,"a",dx=-9)
# ---- b: share of each annual-average impact contributed by events more frequent than a given return period
b=axes_mm(fig,117,70,59,36); AC=F["annual_cum"]; rpd=np.array(AC["rp"]); SH=F["annual_share_frequent"]
for k,col,lab,ls in (("loss",SURGE10,"Structure loss","-"),("basement_lots",RAIN_EXT,"Basement lots","-"),("entrances",PROT,"Station entrances","--")):
    b.plot(rpd,100*np.array(AC[k]),ls,color=col,lw=1.0,label=lab)
b.set_xscale("log"); b.set_xlim(1,1e4); b.set_ylim(0,100); b.set_xlabel("Surge return period (yr)"); b.set_ylabel("Share of annual average\nfrom more frequent events (%)"); b.grid(which="major")
b.axvline(50,color="#9A9A9A",lw=0.5,ls=":"); b.text(1.3,97,"events more frequent\nthan 1-in-50 yr:\nlots %d%%, loss %d%%,\nentrances %d%%"%(round(100*SH["basement_lots"]["50"]),round(100*SH["loss"]["50"]),round(100*SH["entrances"]["50"])),fontsize=6.5,va="top")
b.axvline(139,color=SANDY,ls=":",lw=0.7); b.text(150,4,"Sandy",color=SANDY,fontsize=6.5,va="bottom")
b.legend(loc="center right",fontsize=6.5)   # clear of the Sandy label
panel(b,"b",dx=-10)
# ---- c: model against model, same depth thresholds, where surge is modelled
c=axes_mm(fig,117,10,59,40); S_=ME["surge"]; R_=ME["rain"]
c.plot(S_["rp"],np.array(S_["4in"])/1e3,"-",color=SURGE100,lw=1.0,label="Surge, ≥4 in")
c.plot(S_["rp"],np.array(S_["12in"])/1e3,"--",color=SURGE100,lw=1.0,label="Surge, ≥1 ft")
for k_,col in (("moderate",RAIN_MOD),("extreme",RAIN_EXT)):
    c.plot([R_[k_]["rp"]],[R_[k_]["4in"]/1e3],"o",color=col,ms=3.5); c.plot([R_[k_]["rp"]],[R_[k_]["12in"]/1e3],"^",mfc="white",mec=col,mew=0.8,ms=3.5)
c.text(R_["moderate"]["rp"]*1.3,0.3,"Rain, moderate",color=RAIN_MOD,fontsize=6.5,va="bottom")   # below the surge curves
c.text(R_["extreme"]["rp"]*1.25,R_["extreme"]["12in"]/1e3-1.5,"Rain, extreme",color=RAIN_EXT,fontsize=6.5,va="top",ha="right")   # below the markers, clear of the surge curves
c.legend(handles=[Line2D([],[],color=SURGE100,lw=1.0,label="Surge, ≥4 in"),Line2D([],[],color=SURGE100,lw=1.0,ls="--",label="Surge, ≥1 ft"),
                  Line2D([],[],marker="o",ls="",color=INK,ms=3,label="Rain scenario, ≥4 in"),Line2D([],[],marker="^",ls="",mfc="white",mec=INK,mew=0.8,ms=3,label="Rain scenario, ≥1 ft")],
         loc="upper left",fontsize=6.5)
c.set_xscale("log"); c.set_xlim(2,1000); c.set_ylim(0,45); c.set_xlabel("Return period (yr)"); c.set_ylabel("Basement lots in\nmodelled footprint (thousand)"); c.grid(which="major")
panel(c,"c",dx=-10)
save(fig,"Fig2")
json.dump(dict(surge_only=n_s,rain_only=n_r,both=n_b,stations=dict(surge_only=int((us&~up).sum()),rain_only=int((up&~us).sum()),both=int((us&up).sum()),other=int((~us&~up).sum())),
               annual_share_frequent=F["annual_share_frequent"],model_exposure=ME),open("nc_fig2.json","w"),indent=1)
