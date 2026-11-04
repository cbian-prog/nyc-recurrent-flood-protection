"""Area statistics for Paper A with true cell areas.
The elevation grids are geographic (0.00018 deg, about 15 x 20 m; 0.0005 deg, about 42 x 55 m), so cell area is computed per row
from latitude instead of assuming 20 x 20 m or 55 x 55 m. City-wide statistics of the rainfall footprint (area, share of land,
share outside the flood map, sensors and 311 complaints inside) use the DEP polygons in UTM 18N; statistics that need the modelled
surge reach are computed inside the surge model domain (74.12-73.72 W, 40.53-40.92 N). Sea-level-rise areas are clipped to the
five boroughs. Output: nc_areas.json."""
import glob, json, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio import features
from scipy import ndimage
from shapely.geometry import box
UTM="EPSG:32618"
bg=gpd.read_file("data/nfip/tl_2020_36_bg.shp"); bg=bg[bg.COUNTYFP.isin(["061","047","081","005","085"])]
land_m2=float(bg.ALAND.sum()); print("NYC land (ALAND) km2", round(land_m2/1e6,1))
boro=bg.to_crs(UTM).dissolve(by="COUNTYFP"); city=boro.geometry.buffer(0).union_all()
gd=gpd.read_file(glob.glob("data/pluvial/NYCFloodStormwaterFloodMaps/*Extreme*/*.gdb")[0]).to_crs(UTM)
rain=gd[gd.Flooding_Category.isin([1,2])].geometry.buffer(0).union_all().intersection(city)
print("rain extreme area citywide km2", round(rain.area/1e6,2), "share of land", round(100*rain.area/land_m2,2))
Z=gpd.read_file("data/fema_nfhl/nfhl_zones_nyc.geojson"); Z=Z[Z.SFHA_TF=="T"].to_crs(UTM); sf=Z.geometry.buffer(0).union_all()
print("rain outside SFHA citywide", round(100*rain.difference(sf).area/rain.area,1))
dom=gpd.GeoSeries([box(-74.12,40.53,-73.72,40.92)],crs=4326).to_crs(UTM).iloc[0]
print("rain share inside domain", round(100*rain.intersection(dom).area/rain.area,1), "| city land share in domain (polygon)", round(100*city.intersection(dom).area/city.area,1))
# raster stats with true pixel area per row
with rasterio.open("data/manhattan_dem_20m.tif") as r: dem,tr,nod=r.read(1),r.transform,r.nodata
lat=tr.f+(np.arange(dem.shape[0])+.5)*tr.e; pa=(abs(tr.a)*111320*np.cos(np.radians(lat)))*(abs(tr.e)*110950); A=np.repeat(pa[:,None],dem.shape[1],1)
print("pixel area m2 range", pa.min().round(1), pa.max().round(1))
bgw=bg.to_crs(4326); cityr=features.rasterize(((x,1) for x in bgw.dissolve(by="COUNTYFP").geometry.buffer(0)),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool)
dep=features.rasterize(((x,1) for x in gd.to_crs(4326)[gd.Flooding_Category.isin([1,2])].geometry),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool)
sfr=features.rasterize(((g,1) for g in Z.to_crs(4326).geometry if g is not None),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool)
valid=dem!=nod; dv=np.where(valid,dem,1e4); seed=np.zeros(dem.shape,bool); seed[0,:]=seed[-1,:]=seed[:,0]=seed[:,-1]=True; seed&=dv<-1
c=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in c)); W=np.array([c[str(t)] for t in T])
wl=lambda rp: float(np.interp(np.log(rp),np.log(T),W))-0.063
def surge(w):
    b=(dv<w)&valid; lab,_=ndimage.label(b); return np.isin(lab,np.unique(lab[seed&(lab>0)]))&b&(dv>=0)
s100,s1000=surge(wl(100)),surge(wl(1000)); rr=dep&cityr&valid&(dem>=0); landr=cityr&valid&(dem>=0)
ar=lambda m: float(A[m].sum())
print("domain: rain km2", round(ar(rr)/1e6,1), "(old x400:", round(rr.sum()*400/1e6,1),")", "| out SFHA", round(100*ar(rr&~sfr)/ar(rr),1), "| out s100", round(100*ar(rr&~s100)/ar(rr),1), "| out s1000", round(100*ar(rr&~s1000)/ar(rr),1))
print("domain: SFHA land km2", round(ar(sfr&landr)/1e6,1), "| s100 land km2", round(ar(s100&cityr)/1e6,1), "| SFHA inside s100", round(100*ar(sfr&landr&s100)/ar(sfr&landr),1), "| rain share of domain land", round(100*ar(rr)/ar(landr),2))
print("domain land km2", round(ar(landr)/1e6,1))
# sensors and 311 citywide point-in-polygon
S=pd.DataFrame(json.load(open("data/floodnet/sensors.json"))); S["lat"]=pd.to_numeric(S.latitude,errors="coerce"); S["lon"]=pd.to_numeric(S.longitude,errors="coerce"); S=S.dropna(subset=["lat","lon"])
gs=gpd.GeoSeries(gpd.points_from_xy(S.lon,S.lat),crs=4326).to_crs(UTM)
P=pd.read_csv("data/sf_points.csv").dropna(subset=["lat","lon"]); gp=gpd.GeoSeries(gpd.points_from_xy(P.lon,P.lat),crs=4326).to_crs(UTM)
from shapely.prepared import prep
pr=prep(rain); si=np.mean([pr.contains(p) for p in gs]); share=rain.area/land_m2
pi=np.mean([pr.contains(p) for p in gp])
print("sensors", len(S), "inside", round(100*si,1), "% enrichment", round(si/share,2), "| 311 inside", round(100*pi,1), "%")
# SLR new land NYC only, true pixel area
with rasterio.open("data/slr/nyc55_0ft.tif") as r: f0=r.read(1); t5=r.transform
out={}
lat5=t5.f+(np.arange(f0.shape[0])+.5)*t5.e; pa5=(abs(t5.a)*111320*np.cos(np.radians(lat5)))*(abs(t5.e)*110950); A5=np.repeat(pa5[:,None],f0.shape[1],1)
c5=features.rasterize(((x,1) for x in bgw.dissolve(by="COUNTYFP").geometry.buffer(0)),out_shape=f0.shape,transform=t5,dtype="uint8").astype(bool)
for ft in (1,3,6):
    fn=f"data/slr/nyc55_{ft}ft.tif"
    try:
        with rasterio.open(fn) as r: f=r.read(1)
    except Exception as e: print("missing",fn); continue
    new=(f>=.5)&~(f0>=.5); out[ft]=(round(float(A5[new&c5].sum())/1e6,1), round(float(new.sum())*3025/1e6,1))
print("SLR new land NYC-only true area vs old (window incl NJ, x3025):", out)
OUT_JSON=dict(city_land_km2=land_m2/1e6, rain_km2_city=rain.area/1e6, rain_share_of_city_land=rain.area/land_m2, rain_outside_sfha_city=rain.difference(sf).area/rain.area, rain_share_in_domain=rain.intersection(dom).area/rain.area, city_share_in_domain=city.intersection(dom).area/city.area, domain_rain_km2=ar(rr)/1e6, domain_rain_outside_sfha=ar(rr&~sfr)/ar(rr), domain_rain_outside_s100=ar(rr&~s100)/ar(rr), domain_rain_outside_s1000=ar(rr&~s1000)/ar(rr), domain_sfha_land_km2=ar(sfr&landr)/1e6, domain_s100_land_km2=ar(s100&cityr)/1e6, domain_sfha_inside_s100=ar(sfr&landr&s100)/ar(sfr&landr), sensors_n=int(len(S)), sensors_inside=float(si), sensor_enrichment=float(si/share), complaints_inside=float(pi), slr_new_land_km2_nyc={str(k):v[0] for k,v in out.items()})
json.dump(OUT_JSON,open("nc_areas.json","w"),indent=1); print("saved nc_areas.json")
