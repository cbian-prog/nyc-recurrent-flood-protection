"""Siting data for Extended Data Fig. 1a (read by nc_edfig_record.py): 20 m city and land masks, the DEP extreme rainfall footprint,
FloodNet sensors and 311 street-flooding complaints on a 200 m grid, and the sensor siting enrichment (nc_areas.json).
The former Extended Data Fig. 8 panels (spatial transfer, reporting trend, extrapolation) were dropped from the paper on 2026-11-04.
"""
import glob, json, warnings, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio import features
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm; from matplotlib.lines import Line2D; from matplotlib.patches import Patch
matplotlib.rcParams.update({"font.size":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":10,"axes.titlesize":12})
# ---- a: network map on the 20 m grid, masked to the five boroughs
with rasterio.open("data/manhattan_dem_20m.tif") as r: dem,tr,nod=r.read(1),r.transform,r.nodata
bgs=gpd.read_file("data/nfip/tl_2020_36_bg.shp"); bgs=bgs[bgs.COUNTYFP.isin(["061","047","081","005","085"])].to_crs("EPSG:4326"); boro=bgs.dissolve(by="COUNTYFP"); boro["geometry"]=boro.geometry.buffer(0)
city=features.rasterize(((x,1) for x in boro.geometry),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool); land=city&(dem!=nod)&(dem>=0)
gd=gpd.read_file(glob.glob("data/pluvial/NYCFloodStormwaterFloodMaps/*Extreme*/*.gdb")[0]).to_crs("EPSG:4326")
dep=features.rasterize(((x,1) for x in gd[gd.Flooding_Category.isin([1,2])].geometry),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool)&city
S=pd.DataFrame(json.load(open("data/floodnet/sensors.json"))); S["lat"]=S.latitude.astype(float); S["lon"]=S.longitude.astype(float)
P=pd.read_csv("data/sf_points.csv").dropna(subset=["lat","lon"])
def at(lon,lat,m):
    c=((np.asarray(lon)-tr.c)/tr.a).astype(int); r_=((np.asarray(lat)-tr.f)/tr.e).astype(int); k=(r_>=0)&(r_<dem.shape[0])&(c>=0)&(c<dem.shape[1]); o=np.zeros(len(c),bool); o[k]=m[r_[k],c[k]]; return o
s_in=at(S.lon,S.lat,dep).mean(); land_share=dep[land].sum()/land.sum(); enrich=s_in/land_share
p_in=at(P.lon,P.lat,dep).mean()
# exact point-in-polygon and city-wide shares (nc_areas.py) replace the 20 m raster lookup, which misses points near footprint edges and outside the grid
_A=json.load(open("nc_areas.json")); s_in=_A["sensors_inside"]; land_share=_A["rain_share_of_city_land"]; enrich=_A["sensor_enrichment"]; p_in=_A["complaints_inside"]
print(f"DEP extreme = {100*land_share:.1f}% of five-borough land | sensors inside {100*s_in:.0f}% (enrichment {enrich:.1f}x) | 311 complaints inside {100*p_in:.0f}%")
# 311 density on a 200 m grid
lon_e=np.arange(-74.26,-73.69,0.0024); lat_e=np.arange(40.49,40.92,0.0018); H,_,_=np.histogram2d(P.lon,P.lat,bins=[lon_e,lat_e])
# ---- figure (none: the panel is drawn in nc_edfig_record.py)
