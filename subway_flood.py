"""Subway exposure to flooding: stations, entrances and track alignment vs surge / pluvial / SLR / Sandy.
Vent gratings are not public; entrances (2,120) and underground station entrances are the proxy for openings.
Track depth is not public either; 'Subway' structure = underground, 'Elevated'/'Embankment'/'Viaduct' = above street."""

import glob
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from shapely.geometry import LineString

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

with rasterio.open("data/nyc_dem_55m.tif") as r:
    D, T, N = r.read(1), r.transform, r.nodata; W, H = r.width, r.height
base = np.where((D == N) | (D < 0), 0.0, 1.0); lonc = T.c + (np.arange(W) + .5) * T.a
base[((D == N) | (D < -500)) & (lonc[None, :] < -74.24)] = 1.0; land = base == 1
rows, cols = np.mgrid[0:H, 0:W]; LON = T.c + (cols + .5) * T.a; LAT = T.f + (rows + .5) * T.e


def grid_from(path):
    with rasterio.open(path) as s:
        a, t, shp = s.read(1), s.transform, s.shape
    cc = ((LON - t.c) / t.a).astype(int); rr = ((LAT - t.f) / t.e).astype(int)
    k = (rr >= 0) & (rr < shp[0]) & (cc >= 0) & (cc < shp[1]); v = np.zeros(D.shape, a.dtype); v[k] = a[rr[k], cc[k]]; return v


layers = {}
for c in (1, 2, 3):
    v = grid_from(f"data/slosh_nyc_full/cat{c}.tif"); layers[f"surge Cat{c}"] = (v >= 1) & (v <= 21) & land
    layers[f"surge Cat{c} depth_ft"] = np.where(layers[f"surge Cat{c}"], v.astype(float) - .5, 0)
gd = gpd.read_file(glob.glob("data/pluvial/NYCFloodStormwaterFloodMaps/*Extreme*/*.gdb")[0]).to_crs("EPSG:4326")
layers["pluvial extreme"] = features.rasterize(((x, 1) for x in gd[gd.Flooding_Category.isin([1, 2])].geometry), out_shape=D.shape, transform=T, dtype="uint8").astype(bool)
layers["pluvial deep ≥1 ft"] = features.rasterize(((x, 1) for x in gd[gd.Flooding_Category == 2].geometry), out_shape=D.shape, transform=T, dtype="uint8").astype(bool)
sz = gpd.read_file("data/sandy_zone.geojson").to_crs("EPSG:4326")
layers["Sandy 2012 observed"] = features.rasterize(((x, 1) for x in sz.geometry), out_shape=D.shape, transform=T, dtype="uint8").astype(bool)
for ft in (1, 3):
    p = f"data/slr/nyc55_{ft}ft.tif"
    if glob.glob(p):
        with rasterio.open(p) as r: f = r.read(1)
        layers[f"SLR +{ft} ft tidal"] = (f >= .5) & land


def at(lon, lat, arr):
    c = ((np.asarray(lon) - T.c) / T.a).astype(int); r_ = ((np.asarray(lat) - T.f) / T.e).astype(int)
    k = (r_ >= 0) & (r_ < H) & (c >= 0) & (c < W); out = np.zeros(len(c), arr.dtype); out[k] = arr[r_[k], c[k]]; return out


# ---- stations & entrances ----
S = pd.DataFrame(json.load(open("data/subway/stations.json")))
S["lat"] = S.gtfs_latitude.astype(float); S["lon"] = S.gtfs_longitude.astype(float)
S["underground"] = S.structure.eq("Subway")
E = pd.DataFrame(json.load(open("data/subway/entrances.json")))
E["lat"] = E.entrance_latitude.astype(float); E["lon"] = E.entrance_longitude.astype(float)
E = E.merge(S[["station_id", "structure", "underground"]], on="station_id", how="left")
print(f"stations {len(S)} (underground {int(S.underground.sum())}) | entrances {len(E)} (to underground stations {int(E.underground.fillna(False).sum())})")
print(S.structure.value_counts().to_dict())

haz_keys = ["surge Cat1", "surge Cat2", "surge Cat3", "pluvial extreme", "pluvial deep ≥1 ft", "Sandy 2012 observed"] + [k for k in layers if k.startswith("SLR")]
print(f"\n{'footprint':24}{'stations':>10}{'underground':>13}{'entrances→underground':>24}")
res = {}
for k in haz_keys:
    st = at(S.lon, S.lat, layers[k]).astype(bool); en = at(E.lon, E.lat, layers[k]).astype(bool) & E.underground.fillna(False).values
    res[k] = dict(stations=int(st.sum()), underground=int((st & S.underground.values).sum()), entrances_underground=int(en.sum()))
    print(f"{k:24}{st.sum():>10}{(st & S.underground.values).sum():>13}{en.sum():>24}")

# ---- track alignment from GTFS shapes ----
sh = pd.read_csv("data/subway/gtfs/shapes.txt"); tr = pd.read_csv("data/subway/gtfs/trips.txt", usecols=["route_id", "shape_id"]).drop_duplicates()
sh = sh.merge(tr, on="shape_id", how="left")
seg = []
for sid, g in sh.groupby("shape_id"):
    g = g.sort_values("shape_pt_sequence"); seg.append((sid, g.route_id.iloc[0], LineString(zip(g.shape_pt_lon, g.shape_pt_lat))))
L = gpd.GeoDataFrame(seg, columns=["shape_id", "route", "geometry"], crs="EPSG:4326")
# de-duplicate: keep the longest shape per route, then union all -> per-cell "track present"
L["len"] = L.geometry.length; Lr = L.sort_values("len", ascending=False).drop_duplicates("route")
track = features.rasterize(((x, 1) for x in Lr.geometry), out_shape=D.shape, transform=T, dtype="uint8").astype(bool)
# underground vs elevated per cell: nearest station structure
from scipy.spatial import cKDTree
kd = cKDTree(np.c_[S.lon, S.lat]); ty, tx = np.where(track)
_, idx = kd.query(np.c_[LON[ty, tx], LAT[ty, tx]]); ug_cell = S.underground.values[idx]
km_cell = 55 / 1000
print(f"\ntrack cells {track.sum():,} (~{track.sum()*km_cell:.0f} km of route-km incl. shared trunks; underground share {100*ug_cell.mean():.0f}%)")
print(f"{'footprint':24}{'track km':>10}{'underground km':>16}")
for k in haz_keys:
    m = layers[k][ty, tx]; res[k]["track_km"] = float(m.sum() * km_cell); res[k]["underground_km"] = float((m & ug_cell).sum() * km_cell)
    print(f"{k:24}{m.sum()*km_cell:>10.1f}{(m & ug_cell).sum()*km_cell:>16.1f}")

# per-line exposure (Cat1 surge + pluvial deep), underground stations only
print("\nlines with most underground stations in Cat 1 surge reach:")
st1 = at(S.lon, S.lat, layers["surge Cat1"]).astype(bool) & S.underground.values
stp = at(S.lon, S.lat, layers["pluvial deep ≥1 ft"]).astype(bool) & S.underground.values
rows_ = []
for _, s in S.iterrows():
    for rt in str(s.daytime_routes).split():
        rows_.append((rt, s.underground, s.stop_name))
R = pd.DataFrame(rows_, columns=["route", "ug", "name"])
R["cat1"] = R.name.map(dict(zip(S.stop_name, st1))); R["pluv"] = R.name.map(dict(zip(S.stop_name, stp)))
tab = R[R.ug].groupby("route").agg(ug_stations=("name", "size"), in_cat1=("cat1", "sum"), in_pluvial_deep=("pluv", "sum")).sort_values("in_cat1", ascending=False)
print(tab.head(12).to_string())
sandy_st = S[at(S.lon, S.lat, layers["Sandy 2012 observed"]).astype(bool) & S.underground.values]
print(f"\nunderground stations inside the Sandy 2012 inundation zone ({len(sandy_st)}): " + ", ".join(sandy_st.stop_name.head(15)) + (" …" if len(sandy_st) > 15 else ""))
json.dump(dict(footprints=res, per_line=tab.to_dict("index")), open("subway_summary.json", "w"), indent=1, default=float)

# ---- map ----
dext = [T.c, T.c + W * T.a, T.f + H * T.e, T.f]
fig, ax = plt.subplots(figsize=(14, 11.5))
ax.imshow(base, extent=dext, cmap=ListedColormap(["#DDE7EC", "#F2F2ED"]), interpolation="nearest")
ax.imshow(np.where(layers["surge Cat1"], 1.0, np.nan), extent=dext, cmap=ListedColormap(["#F1C9AE"]), interpolation="nearest")
ax.imshow(np.where(layers["pluvial deep ≥1 ft"], 1.0, np.nan), extent=dext, cmap=ListedColormap(["#8FB8D8"]), interpolation="nearest")
Lr.plot(ax=ax, color="#4A4A4A", lw=.6, zorder=3)
ug = S[S.underground]
hit1 = at(ug.lon, ug.lat, layers["surge Cat1"]).astype(bool); hitp = at(ug.lon, ug.lat, layers["pluvial deep ≥1 ft"]).astype(bool)
ax.scatter(ug.lon[~hit1 & ~hitp], ug.lat[~hit1 & ~hitp], s=9, c="w", ec="#4A4A4A", lw=.5, zorder=4)
ax.scatter(ug.lon[hitp & ~hit1], ug.lat[hitp & ~hit1], s=30, c="#2171B5", ec="k", lw=.5, zorder=5)
ax.scatter(ug.lon[hit1], ug.lat[hit1], s=36, c="#C9653A", ec="k", lw=.5, zorder=6)
ax.scatter(sandy_st.lon, sandy_st.lat, s=110, facecolors="none", ec="#8E1B1B", lw=1.4, zorder=7)
ax.set_xlim(-74.05, -73.73); ax.set_ylim(40.56, 40.90); ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)")
ax.set_title("Subway and flooding, New York City — underground stations inside flood footprints\n"
             f"{int(hit1.sum())} underground stations in Cat 1 surge reach (1-in-45 yr), {int((hitp & ~hit1).sum())} more in DEP deep pluvial (≥1 ft), "
             f"{len(sandy_st)} were inside the Sandy 2012 inundation", fontsize=11.5)
h = [Patch(fc="#F1C9AE", label="Cat 1 storm-surge reach"), Patch(fc="#8FB8D8", label="DEP pluvial, deep ≥1 ft (extreme)"),
     Line2D([], [], color="#4A4A4A", lw=1, label="Subway routes (GTFS)"),
     Line2D([], [], marker="o", ls="", mfc="#C9653A", mec="k", ms=8, label="Underground station in surge reach"),
     Line2D([], [], marker="o", ls="", mfc="#2171B5", mec="k", ms=7, label="Underground station in deep pluvial only"),
     Line2D([], [], marker="o", ls="", mfc="none", mec="#8E1B1B", ms=12, label="Inside Sandy 2012 inundation zone"),
     Line2D([], [], marker="o", ls="", mfc="w", mec="#4A4A4A", ms=5, label="Other underground station")]
ax.legend(handles=h, loc="lower right", fontsize=9, framealpha=.95)
plt.tight_layout(); plt.savefig("subway_flood_map_en.png", dpi=140, bbox_inches="tight"); print("saved subway_flood_map_en.png")
