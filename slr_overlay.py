"""NOAA SLR inundation (MHHW + X ft, hydrologically connected bathtub, 3 m) -> 55 m NYC grid,
overlaid with SLOSH surge reach and tidal FloodNet sensors. Also counts basement lots per scenario.
Run after data/slr/NY_M_slr_depth_{0,1,2,3,6}ft.tif are downloaded."""

import json
import nc_events   # sensor table with the paper's tidal flag (nc_events.sensors)
import subprocess

import numpy as np
import pandas as pd
import rasterio

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FT = [0, 1, 2, 3, 6]
# NPCC (2019) NYC projections relative to 2000-04: 2050s mid-range 11-21 in, 90th pct 30 in; 2080s 90th pct 58 in; 2100 high-end ~75 in
LABEL = {0: "today's MHHW (high-tide line)", 1: "+1 ft  (~2050s mid-range)", 2: "+2 ft  (~2050s 90th pct)",
         3: "+3 ft  (~2080s mid-range)", 6: "+6 ft  (~2100 high-end)"}

with rasterio.open("data/nyc_dem_55m.tif") as r:
    D, T, N = r.read(1), r.transform, r.nodata
    W, H = r.width, r.height
xmin, ymax = T.c, T.f; xmax, ymin = T.c + W * T.a, T.f + H * T.e
frac = {}
for ft in FT:
    src = f"data/slr/NY_M_slr_depth_{ft}ft.tif"; b = f"data/slr/bin_{ft}ft.tif"; o = f"data/slr/nyc55_{ft}ft.tif"
    subprocess.run(["gdal_calc.py", "-A", src, f"--outfile={b}", "--calc=(A>0)*1", "--type=Byte", "--NoDataValue=255",
                    "--co=COMPRESS=DEFLATE", "--co=TILED=YES", "--quiet", "--overwrite"], check=True)
    subprocess.run(["gdalwarp", "-q", "-overwrite", "-t_srs", "EPSG:4326", "-te", str(xmin), str(ymin), str(xmax), str(ymax),
                    "-ts", str(W), str(H), "-r", "average", "-ot", "Float32", "-dstnodata", "-1", "-co", "COMPRESS=DEFLATE", b, o], check=True)
    with rasterio.open(o) as r:
        f = r.read(1); f[f < 0] = np.nan
    frac[ft] = f
    print(f"{ft} ft: NYC-window cells >=50% inundated: {int(np.nansum(f >= .5)):,}")

base = np.where((D == N) | (D < 0), 0.0, 1.0); lonc = T.c + (np.arange(W) + .5) * T.a
base[((D == N) | (D < -500)) & (lonc[None, :] < -74.24)] = 1.0
land = base == 1
# newly inundated land per scenario (relative to today's MHHW), classified by the lowest scenario that floods it
slr = np.zeros(D.shape, "int8")  # 0 none, else ft
for ft in FT[::-1]:
    slr[(frac[ft] >= .5) & land] = ft
slr[(frac[0] >= .5) & land] = -1  # already tidal today
# NOTE: 55 x 55 m overstates the 0.0005-deg cell (~42 x 55 m) and this window includes the New Jersey shore;
# the areas reported in the paper come from nc_areas.py (five boroughs only, true cell area).
km2 = {ft: float(((slr == ft) & land).sum() * 55 * 55 / 1e6) for ft in FT[1:]}
print("newly inundated land at MHHW (km², NYC window incl. NJ shore):", {k: round(v, 1) for k, v in km2.items()})

# basement lots per scenario
p = pd.read_csv("data/pluto_bsmt.csv", low_memory=False).dropna(subset=["latitude", "longitude"])
c = ((p.longitude.values - T.c) / T.a).astype(int); r_ = ((p.latitude.values - T.f) / T.e).astype(int)
ok = (r_ >= 0) & (r_ < H) & (c >= 0) & (c < W)
v = np.full(len(p), 0, "int8"); v[ok] = slr[r_[ok], c[ok]]
p["small"] = p.bldgclass.str[0].isin(["A", "B"]) | p.bldgclass.isin(["C0", "C1", "C2", "C3"])
p["informal"] = 114000 * p.small / p.small.sum()
tab = {}
for ft in FT[1:]:
    m = (v != 0) & (v <= ft) & (v > 0) | (v == -1)
    tab[ft] = dict(lots=int(m.sum()), informal=float(p.informal[m].sum()))
    print(f"basement lots within daily high-tide reach at +{ft} ft: {m.sum():,} (allocated informal units {p.informal[m].sum():,.0f})")
json.dump(dict(newly_inundated_km2=km2, basement=tab), open("slr_summary.json", "w"), indent=1)

# SLOSH Cat1 outline for context
with rasterio.open("data/slosh_nyc_full/cat1.tif") as s:
    a, t, shp = s.read(1), s.transform, s.shape
rows, cols = np.mgrid[0:H, 0:W]; lon = T.c + (cols + .5) * T.a; lat = T.f + (rows + .5) * T.e
cc = ((lon - t.c) / t.a).astype(int); rr = ((lat - t.f) / t.e).astype(int); k = (rr >= 0) & (rr < shp[0]) & (cc >= 0) & (cc < shp[1])
vv = np.zeros(D.shape, "uint8"); vv[k] = a[rr[k], cc[k]]; cat1 = (vv >= 1) & (vv <= 21) & land

S = nc_events.sensors()
S["lat"] = pd.to_numeric(S.latitude, errors="coerce"); S["lon"] = pd.to_numeric(S.longitude, errors="coerce")
import nc_events; E = nc_events.load(); S["n"] = S.sensor_id.map(E.groupby("sensor_id").size()).fillna(0)
Tt = S[S.tidally_influenced == "Yes"].dropna(subset=["lat", "lon"])

dext = [T.c, T.c + W * T.a, T.f + H * T.e, T.f]
fig, ax = plt.subplots(figsize=(14, 11.5))
ax.imshow(base, extent=dext, cmap=ListedColormap(["#DDE7EC", "#F2F2ED"]), interpolation="nearest")
ax.imshow(np.where(cat1, 1.0, np.nan), extent=dext, cmap=ListedColormap(["#F3D9C4"]), interpolation="nearest", alpha=.9)
cols_ = {-1: "#08306B", 1: "#2171B5", 2: "#4292C6", 3: "#6BAED6", 6: "#BDD7E7"}
for ft in (6, 3, 2, 1, -1):
    ax.imshow(np.where(slr == ft, 1.0, np.nan), extent=dext, cmap=ListedColormap([cols_[ft]]), interpolation="nearest")
ax.scatter(Tt.lon, Tt.lat, s=14 + 5 * np.sqrt(Tt.n), facecolors="none", ec="#B33A30", lw=1.1, zorder=5)
ax.set_xlim(-74.27, -73.69); ax.set_ylim(40.49, 40.92); ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)")
ax.set_title("Sea-level rise and saltwater reach, New York City\nNOAA SLR inundation (MHHW + X ft, hydrologically connected) over SLOSH Cat 1 surge reach; scenario years per NPCC 2019", fontsize=12)
h = [Patch(fc="#08306B", label="Tidal today (MHHW)"), Patch(fc="#2171B5", label=f"{LABEL[1]}  +{km2[1]:.0f} km²"),
     Patch(fc="#4292C6", label=f"{LABEL[2]}  +{km2[2]:.0f} km²"), Patch(fc="#6BAED6", label=f"{LABEL[3]}  +{km2[3]:.0f} km²"),
     Patch(fc="#BDD7E7", label=f"{LABEL[6]}  +{km2[6]:.0f} km²"), Patch(fc="#F3D9C4", label="Cat 1 storm-surge reach (1-in-45 yr)"),
     Line2D([], [], marker="o", ls="", mfc="none", mec="#B33A30", ms=9, label=f"Tidally influenced FloodNet sensor (n={len(Tt)})")]
ax.legend(handles=h, loc="lower right", fontsize=9, framealpha=.95)
for nm, (x, y) in {"Jamaica Bay": (-73.86, 40.61), "Rockaway": (-73.80, 40.575), "Coney Island": (-73.97, 40.565), "Red Hook": (-74.02, 40.675),
                   "Lower Manhattan": (-74.03, 40.715), "Staten Island\nSouth Shore": (-74.17, 40.52), "Flushing": (-73.83, 40.77), "Throgs Neck": (-73.80, 40.81)}.items():
    ax.annotate(nm, (x, y), fontsize=8.5, color="#333", ha="center", style="italic", zorder=7)
plt.tight_layout(); plt.savefig("slr_saltwater_map_en.png", dpi=140, bbox_inches="tight"); print("saved slr_saltwater_map_en.png")
