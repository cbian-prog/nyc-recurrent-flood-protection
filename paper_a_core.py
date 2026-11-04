"""Paper A core surge model: NSI exposure x USACE generic depth-damage x NACCS water-level hazard curve.

Exposure   : USACE National Structure Inventory (val_struct, num_story, occtype, found_type, found_ht) for the 5 boroughs.
Vulnerability (main) : USACE generic curves EGM 01-03 (no basement) / EGM 04-01 (with basement), depth relative to first floor
             (ground depth - found_ht). Buildings taller than 3 storeys: curve applies to the lowest-3-storey share of value (3/N).
             No free parameter; Sandy is a validation, not a calibration.
Vulnerability (alt)  : this project's storey-fraction curve k*min(1,d/2)*min(1,ceil(d/3)/N), k recalibrated on Sandy with NSI values.
Hazard     : NACCS SP 7672 water-level curve (MSL -> NAVD88 -0.063 m), connected bathtub on the 20 m DEM.
Outputs    : data/paper_a/depths.npz (per-structure depth by water level), paper_a_core.json.
"""

import json
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage

warnings.filterwarnings("ignore")
FT = 0.3048
NAVD_MINUS_MSL = 0.063
SANDY_WL = 3.44  # m NAVD88 at The Battery
SANDY_LOSS = (11.4e9, 12.35e9, 13.3e9)

# ---------------- USACE generic structure curves (depth in ft relative to first floor, damage %) ----------------
D_NB = np.arange(-2, 17)
D_WB = np.arange(-8, 17)
CURVES = {
    "1S_NB": (D_NB, [0, 2.5, 13.4, 23.3, 32.1, 40.1, 47.1, 53.2, 58.6, 63.2, 67.2, 70.5, 73.2, 75.4, 77.2, 78.5, 79.5, 80.2, 80.7]),
    "2S_NB": (D_NB, [0, 3.0, 9.3, 15.2, 20.9, 26.3, 31.4, 36.2, 40.7, 44.9, 48.8, 52.4, 55.7, 58.7, 61.4, 63.8, 65.9, 67.7, 69.2]),
    "1S_WB": (D_WB, [0, 0.7, 0.8, 2.4, 5.2, 9.0, 13.8, 19.4, 25.5, 32.0, 38.7, 45.5, 52.2, 58.6, 64.5, 69.8, 74.2, 77.7, 80.1, 81.1, 81.1, 81.1, 81.1, 81.1, 81.1]),
    "2S_WB": (D_WB, [1.7, 1.7, 1.9, 2.9, 4.7, 7.2, 10.2, 13.9, 17.9, 22.3, 27.0, 31.9, 36.9, 41.9, 46.9, 51.8, 56.4, 60.8, 64.8, 68.4, 71.4, 73.7, 75.4, 76.4, 76.4]),
}


def usace_frac(d_ff_ft, key):
    x, y = CURVES[key]
    return np.interp(d_ff_ft, x, np.array(y) / 100.0, left=0.0, right=y[-1] / 100.0)


# ---------------- exposure ----------------
s = pd.read_parquet("data/nsi_nyc/nsi.parquet")
s = s[s.val_struct > 0].reset_index(drop=True)
N = np.clip(s.num_story.fillna(1).values, 1, 110)
val = s.val_struct.values.astype("float64")
wb = (s.found_type == "B").values
one_story = N < 1.5
key = np.where(one_story, np.where(wb, "1S_WB", "1S_NB"), np.where(wb, "2S_WB", "2S_NB"))
found_ft = np.clip(s.found_ht.fillna(0).values, 0, 15)
share3 = np.minimum(1.0, 3.0 / N)
print(f"NSI structures {len(s):,} | value {val.sum()/1e9:,.0f} bn | with basement {100*wb.mean():.0f}% | 1-storey {100*one_story.mean():.0f}% | >3 storeys {100*(N>3).mean():.0f}% ({100*val[N>3].sum()/val.sum():.0f}% of value)")

# ---------------- hazard: per-structure depth at a ladder of water levels ----------------
with rasterio.open("data/manhattan_dem_20m.tif") as r:
    dem, tr, nod = r.read(1), r.transform, r.nodata
valid = dem != nod; dem_v = np.where(valid, dem, 1e4)
seed = np.zeros(dem.shape, bool); seed[0, :] = seed[-1, :] = seed[:, 0] = seed[:, -1] = True; seed &= dem_v < -1.0
col = ((s.x.values - tr.c) / tr.a).astype(int); row = ((s.y.values - tr.f) / tr.e).astype(int)
inb = (row >= 0) & (row < dem.shape[0]) & (col >= 0) & (col < dem.shape[1])
print(f"structures inside DEM window: {inb.sum():,} ({val[inb].sum()/1e9:,.0f} bn)")
os.makedirs("data/paper_a", exist_ok=True)
levels = np.round(np.arange(0.5, 7.51, 0.25), 2)
if os.path.exists("data/paper_a/depths.npz"):
    z = np.load("data/paper_a/depths.npz"); depths = z["depths"]; assert np.allclose(z["levels"], levels)
else:
    depths = np.zeros((len(levels), len(s)), "float32")
    for i, w in enumerate(levels):
        below = (dem_v < w) & valid; lab, _ = ndimage.label(below)
        conn = np.isin(lab, np.unique(lab[seed & (lab > 0)])) & below & (dem_v >= 0)
        d = np.zeros(len(s), "float32"); d[inb] = np.where(conn, w - dem_v, 0.0)[row[inb], col[inb]]; depths[i] = d
    np.savez_compressed("data/paper_a/depths.npz", depths=depths, levels=levels)
# Sandy observed footprint depth
g = gpd.read_file("data/sandy_zone.geojson").to_crs("EPSG:4326")
obs = features.rasterize(((x, 1) for x in g.geometry), out_shape=dem.shape, transform=tr, dtype="uint8").astype(bool)
fl = np.zeros(len(s), bool); fl[inb] = obs[row[inb], col[inb]]
d_sandy = np.where(fl & inb, np.maximum(SANDY_WL - np.where(valid, dem, 1e4)[np.clip(row, 0, dem.shape[0] - 1), np.clip(col, 0, dem.shape[1] - 1)], 0), 0.0)


def loss_usace(d_m, valuation=val):
    d_ff = d_m / FT - found_ft
    f = np.zeros(len(s));
    for k in CURVES: m = key == k; f[m] = usace_frac(d_ff[m], k)
    return valuation * f * share3 * (d_m > 0)


def loss_storey(d_m, k, valuation=val):
    return valuation * np.clip(k * np.minimum(1, d_m / 2) * np.minimum(1, np.ceil(np.maximum(d_m, 0) / 3.0) / N), 0, 1)


# ---------------- Sandy validation / calibration ----------------
L_us = loss_usace(d_sandy).sum(); base = loss_storey(d_sandy, 1.0).sum(); k_nsi = SANDY_LOSS[1] / base
print(f"\nSandy footprint: structures flooded {int((d_sandy>0).sum()):,}, exposed value {val[d_sandy>0].sum()/1e9:.0f} bn")
print(f"  USACE generic curves (no free parameter): {L_us/1e9:.2f} bn vs observed 12.35 (11.4–13.3) -> ratio {L_us/SANDY_LOSS[1]:.2f}")
print(f"  storey-fraction curve: k recalibrated on NSI values = {k_nsi:.3f} (was 0.477 on PLUTO $400/sqft)")

# ---------------- hazard curve and AAI ----------------
curve = json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]
T = np.array(sorted(int(k) for k in curve)); WLm = np.array([curve[str(t)] for t in T])
aeps = np.logspace(0, -4, 120)
wl_navd = np.interp(np.log(1 / aeps), np.log(T), WLm) - NAVD_MINUS_MSL


def aai_of(loss_fn, wl_shift=0.0):
    Lw = np.array([loss_fn(depths[i]).sum() for i in range(len(levels))])
    L = np.interp(wl_navd + wl_shift, levels, Lw, left=0.0)
    return float(-np.trapz(L, aeps)), L


res = {}
for name, fn in (("USACE generic, NSI", loss_usace), ("storey curve k_nsi, NSI", lambda d: loss_storey(d, k_nsi))):
    a, L = aai_of(fn)
    rp = {int(t): float(np.interp(np.log(t), np.log(1 / aeps), L)) / 1e9 for t in (10, 100, 143, 1000)}
    res[name] = dict(aai_m=a / 1e6, loss_bn_by_rp=rp); print(f"{name:26} AAI {a/1e6:6.0f} M/yr | loss bn @10/100/143/1000 yr: " + "/".join(f"{rp[t]:.1f}" for t in (10, 100, 143, 1000)))
# PLUTO-basis reference (from naccs_aai.py): 754 M/yr with k=0.477 on $400/sqft
res["reference PLUTO $400/sqft, storey k=0.477"] = dict(aai_m=754.0)
res["sandy"] = dict(usace_bn=L_us / 1e9, k_nsi=k_nsi, exposed_bn=val[d_sandy > 0].sum() / 1e9)
json.dump(res, open("paper_a_core.json", "w"), indent=1)
