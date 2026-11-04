"""Basement-dwelling exposure to flooding in NYC.

Exposure  : PLUTO lots with a below-grade basement (bsmtcode 2/4), residential classes A/B/C/D/S.
Occupancy : no register exists (informal units). Pratt Center's 2008 census-vs-DOB gap of ~114,000 units is
            allocated to 1-3 family and small walk-up lots (A, B, C0-C3) proportionally, giving an expected
            informal-unit count per lot. This is a citywide-rate allocation, not a per-lot fact.
Hazard    : DEP stormwater maps (Moderate-current, Extreme), SLOSH Cat1/Cat3, FloodNet/311 observed cells.
Frequency : expected street-ponding days per year at each lot = sum over MRMS hourly-max rain bins of
            P(flood | bin, FloodNet) x days/yr in that bin at the lot's 1 km MRMS cell (2020-11..2026-08).
"""

import glob
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

INFORMAL_UNITS = 114_000  # Pratt Center 2008, census vs DOB gap

p = pd.read_csv("data/pluto_bsmt.csv", low_memory=False).dropna(subset=["latitude", "longitude"])
p["cls"] = p.bldgclass.str[0]
p["small"] = p.cls.isin(["A", "B"]) | p.bldgclass.isin(["C0", "C1", "C2", "C3"])
p["units"] = pd.to_numeric(p.unitsres, errors="coerce").fillna(0)
w = np.where(p.small, 1.0, 0.0)
p["informal"] = INFORMAL_UNITS * w / w.sum()
print(f"lots with below-grade basement (residential): {len(p):,} | 1-3 family & small walk-ups: {int(p.small.sum()):,}")

with rasterio.open("data/manhattan_dem_20m.tif") as r:
    dem, tr, nod = r.read(1), r.transform, r.nodata
col = ((p.longitude.values - tr.c) / tr.a).astype(int); row = ((p.latitude.values - tr.f) / tr.e).astype(int)
inb = (row >= 0) & (row < dem.shape[0]) & (col >= 0) & (col < dem.shape[1])
print(f"inside model window: {int(inb.sum()):,}")


def lot_mask(shapes):
    m = features.rasterize(((x, 1) for x in shapes), out_shape=dem.shape, transform=tr, dtype="uint8").astype(bool)
    out = np.zeros(len(p), bool); out[inb] = m[row[inb], col[inb]]; return out


haz = {}
for lab, pat in (("DEP moderate (1-in-18 yr)", "*Moderate*Current*"), ("DEP extreme (1-in-458 yr)", "*Extreme*")):
    g = gpd.read_file(glob.glob(f"data/pluvial/NYCFloodStormwaterFloodMaps/{pat}/*.gdb")[0]).to_crs("EPSG:4326")
    haz[lab] = lot_mask(g[g.Flooding_Category.isin([1, 2])].geometry)
    haz[lab + " — deep ≥1 ft"] = lot_mask(g[g.Flooding_Category == 2].geometry)
for c in (1, 3):
    with rasterio.open(f"data/slosh/nyc_slosh_cat{c}.tif") as s:
        a, t, shp = s.read(1), s.transform, s.shape
    sc = ((p.longitude.values - t.c) / t.a).astype(int); sr = ((p.latitude.values - t.f) / t.e).astype(int)
    ok = (sr >= 0) & (sr < shp[0]) & (sc >= 0) & (sc < shp[1])
    cls = np.zeros(len(p), "uint8"); cls[ok] = a[sr[ok], sc[ok]]
    haz[f"SLOSH Cat{c} surge"] = (cls >= 1) & (cls <= 21)
# observed: 311 or FloodNet within ~200 m
p311 = pd.read_csv("data/sf_points.csv")
S = pd.DataFrame(json.load(open("data/floodnet/sensors.json")))
S["lat"] = pd.to_numeric(S.latitude, errors="coerce"); S["lon"] = pd.to_numeric(S.longitude, errors="coerce")
import nc_events; E = nc_events.load()
S = S[S.sensor_id.isin(E.sensor_id)].dropna(subset=["lat", "lon"])
obs = np.zeros(dem.shape, bool)
for df, la, lo in ((p311, "lat", "lon"), (S, "lat", "lon")):
    c = ((df[lo].values - tr.c) / tr.a).astype(int); r_ = ((df[la].values - tr.f) / tr.e).astype(int)
    k = (r_ >= 0) & (r_ < dem.shape[0]) & (c >= 0) & (c < dem.shape[1]); obs[r_[k], c[k]] = True
from scipy.ndimage import binary_dilation
obs = binary_dilation(obs, iterations=10)
m = np.zeros(len(p), bool); m[inb] = obs[row[inb], col[inb]]; haz["311 / FloodNet observed within 200 m"] = m

print(f"\n{'hazard footprint':42}{'lots':>9}{'legal res. units':>18}{'expected informal units':>25}")
res = {}
for lab, mk in haz.items():
    res[lab] = dict(lots=int(mk.sum()), units=int(p.units[mk].sum()), informal=float(p.informal[mk].sum()))
    print(f"{lab:42}{mk.sum():>9,}{int(p.units[mk].sum()):>18,}{p.informal[mk].sum():>25,.0f}")
any_pluv = haz["DEP extreme (1-in-458 yr)"]; any_surge = haz["SLOSH Cat3 surge"]
print(f"\npluvial-only (extreme, outside Cat3): lots {int((any_pluv & ~any_surge).sum()):,}, informal {p.informal[any_pluv & ~any_surge].sum():,.0f}")
print(f"in DEP extreme AND observed: lots {int((any_pluv & haz['311 / FloodNet observed within 200 m']).sum()):,}")

# ---- frequency: expected ponding days/yr from MRMS rain climatology x FloodNet flood rates ----
rate = {(0, 0.5): 0.0061, (0.5, 1.0): 0.076, (1.0, 1.75): 0.192, (1.75, 99): 0.364}  # from zhao_test2_mrms.py, in/hr
files = sorted(glob.glob("data/mrms/*.npz")); z0 = np.load(files[0]); a_, _, c0, _, e_, f0 = z0["transform"]
shape = z0["h00"].shape
daymax = []
for f in files:
    z = np.load(f); hs = [z[k] for k in z.files if k.startswith("h")]
    if len(hs) >= 20: daymax.append(np.max(np.stack(hs), 0) / 25.4)
daymax = np.stack(daymax); nyr = len(daymax) / 365.25 * (365.25 / 365.25)
# days fetched are rainy-ish days only; other days have max ~0 and fall in the lowest bin -> use full 5.8-yr span
span_yr = (pd.Timestamp("2026-08-22") - pd.Timestamp("2020-11-01")).days / 365.25
pond = np.zeros(shape)
for (lo, hi), r_ in rate.items():
    n = ((daymax > lo) & (daymax <= hi)).sum(0) / span_yr
    if lo == 0: n = 365.25 - ((daymax > 0.5).sum(0) / span_yr)  # all remaining days
    pond += r_ * n
mc = ((p.longitude.values - c0) / a_).astype(int); mr = ((p.latitude.values - f0) / e_).astype(int)
okm = (mr >= 0) & (mr < shape[0]) & (mc >= 0) & (mc < shape[1])
p["pond_days"] = np.nan; p.loc[okm, "pond_days"] = pond[mr[okm], mc[okm]]
print(f"\nexpected street-ponding days/yr at a DEP-mapped lot (MRMS climatology x FloodNet rates): "
      f"median {p.pond_days[any_pluv].median():.1f}, range {p.pond_days[any_pluv].quantile(.1):.1f}–{p.pond_days[any_pluv].quantile(.9):.1f}")
print("  (FloodNet rates are for sensor sites, i.e. known trouble spots; applying them to DEP-mapped lots assumes those lots pond like sensor sites)")
exp_unit_days = (p.informal * p.pond_days)[any_pluv].sum()
print(f"informal basement unit-days of street ponding per year in DEP extreme footprint: {exp_unit_days:,.0f}")

# ---- by borough ----
p["boro"] = p.bbl.astype(str).str[0].map({"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens", "5": "Staten Is."})
tb = pd.DataFrame({"basement lots": p.groupby("boro").size(), "informal units (alloc.)": p.groupby("boro").informal.sum().round(0),
                   "in DEP extreme": p[any_pluv].groupby("boro").informal.sum().round(0),
                   "in Cat3 surge": p[any_surge].groupby("boro").informal.sum().round(0)})
print("\n" + tb.to_string())
json.dump(dict(footprints=res, by_boro=tb.to_dict(), informal_total=INFORMAL_UNITS), open("basement_summary.json", "w"), indent=1)

# ---- figure ----
with rasterio.open("data/nyc_dem_55m.tif") as r:
    D, Dt, Dn = r.read(1), r.transform, r.nodata
dext = [Dt.c, Dt.c + D.shape[1] * Dt.a, Dt.f + D.shape[0] * Dt.e, Dt.f]
base = np.where((D == Dn) | (D < 0), 0.0, 1.0); lonc = Dt.c + (np.arange(D.shape[1]) + 0.5) * Dt.a
base[((D == Dn) | (D < -500)) & (lonc[None, :] < -74.24)] = 1.0
fig, ax = plt.subplots(1, 2, figsize=(17, 8.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax[0].imshow(base, extent=dext, cmap=ListedColormap(["#DDE7EC", "#F2F2ED"]), interpolation="nearest")
sm = p.small.values
ax[0].scatter(p.longitude[sm & ~any_pluv & ~any_surge], p.latitude[sm & ~any_pluv & ~any_surge], s=.3, c="#BFC7CC", lw=0)
ax[0].scatter(p.longitude[sm & any_surge & ~any_pluv], p.latitude[sm & any_surge & ~any_pluv], s=.8, c="#1B4965", lw=0)
ax[0].scatter(p.longitude[sm & any_pluv & ~any_surge], p.latitude[sm & any_pluv & ~any_surge], s=.8, c="#C9772E", lw=0)
ax[0].scatter(p.longitude[sm & any_pluv & any_surge], p.latitude[sm & any_pluv & any_surge], s=.8, c="#8E1B1B", lw=0)
ax[0].set_xlim(-74.27, -73.69); ax[0].set_ylim(40.49, 40.92)
ax[0].legend(handles=[Patch(fc="#BFC7CC", label="outside both footprints"), Patch(fc="#C9772E", label="DEP extreme pluvial only"),
                      Patch(fc="#1B4965", label="SLOSH Cat3 surge only"), Patch(fc="#8E1B1B", label="both")], loc="lower right", fontsize=9)
ax[0].set(xlabel="Longitude (°E)", ylabel="Latitude (°N)",
          title=f"(a) 1–3 family & small walk-up lots with a below-grade basement (n={int(sm.sum()):,})")
tb2 = tb[["in DEP extreme", "in Cat3 surge"]]
tb2.plot.barh(ax=ax[1], color=["#C9772E", "#1B4965"])
ax[1].set(xlabel="expected informal basement units in footprint (114,000 allocated citywide)", ylabel="",
          title="(b) Where the exposed informal units are")
for cont in ax[1].containers: ax[1].bar_label(cont, fmt="%.0f", fontsize=8, padding=2)
fig.suptitle("Basement dwellings and flooding, NYC — exposure that the asset-loss model cannot see", fontsize=14)
plt.tight_layout(); plt.savefig("basement_exposure_en.png", dpi=140, bbox_inches="tight"); print("saved basement_exposure_en.png")
