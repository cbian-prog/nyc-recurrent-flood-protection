"""Surge AAI from a WATER-LEVEL hazard curve (NACCS SP 7672, The Battery) instead of storm-category frequencies.

Loss(WL): connected bathtub on the 20 m NAVD88 DEM, PLUTO lots, by-storey curve with the Sandy-calibrated K.
AAI = integral of Loss(WL(AEP)) dAEP over AEP 1 .. 1e-4, WL(AEP) log-interpolated from the digitized NACCS curve.
Datum: NACCS WL is MSL (1983-2001 epoch); NAVD88 zero is 0.063 m above MSL at 8518750, so WL_navd = WL_msl - 0.063.
Sea-level change since the epoch (~+0.10 m to 2026) is applied as a sensitivity case.
Spatial simplification: one Battery curve applied citywide (the CHS per-save-point download would replace this).
"""

import json
import warnings

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, FLOOR_H, SQFT = 0.477, 3.0, 400.0
NAVD_MINUS_MSL = 0.063
curve = json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]
T = np.array(sorted(int(k) for k in curve)); WLm = np.array([curve[str(t)] for t in T])
wl_of_aep = lambda a: np.interp(np.log(1 / a), np.log(T), WLm)  # MSL

p = pd.read_csv("data/pluto_nyc.csv", low_memory=False)
for c in ("bldgarea", "numfloors"):
    p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
p = p[p.bldgarea > 0].reset_index(drop=True)
val = p.bldgarea.values * SQFT; nf = np.clip(p.numfloors.values, 1, None)

with rasterio.open("data/manhattan_dem_20m.tif") as s:
    dem, tr, nod = s.read(1), s.transform, s.nodata
valid = dem != nod; dem_v = np.where(valid, dem, 1e4)
seed = np.zeros(dem.shape, bool); seed[0, :] = seed[-1, :] = seed[:, 0] = seed[:, -1] = True; seed &= dem_v < -1.0
col = ((p.longitude.values - tr.c) / tr.a).astype(int); row = ((p.latitude.values - tr.f) / tr.e).astype(int)
inb = (row >= 0) & (row < dem.shape[0]) & (col >= 0) & (col < dem.shape[1])
dmg = lambda d: np.clip(K * np.minimum(1, d / 2) * np.minimum(1, np.ceil(np.maximum(d, 0) / FLOOR_H) / nf), 0, 1)


def loss_at(wl_navd):
    below = (dem_v < wl_navd) & valid
    lab, _ = ndimage.label(below)
    conn = np.isin(lab, np.unique(lab[seed & (lab > 0)])) & below & (dem_v >= 0)
    depth = np.where(conn, wl_navd - dem_v, 0.0)
    d = np.zeros(len(p)); d[inb] = depth[row[inb], col[inb]]
    return float((val * dmg(d)).sum()), int((d > 0).sum())


levels = np.arange(1.0, 7.01, 0.25)
L = {}
print(f"{'WL navd (m)':>12}{'lots wet':>10}{'loss bn':>10}")
for w in levels:
    L[w], n = loss_at(w); print(f"{w:>12.2f}{n:>10,}{L[w]/1e9:>10.2f}")
Lw = lambda w: np.interp(w, levels, [L[x] for x in levels])
print(f"\nSandy check: bathtub at 3.44 m NAVD88 -> {Lw(3.44)/1e9:.1f} bn vs 12.35 bn used for calibration (observed footprint)")

res = {}
for lab, slc in (("epoch MSL (1983-2001)", 0.0), ("+0.10 m SLC to 2026", 0.10)):
    aeps = np.logspace(0, -4, 120)
    losses = np.array([loss_at(wl_of_aep(a) - NAVD_MINUS_MSL + slc)[0] for a in aeps])  # exact, same grid as insurance_overlay_naccs.py
    aai = float(-np.trapz(losses, aeps))  # aep decreasing
    # contribution by return-period band
    bands = [(1, 10), (10, 50), (50, 100), (100, 500), (500, 1000), (1000, 10000)]
    contrib = {}
    for lo, hi in bands:
        m = (1 / aeps >= lo) & (1 / aeps <= hi)
        contrib[f"{lo}-{hi} yr"] = float(-np.trapz(losses[m], aeps[m]))
    res[lab] = dict(aai_m=aai / 1e6, contrib_m={k: v / 1e6 for k, v in contrib.items()},
                    loss_by_rp_bn={int(t): Lw(wl_of_aep(1 / t) - NAVD_MINUS_MSL + slc) / 1e9 for t in (10, 50, 100, 200, 500, 1000, 10000)})
    print(f"\n{lab}: AAI = {aai/1e6:,.0f} M/yr")
    print("  loss by return period (bn):", {k: round(v, 1) for k, v in res[lab]["loss_by_rp_bn"].items()})
    print("  AAI contribution by band (M):", {k: round(v / 1e6) for k, v in contrib.items()})
print("\nSLOSH-MOM x STORM category approach (this report): 1,411 M/yr; NACCS water-level approach:",
      f"{res['epoch MSL (1983-2001)']['aai_m']:,.0f} M/yr (ratio {res['epoch MSL (1983-2001)']['aai_m']/1411:.2f})")
json.dump(res, open("naccs_aai.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].semilogx(T, WLm, "-o", color="#1B4965", label="NACCS SP 7672 mean, MSL")
ax[0].axhline(3.50, color="#B33A30", ls="--", lw=1); ax[0].text(1.2, 3.55, "Sandy peak 3.50 m MSL", color="#B33A30", fontsize=9)
ax[0].set(xlabel="Return period (yr)", ylabel="Water level (m, MSL)", title="(a) Water-level hazard curve, The Battery (NACCS, digitised)"); ax[0].grid(alpha=.3); ax[0].legend()
rp = np.logspace(0, 4, 200)
ax[1].semilogx(rp, [Lw(wl_of_aep(1 / r) - NAVD_MINUS_MSL) / 1e9 for r in rp], color="#1B4965", label="NACCS water level → bathtub loss")
cat = {45: 25.4, 143: 59.7, 400: 104.4, 1000: 164.2}
ax[1].plot(list(cat), list(cat.values()), "s", color="#C9653A", label="SLOSH MOM by category (Cat 1–4)")
ax[1].set(xlabel="Return period (yr)", ylabel="Direct loss (USD bn)", title="(b) Loss–return period: two frequency bases"); ax[1].grid(alpha=.3); ax[1].legend()
fig.suptitle(f"Storm-surge AAI on a water-level basis: {res['epoch MSL (1983-2001)']['aai_m']:,.0f} M/yr vs 1,411 M/yr on a category basis", fontsize=12)
plt.tight_layout(); plt.savefig("naccs_aai_en.png", dpi=140, bbox_inches="tight"); print("saved naccs_aai_en.png")
