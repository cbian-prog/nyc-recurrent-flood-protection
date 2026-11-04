"""City-comparable NFIP overlay on SLOSH surge loss, exposure from USACE National Structure Inventory.

CITY=nyc|miami.  Two estimates of what NFIP pays:
  upper   : min(block-group loss, block-group building cover)  -- every policy sits on a damaged building
  neutral : policies placed at random among the structures of a block group:
            E[paid_i] = take_bg * min(loss_i, mean cover per contract in bg)
Gap decomposition under the neutral estimate:  loss - paid = (1-take)*loss  [no policy]  +  take*max(loss-cover,0)  [cap]
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio

CITY = os.environ.get("CITY", "nyc")
KS = [float(x) for x in os.environ.get("K", "0.477").split(",")]
FT, FLOOR_H = 0.3048, 3.0
CFG = {
    "nyc": dict(nsi="data/nsi_nyc/nsi.parquet", nfip="data/nfip/all_current.parquet",
                slosh="data/slosh/nyc_slosh_cat{c}.tif",
                freq={1: 0.02220, 2: 0.00700, 3: 0.00250, 4: 0.00100}),
    "miami": dict(nsi="data/miami/nsi.parquet", nfip="data/miami/nfip.parquet",
                  slosh="data/miami/slosh_cat{c}.tif",
                  freq={1: 0.05800, 2: 0.02680, 3: 0.01660, 4: 0.01400, 5: 0.00160}),
}[CITY]

s = pd.read_parquet(CFG["nsi"])
s = s[(s.val_struct > 0)].reset_index(drop=True)
s["bg"] = s.cbfips.astype(str).str[:12]
nf = np.clip(s.num_story.fillna(1).values, 1, 110)
val = s.val_struct.values
# statutory NFIP building limits: 250k per residential unit (RCBAP = 250k x units), 500k non-residential
stat_cap = np.where(s.st_damcat.values == 'RES', 2.5e5 * np.clip(s.resunits.fillna(1).values, 1, None), 5e5)

nfip = pd.read_parquet(CFG["nfip"])
bcov = nfip.groupby("censusGeoid").totalBuildingInsuranceCoverage.sum()
ncon = nfip.groupby("censusGeoid").size()
n_bg = s.groupby("bg").size()
take = (ncon.reindex(n_bg.index).fillna(0) / n_bg).clip(upper=1).reindex(s.bg).values
avgcov = (bcov / ncon).reindex(s.bg).fillna(0).values

depth = {}
for c in CFG["freq"]:
    with rasterio.open(CFG["slosh"].format(c=c)) as r:
        a, t, shp = r.read(1), r.transform, r.shape
    col = ((s.x.values - t.c) / t.a).astype(int); row = ((s.y.values - t.f) / t.e).astype(int)
    ok = (row >= 0) & (row < shp[0]) & (col >= 0) & (col < shp[1])
    cls = np.zeros(len(s), "uint8"); cls[ok] = a[row[ok], col[ok]]
    depth[c] = np.where((cls >= 1) & (cls <= 21), (cls.astype("float32") - 0.5) * FT, 0.0)

print(f"=== {CITY.upper()} === structures {len(s):,} | structure value {val.sum()/1e9:,.0f} bn | "
      f"NFIP contracts {len(nfip):,}, building cover {bcov.sum()/1e9:.1f} bn | "
      f"mean cover/contract {bcov.sum()/len(nfip)/1e3:,.0f} k")
wet = depth[max(CFG['freq'])] > 0
print(f"in largest footprint: {int(wet.sum()):,} structures, {val[wet].sum()/1e9:,.0f} bn | "
      f"structure-count take-up there {100*take[wet].mean():.1f}% | median structure value there {np.median(val[wet])/1e3:,.0f} k | "
      f"share of wet value in structures > 500k: {100*val[wet & (val>5e5)].sum()/val[wet].sum():.0f}%")

result = {}
for K in KS:
    print(f"\n-- K = {K} --")
    print(f"{'':6}{'RP':>7}{'loss bn':>10}{'upper %':>9}{'neutral %':>11}{'gap: no policy':>16}{'gap: cap':>10}{'paid if 100% take-up':>22}")
    tot = dict(loss=0, up=0, neu=0, nop=0, cap=0, full=0); rows = {}
    for c, f in CFG["freq"].items():
        d = depth[c]
        loss = val * np.clip(K * np.minimum(1, d / 2) * np.minimum(1, np.ceil(d / FLOOR_H) / nf), 0, 1)
        lb = pd.Series(loss).groupby(s.bg.values).sum()
        up = np.minimum(lb, bcov.reindex(lb.index).fillna(0)).sum()
        neu = (take * np.minimum(loss, avgcov)).sum()
        nop = ((1 - take) * loss).sum(); cap = (take * np.maximum(loss - avgcov, 0)).sum()
        L = loss.sum(); full = np.minimum(loss, stat_cap).sum()
        rows[c] = dict(rp=1 / f, loss_bn=L / 1e9, upper_pct=100 * up / L, neutral_pct=100 * neu / L,
                       gap_nopolicy_pct=100 * nop / L, gap_cap_pct=100 * cap / L, full_takeup_pct=100 * full / L)
        print(f"Cat{c:<3}{1/f:>7.0f}{L/1e9:>10.1f}{100*up/L:>9.1f}{100*neu/L:>11.1f}{100*nop/L:>16.1f}{100*cap/L:>10.1f}{100*full/L:>22.1f}")
        for k_, v_ in zip(("loss", "up", "neu", "nop", "cap", "full"), (L, up, neu, nop, cap, full)):
            tot[k_] += f * v_
    print(f"AAI  {'':>7}{tot['loss']/1e6:>9,.0f}M{100*tot['up']/tot['loss']:>9.1f}{100*tot['neu']/tot['loss']:>11.1f}"
          f"{100*tot['nop']/tot['loss']:>16.1f}{100*tot['cap']/tot['loss']:>10.1f}{100*tot['full']/tot['loss']:>22.1f}")
    result[str(K)] = dict(events=rows, aai_m=tot["loss"] / 1e6, upper_pct=100 * tot["up"] / tot["loss"],
                          neutral_pct=100 * tot["neu"] / tot["loss"], gap_nopolicy_pct=100 * tot["nop"] / tot["loss"],
                          gap_cap_pct=100 * tot["cap"] / tot["loss"], full_takeup_pct=100 * tot["full"] / tot["loss"])
json.dump(result, open(f"insurance_city_{CITY}.json", "w"), indent=1, default=float)
