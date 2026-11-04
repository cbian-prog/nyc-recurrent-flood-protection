"""Supplementary Methods (valuation): how much do the Fig. 4 insurance shares depend on the structure valuation?
Same surge depths, damage functions (USACE, 3/N rule) and NFIP policies as fig5_gap_nsi.py; only the value of each
structure changes:
  NSI          the valuation used in the paper (NSI replacement value)
  PLUTO        each NSI structure matched to the nearest PLUTO tax-lot centroid within 50 m; the lot's building floor area
               x USD 400 per sq ft is shared among the structures matched to it in proportion to their NSI values;
               unmatched structures keep their NSI value divided by the citywide NSI/PLUTO ratio of matched structures
  NSI x 0.66 / x 1.51   5th and 95th percentiles of the lognormal valuation multiplier (sigma 0.25) in Supplementary Table 4
Output: nc_valuation_sensitivity.json
"""
import json, warnings, numpy as np, pandas as pd
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
src = open("fig5_gap_nsi.py").read(); src = src[:src.index("# per-level totals")]
src = src.replace('import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt', "").replace('matplotlib.rcParams.update', "dict")
exec(src)
# ---- PLUTO valuation per structure
p = pd.read_csv("data/pluto_nyc.csv", low_memory=False, usecols=["bbl", "latitude", "longitude", "bldgarea"])
for c in ("latitude", "longitude", "bldgarea"): p[c] = pd.to_numeric(p[c], errors="coerce")
p = p.dropna().query("bldgarea > 0").reset_index(drop=True)
KX, KY = 111320 * np.cos(np.radians(40.7)), 110950.0
tree = cKDTree(np.c_[p.longitude.values * KX, p.latitude.values * KY])
dist, idx = tree.query(np.c_[s.x.values * KX, s.y.values * KY], distance_upper_bound=50.0)
ok = np.isfinite(dist); lot = np.where(ok, idx, -1)
lot_val = p.bldgarea.values * 400.0
share = pd.Series(np.where(ok, val, 0.0)).groupby(lot).transform("sum").values
val_pluto = np.where(ok, lot_val[np.clip(lot, 0, None)] * val / np.where(share > 0, share, 1), np.nan)
ratio = val[ok].sum() / val_pluto[ok].sum()
val_pluto = np.where(ok, val_pluto, val / ratio)
print(f"structures matched to a lot within 50 m: {ok.mean():.1%} ({val[ok].sum()/val.sum():.1%} of NSI value) | NSI/PLUTO value ratio of matched: {ratio:.2f}")
# ---- the Fig. 4 quantities for a given valuation
interp = lambda y: np.interp(wl, levels, y, left=0)
def run(v):
    tot = {k: np.zeros(len(levels)) for k in ("loss", "upper", "neutral", "full_local", "full_stat")}
    for i in range(len(levels)):
        li = loss_usace(depths[i], valuation=v)
        lb = pd.Series(li).groupby(bg).sum().reindex(bgi).fillna(0).values
        tot["loss"][i] = li.sum(); tot["upper"][i] = np.minimum(lb, cov_bg).sum(); tot["neutral"][i] = (take * np.minimum(li, avgcov)).sum()
        tot["full_local"][i] = np.minimum(li, avgcov).sum(); tot["full_stat"][i] = np.minimum(li, stat_cap).sum()
    out = {"aai_M": float(-np.trapz(interp(tot["loss"]), aeps) / 1e6)}
    out["aai_share"] = {k: float(100 * -np.trapz(interp(tot[k]), aeps) / -np.trapz(interp(tot["loss"]), aeps)) for k in tot if k != "loss"}
    for rp in (10, 100, 1000):
        w_rp = np.interp(np.log(rp), np.log(T), WLm) - NAVD_MINUS_MSL; at_rp = lambda y: float(np.interp(w_rp, levels, y, left=0))   # exactly at the return period
        L = at_rp(tot["loss"])
        out[f"rp{rp}"] = {"loss_bn": float(L / 1e9), **{k: float(100 * at_rp(tot[k]) / L) for k in tot if k != "loss"}}
    return out
RES = {}
for name, v in (("NSI (used)", val), ("PLUTO $400/sq ft", val_pluto), ("NSI x 0.66", val * 0.66), ("NSI x 1.51", val * 1.51)):
    RES[name] = run(v); r = RES[name]
    print(f"{name:18} AAI {r['aai_M']:5.0f} M | upper at 10/100/1000 yr {r['rp10']['upper']:.0f}/{r['rp100']['upper']:.0f}/{r['rp1000']['upper']:.0f}% | "
          f"AAI upper {r['aai_share']['upper']:.0f}% expected {r['aai_share']['neutral']:.0f}% | full local {r['aai_share']['full_local']:.0f}% statutory {r['aai_share']['full_stat']:.0f}%")
json.dump(dict(match_rate=float(ok.mean()), nsi_pluto_ratio=float(ratio), results=RES), open("nc_valuation_sensitivity.json", "w"), indent=1)
# ---- Sandy check: modelled structure loss on the observed Sandy footprint under each valuation (observed structure-only loss 8.0-12.1 bn)
SANDY = {name: float(loss_usace(d_sandy, valuation=v).sum() / 1e9) for name, v in (("NSI (used)", val), ("PLUTO $400/sq ft", val_pluto), ("NSI x 0.66", val * 0.66), ("NSI x 1.51", val * 1.51))}
print("Sandy structure loss (bn):", {k: round(x, 1) for k, x in SANDY.items()}, "| observed 8.0-12.1")
J = json.load(open("nc_valuation_sensitivity.json")); J["sandy_loss_bn"] = SANDY; json.dump(J, open("nc_valuation_sensitivity.json", "w"), indent=1)
# ---- valuations consistent with the observed Sandy structure loss (8.0-12.1 bn): NSI x 8.0/10.5 to x 12.1/10.5
lo_m, hi_m = 8.0 / SANDY["NSI (used)"], 12.1 / SANDY["NSI (used)"]
J = json.load(open("nc_valuation_sensitivity.json"))
for name, m in ((f"NSI x {lo_m:.2f} (Sandy low)", lo_m), (f"NSI x {hi_m:.2f} (Sandy high)", hi_m)):
    r = run(val * m); J["results"][name] = r
    print(f"{name:26} AAI {r['aai_M']:5.0f} M | upper at 10/100/1000 yr {r['rp10']['upper']:.0f}/{r['rp100']['upper']:.0f}/{r['rp1000']['upper']:.0f}% | "
          f"AAI upper {r['aai_share']['upper']:.0f}% expected {r['aai_share']['neutral']:.0f}% | full local {r['aai_share']['full_local']:.0f}% statutory {r['aai_share']['full_stat']:.0f}% | expected 10/1000 {r['rp10']['neutral']:.0f}/{r['rp1000']['neutral']:.0f}%")
json.dump(J, open("nc_valuation_sensitivity.json", "w"), indent=1)
