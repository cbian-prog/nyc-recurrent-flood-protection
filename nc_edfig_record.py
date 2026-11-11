"""Extended Data Fig. 1 | How to read the street-flood record: where the sensors are, how floods cluster in storms and places.
a  FloodNet sensors and 311 street-flooding complaints over the extreme rainfall footprint (siting)
b  concentration in a few storms: cumulative share of flooding against storm days ranked (days with >= 100 sensors reporting)
c  flood probability on a rainy day by maximum hourly rain: observed, model average over places, typical place; band, 10-90% of places
d  street floods per year at basement lots: rainfall footprint (typical lot, 90th percentile of places; storm-day bootstrap 95%
   intervals) against lots inside the 1-in-100-yr surge reach (modelled)
e  odds ratio of flooding at high tide at the same rain, near open water and inland (storm-day clustered 95% intervals)
Replaces former Extended Data Fig. 4 and panels a-d of the former Supplementary Fig. 1 (2026-11-04).
"""
import json, numpy as np, pandas as pd, rasterio
O = {}; exec(open("figED8_observation.py").read().split("# ---- figure")[0], O)       # siting data
from nc_style import *
import nc_style as _ns, importlib; importlib.reload(_ns); from nc_style import *
from matplotlib.colors import ListedColormap, LogNorm; from matplotlib.lines import Line2D; from matplotlib.patches import Patch
SS = json.load(open("nc_storm_stats.json")); SM = json.load(open("nc_storm_model.json")); N = json.load(open("nc_numbers.json")); TE = json.load(open("nc_tide_effect.json"))
SD = pd.read_parquet("data/paper_a/sensor_days_model.parquet").reset_index(drop=True)
expit = lambda x: 1 / (1 + np.exp(-x)); BINS = ["<0.5", "0.5–1", "1–1.75", "≥1.75"]
fig = plt.figure(figsize=(FULL, 112 * MM))
# a: siting
with rasterio.open("data/nyc_dem_55m.tif") as r: Dm, Dt, Dn = r.read(1), r.transform, r.nodata
dext = [Dt.c, Dt.c + Dm.shape[1] * Dt.a, Dt.f + Dm.shape[0] * Dt.e, Dt.f]; water = np.where((Dm == Dn) | (Dm < 0), 1, 0)
lonc = Dt.c + (np.arange(Dm.shape[1]) + .5) * Dt.a; water[((Dm == Dn) | (Dm < -500)) & (lonc[None, :] < -74.24)] = 0
tr, dem = O["tr"], O["dem"]; ext = [tr.c, tr.c + dem.shape[1] * tr.a, tr.f + dem.shape[0] * tr.e, tr.f]
a = axes_mm(fig, 13, 41, 66, 66)
a.imshow(water, extent=dext, cmap=ListedColormap([LAND, WATER]), interpolation="nearest", rasterized=True)
H = O["H"]; a.imshow(np.ma.masked_where(H.T == 0, H.T), extent=[O["lon_e"][0], O["lon_e"][-1], O["lat_e"][0], O["lat_e"][-1]], origin="lower", cmap="Greys", norm=LogNorm(vmin=1, vmax=50), alpha=0.9, interpolation="nearest", rasterized=True)
a.imshow(np.ma.masked_where(~O["dep"], np.ones(dem.shape)), extent=ext, cmap=ListedColormap([RAIN_EXT]), alpha=0.55, interpolation="nearest", rasterized=True)
O["boro"].boundary.plot(ax=a, color=INK, lw=0.3); a.scatter(O["S"].lon, O["S"].lat, s=1.3, c=INK, lw=0, zorder=5)
a.set_xlim(-74.26, -73.70); a.set_ylim(40.49, 40.92); a.set_aspect(1 / np.cos(np.radians(40.7))); a.set_xticks([-74.2, -74.0, -73.8]); a.set_yticks([40.5, 40.6, 40.7, 40.8, 40.9])
a.set_xlabel("Longitude (°E)"); a.set_ylabel("Latitude (°N)"); [a.spines[s_].set_visible(True) for s_ in ("top", "right")]
a.legend(handles=[Patch(fc=RAIN_EXT, alpha=0.55, label=f"Extreme-rain footprint ({100 * O['land_share']:.1f}% of land)"), Patch(fc="#7A7A7A", label="311 street-flooding complaints"),
                  Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=2, label=f"FloodNet sensor ({100 * O['s_in']:.0f}% inside; {O['enrich']:.1f}×)")], loc="upper left", fontsize=6, frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
panel(a, "a", dx=-10)
# b: storms
b = axes_mm(fig, 95, 74, 35, 33)
D = SD.groupby("date").agg(active=("sid", "size"), nflood=("y", "sum")); D = D[D.active >= 100]; fr = np.sort((D.nflood / D.active).values)[::-1]
cum = np.cumsum(fr) / fr.sum(); b.step(np.arange(1, len(cum) + 1), 100 * cum, where="post", color=RAIN_EXT, lw=1.0)
for k in (5, 10): b.plot([k], [100 * cum[k - 1]], "o", ms=2.5, color=RAIN_EXT); b.text(k + 2, 100 * cum[k - 1] - 7, f"top {k}: {100 * cum[k - 1]:.0f}%", fontsize=6.5)
b.set_xlim(0, (D.nflood > 0).sum() + 2); b.set_ylim(0, 102); b.set_xlabel("Storm days, ranked"); b.set_ylabel("Cumulative share\nof flooding (%)"); b.grid(which="major")
panel(b, "b", dx=-12)
# c: places
c = axes_mm(fig, 145, 74, 33, 33); x = np.arange(4)
obs = [N["flood_rate_by_bin"][k] for k in ("0-0.5", "0.5-1.0", "1.0-1.75", "1.75-2.5")]
bb = np.array(SM["any"]["b"]); ss = SM["any"]["point"]["s_site"]; GX, GW = np.polynomial.hermite_e.hermegauss(40); GW = GW / GW.sum()
c.fill_between(x, 100 * expit(bb - 1.2816 * ss), 100 * expit(bb + 1.2816 * ss), color=RAIN_EXT, alpha=0.15, lw=0, label="10–90% of places")
c.plot(x, 100 * np.array(obs), "s", ms=2.5, color=INK, label="observed")
o12 = np.array([N["flood_rate_by_bin_by_depth"]["12in"][k] for k in ("0-0.5", "0.5-1.0", "1.0-1.75", "1.75-2.5")])   # no 1 ft flood in the lightest bin
c.plot(x[o12 > 0], 100 * o12[o12 > 0], "^", ms=2.8, mfc="white", mec=INK, mew=0.6, label="observed, ≥1 ft")
c.axvline(2.5, color=PROT, ls="--", lw=0.7, label="sewer design intensity")
c.plot(x, [100 * (expit(v + ss * GX) * GW).sum() for v in bb], "-", color=RAIN_EXT, lw=0.9, label="average place")
c.plot(x, 100 * expit(bb), "--", color=RAIN_EXT, lw=0.9, label="typical place")
c.set_yscale("log"); c.set_ylim(0.001, 100); c.set_xticks(x); c.set_xticklabels(BINS, fontsize=6.5); c.set_xlabel("Max hourly rain (in h$^{-1}$)"); c.set_ylabel("Flood probability per\nrainy day (%)")
c.legend(loc="lower right", fontsize=5.6, frameon=True, facecolor="white", edgecolor="none", framealpha=1, borderpad=0.2); c.grid(which="major"); panel(c, "c", dx=-12)
# d: lots
d = axes_mm(fig, 123, 10, 55, 44); rows = []
for dk, lab in (("any", "any depth"), ("12in", "≥1 ft")):
    p, ci = SM[dk]["point"], SM[dk]["ci95"]; s_ = N[f"surge_floods_per_year_s100_lots_{dk}"]
    rows += [(f"Rain lots, typical ({lab})", p["lots_q50"], ci["lots_q50"], RAIN_EXT), (f"Rain lots, 90th pct. ({lab})", p["lots_q90"], ci["lots_q90"], RAIN_EXT),
             (f"Surge lots, median ({lab})", s_["median"], None, SURGE100), (f"Surge lots, 90th pct. ({lab})", s_["q90"], None, SURGE100)]
for i, (lab, v, ci, col) in enumerate(rows):
    y = len(rows) - 1 - i; d.plot([v], [y], "o" if "Rain" in lab else "s", ms=2.8, color=col, mfc=col if "Rain" in lab else "white")
    if ci: d.plot(ci, [y, y], color=col, lw=0.8)
d.set_xscale("log"); d.set_xlim(3e-3, 3); d.set_yticks(range(len(rows))); d.set_yticklabels([r[0] for r in rows][::-1], fontsize=6); d.tick_params(axis="y", length=0)
d.set_xlabel("Street floods per year at a lot"); d.grid(axis="x", which="major")
top = d.secondary_xaxis("top", functions=(lambda v: 1 / np.maximum(v, 1e-9), lambda v: 1 / np.maximum(v, 1e-9))); top.set_xticks([100, 10, 1]); top.set_xticklabels(["100", "10", "1"]); top.set_xlabel("Return period (yr)", labelpad=1); top.tick_params(labelsize=6.5)
panel(d, "d", dx=-31)
# e: tide
e = axes_mm(fig, 37, 10, 42, 18); ylab = []
for i, (dk, lab) in enumerate((("any", "any depth"), ("4in", "≥4 in"))):
    for j, (loc, col) in enumerate((("near", PROT), ("inland", "#8C8C8C"))):
        r = TE[dk][loc]["plus_storm_wetness"]; y = i * 2.5 + j
        e.plot([r["odds_ratio"]], [y], "o", ms=2.8, color=col); e.plot(r["ci95"], [y, y], color=col, lw=0.8); ylab.append((y, f"{'<1 km of water' if loc == 'near' else 'inland'}, {lab}"))
e.axvline(1, color=INK, lw=0.5, ls=":"); e.set_xscale("log"); e.set_xlim(0.2, 30); e.set_yticks([v for v, _ in ylab]); e.set_yticklabels([t for _, t in ylab], fontsize=6); e.set_ylim(4.1, -0.6)
e.set_xlabel("Odds ratio of flooding at high tide"); e.grid(axis="x", which="major"); e.tick_params(axis="y", length=0); panel(e, "e", dx=-25)
import os; save(fig, "ED_Fig1", outdirs=tuple(os.environ["FIG_OUTDIR"].split(",")) if os.environ.get("FIG_OUTDIR") else ("figures",))
