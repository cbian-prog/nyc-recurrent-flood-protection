"""Extended Data Fig. 2 | Can the surge model be trusted? Validation, hazard curve and vulnerability in one figure.
a  modelled footprint at Sandy's observed peak against the observed zone (hit, observed only, modelled only)
b  Sandy structure loss from the uncalibrated model against the observed range
c  Monte Carlo distribution of the surge annual average impact (AAI)
d  still-water level against return period at The Battery (NACCS; combined, tropical, extratropical) with Sandy
e  the four NACCS save points used to check the single curve
f  still-water level reconstructed at each save point, with offsets from The Battery
g  USACE generic depth-damage functions (and the storey-fraction curve kept as a sensitivity)
h  structure value by storey class and the share the functions apply to (3/N)
i  surge AAI under alternative storey rules
Panels come from figED1_validation.py, figED3_hazard_basis.py data and figED5_vulnerability.py, each prepared in its own namespace.
Replaces former Extended Data Figs 1, 2 and 7 (2026-11-04); the category-basis panel moved to the two-city Supplementary figure.
"""
import json, numpy as np, rasterio, geopandas as gpd
V = {}; exec(open("figED1_validation.py").read().split("# ---- figure")[0], V)         # Sandy validation and Monte Carlo
U = {}; exec(open("figED5_vulnerability.py").read().split("# ---- figure")[0], U)       # vulnerability
from nc_style import *   # after the source scripts, which change matplotlib defaults
import matplotlib as _mpl, nc_style as _ns, importlib; importlib.reload(_ns); from nc_style import *
from matplotlib.colors import ListedColormap; from matplotlib.patches import Patch
cv = json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T = np.array(sorted(int(k) for k in cv)); WL = np.array([cv[str(t)] for t in T])
cc = json.load(open("data/naccs/battery_naccs_tc_xc_curves.json")); Ttc = np.array(sorted(int(k) for k in cc["TC"])); WLtc = np.array([cc["TC"][str(t)] for t in Ttc])
Txc = np.array(sorted(int(k) for k in cc["XC"])); WLxc = np.array([cc["XC"][str(t)] for t in Txc]); SANDY_MSL = 3.44 + 0.063
J = json.load(open("data/naccs/chs/jpm_points.json")); rps = [10, 45, 100, 143, 400, 1000]
SP = {7672: ("The Battery", -74.0133, 40.7000), 11982: ("Jamaica Bay W", -73.8091, 40.6362), 12049: ("Jamaica Bay E", -73.7705, 40.6097), 14200: ("Rockaway", -73.7845, 40.5879)}
M = np.load("data/paper_a/masks_20m.npz"); s100, city = M["s100"], M["city"]
tr, dem, valid = V["tr"], V["dem"], V["valid"]; ext = [tr.c, tr.c + dem.shape[1] * tr.a, tr.f + dem.shape[0] * tr.e, tr.f]
water = np.where(valid & (dem < 0), 1, 0); boro = V["boro"]
fig = plt.figure(figsize=(FULL, 160 * MM))
def nycmap(ax):
    ax.set_xlim(-74.12, -73.72); ax.set_ylim(40.53, 40.92); ax.set_aspect(1 / np.cos(np.radians(40.7))); ax.set_xticks([-74.1, -73.9, -73.7]); ax.set_yticks([40.6, 40.7, 40.8, 40.9])
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)"); [ax.spines[s_].set_visible(True) for s_ in ("top", "right")]
# a: Sandy footprint
a = axes_mm(fig, 12, 104, 44, 50)
cls = np.zeros(dem.shape, "uint8"); cls[V["hit"]] = 1; cls[V["miss"]] = 2; cls[V["fa"]] = 3
a.imshow(water, extent=ext, cmap=ListedColormap([LAND, WATER]), interpolation="nearest", rasterized=True)
a.imshow(np.ma.masked_equal(cls, 0), extent=ext, cmap=ListedColormap([PROT, SANDY, SURGE100]), vmin=1, vmax=3, interpolation="nearest", rasterized=True)
boro.boundary.plot(ax=a, color=INK, lw=0.35); nycmap(a)
a.legend(handles=[Patch(fc=PROT, label=f"Hit ({100 * V['POD']:.0f}% of observed)"), Patch(fc=SANDY, label="Observed only"), Patch(fc=SURGE100, label=f"Modelled only ({100 * V['FAR']:.0f}% of modelled)")],
         loc="upper left", title=f"Sandy footprint, CSI {V['CSI']:.2f}", alignment="left", fontsize=4.6, title_fontsize=4.8)
panel(a, "a", dx=-10)
# b: Sandy loss
SL = V["SANDY_LOSS"]; L_us = V["L_us"]
b = axes_mm(fig, 76, 114, 16, 38)
b.bar([0], [L_us], 0.6, color=PROT); b.axhspan(SL[0] / 1e9, SL[2] / 1e9, color=SANDY, alpha=0.15, lw=0); b.axhline(SL[1] / 1e9, color=SANDY, lw=0.8)
b.text(0, L_us - 0.3, f"{L_us:.1f}", ha="center", va="top", color="white", fontsize=6); b.text(0.45, SL[2] / 1e9 + 0.2, "observed direct\n11.4–13.3", color=SANDY, fontsize=5.5, ha="center", va="bottom")
b.axhspan(SL[0] / 1e9 * 0.70, SL[2] / 1e9 * 0.909, color=PROT, alpha=0.10, lw=0, hatch="////", ec="#9A9A9A")
b.text(0.95, SL[0] / 1e9 * 0.70 - 0.2, f"structure\nonly\n{SL[0] / 1e9 * 0.70:.1f}–{SL[2] / 1e9 * 0.909:.1f}", color=PROT, fontsize=5.5, ha="center", va="top")
b.set_xlim(-0.6, 1.5); b.set_ylim(0, 16); b.set_xticks([0]); b.set_xticklabels(["Untuned\nmodel"]); b.set_ylabel("Sandy loss (USD bn)")
panel(b, "b", dx=-11)
# c: Monte Carlo
q, out, aai1 = V["q"], V["out"], V["aai1"]
c = axes_mm(fig, 110, 114, 67, 38)
c.hist(out / 1e6, bins=50, color=NEUTRAL, edgecolor="none")
for v, ls in ((q[0], ":"), (q[2], "-"), (q[4], ":")): c.axvline(v, color=INK, ls=ls, lw=0.7)
c.axvline(aai1, color=SURGE10, lw=1.0)
c.text(0.98, 0.97, f"median {q[2]:.0f}\n5–95% {q[0]:.0f}–{q[4]:.0f}", transform=c.transAxes, ha="right", va="top", fontsize=6)
c.text(0.98, 0.70, "point estimate %d" % aai1, transform=c.transAxes, color=SURGE10, fontsize=6, ha="right", va="top")
c.set_xlabel("Surge AAI (M USD yr$^{-1}$)"); c.set_ylabel("Draws"); c.set_xlim(150, 1300)
panel(c, "c", dx=-11)
# d: hazard curve
d = axes_mm(fig, 14, 60, 58, 34)
d.fill_between(T, WL - .1, WL + .1, color=PROT, alpha=0.15, lw=0); d.semilogx(T, WL, "-o", color=PROT, ms=2.5, lw=1.0, label="Combined (used)")
d.semilogx(Ttc, WLtc, "--", color=SURGE10, lw=0.8, label="Tropical"); d.semilogx(Txc, WLxc, ":", color=TIDE, lw=1.0, label="Extratropical")
d.plot([139], [SANDY_MSL], "D", color=SANDY, ms=3, label="Sandy (≈1-in-140 yr)")
d.set_xlim(1, 1e4); d.set_ylim(0, 7); d.set_xlabel("Return period (yr)"); d.set_ylabel("Still-water level at\nThe Battery (m MSL)"); d.grid(which="major"); d.legend(loc="upper left", fontsize=5.5)
panel(d, "d", dx=-11)
# e: save points
e = axes_mm(fig, 88, 58, 30, 38)
e.imshow(water, extent=ext, cmap=ListedColormap([LAND, WATER]), interpolation="nearest", rasterized=True)
e.imshow(np.where(s100 & city, 1.0, np.nan), extent=ext, cmap=ListedColormap([SURGE100]), interpolation="nearest", rasterized=True); boro.boundary.plot(ax=e, color=INK, lw=0.3)
OFF = {7672: (-5, -4, "right", "top"), 11982: (-3, 5, "right", "bottom"), 12049: (5, 1, "left", "center"), 14200: (-3, -5, "right", "top")}
for k, (nm, lo, la) in SP.items():
    e.plot(lo, la, "*" if k == 7672 else "o", color=PROT, ms=6 if k == 7672 else 3.5, mec="white", mew=0.4); dx, dy, ha, va = OFF[k]
    e.annotate(str(k), (lo, la), xytext=(dx, dy), textcoords="offset points", fontsize=5.5, ha=ha, va=va)
nycmap(e); panel(e, "e", dx=-11)
# f: save point curves
f = axes_mm(fig, 134, 60, 43, 34); sty = {7672: (PROT, "-"), 11982: (SURGE10, "--"), 12049: (SURGE100, "-."), 14200: (TIDE, ":")}
for k, (col, ls) in sty.items(): f.semilogx(rps, [J[str(k)][str(r)] for r in rps], ls, marker="o", ms=2.3, color=col, lw=0.9, label=f"{k} {SP[k][0]}")
off = {k: [J[str(k)][str(r)] - J["7672"][str(r)] for r in (100, 1000)] for k in (11982, 12049, 14200)}
f.set_xlabel("Return period (yr)"); f.set_ylabel("Reconstructed still-water\nlevel (m)"); f.grid(which="major"); f.legend(loc="upper left", fontsize=5)
f.text(0.98, 0.04, "offset at 100 / 1,000 yr:\n" + "\n".join(f"{k}: {v[0]:+.2f} / {v[1]:+.2f} m" for k, v in off.items()) + "\nannual loss −2.3%", transform=f.transAxes, ha="right", va="bottom", fontsize=5)
panel(f, "f", dx=-14)
# g: damage functions
g = axes_mm(fig, 14, 12, 46, 34); FT = U["FT"]; kn = U["k_nsi"]
gsty = {"1S_NB": (SURGE10, "-", "1 storey, no basement"), "2S_NB": (SURGE10, "--", "2+ storeys, no basement"), "1S_WB": (RAIN_EXT, "-", "1 storey, basement"), "2S_WB": (RAIN_EXT, "--", "2+ storeys, basement")}
for k, (c_, ls, lab) in gsty.items(): x, y = U["CURVES"][k]; g.plot(x, y, ls, color=c_, lw=0.9, label=lab)
dd = np.linspace(0, 16, 80)
g.plot(dd, 100 * np.clip(kn * np.minimum(1, dd * FT / 2) * np.minimum(1, np.ceil(dd * FT / 3) / 1), 0, 1), ":", color=PROT, lw=0.9, label=f"Storey curve, N = 1 (k = {kn:.2f})")
g.set_xlim(-8, 16); g.set_ylim(0, 100); g.set_xlabel("Depth above first floor (ft)"); g.set_ylabel("Structure damage (%)"); g.grid(which="major"); g.legend(loc="upper left", fontsize=4.8)
panel(g, "g", dx=-12)
# h: value by storey
h = axes_mm(fig, 74, 12, 50, 34); labs, vb, inplay = U["labs"], U["vb"], U["inplay"]; xb = np.arange(len(labs))
h.bar(xb, vb, 0.65, color=NEUTRAL, label="Structure value"); h.bar(xb, inplay, 0.65, color=PROT, label="Value the functions apply to (3/N)")
for i in range(len(labs)): h.text(i, vb.iloc[i] + 6, f"{100 * inplay.iloc[i] / max(vb.iloc[i], 1e-9):.0f}%", ha="center", va="bottom", fontsize=5.5)
h.set_xticks(xb); h.set_xticklabels(labs); h.set_xlabel("Storeys"); h.set_ylabel("Structure value\n(USD bn)"); h.set_ylim(0, vb.max() * 1.35); h.legend(loc="upper right", fontsize=5.5)
panel(h, "h", dx=-12)
# i: AAI by rule
res = U["res"]; names = list(res); v_ = [res[n] for n in names]; cols = [SURGE10] + [NEUTRAL] * 4 + [PROT_LIGHT]
i_ = axes_mm(fig, 153, 12, 24, 34)
i_.barh(range(len(names)), v_, 0.65, color=cols); [i_.text(v + 12, k, f"{v:.0f}", va="center", fontsize=5.5) for k, v in enumerate(v_)]
i_.set_yticks(range(len(names))); i_.set_yticklabels(names, fontsize=5.5); i_.invert_yaxis(); i_.set_xlabel("Surge AAI (M USD yr$^{-1}$)"); i_.set_xlim(0, max(v_) * 1.25); i_.tick_params(axis="y", length=0)
panel(i_, "i", dx=-27)
save(fig, "ED_Fig2", outdirs=("figures",))
