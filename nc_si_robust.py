"""Supplementary Fig. 1 | How robust is the street-flood frequency from rain? Duration rule, averaging time, transfer, record length.
a-d  duration criterion (data from nc_si_duration.py): 311 complaints by drain time; complaint odds ratio for floods draining
     >= T against < T; AUC of the hourly rain maximum by threshold; 1-ft frequency at rain-footprint lots relative to 30 min
e-g  averaging time (data from nc_si_subhourly.py): Atlas 14 intensity-duration curves; share of flood days below 1.75 in/h
     by averaging time; ratio of 30- and 15-min peak to clock-hour intensity at the airport gauges
h    borough transfer of flood rates (nc_si_transfer.json)
i    rainy days per year by bin at the median basement lot: radar 2020-2026 against the long-run rate used (nc_rain_climatology.json)
Replaces former Supplementary Figs 2 and 5 and panels e-f of the former Supplementary Fig. 1 (2026-11-04).
"""
import json, numpy as np, pandas as pd
Dd = {}; exec(open("nc_si_duration.py").read().split("# ---- figure")[0], Dd)
Hs = {}; exec(open("nc_si_subhourly.py").read().split("# ---------------- figure")[0], Hs)
from nc_style import *
import nc_style as _ns, importlib; importlib.reload(_ns); from nc_style import *
OUT, LAB = Dd["OUT"], Dd["LAB"]; TR = json.load(open("nc_si_transfer.json")); CL = json.load(open("nc_rain_climatology.json"))
fig = plt.figure(figsize=(FULL, 156 * MM))
# a
a = axes_mm(fig, 14, 114, 50, 34); g = ["none"] + LAB; v = [100 * OUT["complaints"][x]["share"] for x in g]
a.bar(range(len(g)), v, 0.68, color=[NEUTRAL] + [RAIN_EXT] * len(LAB))
for i_, x in enumerate(g): a.text(i_, v[i_] + 0.8, f"n={OUT['complaints'][x]['n']:,}", ha="center", va="bottom", fontsize=5.8)
a.axvline(3.5, color=PROT, ls="--", lw=0.7); a.set_xticks(range(len(g))); a.set_xticklabels(["no flood"] + LAB, fontsize=6, rotation=40, ha="right", rotation_mode="anchor"); a.set_ylim(0, 50)
a.set_xlabel("Longest drain time that day (min)"); a.set_ylabel("Days with a 311 complaint\nwithin 250 m (%)"); a.grid(axis="y"); panel(a, "a", dx=-12)
# b
b = axes_mm(fig, 80, 114, 44, 34); SS = pd.DataFrame(OUT["split_scan"]); sig = SS.p < 0.05
b.fill_between(SS["T"], SS.lo, SS.hi, color=RAIN_EXT, alpha=0.2, lw=0); b.plot(SS["T"], SS.OR, "-", color=RAIN_EXT, lw=1, label="adjusted for rain")
b.plot(SS["T"], SS.OR_depth, "--", color=INK, lw=0.8, label="also for depth"); b.axhline(1, color="#9A9A9A", lw=0.5); b.axvline(30, color=PROT, ls="--", lw=0.7)
lo_s, hi_s = SS["T"][sig].min(), SS["T"][sig].max(); b.axvspan(lo_s, hi_s, color="#EEEEEE", zorder=0)
b.text(118, 4.3, f"shaded: P < 0.05 ({lo_s}–{hi_s} min)", ha="right", fontsize=6.5, color="#555555")
b.set_xlim(10, 120); b.set_ylim(0, 6); b.set_xticks([10, 30, 60, 90, 120]); b.set_xlabel("Split at drain time T (min)")
b.set_ylabel("Complaint odds ratio,\n≥ T vs < T"); b.legend(loc="upper right", fontsize=6.5); b.grid(); panel(b, "b", dx=-12)
# c
Ts = list(OUT["by_threshold"]); c = axes_mm(fig, 140, 114, 37, 34)
c.plot(Ts, [OUT["by_threshold"][t]["auc_pooled"] for t in Ts], "-o", ms=2.3, color=PROT, label="all sensors")
c.plot(Ts, [OUT["by_threshold"][t]["auc_within"] for t in Ts], "--s", ms=2.0, color=PROT_LIGHT, label="within sensor")
c.axvline(30, color=PROT, ls="--", lw=0.7); c.set_xticks([0, 30, 60, 90, 120]); c.set_xlabel("Drain-time threshold (min)"); c.set_ylabel("AUC of hourly\nrain maximum")
c.legend(loc="upper right", fontsize=6.5); c.grid(); panel(c, "c", dx=-12)
# d
d = axes_mm(fig, 14, 63, 50, 34); REL = {t: OUT["by_threshold"][t]["lot_1ft"] / OUT["by_threshold"][30]["lot_1ft"] for t in Ts}
d.plot(Ts, [REL[t] for t in Ts], "-o", ms=2.3, color=RAIN_EXT); d.axvline(30, color=PROT, ls="--", lw=0.7); d.axhline(1, color="#9A9A9A", lw=0.5)
d.set_xticks([0, 30, 60, 90, 120]); d.set_xlim(-6, 140); d.set_ylim(0, 1.3); d.set_xlabel("Drain-time threshold (min)"); d.set_ylabel("1-ft frequency at rain\nlots, relative to 30 min"); d.grid()
for t in (0, 45, 60, 120): d.text(t + 3, REL[t] + 0.05, f"{REL[t]:.2f}", fontsize=6.5, color=RAIN_EXT)
d.text(0.97, 0.97, "flood days below design:\n" + ", ".join(f"{100 * OUT['by_threshold'][t]['below_design']:.0f}%" for t in (0, 30, 60)) + "\nat 0, 30 and 60 min", transform=d.transAxes, ha="right", va="top", fontsize=6.5, color=RAIN_EXT)
panel(d, "d", dx=-12)
# e
A_, ARI, DESIGN, out, F = Hs["A"], Hs["ARI"], Hs["DESIGN"], Hs["out"], Hs["F"]
e = axes_mm(fig, 80, 63, 40, 34); durs = ["5-min", "10-min", "15-min", "30-min", "60-min", "2-hr"]; mins = [5, 10, 15, 30, 60, 120]
for rp, col, ls in ((1, "#C6DBEF", "-"), (2, "#6BAED6", "-"), (5, "#2171B5", "-"), (10, "#08306B", "-"), (100, INK, "--")):
    k = list(ARI).index(rp); e.plot(mins, [float(A_.loc[dd].values[k]) for dd in durs], ls, color=col, lw=0.9, label=f"1-in-{rp} yr")
e.axhline(DESIGN, color=PROT, ls=":", lw=0.8); e.text(5.3, DESIGN * 0.93, "1.75 in h$^{-1}$", ha="left", va="top", fontsize=6.5, color=PROT)
e.set_xscale("log"); e.set_yscale("log"); e.set_xticks(mins); e.set_xticklabels([str(m) for m in mins]); e.minorticks_off()
e.set_xlabel("Duration (min)"); e.set_ylabel("Rainfall intensity (in h$^{-1}$)"); e.legend(loc="upper right", fontsize=6); e.grid(which="major"); panel(e, "e", dx=-12)
# f
f = axes_mm(fig, 135, 63, 42, 34); keys = [("hourly_mrms", "Hourly\nradar\n(used)"), ("est_30min", "Est.\n30-min\npeak"), ("est_15min", "Est.\n15-min\npeak"), ("worst_case_15min", "Hour's\nrain in\n15 min")]; x = np.arange(4)
for k_, (lab, col, ttl) in enumerate((("any", RAIN_EXT, "any depth"), ("12in", RAIN_MOD, "≥1 ft"))):
    v = [100 * out[f"below_design_{lab}"][k] for k, _ in keys]; f.bar(x + (k_ - 0.5) * 0.36, v, 0.36, color=col, label=ttl)
    [f.text(x[i] + (k_ - 0.5) * 0.36, v[i] + 1, f"{v[i]:.0f}", ha="center", va="bottom", fontsize=6) for i in range(4)]
f.set_xticks(x); f.set_xticklabels([t for _, t in keys], fontsize=6); f.set_ylim(0, 112); f.set_ylabel("Flood days below\n1.75 in h$^{-1}$ (%)"); f.legend(loc="upper right", fontsize=6.5); f.grid(axis="y")
panel(f, "f", dx=-12)
# g
gx = axes_mm(fig, 14, 12, 22, 34); rr = F[["r15", "r30"]].dropna()
gx.boxplot([rr.r30, rr.r15], widths=0.5, showfliers=False, medianprops=dict(color=INK), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
gx.set_xticks([1, 2]); gx.set_xticklabels(["30 min", "15 min"], fontsize=6.5); gx.set_ylabel("Peak / hourly\nintensity at gauge"); gx.set_ylim(0.8, None); gx.grid(axis="y"); panel(gx, "g", dx=-13)
# h
h = axes_mm(fig, 52, 12, 58, 34); B5 = ["Queens", "Brooklyn", "Bronx", "Manhattan", "Staten Island", "all"]; xb = np.arange(6)
for k_, (dk, col, lab) in enumerate((("any", RAIN_EXT, "any depth"), ("12in", RAIN_MOD, "≥1 ft"))):
    v = [TR["lobo"][dk][b_]["ratio"] or 0 for b_ in B5]; h.bar(xb + (k_ - 0.5) * 0.36, v, 0.36, color=col, label=lab)
h.text(2 + 0.18, 0.05, "no 1-ft\ndays", ha="center", va="bottom", fontsize=5.5, rotation=90); h.axhline(1, color=INK, ls=":", lw=0.6)
h.set_xticks(xb); h.set_xticklabels(["Queens", "Brooklyn", "Bronx", "Manhattan", "Staten Is.", "All"], fontsize=6.5, rotation=30, ha="right", rotation_mode="anchor"); h.set_ylabel("Predicted / observed\n(held-out borough)")
h.set_ylim(0, max(2.0, max(TR["lobo"][d_][b_]["ratio"] or 0 for d_ in ("any", "12in") for b_ in B5) * 1.12)); h.legend(loc="upper left", fontsize=6.5); h.grid(axis="y"); panel(h, "h", dx=-12)
# i
i_ = axes_mm(fig, 127, 12, 50, 34); obs_r = np.array(CL["radar_median_lot_rate_per_year"]); ratio = np.array(CL["ratio_by_bin"]); lr = obs_r * ratio; xb4 = np.arange(4)
i_.bar(xb4 - 0.18, obs_r, 0.36, color="#9A9A9A", label="radar, 2020–2026"); i_.bar(xb4 + 0.18, lr, 0.36, color=RAIN_EXT, label="long-run rate used")
for k in range(4): i_.text(k, max(obs_r[k], lr[k]) * 1.25, f"×{ratio[k]:.2f}", ha="center", fontsize=6.5)
i_.set_yscale("log"); i_.set_ylim(0.03, 300); i_.set_xticks(xb4); i_.set_xticklabels(["<0.5", "0.5–1", "1–1.75", "≥1.75"]); i_.set_xlabel("Maximum hourly rain (in h$^{-1}$)")
i_.set_ylabel("Rainy days per year\n(median basement lot)"); i_.legend(loc="upper right", fontsize=6.5); i_.grid(axis="y", which="major"); panel(i_, "i", dx=-12)
save(fig, "SI_Fig1", outdirs=("figures",))
