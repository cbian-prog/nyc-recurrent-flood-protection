"""Data for Supplementary Fig. 1a-d: why a flood is counted when its water takes at least 30 min to drain.
FloodNet records any series of one-minute readings above 10 mm; no public standard sets a duration. On the paper's
40,676 rainy sensor-days at non-tidal sensors, with every event (drain time = maximum depth to dry, full NYC Open Data
table aq7i-eu5q):
 a  share of flood days with a 311 'Street Flooding (SJ)' complaint within 250 m on the same or next local day, by the
    day's longest drain time, against rainy days without an event; odds ratios adjusted for the hourly rain maximum
 b  skill (ROC AUC) of the hourly rain maximum for flood days counted at each drain-time threshold (are short events noise?)
 c  headline results at each threshold: share of flood days below the 1.75 in/h design intensity, and the ratio of
    1-ft flood frequency at rain-exposed to surge-exposed basement lots
Output: nc_si_duration.json, SI_Fig6
"""
import json, warnings, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
exec(open("nc_data.py").read().split("# ---------------------------------------------------------------- 4.")[0])   # ponding() and the lot cells of the rain footprint
N0 = json.load(open("nc_numbers.json")); SURGE_1FT = N0["surge_floods_per_year_s100_lots_12in"]["median"]
SDp = pd.read_parquet("data/paper_a/sensor_days.parquet")[["sid", "date", "pmax", "lat", "lon"]]; SDp["date"] = pd.to_datetime(SDp.date)
Ff = pd.DataFrame(json.load(open("data/floodnet/events_full_20261103/events_full.json"))); Ff["t0"] = pd.to_datetime(Ff.flood_start_time).dt.floor("min")
Oo = pd.DataFrame(json.load(open("data/floodnet/events.json"))); Oo["t0"] = pd.to_datetime(Oo.flood_start_time).dt.floor("min")
Ev = Ff.merge(Oo[["sensor_id", "t0"]], on=["sensor_id", "t0"])                           # every event of the paper's record
_C = pd.read_csv("data/paper_a/event_rain_class.csv"); _C["t0"] = pd.to_datetime(_C.flood_start_time).dt.floor("min")   # same cleaning as nc_events.load:
Ev = Ev.merge(_C[["sensor_id", "t0", "cls"]], on=["sensor_id", "t0"], how="left"); Ev = Ev[~Ev.cls.isin(["fog, no rain", "dry, no fog"])].drop(columns="cls")   # no rain, no flood
for c in ("drain_time_mins", "max_depth_inches"): Ev[c] = pd.to_numeric(Ev[c], errors="coerce")
Ev["date"] = pd.to_datetime(Ev.flood_start_time).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
day = Ev.groupby(["sensor_id", "date"]).agg(drain=("drain_time_mins", "max"), dmax=("max_depth_inches", "max")).reset_index().rename(columns={"sensor_id": "sid"})
SD = SDp.merge(day, on=["sid", "date"], how="left"); SD["y"] = SD.drain.notna().astype(int); SD["dmax"] = SD.dmax.fillna(0)
OUT = {"sensor_days": int(len(SD)), "flood_days_all_events": int(SD.y.sum()), "events": int(len(Ev)),
       "share_events_drain_lt30": float((Ev.drain_time_mins < 30).mean()), "share_events_drain_lt60": float((Ev.drain_time_mins < 60).mean())}
# ---- a. 311 complaints within 250 m, same or next local day (complaint record ends 2026-07-29)
Q = pd.read_csv("data/sf_points.csv"); Q["d"] = pd.to_datetime(Q.created_date).dt.normalize()
qlat, qlon, qd = Q.latitude.values, Q.longitude.values, Q.d.values.astype("datetime64[D]"); KX, KY, R = 111320 * np.cos(np.radians(40.7)), 110950.0, 250.0
def near(lat, lon, d):
    m = (np.abs(qlat - lat) * KY < R) & (np.abs(qlon - lon) * KX < R)
    if not m.any(): return 0
    dd = qd[m][np.hypot((qlat[m] - lat) * KY, (qlon[m] - lon) * KX) < R]; d0 = np.datetime64(d, "D"); return int(((dd == d0) | (dd == d0 + 1)).any())
A = SD[SD.date <= pd.Timestamp("2026-07-28")].copy()
A["c311"] = [near(a_, b_, d_) for a_, b_, d_ in zip(A.lat.values, A.lon.values, A.date.values)]
EDG = [0, 10, 20, 30, 45, 60, 120, 1e9]; LAB = ["<10", "10–20", "20–30", "30–45", "45–60", "60–120", "≥120"]
A["grp"] = np.where(A.y == 0, "none", pd.cut(A.drain, EDG, labels=LAB, right=False).astype(str))
A["g"] = pd.Categorical(A.grp, categories=["none"] + LAB); A["lp"] = np.log1p(A.pmax)
m = smf.logit("c311 ~ C(g) + lp", A).fit(disp=0); ci = m.conf_int()
OUT["complaints"] = {g: dict(n=int((A.grp == g).sum()), share=float(A.c311[A.grp == g].mean()), pmax_median=float(A.pmax[A.grp == g].median()),
                             depth_median=float(A.dmax[A.grp == g].median())) for g in ["none"] + LAB}
for l in LAB:
    k = f"C(g)[T.{l}]"; OUT["complaints"][l].update(OR=float(np.exp(m.params[k])), OR_lo=float(np.exp(ci.loc[k, 0])), OR_hi=float(np.exp(ci.loc[k, 1])))
# below and above 30 min, and the same with depth held fixed (flood days only)
A["ge30"] = np.where(A.y == 0, "none", np.where(A.drain >= 30, "ge30", "lt30")); A["g2"] = pd.Categorical(A.ge30, categories=["none", "lt30", "ge30"])
m2 = smf.logit("c311 ~ C(g2) + lp", A).fit(disp=0)
F_ = A[A.y == 1].assign(ld=lambda x: np.log(x.dmax.clip(lower=0.4)), long=lambda x: (x.drain >= 30).astype(int)); m3 = smf.logit("c311 ~ long + lp + ld", F_).fit(disp=0)
OUT["split30"] = dict(share_lt30=float(A.c311[A.ge30 == "lt30"].mean()), share_ge30=float(A.c311[A.ge30 == "ge30"].mean()), share_none=float(A.c311[A.ge30 == "none"].mean()),
                      OR_lt30=float(np.exp(m2.params["C(g2)[T.lt30]"])), OR_ge30=float(np.exp(m2.params["C(g2)[T.ge30]"])),
                      OR_ge30_vs_lt30_given_depth=float(np.exp(m3.params["long"])), p_given_depth=float(m3.pvalues["long"]))
# where does lasting longer matter? flood days split at T: complaint odds of >= T against < T (rain-adjusted; and with depth held fixed)
F_ = A[A.y == 1].assign(ld=lambda x: np.log(x.dmax.clip(lower=0.4))); OUT["split_scan"] = []
for T in range(10, 125, 5):
    F_["L"] = (F_.drain >= T).astype(int)
    if F_.L.sum() < 20 or (1 - F_.L).sum() < 20: continue
    m_ = smf.logit("c311 ~ L + lp", F_).fit(disp=0); md = smf.logit("c311 ~ L + lp + ld", F_).fit(disp=0); ci_ = m_.conf_int().loc["L"]
    OUT["split_scan"].append(dict(T=T, n_ge=int(F_.L.sum()), share_lt=float(F_.c311[F_.L == 0].mean()), share_ge=float(F_.c311[F_.L == 1].mean()),
                                  OR=float(np.exp(m_.params.L)), lo=float(np.exp(ci_[0])), hi=float(np.exp(ci_[1])), p=float(m_.pvalues.L),
                                  OR_depth=float(np.exp(md.params.L)), p_depth=float(md.pvalues.L)))
print(json.dumps(OUT, indent=1))
# ---- b, c. thresholds
BIN = pd.cut(SD.pmax, [0, 0.5, 1.0, DESIGN, 2.5], labels=["0-0.5", "0.5-1.0", "1.0-1.75", "1.75-2.5"])
ev_nt = day.copy(); OUT["by_threshold"] = {}
for T in (0, 10, 20, 30, 45, 60, 90, 120):
    y = (SD.drain >= T) if T > 0 else SD.drain.notna()
    # a day counts at threshold T if one of its events drains in >= T min; its depth is the deepest such event
    kk = Ev[Ev.drain_time_mins >= T].groupby(["sensor_id", "date"]).max_depth_inches.max().rename("dT").reset_index().rename(columns={"sensor_id": "sid"})
    S_ = SD[["sid", "date", "pmax"]].merge(kk, on=["sid", "date"], how="left"); y = S_.dT.notna(); y12 = y & (S_.dT >= 12)
    ra = {b: float(y[BIN == b].mean()) for b in BIN.cat.categories}; r12 = {b: float(y12[BIN == b].mean()) for b in BIN.cat.categories}
    p12 = ponding(r12)[pm[sel], pc[sel]]; pa = ponding(ra)[pm[sel], pc[sel]]
    w = [roc_auc_score(y[S_.sid == s], S_.pmax[S_.sid == s]) for s in S_.sid.unique() if 3 <= y[S_.sid == s].sum() < (S_.sid == s).sum() - 2]
    OUT["by_threshold"][T] = dict(flood_days=int(y.sum()), below_design=float((S_.pmax[y] < DESIGN).mean()), auc_pooled=float(roc_auc_score(y, S_.pmax)),
                                  auc_within=float(np.mean(w)), n_sensors_within=len(w), lot_any=float(np.median(pa)), lot_1ft=float(np.median(p12)),
                                  ratio_1ft=float(np.median(p12) / SURGE_1FT))
    print(T, {k: round(v, 3) if isinstance(v, float) else v for k, v in OUT["by_threshold"][T].items()})
json.dump(OUT, open("nc_si_duration.json", "w"), indent=1)
# ---- figure
from nc_style import *
fig = plt.figure(figsize=(FULL, 110 * MM))
a = axes_mm(fig, 13, 62, 70, 40); g = ["none"] + LAB; v = [100 * OUT["complaints"][x]["share"] for x in g]
a.bar(range(len(g)), v, 0.68, color=[NEUTRAL] + [RAIN_EXT] * len(LAB))
for i_, x in enumerate(g):
    a.text(i_, v[i_] + 0.8, f"n={OUT['complaints'][x]['n']:,}", ha="center", va="bottom", fontsize=5)
a.axvline(3.5, color=PROT, ls="--", lw=0.7)
a.set_xticks(range(len(g))); a.set_xticklabels(["no\nflood"] + LAB, fontsize=5.5); a.set_ylim(0, 47)
a.set_xlabel("Longest drain time that day (min)"); a.set_ylabel("Days with a 311 complaint within\n250 m, same or next day (%)"); a.grid(axis="y")
panel(a, "a", dx=-11)
b = axes_mm(fig, 106, 62, 70, 40); SS = pd.DataFrame(OUT["split_scan"]); sig = SS.p < 0.05
b.fill_between(SS["T"], SS.lo, SS.hi, color=RAIN_EXT, alpha=0.2, lw=0); b.plot(SS["T"], SS.OR, "-", color=RAIN_EXT, lw=1, label="adjusted for rain")
b.plot(SS["T"], SS.OR_depth, "--", color=INK, lw=0.8, label="also for depth")
b.axhline(1, color="#9A9A9A", lw=0.5); b.axvline(30, color=PROT, ls="--", lw=0.7)
lo_s, hi_s = SS["T"][sig].min(), SS["T"][sig].max(); b.axvspan(lo_s, hi_s, color="#EEEEEE", zorder=0)
b.text((lo_s + hi_s) / 2, 5.6, f"P < 0.05: {lo_s}–{hi_s} min", ha="center", fontsize=5.5, color="#555555")
b.set_xlim(10, 120); b.set_ylim(0, 6); b.set_xticks([10, 30, 60, 90, 120]); b.set_xlabel("Split at drain time T (min)")
b.set_ylabel("Complaint odds ratio,\nfloods draining ≥ T vs < T"); b.legend(loc="upper right"); b.grid(); panel(b, "b", dx=-11)
Ts = list(OUT["by_threshold"]); c = axes_mm(fig, 13, 10, 70, 36)
c.plot(Ts, [OUT["by_threshold"][t_]["auc_pooled"] for t_ in Ts], "-o", ms=2.3, color=PROT, label="all sensors")
c.plot(Ts, [OUT["by_threshold"][t_]["auc_within"] for t_ in Ts], "--s", ms=2.0, color=PROT_LIGHT, label="within sensor")
c.axvline(30, color=PROT, ls="--", lw=0.7); c.set_xticks([0, 30, 60, 90, 120]); c.set_xlabel("Drain-time threshold (min)"); c.set_ylabel("AUC of hourly rain maximum")
c.legend(loc="lower left"); c.grid(); panel(c, "c", dx=-11)
d = axes_mm(fig, 106, 10, 70, 36)
REL = {t_: OUT["by_threshold"][t_]["lot_1ft"] / OUT["by_threshold"][30]["lot_1ft"] for t_ in Ts}   # 1-ft frequency at rain-footprint lots relative to the 30-min rule
d.plot(Ts, [REL[t_] for t_ in Ts], "-o", ms=2.3, color=RAIN_EXT)
d.axvline(30, color=PROT, ls="--", lw=0.7); d.set_xticks([0, 30, 60, 90, 120]); d.set_ylim(0, 1.3); d.axhline(1, color="#9A9A9A", lw=0.5); d.set_xlabel("Drain-time threshold (min)")
d.set_ylabel("1-ft flood frequency at rain\nlots, relative to 30 min"); d.grid()
for t_ in (0, 45, 60, 120): d.text(t_ + 3, REL[t_] + 0.05, f"{REL[t_]:.2f}", fontsize=5.5, color=RAIN_EXT)
d.text(0.97, 0.05, "flood days below design: " + ", ".join(f"{100*OUT['by_threshold'][t_]['below_design']:.0f}%" for t_ in (0, 30, 60)) + "\nat 0, 30 and 60 min",
       transform=d.transAxes, ha="right", va="bottom", fontsize=5.5, color=RAIN_EXT)
panel(d, "d", dx=-11)
save(fig, "unused_SI_duration_only", outdirs=("figures/explore",))   # panels a-d of SI Fig. 1 (nc_si_robust.py)
