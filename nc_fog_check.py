"""Do FloodNet sensors record 'floods' in fog without rain, and does the 30-min drain rule remove them?

Every event in the full FloodNet table (all drain times) at the paper's non-tidal sensors is classed by the weather when it
started: rain (radar rain >= 0.02 in in any hour from 3 h before to 1 h after the start, at the sensor or the 8 surrounding
1 km cells, or precipitation reported at Central Park, LaGuardia or JFK within 1 h), fog without rain (neither, and fog or
mist (FG, BR) reported at one of the three airports within 1 h), or dry without fog. Airport reports: Iowa Environmental
Mesonet ASOS archive (routine and special METARs). Writes data/paper_a/event_rain_class.csv, which nc_events.load uses to drop
non-tidal events with no rain evidence. Finding (2026-11-04): fog without rain is rare (5 of 1,116 events), so fog
does not explain short events; the 30-min rule removes mainly brief, shallow wetting in light rain, but also about half of
the events in heavy rain, some of them deep. Outputs: nc_fog_check.json and figures/explore/fog_check.png (what the
30-min rule removes, by radar rain; the night of 11-12 March 2026 as an example).
"""
import glob, json, numpy as np, pandas as pd
import nc_events
S = nc_events.sensors(); S["lat"] = pd.to_numeric(S.latitude, errors="coerce"); S["lon"] = pd.to_numeric(S.longitude, errors="coerce")
S = S[(S.tidally_influenced != "Yes")].dropna(subset=["lat", "lon"]).set_index("sensor_id")
F = pd.DataFrame(json.load(open(nc_events.FULL))); F = F[F.sensor_id.isin(S.index)].copy()
F["t"] = pd.to_datetime(F.flood_start_time)                                                     # UTC
for c in ("max_depth_inches", "drain_time_mins", "duration_mins"): F[c] = pd.to_numeric(F[c], errors="coerce")
F = F.dropna(subset=["drain_time_mins"])
# radar
files = sorted(glob.glob("data/mrms/*.npz")); z0 = np.load(files[0]); a, _, c0, _, e, f0 = z0["transform"]; sh = z0["h00"].shape
H = {}
for f in files:
    z = np.load(f); day = pd.Timestamp(f.split("/")[-1][:8])
    for k in z.files:
        if k.startswith("h"): v = z[k].astype("float32"); v[v < 0] = 0; H[day + pd.Timedelta(hours=int(k[1:]))] = v / 25.4
def radar(t, lat, lon):
    r, c = int((lat - f0) / e), int((lon - c0) / a); hs = [t.floor("h") + pd.Timedelta(hours=h) for h in range(-2, 3)]
    have = [h for h in hs if h in H]
    return max(H[h][max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].max() for h in have) if have else np.nan
F["radar_in"] = [radar(t, S.at[s, "lat"], S.at[s, "lon"]) for t, s in zip(F.t, F.sensor_id)]
# airports
W = pd.read_csv("data/asos_hourly/wx_NYC_LGA_JFK_2020_2026.csv"); W["t"] = pd.to_datetime(W.valid)
W["wx"] = W.wxcodes.fillna("M").astype(str); W["p"] = pd.to_numeric(W.p01i.replace("T", "0.001"), errors="coerce").fillna(0)
W["precip"] = W.wx.str.contains("RA|DZ|SN|TS|PL|UP|SH|GR|GS") | (W.p > 0)
W["fog"] = W.wx.str.contains("FG|BR") & ~W.precip
W = W.set_index("t").sort_index()
def airport(t):
    w = W[t - pd.Timedelta(hours=1): t + pd.Timedelta(hours=1)]
    return bool(w.precip.any()), bool(w.fog.any())
ap = [airport(t) for t in F.t]; F["ap_precip"] = [x[0] for x in ap]; F["ap_fog"] = [x[1] for x in ap]
F["cls"] = np.where((F.radar_in >= 0.02) | F.ap_precip, "rain", np.where(F.ap_fog, "fog, no rain", "dry, no fog"))
F["kept"] = F.drain_time_mins >= nc_events.PAPER_MIN_DRAIN
F[["sensor_id", "flood_start_time", "cls", "radar_in", "ap_precip", "ap_fog"]].to_csv("data/paper_a/event_rain_class.csv", index=False)   # read by nc_events.load
G = F.groupby("cls").agg(events=("t", "size"), kept=("kept", "sum"), median_drain_min=("drain_time_mins", "median"), median_depth_in=("max_depth_inches", "median"),
                          sensors=("sensor_id", "nunique"))
G["share_removed"] = 1 - G.kept / G.events
print(G.round(2).to_string())
nights = F[F.cls == "fog, no rain"].t.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.normalize().value_counts()
OUT = dict(by_class=G.reset_index().to_dict("records"), fog_events_top_days={str(k.date()): int(v) for k, v in nights.head(8).items()},
           paper_floods_total=int(F.kept.sum()), paper_floods_fog_no_rain=int(F[(F.cls == "fog, no rain") & F.kept].shape[0]),
           paper_floods_dry_no_fog=int(F[(F.cls == "dry, no fog") & F.kept].shape[0]))
json.dump(OUT, open("nc_fog_check.json", "w"), indent=1, default=float); print({k: v for k, v in OUT.items() if k != "by_class"})
R = F[F.radar_in.notna()].copy(); R["rc"] = pd.cut(R.radar_in, [-1, 0.05, 0.25, 0.5, 1, 99], labels=["<0.05", "0.05-0.25", "0.25-0.5", "0.5-1", ">=1"])
RC = R.groupby("rc", observed=False).agg(events=("t", "size"), kept=("kept", "sum"))
X = F.assign(tl=F.t.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None))
night = {lab: dict(events=int(len(m)), kept=int(m.kept.sum())) for lab, m in
         (("00:00-02:00 thunderstorm", X[(X.tl >= "2026-03-11 23:00") & (X.tl < "2026-03-12 02:00")]), ("06:00-09:00 light rain", X[(X.tl >= "2026-03-12 06:00") & (X.tl < "2026-03-12 09:00")]))}
OUT.update(by_radar_rain=dict(classes=list(RC.index.astype(str)), events=RC.events.astype(int).tolist(), share_removed=(1 - RC.kept / RC.events).round(3).tolist()),
           removed_median_depth_in=float(F[~F.kept].max_depth_inches.median()), kept_median_depth_in=float(F[F.kept].max_depth_inches.median()),
           removed_share_ge4in=float((F[~F.kept].max_depth_inches >= 4).mean()), march_12_2026=night)
json.dump(OUT, open("nc_fog_check.json", "w"), indent=1, default=float); print(night, OUT["by_radar_rain"])
# ---- figure
from nc_style import *
fig = plt.figure(figsize=(FULL, 70 * MM))
ax = axes_mm(fig, 14, 12, 72, 50); x = np.arange(len(RC))
ax.bar(x, RC.kept, 0.65, color=RAIN_EXT, label="drained in >= 30 min (counted)"); ax.bar(x, RC.events - RC.kept, 0.65, bottom=RC.kept, color="#D2D2D2", label="drained in < 30 min (not counted)")
for i_, (n_, k_) in enumerate(zip(RC.events, RC.kept)): ax.text(i_, n_ + 6, f"{100 * (1 - k_ / n_):.0f}% removed", ha="center", fontsize=5.5)
ax.set_xticks(x); ax.set_xticklabels(["<0.05", "0.05-0.25", "0.25-0.5", "0.5-1", ">=1"]); ax.set_xlabel("Radar rain near the sensor, 2 h either side of the start (in/h)")
ax.set_ylabel("Events at non-tidal sensors"); ax.legend(loc="upper left", fontsize=5.5); ax.grid(axis="y"); panel(ax, "a", dx=-11)
b = axes_mm(fig, 104, 12, 72, 50)
t0 = pd.Timestamp("2026-03-11 23:00"); t1 = pd.Timestamp("2026-03-12 09:00")
Xn = X[(X.tl >= t0) & (X.tl < t1)].sort_values("tl").reset_index(drop=True)
hrs = lambda t: (t - t0) / pd.Timedelta(hours=1)
Wl = W.copy(); Wl.index = Wl.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None); Wl = Wl[(Wl.index >= t0) & (Wl.index < t1)]
for t, r in Wl[Wl.fog].iterrows(): b.axvspan(hrs(t) - 0.05, hrs(t) + 0.05, color="#EDEDED", lw=0, zorder=0)
for t, r in Wl[Wl.precip].iterrows(): b.axvspan(hrs(t) - 0.05, hrs(t) + 0.05, color="#CFE3F3", lw=0, zorder=0)
b.scatter(hrs(Xn.tl[~Xn.kept]), Xn.index[~Xn.kept], s=4, c="#9A9A9A", lw=0, label=f"drained in < 30 min ({int((~Xn.kept).sum())})")
b.scatter(hrs(Xn.tl[Xn.kept]), Xn.index[Xn.kept], s=7, c=RAIN_EXT, lw=0, label=f"drained in >= 30 min ({int(Xn.kept.sum())})")
b.set_xlim(0, 10); b.set_xticks(range(0, 11, 2)); b.set_xticklabels([(t0 + pd.Timedelta(hours=h)).strftime("%H:%M") for h in range(0, 11, 2)])
b.set_xlabel("11-12 March 2026, New York time"); b.set_ylabel("Events, in order of start"); b.set_yticks([])
b.legend(loc="lower right", fontsize=5.5, title="blue: rain or thunder at an airport\ngrey: mist or fog only", title_fontsize=5.5, alignment="left")
panel(b, "b", dx=-6)
fig.savefig("figures/explore/fog_check.png", dpi=300); print("saved figures/explore/fog_check.png")
