"""Storm-level description of rain flooding at FloodNet's non-tidal sensors, with intervals that resample storm days.

A storm day is a New York local day with rain at any sensor (sensor-day table from nc_data.py). Reports: storm days per year by
the day's citywide maximum hourly radar rain and the share of them with a flood anywhere in the network; how concentrated flood
days are in a few storms and in a few sensors; the share of flood days below the 1.75 in/h design intensity (any depth, 1 ft),
with 95% intervals from 2,000 storm-day bootstrap resamples. Because the network grew from 2 sensors (2021) to about 300 (2026),
concentration in a few storms is measured on the days with at least 100 sensors reporting, as the share of the summed
fraction of sensors flooded that falls on the top 5 and top 10 storm days. Output: nc_storm_stats.json
"""
import json, numpy as np, pandas as pd
SD = pd.read_parquet("data/paper_a/sensor_days_model.parquet").reset_index(drop=True)
yrs = (SD.date.max() - SD.date.min()).days / 365.25
D = SD.groupby("date").agg(nflood=("y", "sum"), n1ft=("dmax", lambda x: int((x >= 12).sum())), pmax_city=("pmax", "max"))
cls = pd.cut(D.pmax_city, [0, 0.5, 1, 1.75, 99], labels=["<0.5", "0.5-1", "1-1.75", ">=1.75"])
by = D.groupby(cls, observed=False).agg(days=("nflood", "size"), flood_days=("nflood", lambda x: int((x > 0).sum())), flood_sensor_days=("nflood", "sum"))
F = SD[SD.y == 1]; F1 = F[F.dmax >= 12]
rng = np.random.default_rng(0); dates = F.date.unique(); idx = F.groupby("date").indices
def boot(mask_fn, sub):
    g = sub.groupby("date").indices; u = np.array(list(g)); v = mask_fn(sub).values
    r = [v[np.concatenate([g[d] for d in rng.choice(u, len(u))])].mean() for _ in range(2000)]
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
vc = F.sid.value_counts(); light = F[F.pmax < 0.5]; vl = light.sid.value_counts()
OUT = dict(years=yrs, storm_days=int(len(D)), storm_days_per_year=len(D) / yrs, flood_storm_days=int((D.nflood > 0).sum()), flood_storm_days_per_year=float((D.nflood > 0).sum() / yrs),
           storm_days_1ft=int((D.n1ft > 0).sum()), flood_sensor_days=int(len(F)), flood_sensor_days_1ft=int(len(F1)),
           by_citywide_max=dict(classes=list(by.index.astype(str)), per_year=(by.days / yrs).round(2).tolist(), share_with_flood=(by.flood_days / by.days).round(3).tolist(),
                                flood_sensor_days=by.flood_sensor_days.astype(int).tolist()),
           share_flood_days_top10_storms=float(D.nflood.sort_values(ascending=False).head(10).sum() / D.nflood.sum()),
           share_flood_dates_with_1_2_sensors=float(((D.nflood > 0) & (D.nflood <= 2)).sum() / (D.nflood > 0).sum()),
           sensors_with_flood=int(F.sid.nunique()), sensors=int(SD.sid.nunique()), share_flood_days_top10pct_sensors=float(vc.head(max(1, len(vc) // 10)).sum() / vc.sum()),
           share_flood_days_light_rain=float(len(light) / len(F)), share_light_rain_flood_days_top5_sensors=float(vl.head(5).sum() / len(light)),
           below_design=float((F.pmax <= 1.75).mean()), below_design_ci95=boot(lambda d: d.pmax <= 1.75, F),
           below_design_1ft=float((F1.pmax <= 1.75).mean()), below_design_1ft_ci95=boot(lambda d: d.pmax <= 1.75, F1))
Dn = SD.groupby("date").agg(active=("sid", "size"), nflood=("y", "sum")); Dn = Dn[Dn.active >= 100]; fr = (Dn.nflood / Dn.active).sort_values(ascending=False)
OUT["fixed_network"] = dict(min_active=100, first=str(Dn.index.min().date()), last=str(Dn.index.max().date()), storm_days=int(len(Dn)), flood_storm_days=int((Dn.nflood > 0).sum()),
                            share_top5=float(fr.head(5).sum() / fr.sum()), share_top10=float(fr.head(10).sum() / fr.sum()))
import nc_events
from scipy.stats import spearmanr
_S = nc_events.sensors().set_index("sensor_id"); _r = SD.groupby("sid").y.mean(); _h = pd.to_numeric(_S.loc[_r.index, "lowest_point_height_delta_inches"], errors="coerce")
_ok = _h.notna(); OUT["spearman_flood_rate_vs_low_point_height"] = dict(rho=float(spearmanr(_r[_ok], _h[_ok])[0]), p=float(spearmanr(_r[_ok], _h[_ok])[1]), n=int(_ok.sum()))
json.dump(OUT, open("nc_storm_stats.json", "w"), indent=1)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in OUT.items()}, indent=0)[:2000])
