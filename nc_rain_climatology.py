"""Is the 2020-2026 radar record wetter in heavy hourly rain than usual? Central Park hourly gauge, rainy days by maximum hourly rain.

Gauge: NOAA ISD 725053-94728 (Central Park), hourly METAR precipitation (AA1, 1-h period), 2005-2025 from the NOAA
global-hourly archive on AWS (earlier years are not on it); January-August 2026 from the Iowa Environmental Mesonet ASOS
archive (same METAR reports, p01i). A rainy day is a New York local day with at least 0.1 in; it is binned by its maximum
hourly rain as in nc_data.py. METAR precipitation is complete only from 2013 (hour coverage below 50% before), so:
  bins <0.5 and 0.5-1 in/h: ratio of gauge rainy days per year, 2013-2019 over 2020-2024 (complete years in the radar period);
  bins 1-1.75 and >=1.75 in/h, which carry most 1-ft flooding and which a few years sample poorly: NOAA Atlas 14 annual rate at a
  point (partial-duration series, Central Park, 60-min depths divided by 1.13 to clock hours) over the radar rate at the
  median basement lot (2020-2026). Atlas 14 is a point climatology; a 1 km radar cell is an areal mean and exceeds a threshold
  less often than a point, so this correction is if anything too small.
The ratio by bin rescales the radar rainy-day counts at basement lots in nc_storm_model.py.
Output: nc_rain_climatology.json
"""
import glob, json, numpy as np, pandas as pd
EDG = [(0, 0.5), (0.5, 1.0), (1.0, 1.75), (1.75, 99)]
rows = []
for f in sorted(glob.glob("data/asos_hourly/isd_72505394728/*.csv")):
    d = pd.read_csv(f, usecols=["DATE", "REPORT_TYPE", "AA1"], low_memory=False).dropna(subset=["AA1"])
    d = d[d.REPORT_TYPE.str.strip() == "FM-15"]; a = d.AA1.str.split(",", expand=True)
    d = d[(a[0] == "01") & (a[1] != "9999")]; a = a.loc[d.index]
    rows.append(pd.DataFrame({"t": pd.to_datetime(d.DATE), "in": a[1].astype(int) / 10 / 25.4}))
G = pd.concat(rows)
I = pd.read_csv("data/asos_hourly/NYC_p01i_1948_2026.csv"); I["t"] = pd.to_datetime(I.valid); I = I[I.t >= "2026-01-01"]
I["in"] = pd.to_numeric(I.p01i.replace("T", "0"), errors="coerce"); G = pd.concat([G, I.dropna(subset=["in"])[["t", "in"]]])
G["t"] = G.t.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
G = G.drop_duplicates("t").set_index("t")["in"]
day = G.groupby(G.index.normalize()).agg(["max", "sum", "size"]); day = day[day["size"] >= 20]
day = day[day["sum"] >= 0.1]
def per_year(lo, hi):
    m = day[(day.index >= lo) & (day.index < hi)]; yrs = (pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25
    cov = len(G[(G.index >= lo) & (G.index < hi)]) / (yrs * 8766)
    return np.array([((m["max"] > a) & (m["max"] <= b)).sum() / yrs for a, b in EDG]), cov
base, cb = per_year("2013-01-01", "2020-01-01"); obs, co = per_year("2020-01-01", "2025-01-01")
ARI = np.array([1, 2, 5, 10, 25, 50, 100]); D60 = np.array([1.07, 1.28, 1.61, 1.89, 2.28, 2.57, 2.87])   # Atlas 14 vol. 10, Central Park, in/h
lam = lambda x: float(1 / np.exp(np.interp(np.log(x * 1.13), np.log(D60), np.log(ARI))))           # annual rate of a clock hour above x
atlas = np.array([np.nan, np.nan, lam(1.0) - lam(1.75), lam(1.75)])
L = np.load("data/paper_a/lot_rain_bins.npz"); radar_lot = np.median(L["nl"], axis=1)
ratio = np.array([base[0] / obs[0], base[1] / obs[1], atlas[2] / radar_lot[2], atlas[3] / radar_lot[3]])
yrs_ge175 = day[(day.index >= "2020-11-01")]["max"].ge(1.75).groupby(day[(day.index >= "2020-11-01")].index.year).sum()
OUT = dict(gauge_baseline="2013-2019", gauge_observed="2020-2024", gauge_rainy_days_per_year_baseline=base.tolist(), gauge_rainy_days_per_year_observed=obs.tolist(),
           hour_coverage_baseline=cb, hour_coverage_observed=co, atlas14_rate_per_year=[None, None, atlas[2], atlas[3]], radar_median_lot_rate_per_year=radar_lot.tolist(),
           ratio_by_bin=ratio.tolist(), central_park_days_ge_design_2013_2019=int(day[(day.index >= "2013-01-01") & (day.index < "2020-01-01")]["max"].ge(1.75).sum()),
           central_park_days_ge_design_2020_11_to_2026_08=int(day[day.index >= "2020-11-01"]["max"].ge(1.75).sum()))
json.dump(OUT, open("nc_rain_climatology.json", "w"), indent=1)
print({k: (np.round(v, 3).tolist() if isinstance(v, list) and None not in v else v) for k, v in OUT.items()})
