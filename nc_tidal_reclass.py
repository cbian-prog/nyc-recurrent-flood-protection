"""Sensors that FloodNet marks as not tidally influenced but that behave as tidal.

Rule (fixed before looking at the paper's results): a sensor is reclassified as tidal when it lies within 1 km of open water
(55 m DEM water mask), has at least 3 floods (paper definition, nc_events) on days with less than 0.1 in of radar rain at the
sensor, and at least two-thirds of those dry-day floods start within 2 h of a Battery water level above MHHW.
Output: data/paper_a/tidal_reclass.json (sensor ids and the evidence for each).
"""
import glob, json, numpy as np, pandas as pd, rasterio
from scipy import ndimage
import nc_events
S = pd.DataFrame(json.load(open("data/floodnet/sensors.json"))); S["lat"] = pd.to_numeric(S.latitude); S["lon"] = pd.to_numeric(S.longitude)
S = S.dropna(subset=["lat", "lon"]); N = S[S.tidally_influenced != "Yes"].copy()
with rasterio.open("data/nyc_dem_55m.tif") as r_: D, Dt, Dn = r_.read(1), r_.transform, r_.nodata
dist = ndimage.distance_transform_edt(~((D == Dn) | (D < 0.5))) * 55.0
N["to_water_m"] = [float(dist[int((la - Dt.f) / Dt.e), int((lo - Dt.c) / Dt.a)]) for la, lo in zip(N.lat, N.lon)]
# daily radar rain at each sensor (local days), as in nc_data.py
files = sorted(glob.glob("data/mrms/*.npz")); z0 = np.load(files[0]); a, _, c0, _, e, f0 = z0["transform"]; sh = z0["h00"].shape
N["r"] = ((N.lat - f0) / e).astype(int); N["c"] = ((N.lon - c0) / a).astype(int); N = N[(N.r >= 0) & (N.r < sh[0]) & (N.c >= 0) & (N.c < sh[1])]
hrs, vals = [], []
for f in files:
    z = np.load(f); day = pd.Timestamp(f.split("/")[-1][:8])
    for k in z.files:
        if k.startswith("h"): hrs.append(day + pd.Timedelta(hours=int(k[1:]))); vals.append(np.maximum(z[k][N.r.values, N.c.values], 0) / 25.4)
H = pd.DataFrame(np.array(vals), index=pd.DatetimeIndex(hrs), columns=N.sensor_id.values)
H.index = H.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None); ptot = H.groupby(H.index.normalize()).sum()
T = pd.read_csv("data/coops/battery_hourly_mhhw_utc.csv"); T["t"] = pd.to_datetime(T.t); T = T.set_index("t").v.astype(float).rolling(5, center=True).max()
E = nc_events.load(rain_evidence=False); E = E[E.sensor_id.isin(N.sensor_id)].copy(); E["t"] = pd.to_datetime(E.flood_start_time)   # UTC
E["day"] = E.t.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
E["rain_day_in"] = [ptot.at[d, s] if d in ptot.index else 0.0 for d, s in zip(E.day, E.sensor_id)]   # no radar file: no rain at Central Park, no event-day pull
E["tide_hi"] = T.reindex(E.t.dt.floor("h")).values > 0
dry = E[E.rain_day_in < 0.1].groupby("sensor_id").agg(dry_floods=("t", "size"), dry_at_high_tide=("tide_hi", "sum"))
N = N.merge(dry, left_on="sensor_id", right_index=True, how="left").fillna({"dry_floods": 0, "dry_at_high_tide": 0})
R = N[(N.to_water_m <= 1000) & (N.dry_floods >= 3) & (N.dry_at_high_tide >= 2 / 3 * N.dry_floods)]
out = {r.sensor_id: dict(name=r.sensor_name, to_water_m=round(r.to_water_m), dry_floods=int(r.dry_floods), dry_at_high_tide=int(r.dry_at_high_tide)) for r in R.itertuples()}
json.dump(out, open("data/paper_a/tidal_reclass.json", "w"), indent=1); print(len(out), "sensors reclassified as tidal:"); [print(" ", v) for v in out.values()]
