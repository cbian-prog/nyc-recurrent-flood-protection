"""Package the cleaned FloodNet flood record used in Paper A, with every filtering step visible, for release with the paper.

Writes release/floodnet_cleaned/:
  floods_clean.csv     usable floods: events that pass every check (the paper uses those with in_paper_record = True)
  excluded_events.csv  events removed, each with an exclusion_reason code (see README)
  events_flagged.csv   every event in the FloodNet event table (NYC Open Data aq7i-eu5q, downloaded 2026-11-03), one row each,
                       with all flags, for checking
  sensors.csv          sensor table (kb2e-tjy3) with the paper's tidal flag and the evidence for each reclassified sensor
  sensor_days_rain.csv rainy sensor-days at non-tidal sensors used by the storm-and-place model (nc_storm_model.py)
  step_counts.json     events remaining after each step
  README.md            sources, steps, column definitions, terms of use
Requires the outputs of nc_tidal_reclass.py, nc_fog_check.py and nc_data.py. The per-minute depth profiles are not copied;
they are in the source table.
"""
import json, os, shutil, numpy as np, pandas as pd
import nc_events
OUTDIR = "floodnet_cleaned"; os.makedirs(OUTDIR, exist_ok=True)
F = pd.DataFrame(json.load(open(nc_events.FULL)))
for c in ("max_depth_inches", "drain_time_mins", "duration_mins", "onset_time_mins", "duration_above_4_inches_mins", "duration_above_12_inches_mins", "duration_above_24_inches_mins"):
    F[c] = pd.to_numeric(F[c], errors="coerce")
F["_k"] = pd.to_datetime(F.flood_start_time).dt.floor("min")
P = pd.DataFrame(json.load(open("data/floodnet/events.json"))); P["_k"] = pd.to_datetime(P.flood_start_time).dt.floor("min")
F["in_paper_record"] = F.set_index(["sensor_id", "_k"]).index.isin(P.set_index(["sensor_id", "_k"]).index)
S = nc_events.sensors(); S["tidal_floodnet"] = S.tidally_influenced_floodnet.eq("Yes"); S["tidal_paper"] = S.tidally_influenced.eq("Yes")
R = json.load(open("data/paper_a/tidal_reclass.json")); S["reclassified_tidal"] = S.sensor_id.isin(R)
F = F.merge(S[["sensor_id", "tidal_floodnet", "tidal_paper", "reclassified_tidal"]], on="sensor_id", how="left")
F["sensor_known"] = F.tidal_paper.notna(); F[["tidal_floodnet", "tidal_paper", "reclassified_tidal"]] = F[["tidal_floodnet", "tidal_paper", "reclassified_tidal"]].fillna(False).astype(bool)
C = pd.read_csv("data/paper_a/event_rain_class.csv"); C["_k"] = pd.to_datetime(C.flood_start_time).dt.floor("min")
F = F.merge(C[["sensor_id", "_k", "cls", "radar_in", "ap_precip", "ap_fog"]], on=["sensor_id", "_k"], how="left")
F["rain_class"] = np.where(F.tidal_paper.fillna(False).astype(bool), "not classed (tidal sensor)", F.cls.fillna("not classed"))
T = pd.read_csv("data/coops/battery_hourly_mhhw_utc.csv"); T["t"] = pd.to_datetime(T.t); T = T.set_index("t").v.astype(float).rolling(5, center=True).max()
F["battery_max_above_mhhw_m"] = T.reindex(F._k.dt.floor("h")).values.round(3)
F["drain_ge_30min"] = F.drain_time_mins >= nc_events.PAPER_MIN_DRAIN
F["counted_as_flood"] = F.in_paper_record & F.sensor_known & F.drain_ge_30min & (F.tidal_paper | F.rain_class.eq("rain"))
F["no_rain_nontidal"] = F.sensor_known & ~F.tidal_paper & ~F.rain_class.eq("rain")
F["exclusion_reason"] = np.select([~F.sensor_known, ~F.drain_ge_30min & F.no_rain_nontidal, ~F.drain_ge_30min, F.no_rain_nontidal],
                                  ["NO_SENSOR", "SHORT+NO_RAIN", "SHORT", "NO_RAIN"], default="")
F["passes_checks"] = F.exclusion_reason.eq("")
# check against the loader the paper uses
L = nc_events.load(); L["_k"] = pd.to_datetime(L.flood_start_time).dt.floor("min")
assert set(zip(L.sensor_id, L._k)) == set(zip(F.sensor_id[F.counted_as_flood], F._k[F.counted_as_flood])), "package and nc_events.load() disagree"
F["start_local"] = F._k.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
cols = ["sensor_id", "sensor_name", "flood_start_time", "flood_end_time", "start_local", "max_depth_inches", "onset_time_mins", "drain_time_mins", "duration_mins",
        "duration_above_4_inches_mins", "duration_above_12_inches_mins", "duration_above_24_inches_mins", "tidal_floodnet", "tidal_paper", "reclassified_tidal",
        "rain_class", "radar_in", "ap_precip", "ap_fog", "battery_max_above_mhhw_m", "in_paper_record", "drain_ge_30min", "passes_checks", "exclusion_reason", "counted_as_flood"]
G = F[cols].rename(columns={"flood_start_time": "flood_start_utc", "flood_end_time": "flood_end_utc", "start_local": "flood_start_new_york", "radar_in": "radar_max_in_per_h_nearby",
                             "ap_precip": "airport_precip_within_1h", "ap_fog": "airport_fog_within_1h"}).sort_values(["flood_start_utc", "sensor_id"])
G.to_csv(f"{OUTDIR}/events_flagged.csv", index=False); G[G.passes_checks].to_csv(f"{OUTDIR}/floods_clean.csv", index=False)
G[~G.passes_checks].to_csv(f"{OUTDIR}/excluded_events.csv", index=False)
assert (G.counted_as_flood == (G.passes_checks & G.in_paper_record)).all()
n = lambda m: int(m.sum())
steps = {"1 events in the source table (to 2026-10-26)": len(F),
         "2 in the paper's record (to 2026-08-11)": n(F.in_paper_record),
         "3 and draining in at least 30 min": n(F.in_paper_record & F.drain_ge_30min),
         "4 of which at tidal sensors (FloodNet flag)": n(F.in_paper_record & F.drain_ge_30min & F.tidal_floodnet),
         "5 of which at tidal sensors (paper flag, +3 reclassified)": n(F.in_paper_record & F.drain_ge_30min & F.tidal_paper),
         "6a at sensors missing from the sensor table (dropped)": n(F.in_paper_record & F.drain_ge_30min & ~F.sensor_known),
         "6b at non-tidal sensors without rain evidence (dropped)": n(F.in_paper_record & F.drain_ge_30min & F.no_rain_nontidal),
         "7 counted as floods": n(F.counted_as_flood),
         "7a tidal": n(F.counted_as_flood & F.tidal_paper), "7b non-tidal (rain)": n(F.counted_as_flood & ~F.tidal_paper),
         "all events: excluded NO_SENSOR": n(F.exclusion_reason.eq("NO_SENSOR")), "all events: excluded SHORT": n(F.exclusion_reason.eq("SHORT")), "all events: excluded NO_RAIN": n(F.exclusion_reason.eq("NO_RAIN")),
         "all events: excluded SHORT+NO_RAIN": n(F.exclusion_reason.eq("SHORT+NO_RAIN")), "all events: usable (floods_clean.csv)": n(F.passes_checks),
         "usable after the paper's cutoff (not analysed)": n(F.passes_checks & ~F.in_paper_record)}
json.dump(steps, open(f"{OUTDIR}/step_counts.json", "w"), indent=1); print(json.dumps(steps, indent=1))
# sensors
S["lat"] = pd.to_numeric(S.latitude, errors="coerce"); S["lon"] = pd.to_numeric(S.longitude, errors="coerce")
ev = G.groupby("sensor_id").agg(events_in_record=("in_paper_record", "sum"), floods_counted=("counted_as_flood", "sum"))
for k in ("dry_floods", "dry_at_high_tide", "to_water_m"): S[f"reclass_{k}"] = S.sensor_id.map({s: v[k] for s, v in R.items()})
Sx = S.merge(ev, left_on="sensor_id", right_index=True, how="left").fillna({"events_in_record": 0, "floods_counted": 0})
Sx[["sensor_id", "sensor_name", "borough", "street_name", "lat", "lon", "date_installed", "date_removed", "lowest_point_height_delta_inches", "tidal_floodnet", "tidal_paper",
    "reclassified_tidal", "reclass_dry_floods", "reclass_dry_at_high_tide", "reclass_to_water_m", "events_in_record", "floods_counted"]].to_csv(f"{OUTDIR}/sensors.csv", index=False)
# sensor-days
SD = pd.read_parquet("data/paper_a/sensor_days_model.parquet").reset_index(drop=True)
SD.rename(columns={"sid": "sensor_id", "date": "date_new_york", "y": "flood", "dmax": "max_flood_depth_in", "pmax": "max_hourly_radar_rain_in", "tide_mhhw": "battery_above_mhhw_m_at_peak_rain",
                   "hi": "high_tide", "to_water": "distance_to_water_m", "near": "within_1km_of_water"})[
    ["sensor_id", "date_new_york", "flood", "max_flood_depth_in", "max_hourly_radar_rain_in", "battery_above_mhhw_m_at_peak_rain", "high_tide", "distance_to_water_m", "within_1km_of_water"]
].to_csv(f"{OUTDIR}/sensor_days_rain.csv", index=False)
print("wrote", OUTDIR, {f: os.path.getsize(f"{OUTDIR}/{f}") for f in os.listdir(OUTDIR)})
