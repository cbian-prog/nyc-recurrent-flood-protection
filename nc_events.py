"""FloodNet events as counted in the paper.

FloodNet records an event for any series of one-minute depth readings above 10 mm. The paper counts an event as a flood
when its water takes at least 30 min to drain from maximum depth to dry (Supplementary Note: short events are not noise,
but 311 complaints reach their plateau at 30 min and the 1-ft results are stable up to it). Drain times come from the
full NYC Open Data table (aq7i-eu5q, downloaded 2026-11-03). MIN_DRAIN_MIN=0 in the environment keeps every event.
"""
import os, json, pandas as pd

RECLASS = os.environ.get("TIDAL_RECLASS", "1") != "0"
RAIN_EVIDENCE = os.environ.get("RAIN_EVIDENCE", "1") != "0"   # RAIN_EVIDENCE=0 keeps non-tidal events that had no rain (nc_fog_check.py)   # TIDAL_RECLASS=0 keeps FloodNet's own tidal flags

PAPER_MIN_DRAIN = 30.0
MIN_DRAIN = float(os.environ.get("MIN_DRAIN_MIN", PAPER_MIN_DRAIN))
FULL = "data/floodnet/events_full_20261103/events_full.json"

def load(path="data/floodnet/events.json", rain_evidence=None):
    """paper floods. Cleaning (on unless rain_evidence=False or RAIN_EVIDENCE=0): events at sensors missing from the sensor
    table are dropped, and at non-tidal sensors events with no rain at the sensor (radar) or at the three airports are dropped
    (nc_fog_check.py: fog or dry weather, often a high tide). nc_tidal_reclass.py reads the uncleaned events."""
    rain_evidence = RAIN_EVIDENCE if rain_evidence is None else rain_evidence
    E = pd.DataFrame(json.load(open(path)))
    if MIN_DRAIN > 0:
        F = pd.DataFrame(json.load(open(FULL)))[["sensor_id", "flood_start_time", "drain_time_mins"]]
        F["_k"] = pd.to_datetime(F.flood_start_time).dt.floor("min"); E["_k"] = pd.to_datetime(E.flood_start_time).dt.floor("min")
        E = E.merge(F[["sensor_id", "_k", "drain_time_mins"]], on=["sensor_id", "_k"], how="left").drop(columns="_k")
        E = E[pd.to_numeric(E.drain_time_mins, errors="coerce") >= MIN_DRAIN]
    if rain_evidence:
        E = E[E.sensor_id.isin(sensors().sensor_id)]   # sensors missing from the sensor table cannot be located or classed
    if rain_evidence and os.path.exists("data/paper_a/event_rain_class.csv"):
        C = pd.read_csv("data/paper_a/event_rain_class.csv"); C["_k"] = pd.to_datetime(C.flood_start_time).dt.floor("min")
        E["_k"] = pd.to_datetime(E.flood_start_time).dt.floor("min")
        E = E.merge(C[["sensor_id", "_k", "cls"]], on=["sensor_id", "_k"], how="left").drop(columns="_k")
        E = E[~E.cls.isin(["fog, no rain", "dry, no fog"])].drop(columns="cls")   # tidal sensors are not classed, so they are kept
    return E.reset_index(drop=True)

def suffix():
    """file-name suffix for runs that do not use the paper's definition"""
    return "" if MIN_DRAIN == PAPER_MIN_DRAIN else f"_drain{MIN_DRAIN:.0f}"

def sensors(path="data/floodnet/sensors.json"):
    """FloodNet sensor table with the paper's tidal flag: FloodNet's own flag plus the sensors that nc_tidal_reclass.py
    finds behaving as tidal (near open water, dry-day floods at high tide). The FloodNet flag is kept as tidally_influenced_floodnet."""
    S = pd.DataFrame(json.load(open(path))); S["tidally_influenced_floodnet"] = S.tidally_influenced
    if RECLASS and os.path.exists("data/paper_a/tidal_reclass.json"):
        S.loc[S.sensor_id.isin(json.load(open("data/paper_a/tidal_reclass.json"))), "tidally_influenced"] = "Yes"
    return S
