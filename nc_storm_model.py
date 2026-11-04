"""Storm-and-place model of street flooding at FloodNet's non-tidal sensors, and the flood frequency it implies at basement lots.

Each rainy sensor-day is one observation. Flooding (paper definition, nc_events) at depth class k is modelled as
    logit P = b[rain bin] + g1 * high_tide + g2 * high_tide * near_water + u[storm day] + v[sensor]
with rain bin the day's maximum hourly radar rain at the sensor (<0.5, 0.5-1, 1-1.75, >=1.75 in/h), high_tide the Battery
water level above MHHW within 2 h of the day's peak rain hour, near_water a sensor within 1 km of open water, and normal random
effects for the storm day (shared by every sensor that day) and for the sensor (the place). Fitted by variational Bayes
(statsmodels BinomialBayesMixedGLM). This two-effect model partitions the variation between storms and places. Storm effects
are not independent of the rain bin (the heavy-rain days come from the few long, wet storms), so averaging a fitted storm
effect out at a given bin understates flooding on heavy-rain days; frequencies are therefore predicted from the same model
without the storm effect, whose bin coefficients average over the storms that produced each bin, and the dependence between
sensor-days of one storm is carried by the storm-day bootstrap. Inputs from nc_data.py: data/paper_a/sensor_days_model.parquet, lot_rain_bins.npz.

Basement lots in the extreme rainfall footprint are given the flood frequency of a typical place (sensor effect 0), averaged
over storm effects, with the share of rainy days at high tide taken from the sensor-days in each rain bin; the spread across
places draws a sensor effect for each lot. Rainy days per year by bin come from the radar record (2020-2026) and, as the main
estimate, are rescaled to long-run rates (Central Park gauge 2013-2019 for the two lighter bins, NOAA Atlas 14 for the two heavier; ratio by bin, nc_rain_climatology).
Uncertainty: storm-day block bootstrap, refitting the model on resampled storm days (B replicates, parallel).
Output: nc_storm_model.json.
"""
import json, os, sys, warnings, numpy as np, pandas as pd
from multiprocessing import get_context
warnings.filterwarnings("ignore")
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

B = int(os.environ.get("BOOT", "100")); NPROC = int(os.environ.get("NPROC", "8"))
EDG = [(0, 0.5), (0.5, 1.0), (1.0, 1.75), (1.75, 99)]
DEPTHS = {"any": 0.0, "4in": 4.0, "12in": 12.0}
SD = pd.read_parquet("data/paper_a/sensor_days_model.parquet").reset_index(drop=True)
SD["bin"] = np.digitize(SD.pmax, [0.5, 1.0, 1.75], right=True); SD["hin"] = SD.hi * SD.near; SD["d"] = SD.date.astype(str)
L = np.load("data/paper_a/lot_rain_bins.npz"); NL = L["nl"]; LOT_NEAR = (L["to_water"] < 1000).astype(float)
PI = np.array([float(SD.hi[SD.bin == k].mean()) for k in range(4)])                 # share of rainy days at high tide, by bin
CLIM = json.load(open("nc_rain_climatology.json"))["ratio_by_bin"] if os.path.exists("nc_rain_climatology.json") else None
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40); GH_W = GH_W / GH_W.sum()       # standard-normal quadrature
expit = lambda x: 1 / (1 + np.exp(-x))

def fit(df, thr, storm=False):
    y = ((df.y == 1) & (df.dmax >= thr)).astype(int)
    vc = {"storm": "0+C(d)", "site": "0+C(sid)"} if storm else {"site": "0+C(sid)"}
    m = BinomialBayesMixedGLM.from_formula("yy ~ C(bin) + hi + hin", vc, df.assign(yy=y.values)).fit_vb()
    fe = dict(zip(m.model.exog_names, m.fe_mean)); sd = dict(zip(m.model.vcp_names, np.exp(m.vcp_mean)))
    b = np.array([fe["Intercept"]] + [fe["Intercept"] + fe[f"C(bin)[T.{k}]"] for k in (1, 2, 3)])
    return dict(b=b, g1=fe["hi"], g2=fe["hin"], s_storm=sd.get("storm", 0.0), s_site=sd["site"], n_events=int(y.sum()), n_storms=int(df.d[y == 1].nunique()))

def p_typical(par, near, hi, v=0.0):
    """P(flood) on a rainy day in each bin at a place with sensor effect v (0 = typical place)"""
    return expit(par["b"] + par["g1"] * hi + par["g2"] * hi * near + v)

def p_mean_place(par, near, hi):
    return (expit(par["b"][:, None] + par["g1"] * hi + par["g2"] * hi * near + par["s_site"] * GH_X[None, :]) * GH_W[None, :]).sum(1)

def lot_freq(par, nl, near, site=None, mean_place=False):
    out = np.zeros(nl.shape[1])
    for nr in (0.0, 1.0):
        m = near == nr
        if not m.any(): continue
        for hi, w in ((0, 1 - PI), (1, PI)):
            if mean_place: out[m] += (w * p_mean_place(par, nr, hi)) @ nl[:, m]
            elif site is None: out[m] += (w * p_typical(par, nr, hi)) @ nl[:, m]
            else: out[m] += (w[:, None] * expit(par["b"][:, None] + par["g1"] * hi + par["g2"] * hi * nr + site[None, m]) * nl[:, m]).sum(0)
    return out

def summarise(par, thr_key):
    nl_clim = NL * np.array(CLIM)[:, None] if CLIM else NL
    f, f_obs = lot_freq(par, nl_clim, LOT_NEAR), lot_freq(par, NL, LOT_NEAR)
    fm, fm_obs = lot_freq(par, nl_clim, LOT_NEAR, mean_place=True), lot_freq(par, NL, LOT_NEAR, mean_place=True)
    site = np.random.default_rng(0).normal(0, par["s_site"], NL.shape[1]); fs = lot_freq(par, nl_clim, LOT_NEAR, site); fs_obs = lot_freq(par, NL, LOT_NEAR, site)
    return dict(lot_typical=float(np.median(f)), lot_typical_observed_period=float(np.median(f_obs)),
                lot_mean_place=float(np.median(fm)), lot_mean_place_observed_period=float(np.median(fm_obs)),
                lots_q10=float(np.quantile(fs, .1)), lots_q50=float(np.median(fs)), lots_q90=float(np.quantile(fs, .9)),
                lots_q90_observed_period=float(np.quantile(fs_obs, .9)), share_lots_ge_1_per_10yr=float((fs >= 0.1).mean()),
                tide_or_near=float(np.exp(par["g1"] + par["g2"])), tide_or_inland=float(np.exp(par["g1"])), s_site=float(par["s_site"]),
                p_typical_low_tide=p_typical(par, 0, 0).tolist())

def one_boot(seed):
    rng = np.random.default_rng(seed); days = SD.d.unique(); pick = rng.choice(days, len(days))
    parts = [SD[SD.d == dd].assign(d=f"{dd}#{i}") for i, dd in enumerate(pick)]
    df = pd.concat(parts, ignore_index=True); res = {}
    for k, thr in DEPTHS.items():
        try: res[k] = summarise(fit(df, thr), k)
        except Exception as ex: res[k] = None
    return res

if __name__ == "__main__":
    OUT = {"n_sensor_days": int(len(SD)), "n_sensors": int(SD.sid.nunique()), "n_storm_days": int(SD.d.nunique()), "share_high_tide_by_bin": PI.tolist(),
           "climatology_ratio_by_bin": CLIM, "boot_replicates": B}
    for k, thr in DEPTHS.items():
        par = fit(SD, thr); OUT[k] = dict(point=summarise(par, k), n_events=par["n_events"], n_storms=par["n_storms"], b=par["b"].tolist(), g1=par["g1"], g2=par["g2"])
        p2 = fit(SD, thr, storm=True); tot = p2["s_storm"] ** 2 + p2["s_site"] ** 2 + np.pi ** 2 / 3
        OUT[k]["variance_partition"] = dict(s_storm=p2["s_storm"], s_site=p2["s_site"], share_storm=p2["s_storm"] ** 2 / tot, share_site=p2["s_site"] ** 2 / tot,
                                             tide_or_near=float(np.exp(p2["g1"] + p2["g2"])), tide_or_inland=float(np.exp(p2["g1"])))   # tide effect within storms
        print(k, {a: (round(v, 4) if isinstance(v, float) else v) for a, v in OUT[k]["point"].items() if a != "p_typical_low_tide"}, "| events", par["n_events"], "storms", par["n_storms"], flush=True)
    if B > 0:
        with get_context("fork").Pool(NPROC) as pool: boots = pool.map(one_boot, range(1, B + 1))
        for k in DEPTHS:
            ok = [b_[k] for b_ in boots if b_ and b_.get(k)]
            OUT[k]["boot_n"] = len(ok)
            OUT[k]["ci95"] = {a: [float(np.percentile([o[a] for o in ok], 2.5)), float(np.percentile([o[a] for o in ok], 97.5))]
                              for a in ok[0] if a != "p_typical_low_tide"}
            print(k, "storm-day bootstrap 95%:", {a: [round(x, 4) for x in v] for a, v in OUT[k]["ci95"].items()}, flush=True)
    N = json.load(open("nc_numbers.json")); s12 = N["surge_floods_per_year_s100_lots_12in"]["median"]; sany = N["surge_floods_per_year_s100_lots_any"]["median"]
    OUT["rain_to_surge_12in"] = dict(point=OUT["12in"]["point"]["lots_q50"] / s12, ci95=[x / s12 for x in OUT["12in"].get("ci95", {}).get("lots_q50", [np.nan, np.nan])])
    OUT["rain_to_surge_any"] = dict(point=OUT["any"]["point"]["lots_q50"] / sany, ci95=[x / sany for x in OUT["any"].get("ci95", {}).get("lots_q50", [np.nan, np.nan])])
    print("rain:surge 1 ft", OUT["rain_to_surge_12in"], "| any", OUT["rain_to_surge_any"])
    json.dump(OUT, open("nc_storm_model.json", "w"), indent=1)
