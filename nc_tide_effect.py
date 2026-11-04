"""Does a high tide raise the chance that a non-tidal sensor floods at the same rain? (compound rain-tide flooding)

Sensor-day table from nc_data.py (rainy days at non-tidal sensors, paper flood definition and cleaning). Logistic regression
of flooding on the day's maximum hourly radar rain bin, a high tide (Battery above MHHW within 2 h of the day's peak rain hour)
and sensor fixed effects, with standard errors clustered by storm day; fitted separately for sensors within 1 km of open water
and farther inland. A second fit adds the citywide maximum hourly rain of the day (in bins), so that a high tide is not
credited with the wetness of the storms that bring it. Output: nc_tide_effect.json
"""
import json, warnings, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
SD = pd.read_parquet("data/paper_a/sensor_days_model.parquet").reset_index(drop=True)
SD["bin"] = np.digitize(SD.pmax, [0.5, 1.0, 1.75], right=True)
SD["city_bin"] = np.digitize(SD.groupby("date").pmax.transform("max"), [0.5, 1.0, 1.75], right=True)
OUT = {"share_high_tide_rainy_days": float(SD.hi.mean())}
for thr, dk in ((0.0, "any"), (4.0, "4in")):
    SD["yy"] = ((SD.y == 1) & (SD.dmax >= thr)).astype(int); OUT[dk] = {}
    for lab, sub in (("near", SD[SD.near == 1]), ("inland", SD[SD.near == 0])):
        sub = sub[sub.groupby("sid").yy.transform("sum") > 0]                # sensors that never flood carry no information here
        g = pd.factorize(sub.date)[0]; res = {}
        for name, f in (("rain_bin", "yy ~ C(bin) + hi + C(sid)"), ("plus_storm_wetness", "yy ~ C(bin) + C(city_bin) + hi + C(sid)")):
            m = smf.glm(f, sub, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": g})
            res[name] = dict(odds_ratio=float(np.exp(m.params.hi)), ci95=np.exp(m.conf_int().loc["hi"]).tolist(), p=float(m.pvalues.hi))
        res.update(n_sensor_days=int(len(sub)), n_sensors=int(sub.sid.nunique()), floods=int(sub.yy.sum()), storm_days=int(sub.date.nunique()))
        OUT[dk][lab] = res
        print(dk, lab, {k: (round(v["odds_ratio"], 2), [round(x, 2) for x in v["ci95"]], round(v["p"], 3)) for k, v in res.items() if isinstance(v, dict)}, "| floods", res["floods"])
json.dump(OUT, open("nc_tide_effect.json", "w"), indent=1)
