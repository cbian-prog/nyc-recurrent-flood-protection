"""Data for Supplementary Fig. 1h | Do the sensor flood rates transfer between boroughs?

Ponding days at basement lots (Fig. 1c) apply the network's flood rate per rainy day, by maximum hourly radar rain,
to every lot in the rainfall footprint. Extended Data Fig. 6b shows that which sensor floods cannot be predicted across
boroughs; this tests the weaker property the ponding estimate needs: that the borough-average rate transfers.
  a  leave-one-borough-out: rates from the other four boroughs predict the flood days at the held-out borough's sensors;
     ratio predicted / observed, at any depth and at >= 1 ft
  b  ponding days per year at the 20,237 rain-footprint basement lots: pooled rates (used), leave-one-borough-out rates
     (each lot gets the rates of the other boroughs), and the range across the five borough-specific rate curves
Also writes the sensor-day table to data/paper_a/sensor_days.parquet for the sub-hourly test (nc_si_subhourly.py).
"""
import numpy as np, pandas as pd, json
exec(open("nc_data.py").read().split("# ---------------------------------------------------------------- 4.")[0])
boro=dict(zip(Sn.sensor_id,Sn.borough)); SD["boro"]=SD.sid.map(boro)
SD.merge(Sn[["sensor_id","lat","lon"]].rename(columns={"sensor_id":"sid"}),on="sid").to_parquet("data/paper_a/sensor_days.parquet")
edges=[0,0.5,1.0,1.75,99]; SD["bin"]=np.digitize(SD.pmax,edges[1:-1],right=True)
def rates(df,thr): return np.array([((df.y==1)&(df.dmax>=thr))[df.bin==k].mean() if (df.bin==k).any() else 0 for k in range(4)])
BOROS=["Queens","Brooklyn","Bronx","Manhattan","Staten Island"]; out={"lobo":{},"ponding":{}}
for dk,thr in (("any",0.0),("12in",12.0)):
    out["lobo"][dk]={}
    for b in BOROS:
        tr_,te=SD[SD.boro!=b],SD[SD.boro==b]; r=rates(tr_,thr)
        pred=r[te.bin.values].sum(); obs=float(((te.y==1)&(te.dmax>=thr)).sum())
        out["lobo"][dk][b]=dict(pred=float(pred),obs=obs,ratio=float(pred/obs) if obs else None,n_sensors=int(te.sid.nunique()))
    P_=sum(v["pred"] for v in out["lobo"][dk].values()); O_=sum(v["obs"] for v in out["lobo"][dk].values()); out["lobo"][dk]["all"]=dict(pred=P_,obs=O_,ratio=P_/O_)
    print(dk,{b:(round(v["ratio"],2) if v["ratio"] else None) for b,v in out["lobo"][dk].items()})
# ponding at rain-footprint basement lots
lot_boro=pd.Series(P.bbl.astype(str).str[0].map({"1":"Manhattan","2":"Bronx","3":"Brooklyn","4":"Queens","5":"Staten Island"}).values)
sel=np.where(inrain.astype(bool)&okp)[0]; lb=lot_boro.values[sel]
cnt=np.stack([(rainy&(daymax>lo)&(daymax<=hi)).sum(0)/span_yr for lo,hi in ((0,0.5),(0.5,1.0),(1.0,1.75),(1.75,99))])  # rainy days/yr by bin per MRMS cell
C=cnt[:,pm[sel],pc[sel]]                                                                                                 # bins x lots
for dk,thr in (("any",0.0),("12in",12.0)):
    pooled=rates(SD,thr)@C
    lobo=np.zeros(len(sel))
    for b in BOROS:
        m=lb==b
        if m.any(): lobo[m]=rates(SD[SD.boro!=b],thr)@C[:,m]
    per_b={b:float(np.median(rates(SD[SD.boro==b],thr)@C)) for b in BOROS}
    out["ponding"][dk]=dict(pooled_median=float(np.median(pooled)),lobo_median=float(np.median(lobo)),lobo_q25=float(np.quantile(lobo,.25)),lobo_q75=float(np.quantile(lobo,.75)),
                            borough_curve_medians=per_b,borough_curve_range=[min(per_b.values()),max(per_b.values())])
    print("ponding",dk,{k:v for k,v in out["ponding"][dk].items() if k!="borough_curve_medians"},per_b)
json.dump(out,open("nc_si_transfer.json","w"),indent=1)
# ---------------- figure
from nc_style import *
fig=plt.figure(figsize=(FULL,60*MM))
a=axes_mm(fig,13,12,80,40); x=np.arange(6); lab=BOROS+["all"]
for k_,(dk,col,ttl) in enumerate((("any",RAIN_EXT,"any depth"),("12in",RAIN_MOD,"≥1 ft"))):
    v=[out["lobo"][dk][b]["ratio"] or 0 for b in lab]
    a.bar(x+(k_-0.5)*0.36,v,0.36,color=col,label=ttl)
    for i in range(6):
        if out["lobo"][dk][lab[i]]["ratio"] is None: a.text(x[i]+(k_-0.5)*0.36,0.05,"no 1-ft\ndays",ha="center",va="bottom",fontsize=5,rotation=90)
        else: a.text(x[i]+(k_-0.5)*0.36,v[i]+0.03,f"{v[i]:.2f}",ha="center",va="bottom",fontsize=5)
a.axhline(1,color=INK,ls=":",lw=0.6); a.set_xticks(x); a.set_xticklabels(["Queens","Brooklyn","Bronx","Manhattan","Staten Is.","All"]); a.set_ylabel("Predicted / observed flood days\n(held-out borough)")
a.set_ylim(0,max(2.0,max(out["lobo"][d][b]["ratio"] or 0 for d in ("any","12in") for b in lab)*1.15)); a.legend(loc="upper right"); a.grid(axis="y")
panel(a,"a",dx=-11)
b=axes_mm(fig,118,12,58,40); x=np.arange(2)
for i,dk in enumerate(("any","12in")):
    o=out["ponding"][dk]; lo,hi=o["borough_curve_range"]
    b.plot([i-0.15],[o["pooled_median"]],"o",color=RAIN_EXT,ms=3.5,label="Pooled rates (used)" if i==0 else None)
    b.plot([i+0.15],[o["lobo_median"]],"o",mfc="white",mec=RAIN_EXT,mew=0.8,ms=3.5,label="Leave-one-borough-out" if i==0 else None)
    b.plot([i+0.15,i+0.15],[o["lobo_q25"],o["lobo_q75"]],color=RAIN_EXT,lw=0.8)
    b.plot([i,i],[lo,hi],color="#9A9A9A",lw=2.5,alpha=0.6,solid_capstyle="butt",zorder=0,label="Range across borough curves" if i==0 else None)
b.set_yscale("log"); b.set_xticks(x); b.set_xticklabels(["Any depth","≥1 ft"]); b.set_xlim(-0.6,1.6); b.set_ylabel("Ponding days per year at\nrain-footprint basement lots"); b.legend(loc="upper right"); b.grid(axis="y",which="major")
panel(b,"b",dx=-13)
save(fig,"unused_SI_transfer_only",outdirs=("figures/explore",))   # the transfer test is panel h of SI Fig. 1 (nc_si_robust.py)
