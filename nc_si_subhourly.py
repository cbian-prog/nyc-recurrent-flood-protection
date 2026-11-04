"""Data for Supplementary Fig. 1e-g | Is the share of flood days below the design intensity an artefact of hourly rainfall?

The sewer design intensity (1.75 in/h) is compared in the paper with the maximum clock-hour radar (MRMS) rainfall at the sensor.
Street inlets respond to shorter bursts. Three checks with 1-minute ASOS precipitation at LaGuardia, JFK and Newark (IEM archive):
  a  NOAA Atlas 14 at Central Park: the return period of 1.75 in/h depends on the averaging time
  b  share of flood days below 1.75 in/h when the sensor's hourly MRMS maximum is scaled to a 15- or 30-min peak with the ratio
     (max rolling 15/30-min intensity) / (max clock-hour intensity) observed at the nearest gauge on that day;
     direct gauge values at sensors within 5 km of a gauge; and a worst case in which all of the hour's rain falls in 15 min
  c  the gauge ratios on flood days
Output nc_si_subhourly.json, SI_Fig2.
"""
import glob, json, numpy as np, pandas as pd
GAUGES={"LGA":(40.7794,-73.8803),"JFK":(40.6386,-73.7622),"EWR":(40.6825,-74.1694)}
DESIGN=1.75
# ---------------- gauges: daily maxima of rolling 15/30-min and clock-hour intensity (local days)
daily={}
for g in GAUGES:
    f=sorted(glob.glob(f"data/asos1min/{g}_*.csv")); D=pd.concat([pd.read_csv(x,usecols=[2,3],names=["t","p"],header=0) for x in f])
    D["t"]=pd.to_datetime(D.t,utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None); D["p"]=pd.to_numeric(D.p,errors="coerce")
    D=D.dropna().drop_duplicates("t").set_index("t").p.sort_index()
    full=D.reindex(pd.date_range(D.index.min().floor("D"),D.index.max().ceil("D"),freq="1min"))     # missing minutes -> NaN
    r15=full.rolling(15,min_periods=14).sum()*4; r30=full.rolling(30,min_periods=28).sum()*2
    ch=full.resample("1h").sum(min_count=55)
    day=lambda x: x.groupby(x.index.normalize()).max()
    dd=pd.DataFrame({"i15":day(r15),"i30":day(r30),"i60c":day(ch)})
    dd["cover"]=full.notna().groupby(full.index.normalize()).mean()
    daily[g]=dd[dd.cover>=0.9]; print(g,"days with >=90% minutes:",len(daily[g]))
# ---------------- Atlas 14 (Central Park, PDS intensity)
A=pd.read_csv("data/atlas14_centralpark_intensity_pds.csv",skiprows=14,nrows=19,header=None).set_index(0)
ARI=np.array([1,2,5,10,25,50,100,200,500,1000]); A.index=[i.strip().rstrip(":") for i in A.index]
rp_of=lambda dur,i: float(np.exp(np.interp(np.log(i),np.log(A.loc[dur].values.astype(float)),np.log(ARI))))
rp_design={d:rp_of(d,DESIGN) for d in ("15-min","30-min","60-min")}; print("RP of 1.75 in/h:",rp_design)
# ---------------- flood days at sensors
SD=pd.read_parquet("data/paper_a/sensor_days.parquet"); F=SD[SD.y==1].copy()
lat0=np.radians(40.7); dist=lambda la,lo,g: 111.2*np.hypot(la-GAUGES[g][0],(lo-GAUGES[g][1])*np.cos(lat0))
for g in GAUGES: F[f"d_{g}"]=dist(F.lat.values,F.lon.values,g)
F["gauge"]=F[[f"d_{g}" for g in GAUGES]].idxmin(axis=1).str[2:]; F["dist"]=F[[f"d_{g}" for g in GAUGES]].min(axis=1)
def gv(r,col):
    dd=daily[r.gauge]; return dd.at[r.date,col] if r.date in dd.index else np.nan
for c in ("i15","i30","i60c"): F[c]=F.apply(lambda r: gv(r,c),axis=1)
ok=F.i60c>=0.05                                    # a measurable hour at the gauge, so the ratio is defined
F["r15"]=np.where(ok,F.i15/F.i60c,np.nan); F["r30"]=np.where(ok,F.i30/F.i60c,np.nan)
med15,med30=float(np.nanmedian(F.r15)),float(np.nanmedian(F.r30))
F["e15"]=F.pmax*F.r15.fillna(med15); F["e30"]=F.pmax*F.r30.fillna(med30)
out=dict(rp_design_by_duration=rp_design,n_flood_days=int(len(F)),n_with_gauge_ratio=int(ok.sum()),ratio_median=dict(r15=med15,r30=med30),
         ratio_iqr=dict(r15=[float(np.nanquantile(F.r15,.25)),float(np.nanquantile(F.r15,.75))],r30=[float(np.nanquantile(F.r30,.25)),float(np.nanquantile(F.r30,.75))]))
for lab,m in (("any",np.ones(len(F),bool)),("12in",(F.dmax>=12).values)):
    f=F[m]; near=f[(f.dist<=5)&f.i15.notna()]
    out[f"below_design_{lab}"]=dict(n=int(len(f)),hourly_mrms=float((f.pmax<DESIGN).mean()),est_30min=float((f.e30<DESIGN).mean()),est_15min=float((f.e15<DESIGN).mean()),
        worst_case_15min=float((4*f.pmax<DESIGN).mean()),near_gauge_n=int(len(near)),
        near_gauge_60c=float((near.i60c<DESIGN).mean()) if len(near) else None,near_gauge_30=float((near.i30<DESIGN).mean()) if len(near) else None,near_gauge_15=float((near.i15<DESIGN).mean()) if len(near) else None)
    print(lab,{k:(round(v,3) if isinstance(v,float) else v) for k,v in out[f"below_design_{lab}"].items()})
print({k:v for k,v in out.items() if k.startswith(("rp","ratio","n_"))})
json.dump(out,open("nc_si_subhourly.json","w"),indent=1)
# ---------------- figure
from nc_style import *
fig=plt.figure(figsize=(FULL,62*MM))
a=axes_mm(fig,12,12,44,42); durs=["5-min","10-min","15-min","30-min","60-min","2-hr"]; mins=[5,10,15,30,60,120]
for rp,col,ls in ((1,"#C6DBEF","-"),(2,"#6BAED6","-"),(5,"#2171B5","-"),(10,"#08306B","-"),(100,INK,"--")):
    k=list(ARI).index(rp); a.plot(mins,[float(A.loc[d].values[k]) for d in durs],ls,color=col,lw=0.9,label=f"1-in-{rp} yr")
a.axhline(DESIGN,color=PROT,ls=":",lw=0.8); a.text(5.3,DESIGN*0.93,"1.75 in h$^{-1}$",ha="left",va="top",fontsize=5.5,color=PROT)
a.set_xscale("log"); a.set_yscale("log"); a.set_xticks(mins); a.set_xticklabels([str(m) for m in mins]); a.minorticks_off()
a.set_xlabel("Duration (min)"); a.set_ylabel("Rainfall intensity (in h$^{-1}$)"); a.legend(loc="upper right",fontsize=5.5); a.grid(which="major")
panel(a,"a",dx=-10)
b=axes_mm(fig,74,12,64,42); keys=[("hourly_mrms","Hourly radar\n(used)"),("est_30min","Est. 30-min\npeak"),("est_15min","Est. 15-min\npeak"),("worst_case_15min","Hour's rain\nin 15 min")]; x=np.arange(4)
for k_,(lab,col,ttl) in enumerate((("any",RAIN_EXT,"any depth"),("12in",RAIN_MOD,"≥1 ft"))):
    v=[100*out[f"below_design_{lab}"][k] for k,_ in keys]; b.bar(x+(k_-0.5)*0.36,v,0.36,color=col,label=ttl)
    [b.text(x[i]+(k_-0.5)*0.36,v[i]+1,f"{v[i]:.0f}",ha="center",va="bottom",fontsize=5.5) for i in range(4)]
b.set_xticks(x); b.set_xticklabels([t for _,t in keys],fontsize=5.5); b.set_ylim(0,110); b.set_ylabel("Flood days below 1.75 in h$^{-1}$ (%)"); b.legend(loc="upper right"); b.grid(axis="y")
panel(b,"b",dx=-11)
c=axes_mm(fig,152,12,26,42); rr=F[["r15","r30"]].dropna()
c.boxplot([rr.r30,rr.r15],widths=0.5,showfliers=False,medianprops=dict(color=INK),boxprops=dict(lw=0.6),whiskerprops=dict(lw=0.6),capprops=dict(lw=0.6))
c.set_xticks([1,2]); c.set_xticklabels(["30 min","15 min"]); c.set_ylabel("Peak / hourly intensity at gauge"); c.set_ylim(0.8,None); c.grid(axis="y")
panel(c,"c",dx=-11)
save(fig,"unused_SI_subhourly_only",outdirs=("figures/explore",))   # panels e-g of SI Fig. 1 (nc_si_robust.py)
