"""Per-save-point joint-probability (JPM) still-water curves at four USACE CHS save points (The Battery 7672; Jamaica Bay
11982, 12049; Rockaway 14200), built with identical settings so that only differences between points are used
(Methods, 'Spatial check of the single curve'; Extended Data Fig. 2c,d).
Tropical: rate = storm relative probability x (high + low intensity storm recurrence rate) x common scaling, convolved with
the nodal bias/uncertainty (normal, B, U); extratropical: 100 historical storms over a 150-year record, same uncertainty.
Inputs in data/naccs/chs (CHS NACCS SRR table, ADCIRC peaks per save point, CHS dependency files). Output jpm_points.json.
Run from data/naccs/chs:  python ../../../naccs_jpm_points.py   (set CHECK=1 to write jpm_points_check.json instead)"""
import os
import pandas as pd, numpy as np, json, scipy.io as sio
from scipy.stats import norm
from scipy.spatial import cKDTree
SPS=[7672,11982,12049,14200]
def load_peaks(f):
    pk=pd.read_csv(f, skiprows=[1,2]); pk.columns=[c.strip() for c in pk.columns]; pk=pk.dropna(subset=["Storm ID"])
    pk["sid"]=pd.to_numeric(pk["Storm ID"],errors="coerce").astype(int); return pk.set_index("sid")["Water Elevation"].where(lambda s:s>-999)
rows={}
for ch in pd.read_csv("NACCS_TS_Sim0_Post0_ST_Stat_SRR.csv", chunksize=3000, low_memory=False):
    m=ch[ch["Save Point ID"].isin(SPS)]
    for _,r in m.iterrows(): rows[int(r["Save Point ID"])]=r
nd=sio.loadmat("CHS_Dependencies/Grid_Files/CHS-NA/CHS-NA_nodeID_v4.mat",squeeze_me=True)["nodeID"]
comb=sio.loadmat("CHS_Dependencies/Bias_and_Uncertainty/CHS-NA/CHS-NA_Comb_BiasUncertainty_perVG_rta.mat",squeeze_me=True,struct_as_record=False)["Comb"]
ids=comb.ADCIRC_ID; inset=np.isin(nd[:,1],ids); sub=nd[inset]; kd=cKDTree(np.c_[sub[:,3],sub[:,2]])
xs=np.linspace(0,9,1801); RPS=[10,45,100,143,400,1000]
res={}
for sp in SPS:
    r=rows[sp]; lo_,la_=float(r["Save Point Latitude"]),float(r["Save Point Longitude"])  # columns are swapped in the csv header
    d,i=kd.query([lo_,la_]); j=np.searchsorted(ids,int(sub[i,1])); B,U=comb.B_a[j],comb.U_a[j]
    hi,lo=float(r["High Intensity Storm Rate"]),float(r["Low Intensity Storm Rate"]); srp=np.array([float(r[f"Synthetic_{k:04d} SRP"]) for k in range(1,1051)])
    tc=load_peaks(f"peaks/CHS-NA_TS_SimB1RT_Post0_SP{sp:05d}_ADCIRC01_Peaks.csv" if sp==7672 else f"peaks/CHS-NA_TS_SimB1RT_Post0_SP{sp}_ADCIRC01_Peaks.csv").reindex(range(1,1051))
    xc=load_peaks(f"peaks/CHS-NA_XH_SimB1HT_Post0_SP{sp:05d}_ADCIRC01_Peaks.csv" if sp==7672 else f"peaks/CHS-NA_XH_SimB1HT_Post0_SP{sp}_ADCIRC01_Peaks.csv").dropna().values
    ok=tc.notna().values; W=tc.values; lam=srp*(hi+lo)*2000
    a_tc=np.array([(lam[ok]*norm.sf((x-(W[ok]+B))/U)).sum() for x in xs]); a_xc=np.array([norm.sf((x-(xc+B))/U).sum()/150 for x in xs])
    a=a_tc+a_xc; f=lambda aep: float(np.interp(-aep,-a,xs))
    res[sp]={rp:f(1/rp) for rp in RPS}
    print(f"SP{sp} ({lo_:.4f},{la_:.4f}) B {B:+.3f} U {U:.3f} | TC n={int(ok.sum())} max {np.nanmax(W):.2f} | XC max {xc.max():.2f} | JPM WL: "+" ".join(f"RP{rp}:{res[sp][rp]:.2f}" for rp in RPS))
print("\nOffset vs Battery (m):")
for sp in SPS[1:]: print(f"  SP{sp}: "+" ".join(f"RP{rp}:{res[sp][rp]-res[7672][rp]:+.2f}" for rp in RPS))
json.dump({str(k):v for k,v in res.items()}, open("jpm_points_check.json" if os.environ.get("CHECK") else "jpm_points.json","w"), indent=1)
