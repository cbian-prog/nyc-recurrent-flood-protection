"""Data for Supplementary Fig. 3b — the hazard basis (storm category against water level).
a  NACCS still-water hazard curve at The Battery (digitised, ±0.1 m), tropical and extratropical components, Sandy
b  category basis (SLOSH MOM x STORM frequencies) versus water-level basis: event loss at the category return periods, same NSI/USACE loss chain
c  model-version history of the surge AAI (eight working versions on PLUTO exposure, then the publication basis)
d  contribution to the publication-basis AAI by return-period band
"""
import json, warnings, numpy as np, pandas as pd, rasterio
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("paper_a_core.py").read().split("# ---------------- Sandy validation")[0])
matplotlib.rcParams.update({"font.size":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":10,"axes.titlesize":12})
C_WL,C_CAT,C_TC,C_XC="#1B4965","#D1542F","#2E7D5B","#7A5EA8"
# ---- a: curves
cv=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in cv)); WL=np.array([cv[str(t)] for t in T])
cc=json.load(open("data/naccs/battery_naccs_tc_xc_curves.json")); Ttc=np.array(sorted(int(k) for k in cc["TC"])); WLtc=np.array([cc["TC"][str(t)] for t in Ttc]); Txc=np.array(sorted(int(k) for k in cc["XC"])); WLxc=np.array([cc["XC"][str(t)] for t in Txc])
SANDY_MSL=SANDY_WL+NAVD_MINUS_MSL; rp_sandy=float(np.exp(np.interp(SANDY_MSL,WL,np.log(T))))
# ---- b: category basis vs water-level basis on the same loss chain
aeps=np.logspace(0,-4,120); wl0=np.interp(np.log(1/aeps),np.log(T),WL)-NAVD_MINUS_MSL
Lus=np.array([loss_usace(depths[i]).sum() for i in range(len(levels))]); Lwl=np.interp(wl0,levels,Lus,left=0); rp=1/aeps
FREQ={1:0.02220,2:0.00700,3:0.00250,4:0.00100}; cat_loss={}
for c,f in FREQ.items():
    with rasterio.open(f"data/slosh/nyc_slosh_cat{c}.tif") as r: a,t,shp=r.read(1),r.transform,r.shape
    cc_=((s.x.values-t.c)/t.a).astype(int); rr=((s.y.values-t.f)/t.e).astype(int); ok=(rr>=0)&(rr<shp[0])&(cc_>=0)&(cc_<shp[1])
    cls=np.zeros(len(s),"uint8"); cls[ok]=a[rr[ok],cc_[ok]]; d=np.where((cls>=1)&(cls<=21),(cls.astype("float32")-0.5)*FT,0.0)
    cat_loss[c]=loss_usace(d).sum()/1e9
rps_cat=np.array([1/FREQ[c] for c in FREQ]); Lcat=np.array([cat_loss[c] for c in FREQ]); Lwl_at=np.array([np.interp(np.log(r),np.log(rp),Lwl) for r in rps_cat])/1e9
aai_cat=sum(FREQ[c]*cat_loss[c] for c in FREQ)*1e3; aai_wl=-np.trapz(Lwl,aeps)/1e6
print("category basis event loss bn:",{c:round(v,1) for c,v in cat_loss.items()},"| water-level basis at same RP:",np.round(Lwl_at,1),"| ratio",np.round(Lcat/Lwl_at,1))
print(f"AAI category basis (sum f x L) {aai_cat:.0f} M | water-level basis {aai_wl:.0f} M | ratio {aai_cat/aai_wl:.1f}")
# ---- c: version history
vals=[321,3519,6230,1700,1531,1328,1411,754,int(round(aai_wl))]
labs=["bathtub\n(initial)","+ datum\nfix","+ SLOSH\nlevels","+ Sandy\ncalibration","+ STORM\nfrequencies","+ PLUTO\nexposure","+ storey\nvulnerability","+ NACCS\nwater levels","NSI × USACE\n(publication)"]
# ---- d: contribution by band on the publication basis
bands=[(1,10),(10,50),(50,100),(100,500),(500,1000),(1000,10000)]; contrib=[]
for lo,hi in bands:
    m=(rp>=lo)&(rp<=hi); contrib.append(-np.trapz(Lwl[m],aeps[m])/1e6)
contrib=np.array(contrib); print("contribution by band (M):",np.round(contrib),"| share <50 yr:",round(100*contrib[:2].sum()/contrib.sum()))
json.dump(dict(rp_sandy=rp_sandy,cat_loss_bn=cat_loss,wl_loss_at_cat_rp_bn=Lwl_at.tolist(),aai_cat_m=aai_cat,aai_wl_m=aai_wl,versions=dict(zip([l.replace("\n"," ") for l in labs],vals)),contrib_by_band_m=dict(zip([f"{lo}-{hi}" for lo,hi in bands],contrib.tolist()))),open("figED3_hazard_basis.json","w"),indent=1)
# ---- figure
fig,ax=plt.subplots(2,2,figsize=(7.5,7.4),gridspec_kw=dict(hspace=.5,wspace=.32,left=.09,right=.985,top=.97,bottom=.1))
a=ax[0,0]; a.fill_between(T,WL-.1,WL+.1,color=C_WL,alpha=.18,lw=0); a.semilogx(T,WL,"-o",color=C_WL,ms=3.5,lw=1.8,label="Combined (used)")
a.semilogx(Ttc,WLtc,"--",color=C_TC,lw=1.4,label="Tropical"); a.semilogx(Txc,WLxc,":",color=C_XC,lw=1.6,label="Extratropical")
a.plot([rp_sandy],[SANDY_MSL],"o",color="#B33A30",ms=6,label=f"Sandy ({rp_sandy:.0f} yr)"); a.set_xlim(1,1e4); a.set_ylim(0,7); a.set_xlabel("Return period (yr)"); a.set_ylabel("Still-water level (m MSL)"); a.grid(alpha=.3); a.legend(loc="upper left",frameon=False); a.set_title("a",loc="left",fontweight="bold")
b=ax[0,1]; x=np.arange(4); b.bar(x-.2,Lcat,.4,color=C_CAT,label="Category basis\n(SLOSH MOM × STORM)"); b.bar(x+.2,Lwl_at,.4,color=C_WL,label="Water-level basis\n(NACCS)")
for i in range(4): b.text(i-.2,Lcat[i]+1,f"{Lcat[i]/Lwl_at[i]:.1f}×",ha="center",va="bottom",fontsize=10)
b.set_xticks(x); b.set_xticklabels([f"Cat {c}\n{1/FREQ[c]:.0f} yr" for c in FREQ]); b.set_ylabel("Event structure loss (USD bn)"); b.set_ylim(0,max(Lcat)*1.25); b.legend(loc="upper left",frameon=False); b.set_title("b",loc="left",fontweight="bold")
b.text(.03,.66,f"AAI: {aai_cat:.0f} vs {aai_wl:.0f} M yr⁻¹\n(ratio {aai_cat/aai_wl:.1f}×)",transform=b.transAxes,ha="left",va="top",fontsize=10)
c=ax[1,0]; cols=["#9A9A9A"]*8+[C_WL]; c.bar(range(9),vals,color=cols);
for i,v in enumerate(vals): c.text(i,v+90,f"{v:,}",ha="center",va="bottom",fontsize=8.5,rotation=90)
c.set_xticks(range(9)); c.set_xticklabels([l.replace("\n"," ") for l in labs],fontsize=8,rotation=60,ha="right",rotation_mode="anchor"); c.set_ylabel("Surge AAI (M USD yr⁻¹)"); c.set_ylim(0,7800); c.grid(axis="y",alpha=.3); c.set_title("c",loc="left",fontweight="bold")
d=ax[1,1]; sh=["#1B4965","#3B6B8A","#6F97AE","#9EBBCB","#C9D3DA","#E3E8EC"]; d.bar(range(6),contrib,color=sh,edgecolor="none")
for i,v in enumerate(contrib): d.text(i,v+4,f"{100*v/contrib.sum():.0f}%",ha="center",va="bottom",fontsize=10)
d.set_xticks(range(6)); d.set_xticklabels([f"{lo:,}–{hi:,}" for lo,hi in bands],fontsize=9,rotation=35,ha="right",rotation_mode="anchor"); d.set_xlabel("Return-period band (yr)"); d.set_ylabel("AAI contribution (M USD yr⁻¹)"); d.set_ylim(0,contrib.max()*1.25); d.set_title("d",loc="left",fontweight="bold")
plt.savefig("figED3_hazard_basis_en.png",dpi=300,bbox_inches="tight"); plt.savefig("figED3_hazard_basis_en.pdf",bbox_inches="tight"); print("saved figED3_hazard_basis_en.png")
