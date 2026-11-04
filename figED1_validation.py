"""Data for Extended Data Fig. 2a-c — validation of the surge basis on Hurricane Sandy and uncertainty of the AAI.
a  modelled connected-bathtub footprint at Sandy's peak (3.44 m NAVD88) vs the field-verified inundation zone (hit / miss / false alarm)
b  Sandy structure loss: untuned USACE/NSI model vs the observed direct-loss range
c  loss vs return period on three bases (USACE generic on NSI; storey-fraction curve recalibrated on NSI; PLUTO $400/sqft working basis)
d  Monte Carlo distribution of the surge AAI (4,000 draws; same sampling as paper_a_mc.py)
"""
import json, warnings, numpy as np, pandas as pd, geopandas as gpd
from scipy import ndimage
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap; from matplotlib.patches import Patch
exec(open("paper_a_core.py").read().split("# ---------------- Sandy validation")[0])
matplotlib.rcParams.update({"font.size":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":11,"axes.titlesize":12})
# ---- a: footprint skill at Sandy's level
below=(dem_v<SANDY_WL)&valid; lab,_=ndimage.label(below); mod=np.isin(lab,np.unique(lab[seed&(lab>0)]))&below&(dem_v>=0)
boro=gpd.read_file("data/nfip/tl_2020_36_bg.shp"); boro=boro[boro.COUNTYFP.isin(["061","047","081","005","085"])].to_crs("EPSG:4326").dissolve(by="COUNTYFP")
city=features.rasterize(((x,1) for x in boro.geometry.buffer(0)),out_shape=dem.shape,transform=tr,dtype="uint8").astype(bool); mod&=city; obs=obs&city
hit=mod&obs; miss=obs&~mod; fa=mod&~obs; POD=hit.sum()/obs.sum(); FAR=fa.sum()/mod.sum(); CSI=hit.sum()/(hit|miss|fa).sum()
print(f"Sandy footprint skill: POD {100*POD:.0f}% | FAR {100*FAR:.0f}% | CSI {100*CSI:.0f}% | obs cells {int(obs.sum()):,} mod cells {int(mod.sum()):,}")
# ---- b: Sandy loss
L_us=loss_usace(d_sandy).sum()/1e9; base=loss_storey(d_sandy,1.0).sum(); k_nsi=SANDY_LOSS[1]/base
# ---- c: loss vs RP on three bases
curve=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in curve)); WLm=np.array([curve[str(t)] for t in T])
aeps=np.logspace(0,-4,120); wl0=np.interp(np.log(1/aeps),np.log(T),WLm)-NAVD_MINUS_MSL; rp=1/aeps
Lus=np.array([loss_usace(depths[i]).sum() for i in range(len(levels))]); Lst=np.array([loss_storey(depths[i],1.0).sum() for i in range(len(levels))])
Lstk=np.array([loss_storey(depths[i],k_nsi).sum() for i in range(len(levels))])   # damage capped at 1 per structure, as in Extended Data Fig. 7d
L1=np.interp(wl0,levels,Lus,left=0); L2=np.interp(wl0,levels,Lstk,left=0)
pl=json.load(open("naccs_aai.json"))["epoch MSL (1983-2001)"]["loss_by_rp_bn"]; rp3=np.array(sorted(int(k) for k in pl)); L3=np.array([pl[str(t)] for t in rp3])
aai1=-np.trapz(L1,aeps)/1e6; aai2=-np.trapz(L2,aeps)/1e6
# ---- d: Monte Carlo (identical sampling to paper_a_mc.py)
rng=np.random.default_rng(1); n=4000; out=np.zeros(n); fam=np.zeros(n,int)
for i in range(n):
    shift=rng.uniform(-0.10,0.10); vmult=np.exp(rng.normal(0,0.25)); fam[i]=rng.integers(2)
    if fam[i]==0: scale=rng.triangular(0.76,1.0,1.15)*(1+rng.normal(0,0.03)); L=np.interp(wl0+shift,levels,Lus,left=0)*scale
    else: L=np.interp(wl0+shift,levels,np.minimum(Lst*k_nsi*rng.triangular(0.92,1.0,1.08),val.sum()),left=0)
    out[i]=-np.trapz(L*vmult,aeps)
q=np.percentile(out,[5,25,50,75,95])/1e6; print(f"MC median {q[2]:.0f} | 50% {q[1]:.0f}-{q[3]:.0f} | 90% {q[0]:.0f}-{q[4]:.0f} | USACE point {aai1:.0f} | storey point {aai2:.0f}")
json.dump(dict(pod=POD,far=FAR,csi=CSI,sandy_usace_bn=L_us,k_nsi=k_nsi,aai_usace_m=aai1,aai_storey_m=aai2,mc_pct=dict(zip(["q05","q25","q50","q75","q95"],q.tolist()))),open("figED1_validation.json","w"),indent=1)
# ---- figure
ext=[tr.c,tr.c+dem.shape[1]*tr.a,tr.f+dem.shape[0]*tr.e,tr.f]
cls=np.zeros(dem.shape,"uint8"); cls[hit]=1; cls[miss]=2; cls[fa]=3
water=np.where(valid&(dem<0),1,0)
fig=plt.figure(figsize=(7.5,8.6)); gs=fig.add_gridspec(2,2,width_ratios=[1.45,1],height_ratios=[1.35,1],hspace=.5,wspace=.38,left=.08,right=.985,top=.975,bottom=.06)
ax=fig.add_subplot(gs[0,0]); ax.imshow(water,extent=ext,cmap=ListedColormap(["#FCFCF9","#E4E4E4"]),interpolation="nearest")
ax.imshow(np.ma.masked_equal(cls,0),extent=ext,cmap=ListedColormap(["#1B4965","#D1542F","#F3C77B"]),vmin=1,vmax=3,interpolation="nearest")
boro.boundary.plot(ax=ax,color="black",lw=.5); ax.set_xlim(-74.12,-73.72); ax.set_ylim(40.53,40.92); ax.set_xticks([-74.1,-73.9,-73.7]); ax.set(xlabel="Longitude (°E)",ylabel="Latitude (°N)"); ax.set_title("a",loc="left",fontweight="bold")
ax.legend(handles=[Patch(fc="#1B4965",label=f"Hit ({100*POD:.0f}% of observed)"),Patch(fc="#D1542F",label="Miss"),Patch(fc="#F3C77B",label=f"False alarm ({100*FAR:.0f}% of modelled)")],loc="upper center",bbox_to_anchor=(0.5,-0.17),ncol=3,frameon=False,handlelength=1.1,handletextpad=0.4,columnspacing=1.0,fontsize=10)
ax.text(0.03,0.97,f"CSI {CSI:.2f}",transform=ax.transAxes,ha="left",va="top",fontsize=11)
# b
b=fig.add_subplot(gs[0,1]); b.bar([0],[L_us],color="#1B4965",width=1.0); b.axhspan(SANDY_LOSS[0]/1e9,SANDY_LOSS[2]/1e9,color="#D1542F",alpha=.25,lw=0); b.axhline(SANDY_LOSS[1]/1e9,color="#D1542F",lw=1.4)
b.text(0,L_us-.4,f"{L_us:.1f} bn\nratio {L_us/(SANDY_LOSS[1]/1e9):.2f}",ha="center",va="top",color="white",fontsize=11); b.text(.55,SANDY_LOSS[2]/1e9+.25,"observed direct loss\n11.4–13.3 bn",color="#B33A30",va="bottom",ha="left",fontsize=11)
b.set_xlim(-.6,1.4); b.set_ylim(0,16); b.set_xticks([0]); b.set_xticklabels(["Untuned model"]); b.set_ylabel("Sandy structure loss (USD bn)"); b.set_title("b",loc="left",fontweight="bold")
# c
c=fig.add_subplot(gs[1,0]); m=(rp>=2)&(rp<=1e4)
c.semilogx(rp[m],L1[m]/1e9,"-",color="#1B4965",lw=1.8,label=f"USACE generic, NSI ({aai1:.0f} M/yr)"); c.semilogx(rp[m],L2[m]/1e9,"--",color="#2E7D5B",lw=1.6,label=f"Storey curve, NSI ({aai2:.0f} M/yr)"); c.semilogx(rp3,L3,":",color="#7A5EA8",lw=1.8,label="PLUTO $400/sqft, storey (754 M/yr)")
c.axvspan(110,150,color="#D1542F",alpha=.15,lw=0); c.errorbar([130],[SANDY_LOSS[1]/1e9],yerr=[[ (SANDY_LOSS[1]-SANDY_LOSS[0])/1e9],[(SANDY_LOSS[2]-SANDY_LOSS[1])/1e9]],fmt="o",color="#B33A30",ms=5,capsize=3,label="Sandy observed")
c.set_xlim(2,1e4); c.set_ylim(0,80); c.set_xlabel("Return period (yr)"); c.set_ylabel("Structure loss (USD bn)"); c.grid(alpha=.3); c.legend(loc="upper left",frameon=False,fontsize=10); c.set_title("c",loc="left",fontweight="bold")
# d
d=fig.add_subplot(gs[1,1]); d.hist(out/1e6,bins=60,color="#C9D3DA",edgecolor="none")
for v,ls in ((q[0],":"),(q[2],"-"),(q[4],":")): d.axvline(v,color="black",ls=ls,lw=1.2)
d.axvline(aai1,color="#1B4965",lw=1.8)
d.text(.97,.97,f"median {q[2]:.0f}\n5–95% {q[0]:.0f}–{q[4]:.0f}\nUSACE point {aai1:.0f}",transform=d.transAxes,ha="right",va="top",fontsize=10,linespacing=1.4)
d.set_xlabel("Surge AAI (M USD yr⁻¹)"); d.set_ylabel("Draws"); d.set_title("d",loc="left",fontweight="bold"); d.set_xlim(150,1400)
plt.savefig("figED1_validation_en.png",dpi=300,bbox_inches="tight"); plt.savefig("figED1_validation_en.pdf",bbox_inches="tight"); print("saved figED1_validation_en.png")
