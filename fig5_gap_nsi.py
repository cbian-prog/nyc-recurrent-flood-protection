"""Data for Fig. 4, Supplementary Fig. 3 and Supplementary Table 5 — why NFIP cover falls short, on the publication basis (NSI x USACE x NACCS water-level curve).

Two estimates of what NFIP pays for a surge event:
  upper   : min(block-group loss, block-group building cover) — every policy sits on a damaged building
  neutral : policies placed at random among the structures of a block group:
            E[paid_i] = take_bg * min(loss_i, mean cover per contract in bg)
Gap decomposition under the neutral estimate:
  loss - paid = (1-take)*loss            [no policy]
              + take*max(loss-cover, 0)  [cap: policy exists but cover < loss]
Counterfactuals (per structure, then summed):
  full take-up, local cover     : min(loss_i, mean cover per contract in bg)   — everyone buys the typical local policy
  full take-up, statutory cap   : min(loss_i, 250k x units | 500k non-res)     — everyone buys the maximum NFIP allows
All quantities are computed on the ladder of still-water levels, interpolated to 120 AEP points and integrated to AAI.
Outputs fig5_gap_nsi.json and fig5_gap_nsi_en.png/.pdf.
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":11,"axes.titlesize":12})
exec(open("paper_a_core.py").read().split("# ---------------- Sandy validation")[0])
bg=s.cbfips.astype(str).str[:12].values
nfip=pd.read_parquet("data/nfip/all_current.parquet")
bcov=nfip.groupby("censusGeoid").totalBuildingInsuranceCoverage.sum(); ncon=nfip.groupby("censusGeoid").size()
n_bg=pd.Series(bg).value_counts()
take=(ncon.reindex(n_bg.index).fillna(0)/n_bg).clip(upper=1).reindex(bg).values
avgcov=(bcov/ncon).reindex(bg).fillna(0).values
stat_cap=np.where(s.st_damcat.values=="RES",2.5e5*np.clip(s.resunits.fillna(1).values,1,None),5e5)
bgi=np.unique(bg); cov_bg=bcov.reindex(bgi).fillna(0).values
# hazard curve -> AEP grid
curve=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in curve)); WLm=np.array([curve[str(t)] for t in T])
aeps=np.logspace(0,-4,120); wl=np.interp(np.log(1/aeps),np.log(T),WLm)-NAVD_MINUS_MSL
# tract LMI groups
eq=pd.read_parquet("data/nfip/tract_equity.parquet")[["tractGeoid","lomod_pct"]].dropna().set_index("tractGeoid").lomod_pct
labs=["<35%","35–51%","51–70%","≥70%"]; grp=pd.cut(eq.reindex(pd.Index(bg).str[:11]).values,[0,35,51,70,101],labels=labs,right=False)
grp=pd.Series(grp).astype(str).values
# per-level totals (city) and per-group totals
keys=["loss","upper","neutral","nopolicy","cap","full_local","full_stat"]
tot={k:np.zeros(len(levels)) for k in keys}; byg={g:{k:np.zeros(len(levels)) for k in keys} for g in labs}
for i in range(len(levels)):
    li=loss_usace(depths[i])
    lb=pd.Series(li).groupby(bg).sum().reindex(bgi).fillna(0).values; up_bg=np.minimum(lb,cov_bg)
    parts=dict(loss=li,neutral=take*np.minimum(li,avgcov),nopolicy=(1-take)*li,cap=take*np.maximum(li-avgcov,0),
               full_local=np.minimum(li,avgcov),full_stat=np.minimum(li,stat_cap))
    for k,v in parts.items(): tot[k][i]=v.sum()
    tot["upper"][i]=up_bg.sum()
    for g in labs:
        m=grp==g
        for k,v in parts.items(): byg[g][k][i]=v[m].sum()
        # upper bound by group: block groups nest in tracts, so assign each bg to its tract's group
    up_series=pd.Series(up_bg,index=bgi); bg_grp=pd.Series(grp,index=bg).groupby(level=0).first().reindex(bgi)
    for g in labs: byg[g]["upper"][i]=up_series[bg_grp.values==g].sum()
interp=lambda y: np.interp(wl,levels,y,left=0)
aai=lambda y: float(-np.trapz(interp(y),aeps))
AAI={k:aai(tot[k]) for k in keys}; AAIg={g:{k:aai(byg[g][k]) for k in keys} for g in labs}
RPS=[10,45,100,400,1000]; EV={}
for rp in RPS:
    wl_rp=np.interp(np.log(rp),np.log(T),WLm)-NAVD_MINUS_MSL; EV[rp]={k:float(np.interp(wl_rp,levels,tot[k],left=0)) for k in keys}   # exactly at the return period, as in fig2_impact_rp.py
pct=lambda d,k: 100*d[k]/d["loss"]
print(f"AAI {AAI['loss']/1e6:.0f} M | upper {pct(AAI,'upper'):.0f}% | neutral {pct(AAI,'neutral'):.0f}% | gap no-policy {pct(AAI,'nopolicy'):.0f}% | gap cap {pct(AAI,'cap'):.0f}% | full take-up local {pct(AAI,'full_local'):.0f}% | statutory {pct(AAI,'full_stat'):.0f}%")
print(f"{'RP':>6}{'loss bn':>9}{'upper':>7}{'neutral':>9}{'no-pol':>8}{'cap':>6}{'full-loc':>10}{'full-stat':>11}")
for rp in RPS: d=EV[rp]; print(f"{rp:>6}{d['loss']/1e9:>9.1f}{pct(d,'upper'):>7.0f}{pct(d,'neutral'):>9.0f}{pct(d,'nopolicy'):>8.0f}{pct(d,'cap'):>6.0f}{pct(d,'full_local'):>10.0f}{pct(d,'full_stat'):>11.0f}")
for g in labs: d=AAIg[g]; print(f"LMI {g:>7}: AAI {d['loss']/1e6:6.0f} M | upper {pct(d,'upper'):3.0f}% | neutral {pct(d,'neutral'):3.0f}% | no-policy {pct(d,'nopolicy'):3.0f}% | cap {pct(d,'cap'):3.0f}% | full-loc {pct(d,'full_local'):3.0f}% | stat {pct(d,'full_stat'):3.0f}%")
# structure-count take-up inside the 100-yr footprint, by group
j100=np.argmin(abs(levels-np.interp(np.log(100),np.log(T),WLm)+NAVD_MINUS_MSL)); wet=depths[j100]>0
tk={g:float(take[wet&(grp==g)].mean()) for g in labs}; print("take-up in 100-yr footprint by LMI group:",{g:round(100*v,1) for g,v in tk.items()})
json.dump(dict(aai=AAI,aai_by_lmi=AAIg,events=EV,takeup_100yr_by_lmi=tk,contracts=int(len(nfip)),policy_units=int(nfip.policyCount.sum()),building_cover_bn=float(bcov.sum()/1e9),
               definitions=__doc__),open("fig5_gap_nsi.json","w"),indent=1)
# ---------------- figure ----------------
C_PAID,C_NOP,C_CAP="#1B4965","#D1542F","#EFA35C"
fig=plt.figure(figsize=(7.5,6.4)); gs=fig.add_gridspec(2,2,hspace=0.55,wspace=0.35)
ax=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0])]; lax=fig.add_subplot(gs[1,1]); lax.axis("off")
x=np.arange(len(RPS)+1); rows=[EV[rp] for rp in RPS]+[AAI]
p=[pct(d,"neutral") for d in rows]; n=[pct(d,"nopolicy") for d in rows]; c=[pct(d,"cap") for d in rows]; u=[pct(d,"upper") for d in rows]
ax[0].bar(x,p,color=C_PAID,label="Paid (expected)"); ax[0].bar(x,n,bottom=p,color=C_NOP,label="Gap: no policy"); ax[0].bar(x,c,bottom=np.array(p)+np.array(n),color=C_CAP,label="Gap: cover < loss")
ax[0].scatter(x,u,marker="_",s=260,color="black",lw=1.6,zorder=5,label="Paid (upper bound)")
ax[0].set_xticks(x); ax[0].set_xticklabels([str(r) for r in RPS]+["AAI"],rotation=30,ha="right"); ax[0].set_ylim(0,100); ax[0].set_ylabel("Share of event loss (%)"); ax[0].set_xlabel("Return period (yr)")
h0,l0=ax[0].get_legend_handles_labels(); ax[0].set_title("a",loc="left",fontweight="bold")
rp_grid=1/aeps; m=(rp_grid>=5)&(rp_grid<=2000)
for k,lab,col,ls in [("upper","Current policies, upper bound","black","-"),("neutral","Current policies, expected",C_PAID,"-"),("full_local","Full take-up, local cover","#2E7D5B","--"),("full_stat","Full take-up, statutory cap","#7A5EA8",":")]:
    y=100*interp(tot[k])/np.maximum(interp(tot["loss"]),1); ax[1].semilogx(rp_grid[m],y[m],ls,color=col,lw=1.6,label=lab)
ax[1].set_ylim(0,100); ax[1].set_xlim(5,2000); ax[1].set_xlabel("Return period (yr)"); ax[1].set_ylabel("Insurable share (%)"); ax[1].grid(alpha=.3)
h1,l1=ax[1].get_legend_handles_labels(); ax[1].set_title("b",loc="left",fontweight="bold")
leg_a=lax.legend(h0,l0,loc="upper left",bbox_to_anchor=(-0.05,1.02),frameon=False,title="a, c",title_fontsize=11,alignment="left",handletextpad=0.5); lax.add_artist(leg_a)
lax.legend(h1,l1,loc="upper left",bbox_to_anchor=(-0.05,0.5),frameon=False,title="b",title_fontsize=11,alignment="left",handlelength=1.8)
xg=np.arange(4); pg=[pct(AAIg[g],"neutral") for g in labs]; ng=[pct(AAIg[g],"nopolicy") for g in labs]; cg=[pct(AAIg[g],"cap") for g in labs]; ug=[pct(AAIg[g],"upper") for g in labs]
ax[2].bar(xg,pg,color=C_PAID); ax[2].bar(xg,ng,bottom=pg,color=C_NOP); ax[2].bar(xg,cg,bottom=np.array(pg)+np.array(ng),color=C_CAP); ax[2].scatter(xg,ug,marker="_",s=260,color="black",lw=1.6,zorder=5)
for i,g in enumerate(labs): ax[2].text(i,102,f"{100*tk[g]:.0f}%",ha="center",va="bottom",fontsize=11)
ax[2].text(-0.55,114,"take-up in 100-yr footprint:",fontsize=11,ha="left",va="bottom")
ax[2].set_xticks(xg); ax[2].set_xticklabels(["<35","35–51","51–70","≥70"]); ax[2].set_ylim(0,100); ax[2].set_xlabel("LMI share of tract (%)"); ax[2].set_ylabel("Share of annual loss (%)"); ax[2].set_title("c",loc="left",fontweight="bold",pad=26)
plt.savefig("fig5_gap_nsi_en.png",dpi=300,bbox_inches="tight"); plt.savefig("fig5_gap_nsi_en.pdf",bbox_inches="tight"); print("saved fig5_gap_nsi_en.png")
