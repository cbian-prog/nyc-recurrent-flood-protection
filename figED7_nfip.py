"""Data for Extended Data Fig. 4 — the NFIP policy data and Sandy claims (the two-city comparison was dropped 2026-11-08).
a  in-force NFIP building coverage per contract (36,961 contracts): most sit exactly at the $250,000 residential limit
b  structure-count take-up (contracts per NSI structure) and building cover by borough
c  Sandy: NFIP building + contents claims paid by borough (OpenFEMA NFIP Redacted Claims v3, loss dates 28 Oct-5 Nov 2012) versus modelled untuned structure loss on the observed footprint
d  gap decomposition of the annual loss, New York City versus Miami-Dade, on the common category basis (SLOSH MOM x STORM, storey curve k = 0.477)
"""
import glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("paper_a_core.py").read().split("# ---------------- Sandy validation")[0])
matplotlib.rcParams.update({"font.size":12,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":10,"axes.titlesize":12})
C_PAID,C_NOP,C_CAP="#1B4965","#D1542F","#EFA35C"; BORO={"005":"Bronx","047":"Brooklyn","061":"Manhattan","081":"Queens","085":"Staten Is."}; order=["Queens","Brooklyn","Staten Is.","Manhattan","Bronx"]
nf=pd.read_parquet("data/nfip/all_current.parquet"); cov=nf.totalBuildingInsuranceCoverage.values
edges=[0,1e5,249999,250001,1e6,1e12]; labs=["<100k","100–250k","= 250k","250k–1M",">1M"]; cls=np.digitize(cov,edges)-1
n_cls=pd.Series(cls).value_counts().reindex(range(5)).fillna(0); c_cls=pd.Series(cov).groupby(cls).sum().reindex(range(5)).fillna(0)/1e9
print("contracts by cover class:",n_cls.astype(int).tolist(),"| cover bn:",c_cls.round(1).tolist())
nf["boro"]=nf.censusGeoid.astype(str).str[2:5].map(BORO); s["boro"]=s.cbfips.astype(str).str[2:5].map(BORO)
tk=(nf.groupby("boro").size()/s.groupby("boro").size()).reindex(order); cb=(nf.groupby("boro").totalBuildingInsuranceCoverage.sum()/1e9).reindex(order)
# Sandy claims by borough
# OpenFEMA NFIP Redacted Claims v3 (NfipClaims), pulled by nfip_sandy_claims_v3.py; all 16,498 carry floodEvent "Hurricane Sandy"
cl=pd.read_json("data/nfip/claims_v3_sandy.json"); cl["fips"]=cl.countyCode.astype(str).str.zfill(5)
cl["t"]=pd.to_datetime(cl.dateOfLoss,errors="coerce").dt.tz_localize(None); cl=cl[(cl.t>="2012-10-28")&(cl.t<"2012-11-06")]; cl["boro"]=cl.fips.str[2:5].map(BORO)
paid_b=(cl.groupby("boro").amountPaidOnBuildingClaim.sum()/1e9).reindex(order).fillna(0); paid_c=(cl.groupby("boro").amountPaidOnContentsClaim.sum()/1e9).reindex(order).fillna(0)
mod_b=(pd.Series(loss_usace(d_sandy)).groupby(s.boro.values).sum()/1e9).reindex(order).fillna(0)
print(f"Sandy claims: building {paid_b.sum():.2f} bn, contents {paid_c.sum():.2f} bn, n={len(cl):,} | modelled structure loss {mod_b.sum():.1f} bn | building claims / modelled {100*paid_b.sum()/mod_b.sum():.0f}%")
json.dump(dict(cover_classes=dict(zip(labs,n_cls.astype(int).tolist())),cover_bn=dict(zip(labs,c_cls.tolist())),takeup_by_boro=tk.to_dict(),cover_by_boro_bn=cb.to_dict(),sandy_claims_building_bn=paid_b.to_dict(),sandy_claims_contents_bn=paid_c.to_dict(),sandy_modelled_bn=mod_b.to_dict()),open("figED7_nfip.json","w"),indent=1)
# ---- figure
fig,ax=plt.subplots(2,2,figsize=(7.5,7.2),gridspec_kw=dict(hspace=.55,wspace=.38,left=.09,right=.985,top=.97,bottom=.1))
a=ax[0,0]; x=np.arange(5); a.bar(x,n_cls/1e3,color="#1B4965")
for i in range(5): a.text(i,n_cls.iloc[i]/1e3+.4,f"{c_cls.iloc[i]:.1f} bn",ha="center",va="bottom",fontsize=9)
a.set_xticks(x); a.set_xticklabels(labs,fontsize=10,rotation=30,ha="right",rotation_mode="anchor"); a.set_xlabel("Building cover per contract (USD)"); a.set_ylabel("Contracts (thousand)"); a.set_ylim(0,n_cls.max()/1e3*1.45); a.set_title("a",loc="left",fontweight="bold"); a.grid(axis="y",alpha=.3)
a.text(.03,.97,f"{len(nf):,} contracts, {cov.sum()/1e9:.1f} bn cover\n(labels: cover in class)",transform=a.transAxes,ha="left",va="top",fontsize=9)
b=ax[0,1]; xb=np.arange(5); b.bar(xb,100*tk,color="#1B4965"); b.set_ylabel("Take-up (contracts per 100 structures)"); b.set_xticks(xb); b.set_xticklabels(order,rotation=30,ha="right",rotation_mode="anchor"); b.set_title("b",loc="left",fontweight="bold"); b.grid(axis="y",alpha=.3)
b2=b.twinx(); b2.plot(xb,cb,"o",color="#D1542F",ms=6); b2.set_ylabel("Building cover (USD bn)",color="#D1542F"); b2.tick_params(axis="y",colors="#D1542F"); b2.set_ylim(0,cb.max()*1.2)
c=ax[1,0]; xc=np.arange(5); c.bar(xc-.2,mod_b,.4,color="#C9D3DA",label="Modelled structure loss"); c.bar(xc+.2,paid_b,.4,color=C_PAID,label="NFIP building claims paid"); c.bar(xc+.2,paid_c,.4,bottom=paid_b,color="#6F97AE",label="NFIP contents claims paid")
c.set_xticks(xc); c.set_xticklabels(order,rotation=30,ha="right",rotation_mode="anchor"); c.set_ylabel("Sandy (USD bn)"); c.set_ylim(0,mod_b.max()*1.3); c.legend(frameon=False,fontsize=8.5,loc="upper right",bbox_to_anchor=(1.02,1.0)); c.set_title("c",loc="left",fontweight="bold"); c.grid(axis="y",alpha=.3)
c.text(.97,.60,f"claims {paid_b.sum():.2f} + {paid_c.sum():.2f} bn\nmodelled {mod_b.sum():.1f} bn\nobserved 11.4–13.3 bn",transform=c.transAxes,ha="right",va="top",fontsize=9)
ax[1,1].axis("off")
plt.savefig("figED7_nfip_en.png",dpi=300,bbox_inches="tight"); plt.savefig("figED7_nfip_en.pdf",bbox_inches="tight"); print("saved figED7_nfip_en.png")
