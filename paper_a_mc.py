"""Monte Carlo uncertainty for the Paper A surge AAI (uses data/paper_a/depths.npz from paper_a_core.py).
Sampled: hazard-curve digitisation shift U(-0.1,0.1) m; structure valuation multiplier lognormal(0, 0.25);
curve family {USACE generic, storey-fraction} 50/50; Sandy scaling for USACE from triangular(0.76,1.0,1.15)
(= observed structure-only Sandy loss 8.0-12.1 bn over the 10.48 bn the curves give unscaled); per-depth curve noise N(0, SD from EGM tables) via a global draw.
Not sampled (structural, stated): no coastal defences in the bathtub; single Battery curve; present-day sea level."""
import json, numpy as np, pandas as pd
exec(open("paper_a_core.py").read().split("# ---------------- Sandy validation")[0])  # reuse exposure/hazard setup
curve=json.load(open("data/naccs/battery_naccs_curve.json"))["rp_wl_msl"]; T=np.array(sorted(int(k) for k in curve)); WLm=np.array([curve[str(t)] for t in T])
aeps=np.logspace(0,-4,120); wl0=np.interp(np.log(1/aeps),np.log(T),WLm)-NAVD_MINUS_MSL
Lus=np.array([loss_usace(depths[i]).sum() for i in range(len(levels))])
Lst=np.array([loss_storey(depths[i],1.0).sum() for i in range(len(levels))])   # k applied after
k_nsi=json.load(open("paper_a_core.json"))["sandy"]["k_nsi"]
rng=np.random.default_rng(1); n=4000; out=np.zeros(n); fam=np.zeros(n,int)
for i in range(n):
    shift=rng.uniform(-0.10,0.10); vmult=np.exp(rng.normal(0,0.25)); fam[i]=rng.integers(2)
    if fam[i]==0: scale=rng.triangular(0.76,1.0,1.15)*(1+rng.normal(0,0.03)); L=np.interp(wl0+shift,levels,Lus,left=0)*scale
    else: L=np.interp(wl0+shift,levels,np.minimum(Lst*k_nsi*rng.triangular(0.92,1.0,1.08),val.sum()),left=0)
    out[i]=-np.trapz(L*vmult,aeps)
q=np.percentile(out,[5,25,50,75,95])/1e6
print(f"AAI Monte Carlo (n={n}): median {q[2]:.0f} M/yr | 50% {q[1]:.0f}–{q[3]:.0f} | 90% {q[0]:.0f}–{q[4]:.0f}")
for f,lab in ((0,"USACE"),(1,"storey")): print(f"  {lab}: median {np.median(out[fam==f])/1e6:.0f}, 90% {np.percentile(out[fam==f],5)/1e6:.0f}–{np.percentile(out[fam==f],95)/1e6:.0f}")
json.dump(dict(n=n,q05=q[0],q25=q[1],q50=q[2],q75=q[3],q95=q[4]),open("paper_a_mc.json","w"),indent=1)
