"""Fig. 3 | Where the risk-management instruments reach (R3, the paper's argument).
a coverage gaps grouped by instrument: share of each exposure outside the instrument's coverage
b insurance follows the map, not the hazard: rain-scenario building loss vs NFIP policies by block-group side; insurable share by side"""
import json, numpy as np
from nc_style import *
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
N=json.load(open("nc_numbers.json")); PI=json.load(open("pluvial_insurance_nsi.json"))["Extreme, 2080 SL"]; G=json.load(open("fig5_gap_nsi.json"))
ev100=G["events"]["100"]; surge100_out=100*(1-ev100["neutral"]/ev100["loss"])
TG=N["tidal_sensors_geography"]; TU=N["takeup_per100_by_map"]
SH=json.load(open("nc_si_subhourly.json"))["below_design_any"]
lots_r=N["basement_lots_rain"]; st_r=N["stations_rain"]
rows=[  # (group, label, share outside %, annotation, modelled-reach check %)
 ("Flood map: anchored to coastal geography (FEMA effective SFHA)",None,None,None,None),
 (None,"Extreme-rain flooded area",100*json.load(open("nc_areas.json"))["rain_outside_sfha_city"],f"{json.load(open('nc_areas.json'))['rain_km2_city']:.0f} km²",100*N["rain_area_outside_s100"]),
 (None,"Rain-exposed basement lots",100*N["basement_lots_rain_outside_sfha"]/lots_r,f"{N['basement_lots_rain_outside_sfha']:,} / {lots_r:,}",100*N["basement_lots_rain_outside_s100"]/N["basement_lots_rain_in_domain"]),
 (None,"Rain-exposed underground stations",100*N["stations_rain_outside_sfha"]/st_r,f"{N['stations_rain_outside_sfha']} / {st_r}",100*N["stations_rain_outside_s100"]/st_r),
 ("Drainage standard: anchored to 1.75 in h$^{-1}$",None,None,None,None),
 (None,"Flood days below the standard, hourly rain",100*N["flood_days_below_design"],f"{N['flood_days_below_design_n']} / {N['flood_days']}",None),
 (None,"Same, 30-min peak (estimated)",100*SH["est_30min"],f"15-min peak: {100*SH['est_15min']:.0f}%",None),
 ("Public insurance: follows the flood zone",None,None,None,None),
 (None,"Rain-side building loss (expected)",100-PI["pluvial-only BGs"]["expected_pct"],f"USD {PI['pluvial-only BGs']['loss_bn']:.1f} bn",None),
 (None,"Basement apartments (policy terms)",100.0,"≥50,000 informal units",None),
 ("Reference: coastal sources, inside the flood zone",None,None,None,None),
 (None,"Tidal flood days (flood zone)",100*(1-TG["flood_days_at_sensors_in_sfha"]),f"{TG['in_sfha']} of {TG['n']} tidal sensors inside",100*(1-TG["flood_days_at_sensors_in_s100"])),
 (None,"Surge-side 1-in-100-yr loss (insurance, expected)",surge100_out,f"USD {ev100['loss']/1e9:.1f} bn",None)]
fig=plt.figure(figsize=(FULL,110*MM))
AX0,AXW=57,40   # panel a bar area in mm; headings start 4 mm from the left edge
a=axes_mm(fig,AX0,12,AXW,90)
y=0; yt=[]; yl=[]; ref=False
for g,lab,v,ann,mod in rows:
    if g:
        a.text(-(AX0-4)/AXW,y,g,transform=a.get_yaxis_transform(),fontsize=7.5,fontweight="bold",ha="left",va="center",color=PROT if not g.startswith("Ref") else "#7F7F7F"); ref=g.startswith("Ref"); y-=1; continue
    col=PROT_LIGHT if ref else PROT
    a.barh(y,v,height=0.62,color=col)
    xl=max(v,mod if (mod is not None and mod>v) else v)+(3 if (mod is not None and mod>v) else 1.5)
    a.text(xl,y,f"{v:.0f}%",va="center",fontsize=7,fontweight="bold",color=INK)
    a.text(xl+18,y,ann,va="center",fontsize=7,color="#555555")
    if mod is not None: a.plot([mod],[y],marker="o",ms=3,mfc="white",mec=INK,mew=0.6,ls="")
    yt.append(y); yl.append(lab); y-=1
a.set_yticks(yt); a.set_yticklabels(yl); a.tick_params(axis="y",length=0,pad=3)
a.set_xlim(0,100); a.set_ylim(y+0.4,2.0); a.set_xticks([0,25,50,75,100]); a.grid(axis="x")   # top row holds the key to the open circles
a.set_xlabel("Share outside the instrument's coverage (%)")
a.spines["left"].set_visible(False)
a.plot([2],[1.2],marker="o",ms=3,mfc="white",mec=INK,mew=0.6,ls="",clip_on=False)
a.text(5,1.2,"same, with the modelled 1-in-100-yr\nsurge reach in place of the flood zone",fontsize=7,va="center",color="#555555")
panel(a,"a",dx=-(AX0-4),dy=4)   # level with b
# ---------------- b
b=axes_mm(fig,146,48,32,30)
pr,ps,px=PI["pluvial-only BGs"],PI["surge-side BGs"],PI["outside surge domain BGs"]   # px: block groups where surge is not modelled
loss_r=pr["loss_share_pct"]; loss_s=ps["loss_share_pct"]; loss_x=px["loss_share_pct"]; pol_r=pr["contract_share_pct"]; pol_s=ps["contract_share_pct"]; pol_x=px["contract_share_pct"]; pol_o=100-pol_r-pol_s-pol_x
HX=dict(color="white",ec="#9A9A9A",lw=0.3,hatch="//////")
b.bar(0,loss_r,0.6,color=RAIN_EXT); b.bar(0,loss_s,0.6,bottom=loss_r,color=SURGE100); b.bar(0,loss_x,0.6,bottom=loss_r+loss_s,**HX)
b.bar(1,pol_r,0.6,color=RAIN_EXT); b.bar(1,pol_s,0.6,bottom=pol_r,color=SURGE100); b.bar(1,pol_x,0.6,bottom=pol_r+pol_s,**HX); b.bar(1,pol_o,0.6,bottom=pol_r+pol_s+pol_x,color="#E3E3E3")
for x,(lo,v) in ((0,(0,loss_r)),(0,(loss_r,loss_s)),(1,(0,pol_r)),(1,(pol_r,pol_s)),(1,(pol_r+pol_s+pol_x,pol_o))):
    if v>6: b.text(x,lo+v/2,f"{v:.0f}%",ha="center",va="center",fontsize=7,color="white" if (x,lo)!=(1,pol_r+pol_s+pol_x) else INK)
b.set_xticks([0,1]); b.set_xticklabels(["Rain-scenario\nbuilding loss","NFIP\npolicies"]); b.set_ylim(0,100); b.set_ylabel("Share (%)"); b.set_xlim(-0.55,1.55)
b.legend(handles=[Patch(fc=RAIN_EXT,label="Rain side"),Patch(fc=SURGE100,label="Surge side"),Patch(fc="white",ec="#9A9A9A",lw=0.3,hatch="//////",label="Surge not modelled"),Patch(fc="#E3E3E3",label="Not flooded")],
         title="Block groups",alignment="left",loc="upper left",bbox_to_anchor=(133/180,37/110),bbox_transform=fig.transFigure,borderaxespad=0,handlelength=0.9,labelspacing=0.25,ncol=2,columnspacing=0.8)
c=axes_mm(fig,146,88,32,18)
for x,side,col in ((0,pr,RAIN_EXT),(1,ps,SURGE100)):
    c.plot([x,x],[side["expected_pct"],side["upper_pct"]],color=col,lw=0.8)
    c.plot(x,side["expected_pct"],"o",ms=3.5,color=col); c.plot(x,side["upper_pct"],"o",ms=3.5,mfc="white",mec=col,mew=0.7)
    c.text(x+0.12,max(side["expected_pct"],7),f"{side['expected_pct']:.1f}%",fontsize=7,va="center"); c.text(x+0.12,side["upper_pct"],f"{side['upper_pct']:.0f}%",fontsize=7,va="center")
c.set_xlim(-0.55,1.55); c.set_ylim(0,100); c.set_xticks([0,1]); c.set_xticklabels(["Rain side","Surge side"]); c.set_ylabel("Insurable (%)")
c.legend(handles=[Line2D([],[],marker="o",ls="",color=INK,ms=3,label="expected"),Line2D([],[],marker="o",ls="",mfc="white",mec=INK,mew=0.7,ms=3,label="upper bound")],
         loc="upper left",ncol=1,borderaxespad=0.1)
fig.text(133/180,24/110,f"{100*pr['bgs_nopolicy']/pr['bgs']:.0f}% of rain-side block groups\nhold no policy",fontsize=7,ha="left",va="top",style="italic")
fig.text(133/180,15/110,f"NFIP contracts per 100 structures:\n{TU['all']['in_sfha']:.0f} inside the flood zone, {TU['all']['outside']:.0f} outside;\nwhere Sandy flooded, {TU['sandy_flooded_bgs']['in_sfha']:.0f} inside\nand {TU['sandy_flooded_bgs']['outside']:.0f} outside",fontsize=7,ha="left",va="top")
panel(c,"b",dx=-12,dy=0)
save(fig,"Fig3")
json.dump(dict(rows=[(r[1],r[2],r[3],r[4]) for r in rows if r[1]],policies_rain=pol_r,policies_surge=pol_s,policies_outside=pol_o,policies_outside_domain=pol_x,loss_rain=loss_r,loss_surge=loss_s,loss_outside_domain=loss_x,
               insurable_rain=(pr["expected_pct"],pr["upper_pct"]),insurable_surge=(ps["expected_pct"],ps["upper_pct"]),rain_bgs_nopolicy=pr["bgs_nopolicy"]/pr["bgs"]),open("nc_fig3.json","w"),indent=1)
