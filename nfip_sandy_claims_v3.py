"""Re-pull Hurricane Sandy NFIP claims for the five boroughs from OpenFEMA 'NFIP Redacted Claims - v3' (NfipClaims) and
compare with the FIMA NFIP Redacted Claims v2 pull used in Extended Data Fig. 7c (data/nfip/claims_*.json).
Filter as before: county FIPS 36005/36047/36061/36081/36085, dateOfLoss 28 Oct - 5 Nov 2012; also reports the subset
FEMA labels with floodEvent containing 'Sandy'. Outputs data/nfip/claims_v3_sandy.json and nfip_sandy_claims_v3.json."""
import json, glob, time, urllib.request, urllib.parse, pandas as pd
B="https://www.fema.gov/api/open/v3/NfipClaims"
FIPS={"36005":"Bronx","36047":"Brooklyn","36061":"Manhattan","36081":"Queens","36085":"Staten Is."}
SEL="id,countyCode,dateOfLoss,floodEvent,amountPaidOnBuildingClaim,amountPaidOnContentsClaim,amountPaidOnIncreasedCostOfComplianceClaim,netBuildingPaymentAmount,netContentsPaymentAmount,occupancyType,asOfDate"
def get(params):
    url=B+"?"+urllib.parse.urlencode(params,safe="$,'= ")
    url=url.replace(" ","%20")
    for k in range(5):
        try:
            with urllib.request.urlopen(url,timeout=300) as r: return json.load(r)
        except Exception as e: print("retry",k,e); time.sleep(5*(k+1))
    raise RuntimeError(url)
rows=[]
for f in FIPS:
    flt=f"countyCode eq '{f}' and dateOfLoss ge '2012-10-28' and dateOfLoss le '2012-11-05T23:59:59.999Z'"
    skip=0
    while True:
        d=get({"$filter":flt,"$select":SEL,"$top":10000,"$skip":skip,"$count":"true"})
        recs=d["NfipClaims"]; rows+=recs; n=d["metadata"]["count"]; skip+=len(recs)
        if skip>=n or not recs: break
    print(f,FIPS[f],"claims",n)
C=pd.DataFrame(rows); C.to_json("data/nfip/claims_v3_sandy.json",orient="records")
C["boro"]=C.countyCode.map(FIPS); num=["amountPaidOnBuildingClaim","amountPaidOnContentsClaim","amountPaidOnIncreasedCostOfComplianceClaim","netBuildingPaymentAmount","netContentsPaymentAmount"]
C[num]=C[num].apply(pd.to_numeric,errors="coerce").fillna(0)
sandy=C.floodEvent.astype(str).str.contains("Sandy",case=False)
# v2 pull, same filter as figED7_nfip.py
V=pd.concat([pd.DataFrame(json.load(open(f))["FimaNfipClaims"]).assign(fips=f.split("_")[-1][:5]) for f in glob.glob("data/nfip/claims_*.json") if "v3" not in f])
V["t"]=pd.to_datetime(V.dateOfLoss,errors="coerce"); V=V[(V.t>="2012-10-28")&(V.t<="2012-11-05")]; V["boro"]=V.fips.map(FIPS)
for c in ("amountPaidOnBuildingClaim","amountPaidOnContentsClaim"): V[c]=pd.to_numeric(V[c],errors="coerce").fillna(0)
order=["Queens","Brooklyn","Staten Is.","Manhattan","Bronx"]
out=dict(v3=dict(claims=int(len(C)),building_bn=C.amountPaidOnBuildingClaim.sum()/1e9,contents_bn=C.amountPaidOnContentsClaim.sum()/1e9,icc_bn=C.amountPaidOnIncreasedCostOfComplianceClaim.sum()/1e9,
                 net_building_bn=C.netBuildingPaymentAmount.sum()/1e9,net_contents_bn=C.netContentsPaymentAmount.sum()/1e9,
                 sandy_label_claims=int(sandy.sum()),sandy_label_building_bn=C.amountPaidOnBuildingClaim[sandy].sum()/1e9,
                 flood_events=C.floodEvent.value_counts().head(5).to_dict(),as_of=str(C.asOfDate.max()),
                 building_by_boro_bn=(C.groupby("boro").amountPaidOnBuildingClaim.sum()/1e9).reindex(order).to_dict(),
                 contents_by_boro_bn=(C.groupby("boro").amountPaidOnContentsClaim.sum()/1e9).reindex(order).to_dict()),
         v2=dict(claims=int(len(V)),building_bn=V.amountPaidOnBuildingClaim.sum()/1e9,contents_bn=V.amountPaidOnContentsClaim.sum()/1e9,
                 building_by_boro_bn=(V.groupby("boro").amountPaidOnBuildingClaim.sum()/1e9).reindex(order).to_dict()))
out["v3"]["share_of_observed_direct_central_pct"]=100*out["v3"]["building_bn"]/12.35
json.dump(out,open("nfip_sandy_claims_v3.json","w"),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
