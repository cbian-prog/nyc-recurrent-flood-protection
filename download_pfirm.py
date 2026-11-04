"""Download the 2015 Preliminary FIRM flood-hazard zones for New York City (DFIRM 360497, PRELM_CODE 360497_20150130)
from FEMA's Prelim_NFHL map service, paged by OBJECTID. Output data/fema_nfhl/pfirm2015_zones_nyc.geojson."""
import json, time, urllib.request, urllib.parse
B="https://hazards.fema.gov/arcgis/rest/services/PrelimPending/Prelim_NFHL/MapServer/28/query"
def get(params,tries=5):
    url=B+"?"+urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url,timeout=300) as r: return json.load(r)
        except Exception as e: print("retry",k,e); time.sleep(5*(k+1))
    raise RuntimeError(url)
ids=get(dict(where="DFIRM_ID='360497'",returnIdsOnly="true",f="json"))["objectIds"]; ids=sorted(ids); print("features",len(ids))
feats=[]
for i in range(0,len(ids),200):
    chunk=ids[i:i+200]
    d=get(dict(objectIds=",".join(map(str,chunk)),outFields="OBJECTID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE",outSR=4326,f="geojson"))
    feats+=d["features"]; print(i+len(chunk),"/",len(ids),flush=True)
json.dump(dict(type="FeatureCollection",features=feats),open("data/fema_nfhl/pfirm2015_zones_nyc.geojson","w"))
print("saved",len(feats))
