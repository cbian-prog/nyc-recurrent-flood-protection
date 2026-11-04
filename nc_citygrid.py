"""Citywide lookup grid for basement lots and stations.

The surge DEM (data/manhattan_dem_20m.tif, 0.00018 deg) covers 74.12-73.72 W, 40.53-40.92 N only. Lots are matched to
footprints by the grid cell under their centroid, so footprints defined everywhere (DEP rainfall maps, FEMA map, city)
are rasterised on the same cells extended to the five boroughs; inside the surge window the result is identical.
Surge masks exist only inside the window and are embedded with False outside it.
"""
import numpy as np, rasterio
from rasterio import features
from affine import Affine

with rasterio.open("data/manhattan_dem_20m.tif") as _r: TR_W, SHAPE_W = _r.transform, _r.shape
_W, _S, _E, _N = -74.27, 40.47, -73.69, 40.93                                  # five boroughs with a margin
C0 = int(np.floor((_W - TR_W.c) / TR_W.a)); R0 = int(np.floor((_N - TR_W.f) / TR_W.e))
TR_C = TR_W * Affine.translation(C0, R0)
SHAPE_C = (int(np.ceil((_S - TR_C.f) / TR_C.e)), int(np.ceil((_E - TR_C.c) / TR_C.a)))

def ras(geoms):
    return features.rasterize(((g, 1) for g in geoms if g is not None), out_shape=SHAPE_C, transform=TR_C, dtype="uint8").astype(bool)

def ras_val(pairs, dtype="float32"):
    """pairs of (geometry, value); later pairs overwrite earlier ones"""
    return features.rasterize(((g, v) for g, v in pairs if g is not None), out_shape=SHAPE_C, transform=TR_C, fill=0, dtype=dtype)

def embed(m, fill=0):
    o = np.full(SHAPE_C, fill, dtype=m.dtype); o[-R0:-R0 + SHAPE_W[0], -C0:-C0 + SHAPE_W[1]] = m; return o

def at(lon, lat, m):
    c = ((np.asarray(lon) - TR_C.c) / TR_C.a).astype(int); r = ((np.asarray(lat) - TR_C.f) / TR_C.e).astype(int)
    k = (r >= 0) & (r < SHAPE_C[0]) & (c >= 0) & (c < SHAPE_C[1]); o = np.zeros(len(c), m.dtype); o[k] = m[r[k], c[k]]; return o

def in_window(lon, lat):
    lon, lat = np.asarray(lon), np.asarray(lat)
    return (lon >= TR_W.c) & (lon < TR_W.c + SHAPE_W[1] * TR_W.a) & (lat <= TR_W.f) & (lat > TR_W.f + SHAPE_W[0] * TR_W.e)

def dep_depth(g, lon, lat, cats=((1, 0.20), (2, 0.46))):
    """DEP stormwater depth class at each point (m): 0.20 for 4 in-1 ft, 0.46 for >1 ft, on the citywide cells"""
    d = np.zeros(len(lon))
    for cat, dep in cats: d = np.maximum(d, np.where(at(lon, lat, ras(g[g.Flooding_Category == cat].geometry)), dep, 0.0))
    return d
