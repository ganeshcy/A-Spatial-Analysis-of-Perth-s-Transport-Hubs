"""
Basemap geometry for the web explorer
Perth Public Transport Network Analysis — merged_node_variant

WHY THE BASEMAP IS VECTOR AND EMBEDDED, NOT TILED

    The explorer must render identically in two places: a sandboxed artifact
    view that blocks all network requests, and a public static host. A slippy
    tile layer works in the second and silently fails in the first, so the page
    would look broken in exactly the setting where it is most often opened.

    It would also make a thesis deliverable depend on a third-party tile server
    remaining free and available, and heavy embedding of OSM's own tiles is
    discouraged by their usage policy. A few hundred kilobytes of simplified
    vector geometry removes the dependency entirely: the page keeps its
    self-contained property and there is nothing to attribute, rate-limit or
    outlive the submission.

WHAT IT CONTAINS

    coast   The Greater Perth boundary, dissolved from the 185 SA2s of the
            GCCSA. Its western edge is the coastline; inland it is the
            metropolitan boundary, which is the correct extent for a map of
            this network in any case.
    water   Mesh blocks categorised Water, dissolved — the Swan and Canning
            rivers and the larger lakes. These are what make the map legible
            as Perth rather than as a generic point cloud.
    park    Mesh blocks categorised Parkland, dissolved, keeping only polygons
            above 4 km2: Kings Park, Bold Park, the coastal reserves and the
            Darling Range edge.
    edges   The scheduled network itself. Each consecutive-stop pair becomes an
            index pair into the node array of nodes.json, deduplicated and
            undirected. Edges touching a node outside the scored set are
            dropped, so this is the network as the index sees it.

    Geometry is simplified at 120-200 m and coordinates are rounded to four
    decimal places, about 11 m at this latitude. That is far below the width of
    a drawn stroke at any zoom the explorer offers.

Usage:
    python sii_basemap_export.py            # rerun only when geography changes
"""

import os
import json
import pickle
import importlib.util

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
WEB = os.path.join(OUT, "web")
DATA = os.path.join(BASE, "data", "abs")
ASGS = os.path.join(DATA, "ASGS_2021_Main_Structure_GDA2020.gpkg")

CRS_M = "EPSG:7850"
SIMPLIFY_COAST_M = 200.0
SIMPLIFY_WATER_M = 120.0
SIMPLIFY_PARK_M = 150.0
MIN_COAST_AREA = 4e6
MIN_WATER_AREA = 1.5e6
MIN_PARK_AREA = 4e6
PRECISION = 4


def _rings(geom, crs, min_area):
    import geopandas as gpd
    from shapely.geometry import Polygon
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    keep = [p for p in polys if p.area > min_area]
    if not keep:
        return []
    ser = gpd.GeoSeries(keep, crs=crs).to_crs("EPSG:4326")
    return [[[round(c[0], PRECISION), round(c[1], PRECISION)]
             for c in p.exterior.coords] for p in ser]


def main():
    import geopandas as gpd
    import pandas as pd
    from pyogrio import list_layers

    os.makedirs(WEB, exist_ok=True)
    print("=" * 70)
    print("BASEMAP GEOMETRY")
    print("=" * 70)

    d = pd.read_csv(os.path.join(OUT, "node_master.csv"), low_memory=False)
    bb = (d.lon.min() - .06, d.lat.min() - .06,
          d.lon.max() + .06, d.lat.max() + .06)

    # ------------------------------------------------------------- coastline
    lyr = [l[0] for l in list_layers(ASGS) if "SA2" in l[0]][0]
    sa2 = gpd.read_file(ASGS, layer=lyr, bbox=bb)
    gcc = [c for c in sa2.columns if c.startswith("GCC") and "NAME" in c]
    gp = (sa2[sa2[gcc[0]].astype(str).str.contains("Perth", na=False)]
          if gcc else sa2)
    print(f"  Greater Perth SA2s: {len(gp):,}")
    land = gp.to_crs(CRS_M).geometry.union_all().simplify(SIMPLIFY_COAST_M)
    coast = _rings(land, CRS_M, MIN_COAST_AREA)

    # --------------------------------------------------------- water, parks
    spec = importlib.util.spec_from_file_location(
        "cp", os.path.join(BASE, "sii_catchment_population_merged.py"))
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    mb = cp.load_mesh_blocks()

    wat = mb[mb.MB_CATEGORY_2021 == "Water"].to_crs(CRS_M)
    water = _rings(wat.geometry.union_all().simplify(SIMPLIFY_WATER_M),
                   CRS_M, MIN_WATER_AREA)
    par = mb[mb.MB_CATEGORY_2021 == "Parkland"].to_crs(CRS_M)
    park = _rings(par.geometry.union_all().simplify(SIMPLIFY_PARK_M),
                  CRS_M, MIN_PARK_AREA)

    # --------------------------------------------------------------- edges
    pay = json.load(open(os.path.join(WEB, "nodes.json")))
    idx = {sid: i for i, sid in enumerate(pay["id"])}
    with open(os.path.join(OUT, "graph_full.pkl"), "rb") as f:
        G = pickle.load(f)
    seen, E = set(), []
    for u, v, _ in G.edges(data=True):
        a, b = idx.get(int(u)), idx.get(int(v))
        if a is None or b is None or a == b:
            continue
        k = (a, b) if a < b else (b, a)
        if k in seen:
            continue
        seen.add(k)
        E += [k[0], k[1]]

    bm = {"coast": coast, "water": water, "park": park, "edges": E}
    p = os.path.join(WEB, "basemap.json")
    json.dump(bm, open(p, "w"), separators=(",", ":"))

    print(f"  coast  {len(coast):>3} rings  {sum(len(r) for r in coast):>5} pts")
    print(f"  water  {len(water):>3} rings  {sum(len(r) for r in water):>5} pts")
    print(f"  park   {len(park):>3} rings  {sum(len(r) for r in park):>5} pts")
    print(f"  edges  {len(E) // 2:>5} undirected, drawable within the node set "
          f"(graph holds {G.number_of_edges():,} directed)")
    print(f"\n  wrote {os.path.relpath(p, OUT)}  "
          f"{os.path.getsize(p) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
