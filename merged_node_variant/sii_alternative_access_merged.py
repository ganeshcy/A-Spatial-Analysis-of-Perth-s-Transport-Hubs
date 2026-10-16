"""
Alternative access (L_i)
Perth Public Transport Network Analysis

L_i measures how far a stop is from being substitutable: the degree to which the
transit access available within walking distance of it is concentrated at that
one stop rather than shared with neighbours. It is the component that makes the
composite a RELIANCE index rather than an influence index. A stop can be central,
frequent and surrounded by population and still be readily replaced because three
comparable stops sit a two-minute walk away; another can be modest on every
supply measure and still be the only option anyone in its catchment has.

    L_i = S_i / (S_i + SUM_{j in walk(i), j != i} S_j)

where S is total scheduled weekday departures and walk(i) is the set of stops
falling inside stop i's walk catchment. L is bounded on [0,1]: near 0 where
walkable service is spread across many stops, 1 where this stop is the only
service anyone within walking distance can reach.

WHY NOT A THRESHOLD COUNT
    The obvious formulation counts neighbouring stops meeting a minimum service
    standard and normalises against a saturation count. Applied to Perth with the
    SNAMUTS standard of a 20 minute inter-peak headway [2], it collapses: only
    284 of 11,750 ordinary stops (2.4%) meet that standard at all, so the
    neighbour count is zero almost everywhere and L saturates at 1 for the great
    majority of the network. A variable that takes the same value for ninety per
    cent of cases cannot discriminate between them.

    That collapse is itself a result worth reporting — it reproduces Scheurer's
    benchmarking finding at stop level [2] — so the threshold counts are computed
    and written out as descriptive statistics. They are not used to build L.

    The service-share formulation avoids the problem because it is relative
    rather than absolute: it asks what proportion of locally available service
    this stop provides, which stays informative however sparse that service is.

Catchments are read from the GeoPackages written by sii_catchment_population.py,
so "within walking distance" means the same 400 m network catchment used for
A_i, and no walk-graph tracing is repeated here.

Usage:
    python sii_alternative_access.py
    python sii_alternative_access.py --saturation 3
    python sii_alternative_access.py --strict-routes

Outputs (under output/):
    alternative_access.csv        L and its inputs, per stop
    alternative_access.txt        summary and the service-standard statistics
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import importlib.util

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "output")
GEO_DIR = os.path.join(OUTPUT_DIR, "geojson")
GTFS = os.path.join(BASE, "google_transit")

_spec = importlib.util.spec_from_file_location(
    "cp", os.path.join(BASE, "sii_catchment_population_merged.py"))
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

# SNAMUTS minimum service standard: 20 minutes between services, weekday
# inter-peak. The inter-peak window used throughout is 09:00-16:00.
INTERPEAK_MINUTES = 7 * 60
SNAMUTS_HEADWAY_MIN = 20.0
SNAMUTS_DEPARTURES = INTERPEAK_MINUTES / SNAMUTS_HEADWAY_MIN   # 21.0

# Saturation count for the threshold-based comparison variant only.
DEFAULT_SATURATION = 3


def walk_distance_pairs(base, cutoff=None):
    """
    Stops within `cutoff` metres of each other by pedestrian NETWORK distance.

    Replaces the previous test, which asked whether stop j's point fell inside
    the polygon of stop i's catchment (the buffered union of reachable edges).
    That test was a proxy for "within walking distance" and a poor one. Audited
    on a 400-stop random sample against a direct walk measure it agreed exactly
    for only 47.5% of stops (Spearman 0.77), overcounting for 31.5% and
    undercounting for 21.0%, by a mean of 1.6 alternatives either way.

    Both errors follow from the polygon being a buffered line network:

      - OVERCOUNT: a stop lying within the 40 m band of some reachable edge is
        counted even when the walk to it exceeds the bound — it is near the
        corridor without being near the stop along it.
      - UNDERCOUNT: an edge enters the polygon only when BOTH its endpoints are
        inside the bound, so on a simplified graph with long arterial edges the
        segment carrying a genuine neighbour is dropped whole.

    The undercount is what makes this unusable for selecting captive stops.
    Ranking on alternative access selects for stops reporting few or no
    alternatives, which is exactly where the polygon fails low, so the error
    concentrates in the shortlist: across a random sample the polygon never
    spuriously reported zero alternatives, but among the 40 highest-reliance
    captive stops it did so twice (28842 Orrong Rd After Archer St, with
    neighbours at 280 m and 331 m by network; 25353 Hasler Rd After Teakle Rd).

    Measuring the pairwise distance directly removes the proxy. It is the same
    quantity, and the same method, that sii_transfer_potential_merged.py already
    uses for T — both now go through walk_snap, so a stop cannot be in i's
    catchment for one component and outside it for the other.

    Costs one bounded Dijkstra per stop and does NOT retrace any catchment, so
    this stage no longer depends on the GeoPackages at all (AUDIT_CHECKLIST
    item 10 — stale polygons used to be read silently).
    """
    import networkx as nx
    import osmnx as ox
    import geopandas as gpd
    from collections import defaultdict
    import walk_snap as ws

    if cutoff is None:
        cutoff = cp.CATCHMENT_M

    graph_path = os.path.join(cp.CACHE_DIR, "perth_walk.graphml")
    if not os.path.exists(graph_path):
        sys.exit(f"Missing {graph_path}. Run the catchment stage first "
                 f"(it caches the walk graph).")

    # ---- cache ------------------------------------------------------------
    # The pair set costs one bounded Dijkstra per stop over a 494k-edge graph,
    # and nothing about it changes unless the walk graph, the stop set or the
    # cutoff changes. Caching it makes re-running A with a different opposing
    # radius cheap. The key carries the mtime of BOTH inputs and the cutoff, and
    # a miss recomputes rather than falling back on whatever is on disk --
    # reading a stale intermediate without a vintage check is precisely what
    # went wrong with the catchment GeoPackages (AUDIT_CHECKLIST item 10) and
    # with critic_weights.csv, so the guard is not optional.
    sii_path = os.path.join(OUTPUT_DIR, "sii_bus_full.csv")
    key = "%.0f_%d_%d_%d" % (
        cutoff,
        int(os.path.getmtime(graph_path)),
        int(os.path.getmtime(sii_path)) if os.path.exists(sii_path) else 0,
        len(base),
    )
    cache_p = os.path.join(OUTPUT_DIR, f"_walk_pairs_{key}.csv")
    if os.path.exists(cache_p):
        got = pd.read_csv(cache_p)
        print(f"\n  Reusing cached walk pairs: {len(got):,} pairs "
              f"({os.path.basename(cache_p)})")
        return got
    for stale in [f for f in os.listdir(OUTPUT_DIR)
                  if f.startswith("_walk_pairs_") and f != os.path.basename(cache_p)]:
        print(f"  (cache key changed; ignoring {stale})")

    print(f"\n  Loading walk graph and measuring pairwise distances "
          f"at {cutoff:.0f} m ...")
    G = ox.project_graph(ox.load_graphml(graph_path), to_crs=cp.CRS_METRIC)
    GU, nodes_xy, edge_geom, node_edges, edge_name = ws.build_undirected(G)
    ws.set_geometry_cache(edge_geom)

    pts = gpd.GeoDataFrame(
        base[["stopnumber"]].copy(),
        geometry=gpd.points_from_xy(base["lon"], base["lat"]),
        crs="EPSG:4326").to_crs(cp.CRS_METRIC)
    snaps = ws.snap_stops(pts.geometry, nodes_xy, edge_geom, edge_name)
    ids = base["stopnumber"].to_numpy()

    # A stop is reached via either end of its own host edge; take the minimum.
    from_node = defaultdict(list)
    for j, s in enumerate(snaps):
        from_node[s.u].append((j, s.d_u))
        from_node[s.v].append((j, s.d_v))

    focal, neigh = [], []
    for i, snap in enumerate(snaps):
        if (i + 1) % 2000 == 0:
            print(f"      {i + 1:,} / {len(ids):,}")
        dist = ws.dijkstra_from_snap(GU, snap, cutoff)
        best = {}
        for n, dn in dist.items():
            for j, off in from_node.get(n, ()):
                if j == i:
                    continue
                tot = dn + off
                if tot <= cutoff and (j not in best or tot < best[j]):
                    best[j] = tot
        for j in best:
            focal.append(ids[i]); neigh.append(ids[j])

    # --- Fallback for stops whose walk trace is still unusable -------------
    # A stop on a broken or isolated fragment of the walk graph finds no
    # neighbours and would score A = 1.000 by construction. That is the failure
    # the radial fallback has always existed to prevent; it is simply applied
    # per PAIR here instead of per polygon. The edge-snapping fix cut this
    # population from 682 stops to 140, but it did not empty it.
    cp_path = os.path.join(OUTPUT_DIR, "sii_bus_full_with_catchment.csv")
    if not os.path.exists(cp_path):
        sys.exit("Missing sii_bus_full_with_catchment.csv — needed for "
                 "catchment_suspect, which marks the stops whose walk trace is "
                 "unusable. Run sii_catchment_population_merged.py first.")
    sus = pd.read_csv(cp_path, usecols=["stopnumber", "catchment_suspect"],
                      low_memory=False)
    sus = set(sus.loc[sus["catchment_suspect"].astype(bool), "stopnumber"])
    if not sus:
        sys.exit("No catchment_suspect stops found — the column did not join. "
                 "Refusing to continue: every broken-trace stop would silently "
                 "score A = 1.000. (This is the item-9 bug in its original form.)")
    print(f"    {len(sus):,} stops have an unusable walk trace — "
          f"supplementing those with a straight-line radius")

    XY = np.c_[pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()]
    from scipy.spatial import cKDTree
    tree = cKDTree(XY)
    pos = {s: k for k, s in enumerate(ids)}
    fb_focal, fb_neigh = [], []
    for s in sus:
        k = pos.get(s)
        if k is None:
            continue
        for j in tree.query_ball_point(XY[k], cutoff):
            if j != k:
                fb_focal.append(s); fb_neigh.append(ids[j])

    pairs = pd.DataFrame({"focal": focal + fb_focal,
                          "neighbour": neigh + fb_neigh})
    # A suspect stop may appear in both sets; keep one row per pair.
    pairs = pairs.drop_duplicates().reset_index(drop=True)
    print(f"    {len(pairs):,} focal/neighbour pairs "
          f"({len(fb_focal):,} from the straight-line fallback)")
    pairs.to_csv(cache_p, index=False)
    print(f"    cached to {os.path.basename(cache_p)}")
    return pairs


def load_catchments(base):
    """
    DEPRECATED — no longer called. Superseded by walk_distance_pairs(), which
    measures stop-to-stop network distance directly instead of testing polygon
    containment. See that function's docstring for the audit, and
    AUDIT_CHECKLIST item 10 for why reading these GeoPackages was unsafe (they
    are written only under --export-polygons, so a stale pair was used silently).

    Kept for reference only. Do not reinstate without re-reading that audit.

    Canonical catchment per stop: network, radial where flagged.
    """
    net_p = os.path.join(GEO_DIR, "catchments_network.gpkg")
    rad_p = os.path.join(GEO_DIR, "catchments_radial.gpkg")
    for p in (net_p, rad_p):
        if not os.path.exists(p):
            print(f"Missing {p}\nRun sii_catchment_population_merged.py "
                  f"--method both --dasymetric --export-polygons first.")
            sys.exit(1)

    net = gpd.read_file(net_p).to_crs(cp.CRS_METRIC)
    rad = gpd.read_file(rad_p).to_crs(cp.CRS_METRIC)
    # A_source lives in sii_bus_full_with_catchment.csv, not in the pipeline's
    # own output, so it is joined here rather than assumed present. Without it
    # the fallback silently does not happen and every stop whose walk trace
    # failed is scored against a broken polygon: Karrinyup Rd After Cedric St
    # traces a catchment of 5,018 m2 — a 40 m radius — which contains no other
    # stop, so L = 1.0 by construction rather than by measurement. 682 stops are
    # flagged, and 303 of the 733 stops with no alternative at all are among
    # them. This is not optional.
    if "A_source" not in base.columns:
        cp_path = os.path.join(OUTPUT_DIR, "sii_bus_full_with_catchment.csv")
        if not os.path.exists(cp_path):
            sys.exit("Missing sii_bus_full_with_catchment.csv — needed for "
                     "A_source, which marks the stops whose walk catchment "
                     "failed and must use the radial fallback.")
        src = pd.read_csv(cp_path, usecols=["stopnumber", "A_source"],
                          low_memory=False)
        base = base.merge(src, on="stopnumber", how="left")
    fb = set(base.loc[base["A_source"] == "radial_fallback", "stopnumber"])
    if not fb:
        sys.exit("No radial-fallback stops found — A_source did not join. "
                 "Refusing to run, because silently skipping the fallback "
                 "sets L = 1.0 for every stop with a failed walk trace.")
    cat = pd.concat([net[~net["stopnumber"].isin(fb)],
                     rad[rad["stopnumber"].isin(fb)]], ignore_index=True)
    cat = gpd.GeoDataFrame(cat, geometry="geometry", crs=cp.CRS_METRIC)
    print(f"  Catchments: {len(cat):,} ({len(fb):,} radial fallback)")
    return cat


def stop_routes():
    """Set of weekday route_ids serving each stop, for the strict variant."""
    cal = pd.read_csv(os.path.join(GTFS, "calendar.txt"))
    trips = pd.read_csv(os.path.join(GTFS, "trips.txt"),
                        usecols=["trip_id", "route_id", "service_id"])
    st = pd.read_csv(os.path.join(GTFS, "stop_times.txt"),
                     usecols=["trip_id", "stop_id"])
    wk = set(cal.loc[cal[["monday", "tuesday", "wednesday", "thursday", "friday"]]
                     .sum(axis=1) == 5, "service_id"])
    trips = trips[trips["service_id"].isin(wk)]
    m = st.merge(trips[["trip_id", "route_id"]], on="trip_id")
    return m.groupby("stop_id")["route_id"].agg(set)


def stop_route_directions(node_map=None):
    """
    {stopnumber: {route_id: {direction_id, ...}}} over weekday service.

    Note the weekday filter here is "operates on AT LEAST ONE weekday", which is
    the convention used throughout the pipeline. stop_routes() above instead
    requires all five, which silently drops Friday-only and other partial-week
    patterns; the two should be reconciled, but stop_routes() feeds only the
    optional --strict-routes comparison and is left alone for now.
    """
    cal = pd.read_csv(os.path.join(GTFS, "calendar.txt"))
    wk = set(cal.loc[cal[["monday", "tuesday", "wednesday", "thursday", "friday"]]
                     .sum(axis=1) > 0, "service_id"].astype(str))
    trips = pd.read_csv(os.path.join(GTFS, "trips.txt"),
                        usecols=["trip_id", "route_id", "direction_id",
                                 "service_id"])
    trips["service_id"] = trips["service_id"].astype(str)
    trips = trips[trips["service_id"].isin(wk)]
    st = pd.read_csv(os.path.join(GTFS, "stop_times.txt"),
                     usecols=["trip_id", "stop_id"], low_memory=False)
    m = st.merge(trips[["trip_id", "route_id", "direction_id"]], on="trip_id")
    m["stop_id"] = m["stop_id"].astype(str)
    if node_map:
        m["stop_id"] = m["stop_id"].map(lambda s: str(node_map.get(s, s)))

    out = {}
    for (sid, rid), grp in m.groupby(["stop_id", "route_id"], sort=False):
        out.setdefault(sid, {})[rid] = set(grp["direction_id"].unique())
    return out


def opposing_same_route_pairs(base, radius=200.0, node_map=None):
    """
    Same-route, opposite-direction stops within `radius` metres STRAIGHT-LINE.

    Why straight-line and not network distance. These are the pairs that face
    one another across a carriageway. The walk graph routes between them via the
    nearest mapped crossing, which on a divided arterial can be several hundred
    metres away and is sometimes not mapped at all: at stop 16999 the opposing
    stop is 26 m away and 409 m by network; at 28836 it is 32 m away and 1,025 m
    by network. Field checking established that at least one of those crossings
    exists on the ground and is simply absent from OpenStreetMap.

    A resident crosses the road. Measuring these pairs through the pedestrian
    graph therefore answers a question about crossing infrastructure and the
    completeness of its mapping, not about whether a substitute exists, so the
    straight line is the better measure for this one purpose. The radius is kept
    tight (200 m) so that only the facing stop is admitted and not the next stop
    along the street.

    A pair qualifies when the neighbour serves some route in a direction the
    focal stop does not serve it in. A stop already serving both directions of a
    route gains nothing from the stop opposite, and correctly picks up no pair
    on that route's account.
    """
    from scipy.spatial import cKDTree
    import geopandas as gpd

    rd = stop_route_directions(node_map)
    ids = base["stopnumber"].to_numpy()
    pts = gpd.GeoDataFrame(
        base[["stopnumber"]],
        geometry=gpd.points_from_xy(base["lon"], base["lat"]),
        crs="EPSG:4326").to_crs(cp.CRS_METRIC)
    XY = np.c_[pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()]
    tree = cKDTree(XY)

    focal, neigh = [], []
    for k, s in enumerate(ids):
        here = rd.get(str(s))
        if not here:
            continue
        for j in tree.query_ball_point(XY[k], radius):
            if j == k:
                continue
            there = rd.get(str(ids[j]))
            if not there:
                continue
            for rid, dirs_here in here.items():
                dirs_there = there.get(rid)
                if dirs_there and (dirs_there - dirs_here):
                    focal.append(s); neigh.append(ids[j])
                    break

    out = pd.DataFrame({"focal": focal, "neighbour": neigh}).drop_duplicates()
    print(f"    {len(out):,} same-route opposite-direction pairs "
          f"within {radius:.0f} m straight-line")
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saturation", type=int, default=DEFAULT_SATURATION,
                    help="saturation count for the threshold comparison variant")
    ap.add_argument("--opposing-radius", type=float, default=200.0,
                    metavar="METRES",
                    help="straight-line radius within which a same-route, "
                         "opposite-direction stop counts as an alternative. "
                         "Set 0 to disable and recover the walk-only measure.")
    ap.add_argument("--strict-routes", action="store_true",
                    help="a neighbour only counts as an alternative if it serves "
                         "at least one route this stop does not")
    args = ap.parse_args()

    # Read the pipeline's own output, not sii_bus_full_with_catchment.csv. L needs
    # only stopnumber, the four frequency terms and lon/lat/stopname/suburb, every
    # one of which lives here; the catchment file adds nothing L uses and is
    # regenerated far less often, so reading it meant L was silently computed from
    # whatever vintage of the frequency terms happened to be on disk.
    sii_p = os.path.join(OUTPUT_DIR, "sii_bus_full.csv")
    if not os.path.exists(sii_p):
        print(f"Missing {sii_p}"); sys.exit(1)

    print("=" * 72)
    print("ALTERNATIVE ACCESS (L)")
    print("=" * 72)

    base = pd.read_csv(sii_p, low_memory=False)
    for c in ("F_AM", "F_PM", "F_off", "F_rem"):
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
    # S is ALL weekday departures, not only 07:00-18:00. The earlier definition
    # omitted 25.3% of weekday service, so a stop whose evening service was
    # strong relative to its neighbours received no credit for it -- and since L
    # is a ratio, that bias fell hardest exactly where day/evening mix differs
    # between a stop and its neighbours, which is at interchanges. See
    # AUDIT_CHECKLIST.md item 4.
    base["S"] = base["F_AM"] + base["F_PM"] + base["F_off"] + base["F_rem"]
    S = base.set_index("stopnumber")["S"]
    F_off = base.set_index("stopnumber")["F_off"]

    print(f"\n  Stops: {len(base):,}")
    print(f"  Weekday departures S: median {S.median():.1f}, max {S.max():.1f}")

    pairs = walk_distance_pairs(base)
    n_walk = len(pairs)

    # ---- opposing same-route stops across the carriageway -----------------
    # Added to the walk-network pairs rather than replacing them: a stop within
    # 400 m on foot remains an alternative whatever route it runs, and this adds
    # the facing stop that the walk graph misses. The union can therefore only
    # lower A, never raise it.
    if args.opposing_radius > 0:
        print("  Adding opposing same-route stops ...")
        nm_p = os.path.join(OUTPUT_DIR, "node_map.csv")
        node_map = None
        if os.path.exists(nm_p):
            nm = pd.read_csv(nm_p)
            node_map = dict(zip(nm["stop_id"].astype(str), nm["node_id"]))
        opp = opposing_same_route_pairs(base, args.opposing_radius, node_map)
        known = set(base["stopnumber"])
        opp = opp[opp["focal"].isin(known) & opp["neighbour"].isin(known)]
        pairs = (pd.concat([pairs[["focal", "neighbour"]], opp], ignore_index=True)
                 .drop_duplicates().reset_index(drop=True))
        print(f"    pairs: {n_walk:,} (walk) -> {len(pairs):,} after union "
              f"(+{len(pairs) - n_walk:,} not already within 400 m on foot)")

    if args.strict_routes:
        print("  Applying strict-routes filter ...")
        sr = stop_routes()
        keep = [
            bool(sr.get(n, set()) - sr.get(f, set()))
            for f, n in zip(pairs["focal"], pairs["neighbour"])
        ]
        before = len(pairs)
        pairs = pairs[keep].reset_index(drop=True)
        print(f"    {before:,} -> {len(pairs):,} pairs "
              f"(neighbour must add a route)")

    pairs["S_n"] = S.reindex(pairs["neighbour"]).to_numpy()
    pairs["meets_std"] = (F_off.reindex(pairs["neighbour"]).to_numpy()
                          >= SNAMUTS_DEPARTURES)

    agg = pairs.groupby("focal").agg(
        n_alternatives=("neighbour", "size"),
        neighbour_S=("S_n", "sum"),
        n_alt_meeting_standard=("meets_std", "sum"))

    out = base[["stopnumber", "stopname", "suburb", "S", "F_off"]].copy()
    out = out.merge(agg, left_on="stopnumber", right_index=True, how="left")
    for c in ("n_alternatives", "neighbour_S", "n_alt_meeting_standard"):
        out[c] = out[c].fillna(0)

    # --- L: this stop's share of all walkable service ----------------------
    denom = out["S"] + out["neighbour_S"]
    out["L"] = np.divide(out["S"].to_numpy(), denom.to_numpy(),
                         out=np.ones(len(out)), where=denom.to_numpy() > 0)
    # Where nothing runs anywhere within walking distance, the stop is not
    # substitutable but neither does it provide anything. Treated as fully
    # reliant, consistent with an isolated stop being the only thing there.
    out.loc[denom <= 0, "L"] = 1.0

    # --- threshold variant, for comparison only ----------------------------
    out["L_threshold"] = 1.0 - np.minimum(
        1.0, out["n_alt_meeting_standard"] / max(args.saturation, 1))

    out["L_z"] = ((out["L"] - out["L"].mean()) / out["L"].std(ddof=0)).clip(-3, 3)

    out.to_csv(os.path.join(OUTPUT_DIR, "alternative_access.csv"), index=False)

    # ---------------------------------------------------------------- report
    meets = int((base["F_off"] >= SNAMUTS_DEPARTURES).sum())
    lines = []
    a = lines.append
    a("=" * 72); a("ALTERNATIVE ACCESS (L)"); a("=" * 72); a("")
    a(f"Stops:                                {len(out):,}")
    a(f"Median alternatives within catchment: {out['n_alternatives'].median():.0f}")
    a(f"Stops with no alternative at all:     "
      f"{int((out['n_alternatives'] == 0).sum()):,} "
      f"({(out['n_alternatives'] == 0).mean() * 100:.1f}%)")
    a("")
    a("L (service-share formulation)")
    a(f"  mean {out['L'].mean():.3f}   median {out['L'].median():.3f}   "
      f"sd {out['L'].std():.3f}")
    for p in (10, 25, 50, 75, 90):
        a(f"    p{p:<3} {np.percentile(out['L'], p):.3f}")
    a(f"  stops at L = 1 (sole provider):      "
      f"{int((out['L'] >= 0.999).sum()):,} ({(out['L'] >= 0.999).mean() * 100:.1f}%)")
    a("")
    a("SNAMUTS minimum service standard (20 min inter-peak headway)")
    a(f"  Departures required in the 09:00-16:00 window: {SNAMUTS_DEPARTURES:.0f}")
    a(f"  Stops meeting the standard: {meets:,} of {len(base):,} "
      f"({meets / len(base) * 100:.1f}%)")
    a("")
    a("  This is why L is not built from a threshold count. With the standard")
    a("  met at 2.4% of stops, a neighbour count is zero almost everywhere and")
    a("  the threshold variant saturates:")
    a(f"    L_threshold = 1 at "
      f"{int((out['L_threshold'] >= 0.999).sum()):,} stops "
      f"({(out['L_threshold'] >= 0.999).mean() * 100:.1f}%)")
    a(f"    L (service share) = 1 at "
      f"{int((out['L'] >= 0.999).sum()):,} stops "
      f"({(out['L'] >= 0.999).mean() * 100:.1f}%)")
    a("")
    a("  The threshold collapse is a substantive finding in its own right: it")
    a("  reproduces at stop level the accessibility deficit Scheurer identifies")
    a("  for Perth against Copenhagen [2].")
    a("")
    a("Least substitutable stops with meaningful service (L = 1, highest S):")
    sole = out[(out["L"] >= 0.999) & (out["S"] > 0)].nlargest(10, "S")
    a(sole[["stopname", "suburb", "S", "n_alternatives"]].round(1)
      .to_string(index=False))
    a("")
    a("Most substitutable stops (lowest L):")
    a(out.nsmallest(10, "L")[["stopname", "suburb", "S", "n_alternatives", "L"]]
      .round(3).to_string(index=False))

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUTPUT_DIR, "alternative_access.txt"), "w") as f:
        f.write(text + "\n")
    print("\nWrote alternative_access.csv and alternative_access.txt")
    print("\nNote: the socio-economic blend described in the methodology needs")
    print("ABS Census table G34 (motor vehicles by dwelling). It is published")
    print("at SA2, not SA1, so it enters at the coarser resolution. Not yet")
    print("downloaded, so L here is service-based only.")


if __name__ == "__main__":
    main()
