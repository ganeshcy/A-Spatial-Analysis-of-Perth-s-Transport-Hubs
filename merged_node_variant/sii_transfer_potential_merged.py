"""
Spatial transfer potential (T_spatial) with a distance sweep
Perth Public Transport Network Analysis

WHY THIS EXISTS
    T as originally implemented counts routes reachable through a relation listed
    in the GTFS transfers.txt file. In Transperth's feed that file is an
    operational artefact: 95.4% of its 5,086 pairs have both ends sharing a
    parent_station, with a median minimum transfer time of 179 seconds. It
    documents walking between stands INSIDE one interchange, not connections
    between separate places. Consequently T is non-zero at 0.8% of stops in the
    per-stand run and 0.17% in the merged run, and dropping it entirely changes
    the index ranking by a mean of 1.9 places out of 11,773.

    A passenger's transfer potential is not that. It is whether a stop on a
    different route is a short walk away. This computes that directly.

        T_spatial(i, d) = | routes at stops within d metres' WALK of i
                            − routes already at i |

DEFINITION OF THE THRESHOLD — three variants, which are not interchangeable
    (1) NETWORK WALK DISTANCE (what this script computes)
        Distance from stop i's snapped walk-graph node to stop j's snapped node,
        along the pedestrian network. Respects severance. Directly readable as
        "j is within d metres' walk of i".

    (2) CATCHMENT-POLYGON CONTAINMENT (the first estimate produced, 37% at 400 m)
        Stop j's point falls inside stop i's A-catchment polygon. Because that
        polygon is the reachable street network buffered by EDGE_BUFFER_M, the
        effective test is "j lies within 40 m of a street reachable within 400 m
        of i" — which is not the same as "j is within 400 m walk of i". A stop
        60 m back from a reachable street is excluded even at 380 m walk.

    (3) EUCLIDEAN (the first sweep produced, 51% at 400 m)
        Straight-line distance. Ignores severance entirely — counts neighbours
        across freeways, rail reserves and the river. An upper bound.

    A_i uses (2) by construction, because population is apportioned by area
    overlap with that polygon. T should use (1): it concerns a walk between two
    points, not the land a stop serves, so the band half-width that A depends on
    is irrelevant to it.

EFFICIENCY
    One bounded Dijkstra per stop at the LARGEST threshold, recording the network
    distance to every stop reached. Every smaller threshold is then a filter on
    that table, so the sweep costs one pass rather than one per threshold.

Usage:
    python sii_transfer_potential.py
    python sii_transfer_potential.py --thresholds 100 150 200 250 300 350 400
    python sii_transfer_potential.py --adopt 200      # writes T_spatial at 200 m

Outputs (under output/):
    transfer_threshold_sensitivity.csv   one row per threshold
    transfer_potential_spatial.csv       per-stop T_spatial at each threshold
    transfer_potential.txt               readable summary
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
OUTPUT_DIR = os.path.join(BASE, "output")
GTFS = os.path.join(PARENT, "google_transit")
CACHE = os.path.join(PARENT, "data", "cache")
GRAPHML = os.path.join(CACHE, "perth_walk.graphml")

CRS_METRIC = "EPSG:7850"
DEFAULT_THRESHOLDS = [100, 150, 200, 250, 300, 350, 400]
MAX_SNAP_M = 100.0          # same guard as the catchment stage


def weekday_routes(node_map=None):
    """Set of weekday route_ids serving each stop (or merged node)."""
    cal = pd.read_csv(os.path.join(GTFS, "calendar.txt"))
    trips = pd.read_csv(os.path.join(GTFS, "trips.txt"),
                        usecols=["trip_id", "route_id", "service_id"])
    st = pd.read_csv(os.path.join(GTFS, "stop_times.txt"),
                     usecols=["trip_id", "stop_id"])
    wk = set(cal.loc[cal[["monday", "tuesday", "wednesday", "thursday", "friday"]]
                     .sum(axis=1) == 5, "service_id"])
    trips = trips[trips["service_id"].isin(wk)]
    if node_map is not None:
        st["stop_id"] = st["stop_id"].map(node_map).fillna(st["stop_id"]).astype(int)
    m = st.merge(trips[["trip_id", "route_id"]], on="trip_id")
    return m.drop_duplicates(["stop_id", "route_id"]).groupby("stop_id")["route_id"].agg(set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=int, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--adopt", type=int, default=None,
                    help="write T_spatial at this threshold as the operative column")
    args = ap.parse_args()

    import geopandas as gpd
    import networkx as nx
    import osmnx as ox

    sii_p = os.path.join(OUTPUT_DIR, "sii_bus_full.csv")
    if not os.path.exists(sii_p) or not os.path.exists(GRAPHML):
        print(f"Need {sii_p} and {GRAPHML}.\nRun the catchment stage first "
              f"(it caches the walk graph).")
        sys.exit(1)

    nm_p = os.path.join(OUTPUT_DIR, "node_map.csv")
    node_map = None
    if os.path.exists(nm_p):
        nm = pd.read_csv(nm_p)
        node_map = dict(zip(nm.stop_id, nm.node_id))
        print("  merged-node map found — computing on merged nodes")

    routes = weekday_routes(node_map)
    d = pd.read_csv(sii_p, low_memory=False).dropna(subset=["lon", "lat"])
    print(f"  stops: {len(d):,}")

    print("  loading cached walk graph (slow) ...")
    G = ox.project_graph(ox.load_graphml(GRAPHML), to_crs=CRS_METRIC)

    # Same edge-based snapping as the catchment stage — see walk_snap.__doc__
    # and AUDIT_CHECKLIST item 9. Both stages import the one module so they
    # cannot drift apart.
    import walk_snap as ws
    GU, nodes_xy, edge_geom, node_edges, edge_name = ws.build_undirected(G)
    ws.set_geometry_cache(edge_geom)
    print(f"    {GU.number_of_nodes():,} nodes / {GU.number_of_edges():,} edges")

    pts = gpd.GeoDataFrame(d[["stopnumber"]],
                           geometry=gpd.points_from_xy(d.lon, d.lat),
                           crs="EPSG:4326").to_crs(CRS_METRIC)
    snaps = ws.snap_stops(pts.geometry, nodes_xy, edge_geom, edge_name)
    snap_d = np.array([s.snap_dist for s in snaps])
    ok = snap_d <= MAX_SNAP_M
    print(f"    {int((~ok).sum()):,} stops snap further than {MAX_SNAP_M:.0f} m — excluded")

    ids = d["stopnumber"].to_numpy()

    # Stops no longer sit ON nodes, so a target stop j is reached via either end
    # of its own host edge. Index (node -> [(stop, metres from node to stop)])
    # and take the minimum over the two ends.
    from collections import defaultdict
    from_node = defaultdict(list)
    for j, s in enumerate(snaps):
        if not ok[j]:
            continue
        from_node[s.u].append((j, s.d_u))
        from_node[s.v].append((j, s.d_v))

    DMAX = float(max(args.thresholds))
    print(f"  one bounded Dijkstra per stop at {DMAX:.0f} m ...")
    pair_i, pair_j, pair_d = [], [], []
    for i, snap in enumerate(snaps):
        if not ok[i]:
            continue
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1:,} / {len(ids):,}")
        # Searching to DMAX is sufficient: the distance to stop j is the min
        # over its two host endpoints of (dist to endpoint + offset), and if
        # the minimising endpoint lies beyond DMAX then so does stop j, since
        # offsets are non-negative.
        dist = ws.dijkstra_from_snap(GU, snap, DMAX)
        best = {}
        for n, dn in dist.items():
            for j, off in from_node.get(n, ()):
                if j == i:
                    continue
                tot = dn + off
                if tot <= DMAX and (j not in best or tot < best[j]):
                    best[j] = tot
        for j, tot in best.items():
            pair_i.append(i); pair_j.append(j); pair_d.append(tot)

    pairs = pd.DataFrame({"i": pair_i, "j": pair_j, "d": pair_d})
    print(f"    {len(pairs):,} ordered stop pairs within {DMAX:.0f} m walk")

    own = [routes.get(s, set()) for s in ids]
    out = d[["stopnumber", "stopname", "suburb"]].copy()
    rows = []
    for thr in sorted(args.thresholds):
        sub = pairs[pairs["d"] <= thr]
        T = np.zeros(len(ids)); NB = np.zeros(len(ids))
        for i, grp in sub.groupby("i")["j"]:
            acc = set()
            for j in grp:
                acc |= routes.get(ids[j], set())
            T[i] = len(acc - own[i]); NB[i] = len(grp)
        col = f"T_spatial_{thr}"
        out[col] = T
        out[f"n_neighbours_{thr}"] = NB
        s = pd.Series(T)
        wt = pd.to_numeric(d["sched_trips_weekday"], errors="coerce").reset_index(drop=True)
        rows.append({
            "threshold_m": thr,
            "nodes_T_gt0": int((s > 0).sum()),
            "coverage_pct": round((s > 0).mean() * 100, 1),
            "median_T": float(s.median()),
            "mean_T": round(float(s.mean()), 2),
            "max_T": int(s.max()),
            "mean_neighbours": round(float(NB.mean()), 2),
            "spearman_vs_boardings": round(float(s.corr(wt, method="spearman")), 4),
            "corr_with_neighbour_count": round(float(s.corr(pd.Series(NB))), 3),
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUTPUT_DIR, "transfer_threshold_sensitivity.csv"), index=False)

    if args.adopt:
        out["T_spatial"] = out[f"T_spatial_{args.adopt}"]
        print(f"\n  adopted threshold: {args.adopt} m -> column T_spatial")
    out.to_csv(os.path.join(OUTPUT_DIR, "transfer_potential_spatial.csv"), index=False)

    txt = ["=" * 72, "SPATIAL TRANSFER POTENTIAL — THRESHOLD SENSITIVITY", "=" * 72, "",
           "Threshold is NETWORK WALK DISTANCE between stops, not Euclidean, and",
           "not containment in the A-catchment polygon. See the module docstring:",
           "the three definitions give materially different coverage.", "",
           tbl.to_string(index=False), "",
           "corr_with_neighbour_count is the share of T explained by stop density",
           "alone. It should inform the threshold choice: a shorter threshold is",
           "more behaviourally defensible for a mid-journey walk and carries less",
           "of that confound."]
    text = "\n".join(txt)
    print("\n" + text)
    with open(os.path.join(OUTPUT_DIR, "transfer_potential.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
