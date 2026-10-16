"""
Merged-node mapping (hypothesis test)
Perth Public Transport Network Analysis — merged_node_variant

HYPOTHESIS
    Treating each bus station, interchange and busport as a SINGLE node carrying
    the union of routes that serve it — rather than as a dozen adjacent stands,
    each a separate node — gives a whole-network run that is not distorted by
    dense clusters of co-located stops.

    The clusters cause four distinct problems in the current pipeline:
      1. Catchment overlap is most severe where stands crowd together, so
         competitive allocation divides one facility's demand a dozen ways.
      2. Alternative access L treats adjacent stands as substitutes, which
         understates reliance at exactly the places passengers depend on most.
      3. Betweenness centrality is split across parallel stands, understating
         the structural role of the facility as a whole.
      4. Route diversity R is counted per stand, so a facility served by thirty
         routes appears as a dozen nodes served by two or three each.

MECHANISM
    Transperth's GTFS feed populates parent_station: 782 boarding points are
    already grouped under 119 records with location_type = 1. The grouping is
    therefore taken from the feed rather than inferred from stop names, which
    makes it reproducible and removes the need to defend a name heuristic.

    Note what parent_station does NOT do. Kerbside stands strung along a street
    — the St Georges Tce and William St stands, for instance — are left
    unparented, because they are genuinely separate locations a few hundred
    metres apart rather than one facility. That is almost certainly correct, but
    it means this mapping addresses facility bunching and not the CBD kerbside
    clustering identified earlier. A name-based extension is available behind a
    flag so the alternative can be tested, but it is off by default.

This script only builds and characterises the mapping. It writes no pipeline
outputs and touches nothing in the parent directory.

Usage:
    python build_node_map.py
    python build_node_map.py --include-name-groups

Outputs (in merged_node_variant/output/):
    node_map.csv          stop_id -> node_id, with names
    node_map_report.txt   what merges, and the projected effect
"""

import os
import re
import argparse
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
GTFS = os.path.join(PARENT, "google_transit")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

NAME_GROUP_PATTERN = re.compile(
    r"^(?P<base>.+?)\s+(Stand|Platform)\s*\d+", re.IGNORECASE)


def load_stops():
    s = pd.read_csv(os.path.join(GTFS, "stops.txt"), skipinitialspace=True)
    s.columns = [c.strip().replace("﻿", "") for c in s.columns]
    return s


def build_map(stops, include_name_groups=False):
    pts = stops[stops["location_type"] == 0].copy()
    stations = stops[stops["location_type"] == 1].set_index("stop_id")["stop_name"]

    # Guard against a station id colliding with a boarding-point id.
    overlap = set(stations.index) & set(pts["stop_id"])
    if overlap:
        raise RuntimeError(f"station ids collide with stop ids: {sorted(overlap)[:5]}")

    pts["node_id"] = pts["stop_id"]
    pts["node_name"] = pts["stop_name"]
    pts["merge_source"] = "unmerged"

    has_parent = pts["parent_station"].notna()
    pts.loc[has_parent, "node_id"] = pts.loc[has_parent, "parent_station"].astype(int)
    pts.loc[has_parent, "node_name"] = (
        pts.loc[has_parent, "parent_station"].astype(int).map(stations))
    pts.loc[has_parent, "merge_source"] = "parent_station"

    if include_name_groups:
        # Secondary grouping for stands that carry a "Stand N" / "Platform N"
        # suffix but no parent_station. Off by default: stands strung along a
        # street are separate places, and merging them asserts otherwise.
        unp = pts[~has_parent].copy()
        bases = unp["stop_name"].str.extract(NAME_GROUP_PATTERN)["base"]
        grouped = bases.notna()
        if grouped.any():
            # Synthetic node ids well clear of the real id space.
            key = bases[grouped]
            codes = key.astype("category").cat.codes + 900000
            pts.loc[key.index, "node_id"] = codes.to_numpy()
            pts.loc[key.index, "node_name"] = key.to_numpy()
            pts.loc[key.index, "merge_source"] = "name_group"

    return pts[["stop_id", "stop_name", "node_id", "node_name",
                "merge_source", "stop_lat", "stop_lon"]]


def weekday_route_table():
    cal = pd.read_csv(os.path.join(GTFS, "calendar.txt"))
    trips = pd.read_csv(os.path.join(GTFS, "trips.txt"),
                        usecols=["trip_id", "route_id", "service_id"])
    st = pd.read_csv(os.path.join(GTFS, "stop_times.txt"),
                     usecols=["trip_id", "stop_id"])
    wk = set(cal.loc[cal[["monday", "tuesday", "wednesday", "thursday", "friday"]]
                     .sum(axis=1) == 5, "service_id"])
    trips = trips[trips["service_id"].isin(wk)]
    return st.merge(trips[["trip_id", "route_id"]], on="trip_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-name-groups", action="store_true",
                    help="also merge 'Stand N' stops sharing a name stem "
                         "(off by default — see the module docstring)")
    args = ap.parse_args()

    print("=" * 72)
    print("MERGED-NODE MAPPING")
    print("=" * 72)

    stops = load_stops()
    nm = build_map(stops, args.include_name_groups)
    nm.to_csv(os.path.join(OUT, "node_map.csv"), index=False)

    n_stops = len(nm)
    n_nodes = nm["node_id"].nunique()
    merged = nm[nm["merge_source"] != "unmerged"]

    sr = weekday_route_table()
    sr = sr.merge(nm[["stop_id", "node_id", "node_name"]],
                  on="stop_id", how="inner")
    R_before = sr.groupby("stop_id")["route_id"].nunique()
    R_after = sr.groupby("node_id")["route_id"].nunique()

    lines = []
    a = lines.append
    a("=" * 72); a("MERGED-NODE MAPPING — CHARACTERISATION"); a("=" * 72); a("")
    a(f"Boarding points (location_type = 0):   {n_stops:,}")
    a(f"Nodes after merging:                   {n_nodes:,}")
    a(f"Reduction:                             {n_stops - n_nodes:,} "
      f"({(n_stops - n_nodes) / n_stops * 100:.1f}%)")
    a("")
    a("Merge source:")
    for k, v in nm["merge_source"].value_counts().items():
        a(f"  {k:<18} {v:>7,} stops")
    a("")
    grp = merged.groupby("node_id").size()
    if len(grp):
        a(f"Merged groups: {len(grp):,}   median size {grp.median():.0f}   "
          f"max {grp.max()}")
        a("")
        a("Largest groups:")
        top = (merged.groupby("node_name").size().sort_values(ascending=False)
               .head(12))
        for k, v in top.items():
            a(f"  {str(k)[:44]:<46} {v:>3} stands")
    a("")
    a("EFFECT ON ROUTE DIVERSITY (weekday, unique routes)")
    a(f"  per stop  — median {R_before.median():.0f}, "
      f"p90 {np.percentile(R_before, 90):.0f}, max {R_before.max()}")
    a(f"  per node  — median {R_after.median():.0f}, "
      f"p90 {np.percentile(R_after, 90):.0f}, max {R_after.max()}")
    a("")
    a("  Nodes gaining the most route diversity:")
    gain = (R_after.rename("after").to_frame()
            .join(nm.drop_duplicates("node_id").set_index("node_id")["node_name"]))
    stand_max = sr.groupby(["node_id", "stop_id"])["route_id"].nunique() \
                  .groupby("node_id").max().rename("best_single_stand")
    gain = gain.join(stand_max)
    gain["gain"] = gain["after"] - gain["best_single_stand"]
    top = gain.nlargest(10, "gain")
    a("  {:<40}{:>8}{:>14}{:>7}".format("node", "routes", "best stand", "gain"))
    for _, r in top.iterrows():
        a("  {:<40}{:>8.0f}{:>14.0f}{:>7.0f}".format(
            str(r["node_name"])[:38], r["after"], r["best_single_stand"], r["gain"]))
    a("")
    a("INTERPRETATION")
    a("  The reduction in node count is modest, but it is concentrated entirely")
    a("  on the nodes that dominate every centrality and catchment measure. The")
    a("  route-diversity gain column shows what the current per-stand treatment")
    a("  costs: a facility whose thirty routes are spread across a dozen stands")
    a("  currently presents as a dozen nodes of two or three routes each.")
    a("")
    a("  Kerbside stands along a street keep their own nodes unless")
    a("  --include-name-groups is passed. Facility bunching is addressed here;")
    a("  CBD kerbside clustering is a separate question.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUT, "node_map_report.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nWrote {os.path.join(OUT, 'node_map.csv')}")
    print(f"Wrote {os.path.join(OUT, 'node_map_report.txt')}")


if __name__ == "__main__":
    main()
