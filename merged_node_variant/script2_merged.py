"""
Script 2: Build Edge List from GTFS stop_times
Perth Public Transport Network Analysis

Constructs a weighted directed edge list where:
  - Each node is a stop (STOPID)
  - Each edge is a consecutive stop-pair on a trip
  - Edge weight = number of trips traversing that connection per day
"""

# --- merged_node_variant path + node-map header (inserted) -----------------
# The original of this script carried absolute paths to an earlier project
# location that no longer exists. Paths are now derived from this file.
import os as _os
import pandas as _pd
_BASE   = _os.path.dirname(_os.path.abspath(__file__))
_PARENT = _os.path.dirname(_BASE)
GTFS_DIR    = _os.path.join(_PARENT, "google_transit")
SPATIAL_DIR = _os.path.join(_PARENT, "Spatial_Data")
OUT_DIR     = _os.path.join(_BASE, "output")
_os.makedirs(OUT_DIR, exist_ok=True)

def load_node_map():
    """stop_id -> node_id, from build_node_map.py. Run that first."""
    p = _os.path.join(OUT_DIR, "node_map.csv")
    if not _os.path.exists(p):
        raise SystemExit("Run build_node_map.py first (node_map.csv missing).")
    nm = _pd.read_csv(p)
    return dict(zip(nm.stop_id, nm.node_id)), nm

NODE_MAP, NODE_MAP_DF = load_node_map()

def to_node(series):
    """Remap a Series of stop_ids onto merged node ids."""
    return series.map(NODE_MAP).fillna(series).astype(int)
# ---------------------------------------------------------------------------


import pandas as pd

GTFS_PATH = _os.path.join(GTFS_DIR, "")

# ── 1. Load required GTFS files ───────────────────────────────────────────────
print("Loading GTFS files...")
stop_times = pd.read_csv(GTFS_PATH + "stop_times.txt",
                         usecols=['trip_id','stop_id','stop_sequence'])
trips      = pd.read_csv(GTFS_PATH + "trips.txt",
                         usecols=['trip_id','route_id','service_id'])
routes     = pd.read_csv(GTFS_PATH + "routes.txt",
                         usecols=['route_id','route_type','route_short_name'])
calendar   = pd.read_csv(GTFS_PATH + "calendar.txt")

print(f"  stop_times: {len(stop_times):,} rows")
print(f"  trips:      {len(trips):,} rows")

# ── 2. Filter to weekday trips only ──────────────────────────────────────────
# GTFS route_type codes: 0=Tram, 1=Subway, 2=Rail, 3=Bus, 4=Ferry
weekday_services = calendar[
    (calendar['monday'] == 1) | (calendar['tuesday'] == 1)
]['service_id']

weekday_trips = trips[trips['service_id'].isin(weekday_services)]['trip_id']
stop_times_wd = stop_times[stop_times['trip_id'].isin(weekday_trips)].copy()
print(f"\nWeekday trips: {len(weekday_trips):,}")
print(f"Weekday stop_times rows: {len(stop_times_wd):,}")

# ── 3. Build consecutive stop pairs ──────────────────────────────────────────
print("\nBuilding consecutive stop pairs...")
# --- merged-node remap ------------------------------------------------------
# Collapse each facility's stands onto a single node BEFORE forming consecutive
# pairs, so that a trip calling at two stands of the same interchange produces
# one node rather than an edge between them.
_before = stop_times_wd['stop_id'].nunique()
stop_times_wd['stop_id'] = to_node(stop_times_wd['stop_id'])
print(f"  merged-node remap: {_before:,} stop ids -> "
      f"{stop_times_wd['stop_id'].nunique():,} node ids")

stop_times_wd = stop_times_wd.sort_values(['trip_id', 'stop_sequence'])

# Shift to get the next stop in each trip
stop_times_wd['next_stop'] = (
    stop_times_wd.groupby('trip_id')['stop_id'].shift(-1)
)

# Drop the last stop of each trip (no next stop)
edges_raw = stop_times_wd.dropna(subset=['next_stop']).copy()
edges_raw['next_stop'] = edges_raw['next_stop'].astype(int)

# A trip calling at two stands of the same merged facility now yields an edge
# from a node to itself. These are not movements through the network and would
# inflate the degree and PageRank of exactly the nodes the merge is meant to
# represent fairly.
_loops = int((edges_raw['stop_id'] == edges_raw['next_stop']).sum())
edges_raw = edges_raw[edges_raw['stop_id'] != edges_raw['next_stop']].copy()
print(f"  dropped {_loops:,} intra-facility self-loops")
edges_raw['stop_id']   = edges_raw['stop_id'].astype(int)

# ── 4. Count trips per edge ───────────────────────────────────────────────────
edge_weights = (
    edges_raw.groupby(['stop_id', 'next_stop'])
    .size()
    .reset_index(name='trip_count')
    .rename(columns={'stop_id': 'from_stop', 'next_stop': 'to_stop'})
)

print(f"Unique edges (directed): {len(edge_weights):,}")
print(f"Max trips on a single edge: {edge_weights['trip_count'].max()}")
print(f"Median trips per edge: {edge_weights['trip_count'].median():.1f}")
print()
print("Top 10 busiest connections:")
print(edge_weights.nlargest(10, 'trip_count').to_string())

# ── 5. Add route type annotation ──────────────────────────────────────────────
# Join route_type to edge for filtering (bus-only, rail-only etc.)
trip_routes = trips.merge(routes[['route_id','route_type','route_short_name']],
                           on='route_id')

edges_with_type = (
    edges_raw.merge(trip_routes[['trip_id','route_type']], on='trip_id')
    .groupby(['stop_id','next_stop','route_type'])
    .size()
    .reset_index(name='trip_count')
    .rename(columns={'stop_id': 'from_stop', 'next_stop': 'to_stop'})
)

print()
print("Edges by route_type (0=Tram,2=Rail,3=Bus,4=Ferry):")
print(edges_with_type.groupby('route_type')['trip_count'].agg(['count','sum']))

# ── 6. Save outputs ───────────────────────────────────────────────────────────
edge_weights.to_csv(_os.path.join(OUT_DIR, "edge_weights.csv"), index=False)
edges_with_type.to_csv(_os.path.join(OUT_DIR, "edge_weights_by_type.csv"), index=False)
print("\nSaved: output/edge_weights.csv")
print("Saved: output/edge_weights_by_type.csv")