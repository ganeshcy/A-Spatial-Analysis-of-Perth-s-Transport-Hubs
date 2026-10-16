"""
Script 3 (FINAL): Build NetworkX Graph
Perth Public Transport Network Analysis

Node sources:
  - Shapefile Stops.shp   → 16,535 bus/CAT/school stops (keyed by STOPNUMBER)
  - GTFS stops.txt        → 192 train platform stops (IDs 99000+, not in shapefile)

Edge sources:
  - GTFS stop_times.txt   → consecutive stop pairs per trip (bus + train routes)
  - GTFS transfers.txt    → intermodal transfer links (bus stop ↔ train platform)
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


import geopandas as gpd
import pandas as pd
import networkx as nx
import pickle
import os

#os.makedirs("PARENT_TOKEN/output", exist_ok=True)

GTFS_PATH = _os.path.join(GTFS_DIR, "")

G = nx.DiGraph()

# -- 1-4. Build MERGED nodes --------------------------------------------------
# Each station / interchange / busport becomes one node carrying the union of
# what its stands provide. Geometry is the stand NEAREST THE CENTROID of the
# group rather than the centroid itself: the centroid of a large interchange can
# fall inside the building, and that coordinate is later snapped to the walk
# graph for catchment tracing, where it must correspond to a real boarding
# point. Ties are broken by a seeded random draw so the choice is reproducible.
import numpy as _np

print("Loading shapefile stops...")
stops_shp = gpd.read_file(_os.path.join(SPATIAL_DIR, "Stops/Stops.shp"))
stops_shp = stops_shp[stops_shp['STATUS'] == 'Active'].copy()
stops_shp['sched_trips_weekday'] = (stops_shp['WeekPAM'] + stops_shp['WeekPPM'] +
                               stops_shp['WeekOFF'] + stops_shp['WeekREM'])
stops_shp['sched_trips_weekend'] = (stops_shp['SatPAM']  + stops_shp['SatPPM']  +
                               stops_shp['SatOFF']  + stops_shp['SatREM']  +
                               stops_shp['SunPHPAM']+ stops_shp['SunPHPPM']+
                               stops_shp['SunPHOFF']+ stops_shp['SunPHREM'])
print(f"  Shapefile stops: {len(stops_shp):,}")

print("Loading GTFS stops...")
stops_gtfs = pd.read_csv(GTFS_PATH + "stops.txt", skipinitialspace=True)
stops_gtfs.columns = [c.strip().replace("\ufeff", "") for c in stops_gtfs.columns]
gtfs_pts = stops_gtfs[stops_gtfs['location_type'] == 0].copy()
train_platforms = gtfs_pts[gtfs_pts['stop_id'] >= 99000]
print(f"  GTFS boarding points: {len(gtfs_pts):,} "
      f"(rail platforms: {len(train_platforms):,})")

# --- unified per-stop attribute table ---------------------------------------
shp = stops_shp.assign(
    stop_id=stops_shp['STOPNUMBER'].astype(int),
    lon=stops_shp.geometry.x, lat=stops_shp.geometry.y,
    stopname=stops_shp['STOPNAME'], stoptype=stops_shp['STOPTYPE'],
    suburb=stops_shp['SUBURB'], accessible=stops_shp['ACCESSIBLE'],
    area=stops_shp['AREA'], source='shapefile',
)[['stop_id','lon','lat','stopname','stoptype','suburb','accessible','area',
   'sched_trips_weekday','sched_trips_weekend','source']]

rail = gtfs_pts[gtfs_pts['stop_id'] >= 99000].assign(
    stop_id=lambda d: d['stop_id'].astype(int),
    lon=lambda d: d['stop_lon'].astype(float), lat=lambda d: d['stop_lat'].astype(float),
    stopname=lambda d: d['stop_name'].astype(str),
    stoptype='Rail',
    suburb=lambda d: d['stop_name'].astype(str).str.rsplit(' Platform', n=1).str[0],
    accessible='Y', area='Perth Metro', sched_trips_weekday=0, sched_trips_weekend=0,
    source='gtfs_rail',
)[['stop_id','lon','lat','stopname','stoptype','suburb','accessible','area',
   'sched_trips_weekday','sched_trips_weekend','source']]

allstops = pd.concat([shp, rail[~rail.stop_id.isin(shp.stop_id)]], ignore_index=True)
allstops['node_id'] = to_node(allstops['stop_id'])
print(f"  Unified stop table: {len(allstops):,} stops -> "
      f"{allstops['node_id'].nunique():,} nodes")

# --- representative geometry: stand nearest the group centroid ---------------
_rng = _np.random.default_rng(20260809)

def _representative(g):
    if len(g) == 1:
        return g.index[0]
    # Planar approximation is ample at this scale for choosing among stands
    # metres apart; cos(lat) corrects longitude before comparing distances.
    clat = _np.cos(_np.radians(g['lat'].mean()))
    dx = (g['lon'] - g['lon'].mean()) * clat
    dy = (g['lat'] - g['lat'].mean())
    d2 = (dx * dx + dy * dy).to_numpy()
    best = _np.flatnonzero(d2 <= d2.min() + 1e-18)
    return g.index[best[0] if len(best) == 1 else _rng.choice(best)]

reps = allstops.groupby('node_id', sort=False).apply(_representative)
rep_rows = allstops.loc[reps.to_numpy()].set_index('node_id')

agg = allstops.groupby('node_id').agg(
    n_stands=('stop_id', 'size'),
    sched_trips_weekday=('sched_trips_weekday', 'sum'),
    sched_trips_weekend=('sched_trips_weekend', 'sum'),
)
_nm = NODE_MAP_DF.drop_duplicates('node_id').set_index('node_id')['node_name']

def _modal(col):
    return allstops.groupby('node_id')[col].agg(
        lambda x: x.mode().iat[0] if len(x.mode()) else None)

nodes = pd.DataFrame({
    'lon': rep_rows['lon'], 'lat': rep_rows['lat'],
    'stopname': _nm.reindex(rep_rows.index).fillna(rep_rows['stopname']),
    'stoptype': _modal('stoptype'), 'suburb': _modal('suburb'),
    'area': _modal('area'), 'source': rep_rows['source'],
})
nodes['accessible'] = allstops.groupby('node_id')['accessible'].agg(
    lambda x: 'Y' if (x == 'Y').any() else 'N')
nodes = nodes.join(agg)

multi = nodes[nodes['n_stands'] > 1]
print(f"  Merged nodes: {len(multi):,} (max {int(nodes['n_stands'].max())} stands)")
print(f"  Boardings summed across stands for merged nodes")

for nid, r in nodes.iterrows():
    G.add_node(int(nid), source=r['source'], lon=float(r['lon']), lat=float(r['lat']),
               stopname=str(r['stopname']), stoptype=str(r['stoptype']),
               suburb=str(r['suburb']), accessible=str(r['accessible']),
               area=str(r['area']), sched_trips_weekday=int(r['sched_trips_weekday']),
               sched_trips_weekend=int(r['sched_trips_weekend']), n_stands=int(r['n_stands']))

print("Loading edges and transfers...")
edges     = pd.read_csv(_os.path.join(OUT_DIR, "edge_weights.csv"), skipinitialspace=True)
transfers = pd.read_csv(GTFS_PATH + "transfers.txt", skipinitialspace=True)
# Transfers between stands of one facility are now internal to a node.
transfers['from_stop_id'] = to_node(transfers['from_stop_id'])
transfers['to_stop_id']   = to_node(transfers['to_stop_id'])
_t0 = len(transfers)
transfers = transfers[transfers['from_stop_id'] != transfers['to_stop_id']]
print(f"  Trip edges: {len(edges):,}")
print(f"  Transfer pairs: {_t0:,} -> {len(transfers):,} after dropping intra-node")

nodes_in_graph = set(G.nodes())
print(f"  Total nodes: {G.number_of_nodes():,}")

# -- 5. Add trip edges (bus routes + train routes) ----------------------------
skipped = 0
for _, row in edges.iterrows():
    f, t, w = int(row['from_stop']), int(row['to_stop']), int(row['trip_count'])
    if f in nodes_in_graph and t in nodes_in_graph:
        G.add_edge(f, t, weight=w, edge_type='service')
    else:
        skipped += 1

print(f"  Trip edges added: {G.number_of_edges():,}  (skipped {skipped:,} unmatched)")

# -- 6. Add transfer edges (bus stop ↔ train platform) ----------------------
transfer_added = 0
transfer_skipped = 0
for _, row in transfers.iterrows():
    f = int(row['from_stop_id'])
    t = int(row['to_stop_id'])
    if f in nodes_in_graph and t in nodes_in_graph:
        # Weight = 2: transfers are possible but represent a mode-change cost
        G.add_edge(f, t, weight=2, edge_type='transfer')
        G.add_edge(t, f, weight=2, edge_type='transfer')
        transfer_added += 1
    else:
        transfer_skipped += 1

print(f"  Transfer edges added: {transfer_added * 2:,} (bidirectional)  "
      f"(skipped {transfer_skipped:,})")

# -- 7. Connectivity report ---------------------------------------------------
print(f"\nFinal graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print(f"Graph density: {nx.density(G):.6f}")

undirected = G.to_undirected()
components = sorted(nx.connected_components(undirected), key=len, reverse=True)
print(f"\nConnected components: {len(components):,}")
print(f"Largest component:    {len(components[0]):,} nodes  <- should now include rail")
print(f"Isolated nodes:       {sum(1 for c in components if len(c) == 1):,}")

# Stop type breakdown
type_counts = {}
for n, d in G.nodes(data=True):
    t = d.get('stoptype', '?')
    type_counts[t] = type_counts.get(t, 0) + 1
print("\nNodes by stop type:")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:<25} {c:>5}")

# Rail platform connectivity check
rail_nodes = [n for n, d in G.nodes(data=True) if d.get('stoptype') == 'Rail']
G_rail = G.subgraph(rail_nodes).copy()
print(f"\nRail platform subgraph: {G_rail.number_of_nodes()} nodes, "
      f"{G_rail.number_of_edges()} edges")
print("Sample rail nodes:")
for n in list(rail_nodes)[:5]:
    deg = G.degree(n)
    print(f"  {G.nodes[n]['stopname']:<40} degree={deg}")

# -- 8. Save ------------------------------------------------------------------
with open(_os.path.join(OUT_DIR, "graph_full.pkl"), "wb") as f:
    pickle.dump(G, f)
print("\nSaved: output/graph_full.pkl")
#print("Now re-run 04_centrality.py")