"""
Script 4 (FIXED): Centrality & Influence Analysis
Perth Public Transport Network Analysis

Three fixes applied vs previous version:
  1. Centrality computed on largest connected component only
     (isolates + small regional subgraphs were polluting scores)
  2. School stops excluded from main analysis
     (run separately at end for completeness)
  3. Influence score uses betweenness + pagerank only
     (degree terms were amplifying peripheral stops)
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
import networkx as nx
import pickle
import os

#os.makedirs("PARENT_TOKEN/output", exist_ok=True)

# -- 1. Load graph -------------------------------------------------------------
print("Loading graph...")
with open(_os.path.join(OUT_DIR, "graph_full.pkl"), "rb") as f:
    G = pickle.load(f)
print(f"Full graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# -- 2. Extract largest connected component (LCC) -----------------------------
undirected  = G.to_undirected()
components  = sorted(nx.connected_components(undirected), key=len, reverse=True)
lcc_nodes   = components[0]
G_lcc       = G.subgraph(lcc_nodes).copy()

print(f"\nLargest connected component: {G_lcc.number_of_nodes():,} nodes, "
      f"{G_lcc.number_of_edges():,} edges")
print(f"Nodes excluded (isolated/small components): "
      f"{G.number_of_nodes() - G_lcc.number_of_nodes():,}")

# -- 3. Exclude school stops from main analysis --------------------------------
# School stops are captive-audience routes, not the general network
non_school = [n for n, d in G_lcc.nodes(data=True)
              if d.get('stoptype') != 'School']
G_main = G_lcc.subgraph(non_school).copy()

# Recheck connectivity after school removal
ug2 = G_main.to_undirected()
comps2 = sorted(nx.connected_components(ug2), key=len, reverse=True)
G_main = G_main.subgraph(comps2[0]).copy()

print(f"\nMain analysis graph (LCC, non-school): "
      f"{G_main.number_of_nodes():,} nodes, {G_main.number_of_edges():,} edges")

stop_types = {}
for n, d in G_main.nodes(data=True):
    t = d.get('stoptype','?')
    stop_types[t] = stop_types.get(t, 0) + 1
print("Stop types in main graph:")
for t, c in sorted(stop_types.items(), key=lambda x: -x[1]):
    print(f"  {t:<25} {c:>5}")

# -- 4. Compute centrality measures -------------------------------------------
print("\nComputing betweenness centrality (k=500 approximation)...")
print("  [This takes ~5-10 minutes]")
betweenness = nx.betweenness_centrality(G_main, weight='weight',
                                        normalized=True, k=500)
print("  Done.")

print("Computing degree centrality...")
in_degree  = dict(G_main.in_degree(weight='weight'))
out_degree = dict(G_main.out_degree(weight='weight'))

print("Computing PageRank...")
pagerank = nx.pagerank(G_main, weight='weight', alpha=0.85)

# -- 5. Assemble results -------------------------------------------------------
results = pd.DataFrame({
    'stopnumber'   : list(G_main.nodes()),
    'betweenness'  : [betweenness[n] for n in G_main.nodes()],
    'pagerank'     : [pagerank[n]    for n in G_main.nodes()],
    'in_degree_w'  : [in_degree[n]   for n in G_main.nodes()],
    'out_degree_w' : [out_degree[n]  for n in G_main.nodes()],
})

# Attach stop attributes
node_attrs = pd.DataFrame([
    {'stopnumber': n, **G_main.nodes[n]} for n in G_main.nodes()
])
results = results.merge(node_attrs, on='stopnumber')

# -- 6. Influence score: betweenness + pagerank only --------------------------
# Betweenness = bridge/chokepoint importance
# PageRank    = systemic/recursive importance
# Equal weight -- adjust if your research prioritises one over the other
def norm(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-12)

results['betweenness_norm'] = norm(results['betweenness'])
results['pagerank_norm']    = norm(results['pagerank'])
results['influence_score']  = (results['betweenness_norm'] * 0.5 +
                                results['pagerank_norm']    * 0.5)

# -- 7. Print results ----------------------------------------------------------
cols_display = ['stopnumber','stopname','suburb','stoptype',
                'influence_score','betweenness','pagerank','sched_trips_weekday']

print("\n=== TOP 25 STOPS BY INFLUENCE SCORE ===")
print("(Betweenness + PageRank on largest connected component, non-school stops)")
print(results.nlargest(25, 'influence_score')[cols_display].to_string(index=False))

print("\n=== TOP 15 BY BETWEENNESS (critical chokepoints) ===")
print(results.nlargest(15, 'betweenness')[
    ['stopname','suburb','stoptype','betweenness','sched_trips_weekday']
].to_string(index=False))

print("\n=== TOP 15 BY PAGERANK (systemic hubs) ===")
print(results.nlargest(15, 'pagerank')[
    ['stopname','suburb','stoptype','pagerank','sched_trips_weekday']
].to_string(index=False))

# -- 8. Sanity check: Rail stops only -----------------------------------------
rail_results = results[results['stoptype'] == 'Rail'].nlargest(10, 'influence_score')
print("\n=== TOP 10 RAIL STATIONS BY INFLUENCE ===")
print("(Validation: Perth Central, Elizabeth Quay, Fremantle etc should appear)")
print(rail_results[['stopname','suburb','betweenness','pagerank',
                     'influence_score','sched_trips_weekday']].to_string(index=False))

# -- 9. Save -------------------------------------------------------------------
results.to_csv(_os.path.join(OUT_DIR, "centrality_results.csv"), index=False)
print("\nSaved: output/centrality_results.csv")
print("Run 05_reliance_index.py next (requires ABS data downloads)")