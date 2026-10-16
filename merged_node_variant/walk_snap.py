"""
Edge-based snapping of stops onto the pedestrian walk graph.

WHY THIS MODULE EXISTS
----------------------
The pipeline previously assigned each stop to the *nearest node* of the walk
graph (`osmnx.nearest_nodes`). The graph is a *simplified* OSMnx graph: nodes of
degree two have been dissolved and the intervening geometry survives only as an
edge attribute. On such a graph the nearest surviving vertex is not the nearest
point on the network, and it may belong to a different part of the network
altogether.

Worked failure (AUDIT_CHECKLIST item 9). Stop 16999, Old Coast Rd After Peelwood
Pde, Erskine:

  - nearest node is 17.5 m away, on an unnamed footway off Castleward Place,
    inside the cul-de-sac subdivision *behind* the arterial
  - nearest node actually carrying `Old Coast Road` is 165 m from the stop and
    **791 m from the assigned node** by walk network
  - measured along Old Coast Rd the stop is ~330 m from stop 22576, i.e. inside
    the 400 m bound. The pipeline reported no alternatives and A = 1.000.

Measured over a 500-stop sample, re-snapping moved alternative access by more
than 0.25 for 6.2% of stops, and 15 of 18 stops reporting A = 1.000 fell below
unity. The snap distance gives no warning: 17.5 m passes any tolerance.

WHAT THIS MODULE DOES
---------------------
Each stop is projected onto the nearest *edge*. The resulting point is used as a
virtual graph vertex: a temporary node joined to the edge's two endpoints u and
v with weights equal to the distances along the edge geometry from the projected
point to u and to v. Every bounded Dijkstra then starts from the stop's true
position on the network.

This removes the choice of snapping rule rather than adjudicating between rules.
Matching on street name is NOT the fix and is not used here for anything but a
diagnostic column: where OpenStreetMap maps a footway separately from its
carriageway, the footway is frequently the more appropriate origin for a
pedestrian catchment.

USED BY
-------
  sii_catchment_population_merged.network_catchments()   -> D, and V via
      sii_v_catchment.py, which calls that function
  sii_transfer_potential_merged.py                       -> T

Both import from here so the two stages are provably snapping identically.
"""

from __future__ import annotations

import re
import math
from collections import defaultdict, namedtuple

import numpy as np


# --------------------------------------------------------------------------- #
#  A stop's position on the network
# --------------------------------------------------------------------------- #

EdgeSnap = namedtuple(
    "EdgeSnap",
    [
        "key",        # (u, v) with u <= v — the key used in edge_geom
        "u", "v",     # endpoints
        "d_u", "d_v", # metres along the edge geometry from the stop to u / to v
        "d_along",    # metres from the geometry's own start to the stop
        "length",     # geometry length, metres
        "snap_dist",  # perpendicular distance stop -> nearest point on edge
        "host_name",  # `name` attribute of the host edge, or None
    ],
)

TEMP_NODE = "__stop_virtual__"


# --------------------------------------------------------------------------- #
#  Graph preparation
# --------------------------------------------------------------------------- #

def build_undirected(G):
    """
    Collapse a projected OSMnx MultiDiGraph to an undirected Graph carrying, for
    each node pair, the shortest parallel edge and that edge's geometry.

    Returns (GU, nodes_xy, edge_geom, node_edges).

    `edge_geom` is keyed on (u, v) with u <= v. The geometry's orientation is
    NOT guaranteed to run u -> v; `snap_stops()` resolves orientation per edge
    by comparing the geometry's first coordinate to the node coordinates.
    """
    import networkx as nx
    from shapely.geometry import LineString

    nodes_xy = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

    GU = nx.Graph()
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        length = float(data.get("length", 0.0))
        if GU.has_edge(u, v) and GU[u][v]["length"] <= length:
            continue
        GU.add_edge(u, v, length=length,
                    geometry=data.get("geometry"),
                    name=data.get("name"))

    edge_geom, edge_name = {}, {}
    node_edges = defaultdict(list)
    for u, v, data in GU.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            geom = LineString([nodes_xy[u], nodes_xy[v]])
        key = (u, v) if u <= v else (v, u)
        edge_geom[key] = geom
        edge_name[key] = data.get("name")
        node_edges[u].append(key)
        node_edges[v].append(key)

    return GU, nodes_xy, edge_geom, node_edges, edge_name


# --------------------------------------------------------------------------- #
#  Snapping
# --------------------------------------------------------------------------- #

def snap_stops(points, nodes_xy, edge_geom, edge_name, verbose=True):
    """
    Project each point onto the nearest edge.

    `points` : GeoSeries of stop geometries, already in the graph's projected CRS
    Returns  : list of EdgeSnap, one per point, aligned to `points` order.

    Uses the geopandas spatial index over the *same* edge set the Dijkstra will
    traverse, so the host edge is guaranteed to exist in GU.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    keys = list(edge_geom.keys())
    geoms = [edge_geom[k] for k in keys]
    edges = gpd.GeoDataFrame(geometry=geoms, crs=points.crs)

    if verbose:
        print(f"    snapping {len(points):,} stops to nearest of "
              f"{len(keys):,} walk edges ...")

    # geopandas >= 0.10 / shapely 2: STRtree-backed, no rtree dependency.
    res = edges.sindex.nearest(points.geometry, return_all=False,
                               return_distance=True)
    idx_pairs, dists = res
    # idx_pairs is (2, n): row 0 = input order, row 1 = edge index
    order = np.asarray(idx_pairs[0])
    ematch = np.asarray(idx_pairs[1])
    dmatch = np.asarray(dists)

    # Guard: sindex.nearest is documented to return input indices in order, but
    # do not rely on it.
    edge_of = np.empty(len(points), dtype=np.int64)
    dist_of = np.empty(len(points), dtype=float)
    edge_of[order] = ematch
    dist_of[order] = dmatch

    out = []
    xs = points.geometry.x.to_numpy()
    ys = points.geometry.y.to_numpy()
    for i in range(len(points)):
        key = keys[edge_of[i]]
        geom = edge_geom[key]
        u, v = key
        L = float(geom.length)
        d_along = float(geom.project(Point(xs[i], ys[i])))
        d_along = min(max(d_along, 0.0), L)

        # Resolve geometry orientation: does it start at u or at v?
        c0 = geom.coords[0]
        du_start = math.hypot(c0[0] - nodes_xy[u][0], c0[1] - nodes_xy[u][1])
        dv_start = math.hypot(c0[0] - nodes_xy[v][0], c0[1] - nodes_xy[v][1])
        if du_start <= dv_start:          # geometry runs u -> v
            d_u, d_v = d_along, L - d_along
        else:                             # geometry runs v -> u
            d_u, d_v = L - d_along, d_along

        out.append(EdgeSnap(key=key, u=u, v=v, d_u=d_u, d_v=d_v,
                            d_along=d_along, length=L,
                            snap_dist=float(dist_of[i]),
                            host_name=edge_name.get(key)))

    if verbose:
        sd = np.array([s.snap_dist for s in out])
        print(f"      snap distance: median {np.median(sd):.1f} m, "
              f"p90 {np.percentile(sd, 90):.1f} m, max {sd.max():.1f} m")
    return out


# --------------------------------------------------------------------------- #
#  Bounded search from a snapped position
# --------------------------------------------------------------------------- #

def dijkstra_from_snap(GU, snap, cutoff):
    """
    Bounded Dijkstra from a stop's projected position on its host edge.

    Inserts a temporary vertex joined to the host edge's endpoints, runs the
    search, then removes it. Returns {node: metres}, which EXCLUDES the
    temporary vertex itself.

    Distances are exact for the virtual-node formulation: reaching any node
    requires travelling along the host edge to u or to v first.
    """
    import networkx as nx

    GU.add_node(TEMP_NODE)
    GU.add_edge(TEMP_NODE, snap.u, length=snap.d_u)
    GU.add_edge(TEMP_NODE, snap.v, length=snap.d_v)
    try:
        dist = nx.single_source_dijkstra_path_length(
            GU, TEMP_NODE, cutoff=float(cutoff), weight="length")
    except nx.NodeNotFound:
        dist = {}
    finally:
        GU.remove_node(TEMP_NODE)

    dist.pop(TEMP_NODE, None)
    return dist


def host_edge_reach(snap, cutoff):
    """
    The portion of the stop's own host edge reachable within `cutoff`.

    Returned as (geometry_or_None, length_m). The stop sits ON this edge, so the
    segment around it is always part of the catchment — under the old
    node-snapping the stop's own street segment could be omitted entirely
    whenever one endpoint fell outside the bound.
    """
    from shapely.ops import substring

    lo = max(0.0, snap.d_along - cutoff)
    hi = min(snap.length, snap.d_along + cutoff)
    if hi - lo <= 0.0:
        return None, 0.0
    geom = substring(_geom_cache[snap.key], lo, hi)
    return geom, float(hi - lo)


# `host_edge_reach` needs the geometry; set once by the caller to avoid
# threading edge_geom through every call site.
_geom_cache = {}


def set_geometry_cache(edge_geom):
    global _geom_cache
    _geom_cache = edge_geom


# --------------------------------------------------------------------------- #
#  Diagnostic only: is the host edge the street the stop is named for?
# --------------------------------------------------------------------------- #

_ABBREV = {
    "rd": "road", "st": "street", "av": "avenue", "ave": "avenue",
    "pde": "parade", "dr": "drive", "bvd": "boulevard", "cr": "crescent",
    "cres": "crescent", "tce": "terrace", "wy": "way", "hwy": "highway",
    "pl": "place", "ct": "court", "cl": "close", "gr": "grove", "ln": "lane",
    "pwy": "parkway", "sq": "square", "ret": "retreat", "gdns": "gardens",
    "grn": "green", "vw": "view", "hts": "heights", "ent": "entrance",
    "esp": "esplanade", "cct": "circuit", "ch": "chase", "app": "approach",
}

_SPLIT = re.compile(
    r"\s+(?:After|Before|Between|Opp|Opposite|Stand|At|Cat\s+Id|Nr|Near)\b",
    re.I,
)


def _normalise(s):
    s = str(s).lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9 ']", " ", s)
    return " ".join(_ABBREV.get(w, w) for w in s.split())


def road_of(stopname):
    """The street name embedded in a Transperth stop label."""
    return _normalise(_SPLIT.split(str(stopname))[0])


def host_matches_road(stopname, host_name):
    """
    True if the host edge's name matches the stop's own street.

    DIAGNOSTIC ONLY — never use this to filter or to choose a snap. A False
    result is usually an unnamed footway, which is often the correct pedestrian
    origin. Its purpose is to make the topological-consistency question visible
    in the output, which the old 100 m snap tolerance could not do.
    """
    if host_name is None:
        return None                      # unnamed way: not assessable
    if isinstance(host_name, (list, tuple, set)):
        cands = [_normalise(x) for x in host_name]
    else:
        cands = [_normalise(host_name)]
    rd = road_of(stopname)
    if not rd:
        return None
    return any(rd == c or rd in c or c in rd for c in cands)
