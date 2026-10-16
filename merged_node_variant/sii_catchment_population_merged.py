"""
Catchment population and employment (A_i)
Perth Public Transport Network Analysis

Builds the catchment-demand component A_i for every ordinary bus stop, by two
methods, and joins it to the existing SII table.

  A_i = catchment resident population + catchment employment

Two catchment definitions are computed for each stop:

  radial   400 m Euclidean disc around the stop.
           Cheap, conventional (PTA Bus Planning and Design Guidelines), and
           blind to severance — it will happily count population on the far
           side of a freeway, a rail reserve or the Swan River.

  network  400 m along the walkable street network from the stop.
           Traces the routable OSM walk graph outward to a 400 m path-length
           threshold, takes the reachable edges, and buffers them to form a
           service-area polygon. Respects severance.

Reporting both is the point. The difference between them is a measurable
result: it quantifies how much a radial buffer overstates the population a
Perth stop can actually reach on foot.

Apportionment from SA1/SA2 polygons to catchment polygons is area-weighted:
a catchment that covers 30% of an SA1's area is credited with 30% of that
SA1's population. SA1s in urban Perth are small (median ~200 persons), so the
uniform-density assumption inside each SA1 is tolerable. Employment is only
published at SA2 for place of work, so job counts are apportioned from the
coarser SA2 polygons and are correspondingly coarser.

  ------------------------------------------------------------------------
  REQUIRED INPUT FILES — see ABS_DATA_CHECKLIST.md in the project root.
  The script checks for all of them up front and tells you exactly what is
  missing before doing any work.
  ------------------------------------------------------------------------

Usage:
    python sii_catchment_population.py                  # both methods
    python sii_catchment_population.py --method radial  # skip the OSM stage
    python sii_catchment_population.py --limit 300      # quick smoke test
    python sii_catchment_population.py --in-place       # write into sii_bus_full.csv
    python sii_catchment_population.py --export-polygons

Outputs (all under output/):
    sii_bus_full_with_catchment.csv   SII table + A_i columns
    catchment_comparison.csv          per-stop radial vs network comparison
    catchment_summary.txt             aggregate overstatement statistics
    catchment_comparison.png          diagnostic plots
    geojson/catchments_*.gpkg         catchment polygons (--export-polygons)
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "output")
GEO_DIR    = os.path.join(OUTPUT_DIR, "geojson")
DATA_DIR   = os.path.join(BASE, "data", "abs")
CACHE_DIR  = os.path.join(BASE, "data", "cache")

SII_CSV = os.path.join(OUTPUT_DIR, "sii_bus_full.csv")

# --- ABS inputs (see ABS_DATA_CHECKLIST.md) --------------------------------
# The ASGS Main Structure download is a SINGLE GeoPackage holding every level
# as a separate layer (MB, SA1, SA2, SA3, SA4, GCCSA, STE, AUS), not one file
# per level.
ASGS_GPKG = os.path.join(DATA_DIR, "ASGS_2021_Main_Structure_GDA2020.gpkg")
SA1_LAYER = "SA1_2021_AUST_GDA2020"
SA2_LAYER = "SA2_2021_AUST_GDA2020"
MB_LAYER  = "MB_2021_AUST_GDA2020"   # not used yet — see the dasymetric note below

# Census 2021 GCP table G01, SA1 level, WA. Population by place of usual residence.
SA1_POPULATION = os.path.join(DATA_DIR, "2021Census_G01_WA_SA1.csv")
SA1_POP_ID_COL = "SA1_CODE_2021"
SA1_POP_VAL_COL = "Tot_P_P"

# Census 2021 Working Population Profile table W01, SA2 level, WA.
# Place of WORK, not place of usual residence.
#
# W01 is split across two files because of the column limit. The age-by-sex
# breakdown runs through W01A and spills into W01B; the persons totals live
# only in W01B, which is the part we need. W01A is not read at all.
#
# Note the code column is POW_SA2_CODE_2021, not SA2_CODE_2021 — place-of-work
# geography is a distinct code set. It includes non-spatial special codes
# (e.g. 597979799 "no fixed address", 599999499 "undefined") which carry no
# geometry and are dropped by the inner join against the boundary layer.
SA2_EMPLOYMENT = os.path.join(DATA_DIR, "2021Census_W01B_WA_POW_SA2.csv")
SA2_EMP_ID_COL = "POW_SA2_CODE_2021"
SA2_EMP_VAL_COL = "P_Tot_Tot"

# Restrict to Greater Perth. ASGS GCCSA code for Greater Perth is 5GPER.
GCCSA_FILTER = "5GPER"

# --- Dasymetric weights (--dasymetric) -------------------------------------
# Area weighting assumes people and jobs are spread evenly across a zone. They
# are not: in Greater Perth, Commercial mesh blocks are 0.82% of land area and
# Parkland is 37.7%. Dasymetric apportionment redistributes each zone's count
# across its mesh blocks in proportion to (weight x mesh block area), so the
# count lands where the land use says it plausibly is.
#
# Weights are normalised within each zone, so zone totals are conserved exactly
# whatever values are used here.
#
# These are KNOWLEDGE-DRIVEN parameters in the Wu et al. sense — they encode a
# judgement about which land uses hold residents and jobs. They are exposed here
# rather than buried so they can be varied and sensitivity-tested. The employment
# weights are the more arguable set: the non-zero weight on Residential is there
# because working from home was substantial at the 2021 Census, and Transport is
# held low because that category is mostly road and rail reserve rather than
# workplaces.
POP_MB_WEIGHTS = {
    "Residential":        1.00,
    "Primary Production": 0.05,
    "Other":              0.05,
    "Parkland":           0.00,
    "Commercial":         0.00,
    "Industrial":         0.00,
    "Education":          0.00,
    "Hospital/Medical":   0.00,
    "Transport":          0.00,
    "Water":              0.00,
}
JOB_MB_WEIGHTS = {
    "Commercial":         1.00,
    "Industrial":         1.00,
    "Education":          1.00,
    "Hospital/Medical":   1.00,
    "Other":              0.30,
    "Transport":          0.20,
    "Residential":        0.15,
    "Primary Production": 0.10,
    "Parkland":           0.05,
    "Water":              0.00,
}

# --- Method parameters -----------------------------------------------------
CATCHMENT_M   = 400.0   # PTA bus-stop walking standard

# Half-width of the band drawn around the reachable street network.
#
# This parameter is load-bearing: it alone determines how much smaller a network
# catchment looks than a radial one, so the radial-vs-network result cannot be
# read without it. It represents how far back from a street frontage the land
# that street serves extends — in effect half the depth of a block.
#
# 40 m is set from Perth's suburban block geometry: typical lots run about 30-40 m
# deep, with roughly 70-80 m between parallel streets, so a 40 m half-width
# captures the parcels fronting a street without reaching across the block and
# double-counting the parcels fronting the next one.
#
# Because it is a judgement rather than a measurement, every run emits a
# sensitivity sweep across SENSITIVITY_HALF_WIDTHS so the dependence is visible.
EDGE_BUFFER_M = 40.0

# Half-widths reported in the sensitivity sweep (metres).
SENSITIVITY_HALF_WIDTHS = [20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0]
SENSITIVITY_SAMPLE = 1500   # stops sampled for the sweep

# --- Walk-graph quality guards ---------------------------------------------
# A stop whose snapped node sits on a sparse or disconnected fragment of the walk
# graph produces a near-empty catchment. That is a data defect, not severance,
# and such stops would otherwise dominate the "most overstated" ranking.
MIN_REACH_M  = 400.0   # min total reachable street length for a usable catchment
# Max tolerable distance from the stop to its snapped position. Since the switch
# to edge snapping (AUDIT_CHECKLIST item 9) this is the perpendicular distance to
# the nearest EDGE, not to the nearest node, so it is far smaller and this guard
# rarely binds. It is retained for genuinely unmapped locations. Note it never
# caught the failure it was assumed to: stop 16999 snapped 17.5 m onto entirely
# the wrong sub-network. Topological consistency is reported separately, in the
# `snap_road_match` diagnostic column.
MAX_SNAP_M   = 100.0
CRS_METRIC    = "EPSG:7850"   # GDA2020 / MGA zone 50 — correct for Perth
CRS_GEOG      = "EPSG:7844"   # GDA2020 geographic

PROGRESS_EVERY = 500


# --- standardisation of the catchment term ---------------------------------
# Must match the treatment of the SII components in sii_bus_stops_pipeline*.py.
# A is a right-skewed non-negative count like F, R and T, so if those are
# standardised on a log scale this one has to be as well: otherwise a weight on
# z(A) would express sensitivity to ABSOLUTE change while the same weight on
# z(F) expressed sensitivity to PROPORTIONAL change, and the two would not be
# commensurable inside the weighted sum.
#
# L is deliberately NOT transformed. It is bounded on [0,1] and enters the Hub
# Reliance Index multiplicatively rather than as a summed z-score, so it is not
# part of this comparability requirement.
LOG_TRANSFORM = True

def standardise(series, clip=3.0):
    x = series.astype(float)
    if LOG_TRANSFORM and x.min() >= 0:
        x = np.log1p(x)      # monotone, keeps zeros at zero
    mean, std = x.mean(), x.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-9 * max(1.0, abs(mean)):
        return pd.Series(np.zeros(len(x)), index=x.index), mean, 0.0
    return ((x - mean) / std).clip(-clip, clip), mean, std


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def check_inputs(method, population_only=False):
    """Fail early and loudly, listing every missing file at once."""
    required = [
        (SII_CSV, "Existing SII output. Run sii_bus_stops_pipeline.py first."),
        (ASGS_GPKG, "ASGS Ed3 Main Structure GeoPackage (all levels, one file)."),
        (SA1_POPULATION, "Census 2021 GCP table G01 at SA1, WA."),
    ]
    employment = [
        (SA2_EMPLOYMENT, "Census 2021 WPP table W01B at POW SA2, WA."),
    ]
    if not population_only:
        required = required + employment

    missing = [(p, why) for p, why in required if not os.path.exists(p)]
    if missing:
        print("\nMISSING REQUIRED INPUT FILES\n" + "=" * 72)
        for p, why in missing:
            print(f"  {p}\n      {why}\n")
        print("See ABS_DATA_CHECKLIST.md for where to download each file.")
        if any(p for p, _ in employment if not os.path.exists(p)):
            print("\nEmployment data is REQUIRED by default, because SA1 population")
            print("is place of usual RESIDENCE and carries no information about jobs.")
            print("Perth CBD SA1s hold few residents but very large job counts, so a")
            print("population-only A_i scores the Adelaide Tce / Wellington St / St")
            print("Georges Tce stops near zero — the same stops that top the SII")
            print("ranking. That does not merely weaken the index, it inverts it.")
            print("\nTo proceed anyway (e.g. a provisional run while you wait on the")
            print("Working Population Profile download), pass --population-only.")
        sys.exit(1)

    have_employment = not population_only
    if population_only:
        print("\n" + "!" * 72)
        print("--population-only: A_i excludes employment.")
        print("Catchment demand will be understated at every destination-side stop,")
        print("most severely in the CBD and at activity centres. Treat the resulting")
        print("A_i as provisional and do not report it as the final component.")
        print("!" * 72 + "\n")

    if method in ("network", "both"):
        try:
            import osmnx  # noqa: F401
        except ImportError:
            print("\nThe network catchment stage needs osmnx:")
            print("    pip install osmnx")
            print("Or run with --method radial to skip it.\n")
            sys.exit(1)

    return have_employment


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_stops(limit=None):
    """Ordinary bus stops from the SII table, as a projected GeoDataFrame."""
    import geopandas as gpd

    sii = pd.read_csv(SII_CSV, low_memory=False)
    sii = sii.dropna(subset=["lon", "lat"])
    if limit:
        sii = sii.nlargest(limit, "SII").copy()
        print(f"  --limit active: using the {len(sii)} highest-SII stops only")

    gdf = gpd.GeoDataFrame(
        sii,
        geometry=gpd.points_from_xy(sii["lon"], sii["lat"]),
        crs="EPSG:4326",
    ).to_crs(CRS_METRIC)
    print(f"  Stops loaded: {len(gdf):,}")
    return gdf


def load_zones(boundary_path, layer, value_csv, id_col, val_col, code_field, label):
    """
    Load one layer from the ASGS GeoPackage, attach a count column, and return it
    projected with a precomputed polygon area. Zones with no matching count are
    dropped.
    """
    import geopandas as gpd

    print(f"  Reading {label} boundaries (layer {layer}) ...")

    # The Main Structure GeoPackage is ~900 MB and Australia-wide. Push the
    # Greater Perth filter down into the driver so we never materialise the
    # other ~59,000 SA1s in memory.
    try:
        zones = gpd.read_file(boundary_path, layer=layer,
                              where=f"GCCSA_CODE_2021 = '{GCCSA_FILTER}'")
        print(f"    read {len(zones):,} features (filtered at read time)")
    except Exception:
        print("    driver-level filter unavailable, reading full layer ...")
        zones = gpd.read_file(boundary_path, layer=layer)
        gccsa_cols = [c for c in zones.columns if c.upper().startswith("GCC")
                      and "CODE" in c.upper()]
        if gccsa_cols:
            col = gccsa_cols[0]
            before = len(zones)
            zones = zones[zones[col].astype(str) == GCCSA_FILTER].copy()
            print(f"    filtered {before:,} -> {len(zones):,} on {col}")

    # The boundary layer's own code field varies by vintage and format.
    if code_field not in zones.columns:
        cands = [c for c in zones.columns if c.upper().startswith(code_field[:3])
                 and "CODE" in c.upper()]
        if not cands:
            raise KeyError(
                f"Could not find a {label} code column in {boundary_path}. "
                f"Columns present: {list(zones.columns)}"
            )
        code_field = cands[0]
        print(f"    using boundary code column '{code_field}'")

    counts = pd.read_csv(value_csv)
    if id_col not in counts.columns or val_col not in counts.columns:
        raise KeyError(
            f"{os.path.basename(value_csv)} does not contain "
            f"'{id_col}' and/or '{val_col}'. Columns present: {list(counts.columns)[:15]} ..."
        )
    counts = counts[[id_col, val_col]].copy()
    counts[id_col] = counts[id_col].astype(str).str.strip()
    counts = counts.rename(columns={val_col: "zone_count"})

    zones[code_field] = zones[code_field].astype(str).str.strip()
    zones = zones.merge(counts, left_on=code_field, right_on=id_col, how="inner")

    zones = zones.to_crs(CRS_METRIC)
    zones = zones[zones.geometry.notna() & ~zones.geometry.is_empty].copy()
    zones["zone_area"] = zones.geometry.area
    zones = zones[zones["zone_area"] > 0].copy()
    zones = zones[[code_field, "zone_count", "zone_area", "geometry"]]
    zones = zones.rename(columns={code_field: "zone_id"})

    print(f"    {label}: {len(zones):,} zones, total count {zones['zone_count'].sum():,.0f}")
    return zones.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Catchment construction
# ---------------------------------------------------------------------------
def radial_catchments(stops):
    """400 m Euclidean disc around each stop."""
    import geopandas as gpd

    print(f"\n  Building radial catchments ({CATCHMENT_M:.0f} m) ...")
    polys = stops.geometry.buffer(CATCHMENT_M)
    out = gpd.GeoDataFrame(
        {"stopnumber": stops["stopnumber"].values},
        geometry=polys.values,
        crs=CRS_METRIC,
    )
    out["catchment_area"] = out.geometry.area
    print(f"    {len(out):,} catchments, mean area "
          f"{out['catchment_area'].mean():,.0f} m2")
    return out


def _osmnx_compat():
    """osmnx moved several functions between v1 and v2."""
    import osmnx as ox
    nearest = getattr(getattr(ox, "distance", ox), "nearest_nodes", None)
    if nearest is None:
        nearest = ox.nearest_nodes
    return ox, nearest


def network_catchments(stops):
    """
    400 m walk-network service areas.

    Downloads (and caches) the OSM walk graph for the stop envelope, then for
    each stop runs a Dijkstra bounded at CATCHMENT_M, collects the fully
    reachable edges, and buffers their union by EDGE_BUFFER_M.
    """
    import geopandas as gpd
    import networkx as nx
    from shapely.ops import unary_union
    from collections import defaultdict

    ox, nearest_nodes = _osmnx_compat()
    ox.settings.use_cache = True
    ox.settings.cache_folder = CACHE_DIR
    os.makedirs(CACHE_DIR, exist_ok=True)

    graph_path = os.path.join(CACHE_DIR, "perth_walk.graphml")

    if os.path.exists(graph_path):
        print(f"\n  Loading cached walk graph from {graph_path} ...")
        G = ox.load_graphml(graph_path)
    else:
        print("\n  Downloading OSM walk network (slow, one time only) ...")
        envelope = (stops.geometry.union_all().convex_hull
                    if hasattr(stops.geometry, "union_all")
                    else stops.geometry.unary_union.convex_hull)
        envelope = envelope.buffer(1500)  # margin so edge stops aren't clipped
        envelope_geog = (gpd.GeoSeries([envelope], crs=CRS_METRIC)
                         .to_crs("EPSG:4326").iloc[0])
        G = ox.graph_from_polygon(envelope_geog, network_type="walk",
                                  simplify=True, retain_all=True)
        ox.save_graphml(G, graph_path)
        print(f"    cached to {graph_path}")

    G = ox.project_graph(G, to_crs=CRS_METRIC)
    print(f"    walk graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    # Undirected graph, edge geometry lookup, node -> incident edges index.
    # Built in walk_snap so the transfer-potential stage snaps identically.
    import walk_snap as ws
    GU, nodes_xy, edge_geom, node_edges, edge_name = ws.build_undirected(G)
    ws.set_geometry_cache(edge_geom)

    # EDGE-based snapping. See AUDIT_CHECKLIST item 9 and walk_snap.__doc__:
    # nearest-NODE snapping on a simplified graph put stops on the wrong
    # sub-network (stop 16999 snapped 17.5 m onto a footway 791 m of walk from
    # the arterial it stands on), manufacturing the A = 1.000 population.
    snaps = ws.snap_stops(stops.geometry, nodes_xy, edge_geom, edge_name)
    snap_dist = np.array([s.snap_dist for s in snaps])

    # Diagnostic only — never used to filter or to choose a snap.
    if "stopname" in stops.columns:
        host_match = [ws.host_matches_road(nm, s.host_name)
                      for nm, s in zip(stops["stopname"].values, snaps)]
    else:
        host_match = [None] * len(snaps)

    # Sample a subset up front and retain their raw (unbuffered) reachable
    # network, so the buffer sensitivity sweep can be computed from one trace.
    rng = np.random.default_rng(0)
    sens_idx = set(rng.choice(len(stops),
                              size=min(SENSITIVITY_SAMPLE, len(stops)),
                              replace=False).tolist())
    sens_raw = []

    print(f"    tracing {len(stops):,} service areas "
          f"(half-width {EDGE_BUFFER_M:.0f} m) ...")
    records = []
    for i, (stopnum, snap) in enumerate(zip(stops["stopnumber"].values, snaps)):
        if (i + 1) % PROGRESS_EVERY == 0:
            print(f"      {i + 1:,} / {len(stops):,}")

        dist = ws.dijkstra_from_snap(GU, snap, CATCHMENT_M)

        reachable = set(dist)
        geoms, seen, reach_len = [], set(), 0.0

        # The stop's own host edge. Under node snapping this segment could be
        # dropped entirely whenever one endpoint fell outside the bound — the
        # stop's own street omitted from its own catchment.
        host_g, host_len = ws.host_edge_reach(snap, CATCHMENT_M)
        if host_g is not None:
            geoms.append(host_g)
            reach_len += host_len
            seen.add(snap.key)

        for n in reachable:
            for key in node_edges[n]:
                if key in seen:
                    continue
                u, v = key
                if u in reachable and v in reachable:
                    seen.add(key)
                    geoms.append(edge_geom[key])
                    reach_len += GU[u][v]["length"]

        if geoms:
            raw = unary_union(geoms)
            poly = raw.buffer(EDGE_BUFFER_M)
        else:
            # Isolated snap point: fall back to a small disc so the stop is not
            # silently dropped. It will be flagged as suspect below.
            raw = None
            poly = stops.geometry.iloc[i].buffer(EDGE_BUFFER_M)

        if i in sens_idx and raw is not None:
            sens_raw.append(raw)

        records.append((stopnum, poly, len(reachable), reach_len, snap_dist[i],
                        host_match[i]))

    out = gpd.GeoDataFrame(
        {
            "stopnumber":  [r[0] for r in records],
            "n_reachable": [r[2] for r in records],
            "reach_len_m": [r[3] for r in records],
            "snap_dist_m": [r[4] for r in records],
            # Diagnostic: did the stop land on the street it is named for?
            # None = host way unnamed (usually a footway, often correct).
            "snap_road_match": [r[5] for r in records],
        },
        geometry=[r[1] for r in records],
        crs=CRS_METRIC,
    )
    out["catchment_area"] = out.geometry.area

    n_named = int(out["snap_road_match"].notna().sum())
    if n_named:
        n_bad = int((out["snap_road_match"] == False).sum())  # noqa: E712
        print(f"    snap diagnostic: {n_named:,} stops landed on a NAMED way, "
              f"of which {n_bad:,} ({n_bad / n_named * 100:.1f}%) on a street "
              f"other than the one in the stop name")
        print(f"      (diagnostic only — unnamed footways are often the correct "
              f"pedestrian origin and are not counted here)")

    # Flag stops whose catchment reflects a walk-graph defect rather than the
    # built environment.
    out["catchment_suspect"] = (
        (out["reach_len_m"] < MIN_REACH_M) | (out["snap_dist_m"] > MAX_SNAP_M)
    )
    n_sus = int(out["catchment_suspect"].sum())
    print(f"    mean area {out['catchment_area'].mean():,.0f} m2")
    print(f"    median reachable street length {out['reach_len_m'].median():,.0f} m")
    if n_sus:
        print(f"    {n_sus:,} stops ({n_sus / len(out) * 100:.1f}%) flagged suspect "
              f"(reachable < {MIN_REACH_M:.0f} m or snap > {MAX_SNAP_M:.0f} m)")
        print(f"      these are excluded from the radial-vs-network comparison")

    return out, sens_raw


def sensitivity_sweep(sens_raw):
    """
    Median catchment area at a range of band half-widths, from one trace.

    Buffering is a Minkowski sum with a disc, so the areas at every half-width
    are recoverable from the raw reachable network without re-tracing.
    """
    if not sens_raw:
        return None

    import geopandas as gpd

    disc = np.pi * CATCHMENT_M ** 2
    gs = gpd.GeoSeries(sens_raw, crs=CRS_METRIC)
    rows = []
    for hw in SENSITIVITY_HALF_WIDTHS:
        med = gs.buffer(hw).area.median()
        rows.append({
            "half_width_m": hw,
            "median_area_m2": round(med, 1),
            "pct_of_disc": round(med / disc * 100, 1),
            "implied_radial_over_network": round(disc / med, 2) if med > 0 else np.nan,
            "in_use": hw == EDGE_BUFFER_M,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Apportionment
# ---------------------------------------------------------------------------
def apportion(catchments, zones, label, quiet=False):
    """
    Area-weighted apportionment of a zone count into each catchment.

        contribution = zone_count * (intersection_area / zone_area)

    `zones` may be the raw SA1/SA2 layer (plain area weighting) or a mesh-block
    surface produced by build_weighted_surface() (dasymetric weighting). The
    arithmetic is identical either way; only the granularity of the donor
    polygons changes.

    Returns a Series indexed by stopnumber.
    """
    import geopandas as gpd

    if not quiet:
        print(f"  Apportioning {label} ...")

    cat = catchments[["stopnumber", "geometry"]].reset_index(drop=True)
    zon = zones[["zone_count", "zone_area", "geometry"]].reset_index(drop=True)

    joined = gpd.sjoin(cat, zon, how="inner", predicate="intersects")
    if joined.empty:
        print(f"    WARNING: no overlap between catchments and {label} zones. "
              f"Check that both layers cover Perth and share a CRS.")
        return pd.Series(dtype=float, name=label)

    # Vectorised pairwise intersection. Both GeoSeries are rebuilt on a clean
    # RangeIndex so shapely pairs them positionally rather than by index.
    left = gpd.GeoSeries(cat.geometry.loc[joined.index].values, crs=CRS_METRIC)
    right = gpd.GeoSeries(zon.geometry.loc[joined["index_right"]].values,
                          crs=CRS_METRIC)
    inter_area = left.intersection(right).area.to_numpy()

    zone_area = joined["zone_area"].to_numpy()
    share = np.divide(inter_area, zone_area,
                      out=np.zeros_like(inter_area), where=zone_area > 0)

    out = pd.DataFrame({
        "stopnumber": joined["stopnumber"].to_numpy(),
        "contrib": joined["zone_count"].to_numpy() * share,
    })
    result = out.groupby("stopnumber")["contrib"].sum().rename(label)
    if not quiet:
        print(f"    {label}: {len(result):,} stops with non-empty catchment, "
              f"mean {result.mean():,.1f}")
    return result


def apportion_competitive(catchments, zones, label, supply=None, quiet=False):
    """
    Competitive apportionment: each zone's count is divided among the stops whose
    catchments reach it, instead of being credited in full to every one of them.

    Plain apportion() lets overlapping catchments each claim the same people. In
    Perth that inflated the metropolitan sum to 2.0x actual residents and 2.8x
    actual jobs, and unevenly — roughly 1x at an isolated Hills stop against
    about 8x in the CBD. A stop was being rewarded for having neighbours.

    Allocation follows the competition principle of the two-step floating
    catchment area method: a zone divides its count among the stops that serve
    it in proportion to how much of the zone each catchment covers and how much
    service each stop offers.

        w_ij = a_ij * S_j                        a_ij = share of zone i in catchment j
        A_j  = SUM_i  P_i * w_ij / SUM_k w_ik    S_j  = weekday supply at stop j

    With S_j held constant this reduces to proportional allocation by coverage.
    Either way SUM_j A_j equals the true total over zones reached by at least one
    stop, so the result is conservative by construction and the check below
    enforces it.

    Note on naming: canonical 2SFCA (Luo and Wang) scores accessibility *at
    demand locations*. This is its dual — attributing demand *to supply
    locations* — so describe it as 2SFCA-style competitive allocation rather
    than as 2SFCA itself.

    `supply` is a Series indexed by stopnumber. Pass None for equal weighting.
    """
    import geopandas as gpd

    mode = "proportional" if supply is None else "supply-weighted"
    if not quiet:
        print(f"  Apportioning {label} ({mode} competitive allocation) ...")

    cat = catchments[["stopnumber", "geometry"]].reset_index(drop=True)
    zon = zones[["zone_count", "zone_area", "geometry"]].reset_index(drop=True)

    joined = gpd.sjoin(cat, zon, how="inner", predicate="intersects")
    if joined.empty:
        print(f"    WARNING: no overlap between catchments and {label} zones.")
        return pd.Series(dtype=float, name=label)

    left = gpd.GeoSeries(cat.geometry.loc[joined.index].values, crs=CRS_METRIC)
    right = gpd.GeoSeries(zon.geometry.loc[joined["index_right"]].values,
                          crs=CRS_METRIC)
    inter_area = left.intersection(right).area.to_numpy()

    zone_area = joined["zone_area"].to_numpy()
    a_ij = np.divide(inter_area, zone_area,
                     out=np.zeros_like(inter_area), where=zone_area > 0)

    stopnums = joined["stopnumber"].to_numpy()
    if supply is None:
        s_j = np.ones(len(joined))
    else:
        s_j = supply.reindex(stopnums).to_numpy(dtype=float)
        # A stop with no scheduled weekday service would otherwise claim nothing
        # and, worse, zero out a zone served only by such stops. Give it a floor
        # so it competes weakly rather than not at all.
        s_j = np.nan_to_num(s_j, nan=0.0)
        s_j = np.where(s_j <= 0, 0.1, s_j)

    w_ij = a_ij * s_j

    pairs = pd.DataFrame({
        "stopnumber": stopnums,
        "zone_idx": joined["index_right"].to_numpy(),
        "zone_count": joined["zone_count"].to_numpy(),
        "w": w_ij,
    })
    denom = pairs.groupby("zone_idx")["w"].transform("sum").to_numpy()
    pairs["contrib"] = np.divide(
        pairs["zone_count"].to_numpy() * pairs["w"].to_numpy(), denom,
        out=np.zeros(len(pairs)), where=denom > 0)

    result = pairs.groupby("stopnumber")["contrib"].sum().rename(label)

    # Conservation: the allocated total must equal the count in every zone that
    # at least one catchment reaches. Anything else means demand was invented or
    # lost, which would silently corrupt every weight fitted on this surface.
    reached = zones.iloc[sorted(pairs["zone_idx"].unique())]["zone_count"].sum()
    allocated = result.sum()
    drift = abs(allocated - reached) / max(reached, 1)
    if not quiet:
        status = "OK" if drift < 1e-6 else "MISMATCH"
        print(f"    conservation: reached zones {reached:,.0f} -> allocated "
              f"{allocated:,.0f}  [{status}]")
        print(f"    {label}: {len(result):,} stops, mean {result.mean():,.1f}")
    if drift >= 1e-6:
        raise RuntimeError(
            f"Competitive allocation of {label} drifted by "
            f"{allocated - reached:,.0f} ({drift:.2%}). Refusing to continue.")
    return result


def load_mesh_blocks():
    """Greater Perth mesh blocks with land-use category, projected."""
    import geopandas as gpd

    print("  Reading mesh blocks (layer MB_2021_AUST_GDA2020) ...")
    cols = ["MB_CODE_2021", "MB_CATEGORY_2021", "SA1_CODE_2021", "SA2_CODE_2021"]
    try:
        mb = gpd.read_file(ASGS_GPKG, layer=MB_LAYER,
                           where=f"GCCSA_CODE_2021 = '{GCCSA_FILTER}'",
                           columns=cols)
    except Exception:
        mb = gpd.read_file(ASGS_GPKG, layer=MB_LAYER)
        mb = mb[mb["GCCSA_CODE_2021"].astype(str) == GCCSA_FILTER]
        mb = mb[cols + ["geometry"]]

    mb = mb.to_crs(CRS_METRIC)
    mb = mb[mb.geometry.notna() & ~mb.geometry.is_empty].copy()
    mb["mb_area"] = mb.geometry.area
    mb = mb[mb["mb_area"] > 0].copy()
    for c in ("SA1_CODE_2021", "SA2_CODE_2021"):
        mb[c] = mb[c].astype(str).str.strip()

    print(f"    {len(mb):,} mesh blocks")
    counts = mb["MB_CATEGORY_2021"].value_counts()
    for cat, n in counts.items():
        pct = mb.loc[mb["MB_CATEGORY_2021"] == cat, "mb_area"].sum() \
            / mb["mb_area"].sum() * 100
        print(f"      {cat:<20} {n:>6,}  ({pct:5.2f}% of area)")
    return mb.reset_index(drop=True)


def build_weighted_surface(mb, zones, zone_key, weights, label):
    """
    Redistribute each zone's count across its mesh blocks in proportion to
    (category weight x mesh block area).

    The mesh block table already carries SA1_CODE_2021 and SA2_CODE_2021, so
    zones and mesh blocks are matched by attribute join — no spatial operation
    and no risk of sliver mismatches at shared boundaries.

    Returns a GeoDataFrame shaped like a zone layer (zone_count, zone_area,
    geometry) that can be passed straight to apportion().
    """
    import geopandas as gpd

    print(f"  Building dasymetric surface for {label} ...")
    surf = mb[[zone_key, "MB_CATEGORY_2021", "mb_area", "geometry"]].copy()
    surf = surf.merge(zones[["zone_id", "zone_count"]],
                      left_on=zone_key, right_on="zone_id", how="inner")

    surf["w"] = surf["MB_CATEGORY_2021"].map(weights).fillna(0.0)
    surf["wa"] = surf["w"] * surf["mb_area"]

    denom = surf.groupby("zone_id")["wa"].transform("sum")

    # A zone whose mesh blocks all carry weight zero would lose its entire count.
    # Fall back to plain area weighting within that zone so nothing evaporates.
    fallback = denom <= 0
    n_fallback = int(surf.loc[fallback, "zone_id"].nunique())
    if n_fallback:
        area_denom = surf.groupby("zone_id")["mb_area"].transform("sum")
        surf.loc[fallback, "wa"] = surf.loc[fallback, "mb_area"]
        denom = denom.where(~fallback, area_denom)
        print(f"    {n_fallback:,} zones had no weighted mesh block — "
              f"fell back to area weighting within those zones")

    surf["zone_count"] = surf["zone_count"] * np.divide(
        surf["wa"].to_numpy(), denom.to_numpy(),
        out=np.zeros(len(surf)), where=denom.to_numpy() > 0)

    # Conservation check: the surface must carry the same total as the zones it
    # was built from, restricted to zones the mesh blocks actually cover.
    covered = zones[zones["zone_id"].isin(surf["zone_id"].unique())]
    zone_total, surf_total = covered["zone_count"].sum(), surf["zone_count"].sum()
    drift = abs(surf_total - zone_total) / max(zone_total, 1)
    status = "OK" if drift < 1e-6 else "MISMATCH"
    print(f"    conservation: zones {zone_total:,.0f} -> surface "
          f"{surf_total:,.0f}  [{status}]")
    if status == "MISMATCH":
        raise RuntimeError(
            f"Dasymetric redistribution lost {zone_total - surf_total:,.0f} "
            f"of {label}. Refusing to continue with a non-conservative surface.")

    surf = surf[surf["zone_count"] > 0].copy()
    surf = surf.rename(columns={"mb_area": "zone_area"})
    print(f"    surface: {len(surf):,} weighted mesh blocks")
    return gpd.GeoDataFrame(
        surf[["zone_count", "zone_area", "geometry"]],
        geometry="geometry", crs=CRS_METRIC).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def apportionment_report(merged, comparison_rows):
    """
    Compare dasymetric against plain area weighting on identical catchments.

    Totals are conserved by construction, so this is entirely about
    redistribution: which stops gain, which lose, and how far the ranking moves.
    """
    lines = []
    add = lines.append
    add("=" * 72)
    add("DASYMETRIC vs AREA WEIGHTING")
    add("=" * 72)

    frames = []
    for name, cmp_df in comparison_rows.items():
        sub = merged[["stopnumber", "stopname", "suburb",
                      f"A_pop_{name}", f"A_job_{name}", f"A_{name}"]].copy()
        sub = sub.merge(cmp_df, left_on="stopnumber", right_index=True, how="inner")
        sub["area_A"] = sub["area_pop"] + sub["area_job"]
        sub = sub[sub["area_A"].notna() & sub[f"A_{name}"].notna()]
        if sub.empty:
            continue
        sub["method"] = name

        dasy_rank = sub[f"A_{name}"].rank(ascending=False)
        area_rank = sub["area_A"].rank(ascending=False)
        sub["rank_shift"] = (dasy_rank - area_rank).abs()

        add("")
        add(f"--- {name} catchments (n = {len(sub):,}) ---")
        add(f"  mean A_i, area weighted:   {sub['area_A'].mean():,.1f}")
        add(f"  mean A_i, dasymetric:      {sub[f'A_{name}'].mean():,.1f}")
        add(f"  mean |rank shift|:         {sub['rank_shift'].mean():,.1f} places")
        add(f"  stops moving >50 places:   {int((sub['rank_shift'] > 50).sum()):,}")

        n_area = sub["area_job"].round(1).nunique()
        n_dasy = sub[f"A_job_{name}"].round(1).nunique()
        add(f"  distinct job values:       {n_area:,} (area) -> {n_dasy:,} (dasymetric)")
        add("  ^ the rise is the CBD tie breaking: area weighting gives every")
        add("    stop inside one SA2 an identical job count, dasymetric does not.")

        add("")
        add("  Largest gains under dasymetric weighting:")
        sub["delta"] = sub[f"A_{name}"] - sub["area_A"]
        cols = ["stopname", "suburb", "area_A", f"A_{name}", "delta"]
        add(sub.nlargest(8, "delta")[cols].round(0).to_string(index=False))
        add("")
        add("  Largest losses:")
        add(sub.nsmallest(8, "delta")[cols].round(0).to_string(index=False))

        frames.append(sub)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUTPUT_DIR, "apportionment_summary.txt"), "w") as f:
        f.write(text + "\n")

    if frames:
        pd.concat(frames).to_csv(
            os.path.join(OUTPUT_DIR, "apportionment_comparison.csv"), index=False)
        print("\n  Wrote apportionment_comparison.csv and apportionment_summary.txt")


def comparison_report(df, have_employment, sens_df=None):
    """Quantify how much the radial buffer overstates the walkable catchment."""
    needed = ["A_radial", "A_network", "catchment_area_radial", "catchment_area_network"]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        print("  Only one catchment method was run — the radial vs network")
        print("  comparison needs both. Re-run with --method both to produce it.")
        return

    both = df.dropna(subset=["A_radial", "A_network"]).copy()

    # Drop stops whose network catchment reflects a walk-graph defect. Left in,
    # they head the overstatement ranking with ratios in the thousands and read
    # as severance when they are nothing of the kind.
    n_before = len(both)
    if "catchment_suspect" in both.columns:
        both = both[~both["catchment_suspect"].astype(bool)].copy()
    n_excluded = n_before - len(both)

    if both.empty:
        print("\n  Both methods not available — skipping comparison report.")
        return

    both["A_diff"] = both["A_radial"] - both["A_network"]
    both["A_ratio"] = np.where(both["A_network"] > 0,
                               both["A_radial"] / both["A_network"], np.nan)
    both["area_ratio"] = np.where(both["catchment_area_network"] > 0,
                                  both["catchment_area_radial"]
                                  / both["catchment_area_network"], np.nan)

    lines = []
    add = lines.append
    add("=" * 72)
    add("RADIAL vs NETWORK CATCHMENT COMPARISON")
    add("=" * 72)
    add(f"Stops compared:                 {len(both):,}")
    add(f"Excluded (walk-graph defect):   {n_excluded:,}")
    add(f"Catchment distance:             {CATCHMENT_M:.0f} m")
    add(f"Network band half-width:        {EDGE_BUFFER_M:.0f} m")
    add(f"Employment included in A_i:     {'yes' if have_employment else 'no'}")
    add("")

    if sens_df is not None:
        add("-" * 72)
        add("SENSITIVITY TO THE BAND HALF-WIDTH")
        add("-" * 72)
        add("The half-width sets how much smaller a network catchment looks than")
        add("a radial one, so no figure below can be read without it. Sampled from")
        add(f"{SENSITIVITY_SAMPLE:,} stops; the row in use is marked.")
        add("")
        add("  half-width   median area      % of disc   implied radial/network")
        for _, r in sens_df.iterrows():
            mark = "  <-- in use" if r["in_use"] else ""
            add("  {:>7.0f} m {:>14,.0f} m2 {:>12.1f}% {:>18.2f}{}".format(
                r["half_width_m"], r["median_area_m2"], r["pct_of_disc"],
                r["implied_radial_over_network"], mark))
        add("")
        add("Report this table alongside any radial-vs-network claim.")
        add("")
    add(f"Mean A_i, radial:               {both['A_radial'].mean():,.1f}")
    add(f"Mean A_i, network:              {both['A_network'].mean():,.1f}")
    add(f"Mean overstatement (radial-net):{both['A_diff'].mean():,.1f}")
    add(f"Median ratio radial/network:    {both['A_ratio'].median():,.3f}")
    add("")
    for thresh in (1.25, 1.5, 2.0):
        n = int((both["A_ratio"] > thresh).sum())
        add(f"Stops where radial overstates by >{(thresh-1)*100:.0f}%: "
            f"{n:,} ({n/len(both)*100:.1f}%)")
    add("")
    add("Most severely overstated stops (largest radial/network ratio):")
    cols = ["stopname", "suburb", "A_radial", "A_network", "A_ratio"]
    cols = [c for c in cols if c in both.columns]
    add(both.nlargest(15, "A_ratio")[cols].round(2).to_string(index=False))
    add("")
    add("These are the stops where severance — freeways, rail reserves, the")
    add("river, or simply the absence of a connecting footpath — cuts off part")
    add("of the disc a radial buffer would credit to the stop.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUTPUT_DIR, "catchment_summary.txt"), "w") as f:
        f.write(text + "\n")

    keep = [c for c in ["stopnumber", "stopname", "suburb",
                        "A_pop_radial", "A_job_radial", "A_radial",
                        "A_pop_network", "A_job_network", "A_network",
                        "A_diff", "A_ratio",
                        "catchment_area_radial", "catchment_area_network",
                        "area_ratio"] if c in both.columns]
    both[keep].to_csv(os.path.join(OUTPUT_DIR, "catchment_comparison.csv"),
                      index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        axes[0].scatter(both["A_network"], both["A_radial"], s=5, alpha=0.25)
        hi = float(np.nanmax([both["A_radial"].max(), both["A_network"].max()]))
        axes[0].plot([0, hi], [0, hi], "k--", lw=0.8)
        axes[0].set_xlabel("A_i, network catchment")
        axes[0].set_ylabel("A_i, radial catchment")
        axes[0].set_title("Radial vs network catchment demand")

        axes[1].hist(both["A_ratio"].dropna().clip(0, 4), bins=60)
        axes[1].axvline(1.0, color="k", ls="--", lw=0.8)
        axes[1].set_xlabel("A_radial / A_network")
        axes[1].set_ylabel("Stops")
        axes[1].set_title("Distribution of overstatement")

        axes[2].scatter(both["area_ratio"], both["A_ratio"], s=5, alpha=0.25)
        axes[2].set_xlabel("Area ratio (radial / network)")
        axes[2].set_ylabel("A_i ratio (radial / network)")
        axes[2].set_title("Area loss vs demand loss")
        axes[2].set_xlim(0, 10)
        axes[2].set_ylim(0, 5)

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "catchment_comparison.png"), dpi=140)
        plt.close(fig)
        print("  Wrote catchment_comparison.png")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["radial", "network", "both"],
                    default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the N highest-SII stops (smoke test)")
    ap.add_argument("--in-place", action="store_true",
                    help="write A_i back into sii_bus_full.csv (a .bak is kept)")
    ap.add_argument("--export-polygons", action="store_true",
                    help="also write catchment polygons as GeoPackage for QGIS")
    ap.add_argument("--allocation", choices=["supply", "proportional", "none"],
                    default="supply",
                    help="how overlapping catchments divide a zone's count. "
                         "'supply' = 2SFCA-style, weighted by each stop's weekday "
                         "service (default); 'proportional' = equal split by "
                         "coverage; 'none' = every catchment claims the full "
                         "count (double-counts, kept only for comparison)")
    ap.add_argument("--dasymetric", action="store_true",
                    help="redistribute counts across mesh blocks by land-use "
                         "category instead of weighting by raw zone area")
    ap.add_argument("--population-only", action="store_true",
                    help="run without employment data (provisional only — see the "
                         "warning this prints, A_i will be understated at all "
                         "destination-side stops)")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GEO_DIR, exist_ok=True)

    print("=" * 72)
    print("CATCHMENT DEMAND (A_i)")
    print("=" * 72)

    have_employment = check_inputs(args.method, population_only=args.population_only)

    print("\nSTEP 1: Load stops and ABS zones")
    print("-" * 72)
    stops = load_stops(limit=args.limit)

    sa1 = load_zones(ASGS_GPKG, SA1_LAYER, SA1_POPULATION,
                     SA1_POP_ID_COL, SA1_POP_VAL_COL, "SA1_CODE_2021", "SA1")
    sa2 = None
    if have_employment:
        sa2 = load_zones(ASGS_GPKG, SA2_LAYER, SA2_EMPLOYMENT,
                         SA2_EMP_ID_COL, SA2_EMP_VAL_COL, "SA2_CODE_2021", "SA2")

    print("\nSTEP 2: Build catchments")
    print("-" * 72)
    catch = {}
    sens_df = None
    net_diag = None
    if args.method in ("radial", "both"):
        catch["radial"] = radial_catchments(stops)
    if args.method in ("network", "both"):
        net, sens_raw = network_catchments(stops)
        catch["network"] = net
        net_diag = net[["stopnumber", "n_reachable", "reach_len_m",
                        "snap_dist_m", "snap_road_match",
                        "catchment_suspect"]].copy()
        sens_df = sensitivity_sweep(sens_raw)
        if sens_df is not None:
            sens_df.to_csv(os.path.join(OUTPUT_DIR,
                                        "catchment_sensitivity.csv"), index=False)

    print("\nSTEP 3: Apportion population and employment")
    print("-" * 72)
    result = stops[["stopnumber"]].copy()

    pop_donor, job_donor = sa1, sa2
    if args.dasymetric:
        mb = load_mesh_blocks()
        pop_donor = build_weighted_surface(mb, sa1, "SA1_CODE_2021",
                                           POP_MB_WEIGHTS, "population")
        if sa2 is not None:
            job_donor = build_weighted_surface(mb, sa2, "SA2_CODE_2021",
                                               JOB_MB_WEIGHTS, "employment")
        print()

    comparison_rows = {}

    # Supply weights for competitive allocation: total scheduled weekday
    # departures at each stop, from the existing SII components.
    supply = None
    if args.allocation == "supply":
        base = pd.read_csv(SII_CSV, low_memory=False)
        for c in ("F_AM", "F_PM", "F_off"):
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
        supply = (base["F_AM"] + base["F_PM"] + base["F_off"])
        supply.index = base["stopnumber"]
        print(f"  Supply weights from weekday departures: "
              f"median {supply.median():.1f}, max {supply.max():.1f}")

    def allocate(cg, donor, lab):
        if args.allocation == "none":
            return apportion(cg, donor, lab)
        return apportion_competitive(cg, donor, lab, supply=supply)

    for name, cg in catch.items():
        pop = allocate(cg, pop_donor, f"A_pop_{name}")
        result = result.merge(pop.rename(f"A_pop_{name}"),
                              left_on="stopnumber", right_index=True, how="left")

        if job_donor is not None:
            job = allocate(cg, job_donor, f"A_job_{name}")
            result = result.merge(job.rename(f"A_job_{name}"),
                                  left_on="stopnumber", right_index=True, how="left")
        else:
            result[f"A_job_{name}"] = 0.0

        # When running dasymetric, also compute the plain area-weighted answer
        # for the same catchments so the two can be compared directly.
        if args.dasymetric:
            # Like-for-like: vary ONLY the apportionment surface. The allocation
            # method must match the main run, or the comparison silently folds
            # together two independent changes and reports the sum of both as
            # the effect of dasymetric weighting.
            print(f"  [comparison] recomputing {name} under plain area weighting ...")
            ap = allocate(cg, sa1, "cmp_pop")
            aj = allocate(cg, sa2, "cmp_job") if sa2 is not None else None
            comparison_rows[name] = pd.DataFrame({
                "area_pop": ap,
                "area_job": aj if aj is not None else 0.0,
            })

        result[f"A_pop_{name}"] = result[f"A_pop_{name}"].fillna(0.0)
        result[f"A_job_{name}"] = result[f"A_job_{name}"].fillna(0.0)
        result[f"A_{name}"] = result[f"A_pop_{name}"] + result[f"A_job_{name}"]

        areas = cg.set_index("stopnumber")["catchment_area"]
        result = result.merge(areas.rename(f"catchment_area_{name}"),
                              left_on="stopnumber", right_index=True, how="left")

    print("\nSTEP 4: Standardise and join to the SII table")
    print("-" * 72)

    # z-scored within the ordinary bus stop universe, clipped at +/-3, to match
    # the convention used for every other SII component.
    for name in catch:
        col = f"A_{name}"
        z, mean, std = standardise(result[col])
        result[f"{col}_z"] = z
        if std == 0:
            print(f"  {col}: no usable variation (sd ~ 0) — z set to 0")
        else:
            print(f"  {col}: mean {mean:,.3f}, sd {std:,.3f} "
                  f"({'log' if LOG_TRANSFORM else 'linear'} scale), "
                  f"raw range [{result[col].min():,.0f}, {result[col].max():,.0f}]")

    if net_diag is not None:
        result = result.merge(net_diag, on="stopnumber", how="left")
        result["catchment_suspect"] = result["catchment_suspect"].fillna(False)

    # ----------------------------------------------------------------------
    # Canonical A_i: network catchment, with radial substituted where the walk
    # graph could not support one.
    #
    # The suspect stops are concentrated in the Perth Hills, the Swan Valley and
    # the Kwinana industrial strip — places with little or no mapped footway.
    # Their network catchment is near-empty, which may well be true on foot, but
    # dropping them would remove those regions from the study entirely and bias
    # it toward the serviced inner suburbs. Radial keeps them in the index; the
    # A_source column keeps them identifiable so the substitution is never
    # silent and can be reported as its own class of stop.
    # ----------------------------------------------------------------------
    if {"A_network", "A_radial"}.issubset(result.columns):
        sus = result["catchment_suspect"].astype(bool)
        for part in ("A_pop", "A_job", "A"):
            net_col, rad_col = f"{part}_network", f"{part}_radial"
            result[part] = np.where(sus, result[rad_col], result[net_col])
        result["A_source"] = np.where(sus, "radial_fallback", "network")

        z_, mean_, std_ = standardise(result["A"])
        result["A_z"] = z_

        n_fb = int(sus.sum())
        print(f"\n  Canonical A_i assembled: network for "
              f"{len(result) - n_fb:,} stops, radial fallback for {n_fb:,} "
              f"({n_fb / len(result) * 100:.1f}%)")
        print(f"    A: mean {mean_:,.1f}, sd {std_:,.1f}  -> use A_z in the index")
    elif "A_network" in result.columns:
        result["A"] = result["A_network"]
        result["A_z"] = result["A_network_z"]
        result["A_source"] = "network"
        print("\n  Canonical A_i = network (no radial computed, so no fallback "
              "available for suspect stops)")

    sii = pd.read_csv(SII_CSV, low_memory=False)
    new_cols = [c for c in result.columns if c != "stopnumber"]
    sii = sii.drop(columns=[c for c in new_cols if c in sii.columns], errors="ignore")
    merged = sii.merge(result, on="stopnumber", how="left")

    out_path = os.path.join(OUTPUT_DIR, "sii_bus_full_with_catchment.csv")
    merged.to_csv(out_path, index=False)
    print(f"\n  Wrote {out_path}")
    print(f"  Added columns: {', '.join(new_cols)}")

    if args.in_place:
        if not args.limit:
            bak = SII_CSV + ".bak"
            if not os.path.exists(bak):
                pd.read_csv(SII_CSV, low_memory=False).to_csv(bak, index=False)
                print(f"  Backed up original to {bak}")
            merged.to_csv(SII_CSV, index=False)
            print(f"  Updated {SII_CSV} in place")
        else:
            print("  --in-place ignored because --limit is active "
                  "(would truncate the table).")

    if args.export_polygons:
        for name, cg in catch.items():
            p = os.path.join(GEO_DIR, f"catchments_{name}.gpkg")
            cg.to_file(p, driver="GPKG")
            print(f"  Wrote {p}")

    if comparison_rows:
        print("\nSTEP 5a: Dasymetric vs area weighting")
        print("-" * 72)
        apportionment_report(merged, comparison_rows)

    print("\nSTEP 5: Radial vs network comparison")
    print("-" * 72)
    comparison_report(merged, have_employment, sens_df)

    print("\nDone.")
    print("\nNote: A_i has been computed and standardised, but the SII weight")
    print("vector in sii_bus_stops_pipeline.py is unchanged. Adding A_i to the")
    print("composite is a separate decision — the weights must still sum to 1.")


if __name__ == "__main__":
    main()
