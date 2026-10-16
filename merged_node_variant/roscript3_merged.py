"""
Script C: Clean Hotspot Results + Route Optimisation Candidate Modelling
Perth Public Transport Network Analysis

Filters noise from segment analysis, then produces three things:
  1. Reliable congestion hotspot corridors
  2. Express variant model (skip low-demand stops in peak)
  3. Optimisation priority table for each candidate route
"""
import pandas as pd
import numpy as np

import os as _os
_BASE=_os.path.dirname(_os.path.abspath(__file__)); _PARENT=_os.path.dirname(_BASE)
GTFS = _os.path.join(_PARENT,"google_transit")+_os.sep
OUT  = _os.path.join(_BASE,"output")+_os.sep

def normalize_text_key(s: pd.Series) -> pd.Series:
    """Normalize text join keys (trim, remove BOM, coerce numerics like 83.0 -> 83)."""
    return (s.astype("string")
             .str.replace("\ufeff", "", regex=False)
             .str.strip()
             .str.replace(r"\.0$", "", regex=True))

# ── 1. Load all outputs ───────────────────────────────────────────────────────
hotspots   = pd.read_csv(OUT+"segment_hotspots.csv")
congested  = pd.read_csv(OUT+"congested_routes.csv")
centrality = pd.read_csv(OUT+"centrality_results.csv")
stop_times = pd.read_csv(GTFS + "stop_times.txt",
                         usecols=['trip_id','arrival_time','departure_time',
                                  'stop_id','stop_sequence'])
trips      = pd.read_csv(GTFS + "trips.txt",
                         usecols=['trip_id','route_id','service_id'])
routes     = pd.read_csv(GTFS + "routes.txt",
                         usecols=['route_id','route_short_name'])
stops_gtfs = pd.read_csv(GTFS + "stops.txt")
stops_gtfs.columns = (stops_gtfs.columns
                      .str.replace("\ufeff", "", regex=False)
                      .str.strip())
stops_gtfs = stops_gtfs[['stop_id','stop_name','stop_lat','stop_lon']].copy()

# Normalize merge keys that may come in as mixed object/int/float representations
routes['route_short_name'] = normalize_text_key(routes['route_short_name'])
congested['route_short_name'] = normalize_text_key(congested['route_short_name'])

# ── 2. Clean hotspot segments ─────────────────────────────────────────────────
# Rebuild with trip count per segment so we can filter on sample size
def parse_gtfs_time(t):
    h, m, s = str(t).strip().split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def tag_period(hour):
    if 7 <= hour < 9:   return 'AM_peak'
    if 16 <= hour < 18: return 'PM_peak'   # aligned with the index; was 16.5-18.5
    if 9 <= hour < 16:  return 'off_peak'
    return 'other'

stop_times['dep_sec'] = stop_times['departure_time'].apply(parse_gtfs_time)
stop_times['arr_sec'] = stop_times['arrival_time'].apply(parse_gtfs_time)

congested_route_ids = set(congested['route_id'])
bus_trips = trips[trips['route_id'].isin(congested_route_ids)]

first_dep = (stop_times[stop_times['trip_id'].isin(bus_trips['trip_id'])]
             .sort_values('stop_sequence').groupby('trip_id')['dep_sec'].first())
trip_period = first_dep.apply(lambda s: tag_period(s // 3600)).rename('period')

st = stop_times[stop_times['trip_id'].isin(bus_trips['trip_id'])].copy()
st = st.sort_values(['trip_id','stop_sequence'])
st['next_stop'] = st.groupby('trip_id')['stop_id'].shift(-1)
st['next_arr']  = st.groupby('trip_id')['arr_sec'].shift(-1)
st = st.dropna(subset=['next_stop'])
st['seg_sec'] = st['next_arr'] - st['dep_sec']
st['next_stop'] = st['next_stop'].astype(int)
st = st.join(trip_period, on='trip_id')
st = st.merge(trips[['trip_id','route_id']], on='trip_id')
st = st.merge(routes, on='route_id')

# Aggregate with count
seg_agg = (st[st['period'].isin(['AM_peak','PM_peak','off_peak'])]
           .groupby(['stop_id','next_stop','route_short_name','period'])
           .agg(mean_sec=('seg_sec','mean'), n_trips=('seg_sec','count'))
           .reset_index())

seg_wide = seg_agg.pivot_table(
    index=['stop_id','next_stop','route_short_name'],
    columns='period',
    values=['mean_sec','n_trips']
).reset_index()
seg_wide.columns = ['_'.join(c).strip('_') for c in seg_wide.columns]

seg_wide = seg_wide.rename(columns={
    'mean_sec_AM_peak':'AM_sec', 'mean_sec_PM_peak':'PM_sec',
    'mean_sec_off_peak':'offpeak_sec',
    'n_trips_AM_peak':'n_AM', 'n_trips_PM_peak':'n_PM',
    'n_trips_off_peak':'n_off'
})
seg_wide = seg_wide.dropna(subset=['AM_sec','PM_sec','offpeak_sec'])

# NOISE FILTERS:
#  - offpeak segment >= 90 seconds (avoids rounding ratio explosion)
#  - minimum 3 trips in each period (statistical reliability)
reliable = seg_wide[
    (seg_wide['offpeak_sec'] >= 90) &
    (seg_wide['n_AM']  >= 3) &
    (seg_wide['n_PM']  >= 3) &
    (seg_wide['n_off'] >= 3)
].copy()

reliable['AM_inflation']  = reliable['AM_sec']  / reliable['offpeak_sec']
reliable['PM_inflation']  = reliable['PM_sec']  / reliable['offpeak_sec']
reliable['max_inflation'] = reliable[['AM_inflation','PM_inflation']].max(axis=1)
reliable['worst_period']  = reliable[['AM_inflation','PM_inflation']].idxmax(axis=1).str.replace('_inflation','')
reliable['route_short_name'] = normalize_text_key(reliable['route_short_name'])

# Join stop names and centrality
reliable = reliable.merge(
    stops_gtfs.rename(columns={'stop_id':'stop_id_x','stop_name':'from_name',
                                'stop_lat':'from_lat','stop_lon':'from_lon'}),
    left_on='stop_id', right_on='stop_id_x', how='left')
reliable = reliable.merge(
    centrality[['stopnumber','influence_score']].rename(columns={'stopnumber':'stop_id'}),
    on='stop_id', how='left')

reliable['priority'] = reliable['max_inflation'] * reliable['influence_score'].fillna(0.1)
reliable = reliable.sort_values('priority', ascending=False)

print("=== RELIABLE CONGESTION HOTSPOT SEGMENTS ===")
print(f"(After noise filters: offpeak>=90s, n>=3 per period)")
print(f"Segments remaining: {len(reliable)} (from {len(seg_wide)} total)")
print()
display_cols = ['route_short_name','from_name','offpeak_sec',
                'AM_sec','PM_sec','AM_inflation','PM_inflation',
                'worst_period','n_AM','n_PM','n_off','priority']
print(reliable.head(25)[display_cols].round(2).to_string(index=False))

# ── 3. Identify optimisation candidate routes ---------------------------------
# Corridors where multiple consecutive segments are inflated = strong candidate

# Group by route, count how many reliable hot segments each has
route_hotsegs = (reliable[reliable['max_inflation'] >= 1.15]
                 .groupby('route_short_name')
                 .agg(n_hot_segments    = ('stop_id','count'),
                      mean_AM_inflation = ('AM_inflation','mean'),
                      mean_PM_inflation = ('PM_inflation','mean'),
                      max_inflation     = ('max_inflation','max'))
                 .reset_index()
                 .sort_values('n_hot_segments', ascending=False))

route_hotsegs = route_hotsegs.merge(
    congested[['route_short_name','mean_offpeak_min','mean_AM_min',
               'mean_PM_min','AM_ratio','PM_ratio']],
    on='route_short_name', how='left')

print("\n=== ROUTE OPTIMISATION CANDIDATES ===")
print("(Routes with multiple reliable hot segments, ranked by segment count)")
print(route_hotsegs.to_string(index=False))

# ── 4. Express variant model --------------------------------------------------
# For top candidate routes: which stops can be skipped in AM peak?
# Logic: if a stop has very low sched_trips_weekday AND is not on a hot segment,
# it's a skip candidate for an express service

print("\n=== EXPRESS VARIANT MODELLING ===")

top_routes = route_hotsegs.head(5)['route_short_name'].tolist()

for route_name in top_routes:
    route_id = congested[congested['route_short_name']==route_name]['route_id'].iloc[0]

    # Get a representative AM trip for this route
    route_trips = trips[trips['route_id']==route_id]['trip_id']
    route_st = stop_times[stop_times['trip_id'].isin(route_trips)].copy()
    route_st = route_st.join(trip_period, on='trip_id')
    am_trips = route_st[route_st['period']=='AM_peak']['trip_id'].unique()

    if len(am_trips) == 0:
        continue

    # Get stop sequence for one AM trip
    sample_trip = am_trips[0]
    seq = (stop_times[stop_times['trip_id']==sample_trip]
           .sort_values('stop_sequence')
           .merge(stops_gtfs, on='stop_id')
           .merge(centrality[['stopnumber','sched_trips_weekday','influence_score']],
                  left_on='stop_id', right_on='stopnumber', how='left'))

    seq['dep_sec'] = seq['departure_time'].apply(parse_gtfs_time)
    seq['arr_sec'] = seq['arrival_time'].apply(parse_gtfs_time)

    total_dur = (seq['arr_sec'].iloc[-1] - seq['dep_sec'].iloc[0]) / 60

    # Skip candidates: stops with < 30 daily weekday trips AND influence < 0.3
    skip_mask = ((seq['sched_trips_weekday'].fillna(0) < 30) &
                 (seq['influence_score'].fillna(0) < 0.3))
    skippable = seq[skip_mask]

    # Time saved: sum of dwell + approach time for skipped stops
    # Approximate: each stop skip saves ~30 seconds (average dwell)
    time_saved_min = len(skippable) * 0.5

    print(f"\nRoute {route_name}:")
    print(f"  Full trip duration (AM scheduled): {total_dur:.1f} min, {len(seq)} stops")
    print(f"  Skippable stops in express variant: {len(skippable)} stops")
    print(f"  Estimated time saving: ~{time_saved_min:.0f} min "
          f"({time_saved_min/total_dur*100:.0f}% reduction)")
    if len(skippable) > 0:
        print(f"  Stops to skip: {', '.join(skippable['stop_name'].dropna().head(8).tolist())}")

# ── 5. Save final outputs -----------------------------------------------------
reliable.to_csv(OUT+"reliable_hotspots.csv", index=False)
route_hotsegs.to_csv(OUT+"optimisation_candidates.csv", index=False)
print("\n\nSaved: output/reliable_hotspots.csv")
print("Saved: output/optimisation_candidates.csv")