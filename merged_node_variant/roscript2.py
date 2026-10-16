"""
Script B: Find which specific stop-pairs inflate most in peak hours
Run after Script A.
"""
import os
import pandas as pd
import numpy as np

# Resolved relative to this file, matching the convention in the other merged
# scripts: GTFS lives in the parent directory, outputs in ./output.
BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
GTFS = os.path.join(PARENT, "google_transit") + os.sep
OUT = os.path.join(BASE, "output")

stop_times = pd.read_csv(GTFS + "stop_times.txt",
                         usecols=['trip_id','arrival_time',
                                  'departure_time','stop_id','stop_sequence'])
trips  = pd.read_csv(GTFS + "trips.txt", usecols=['trip_id','route_id','service_id'])
routes = pd.read_csv(GTFS + "routes.txt",
                     usecols=['route_id','route_short_name','route_type'])
congested = pd.read_csv(os.path.join(OUT, "congested_routes.csv"))
centrality = pd.read_csv(os.path.join(OUT, "centrality_results.csv"))

def parse_gtfs_time(t):
    h, m, s = str(t).strip().split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def tag_period(hour):
    if 7 <= hour < 9:   return 'AM_peak'
    if 16.5 <= hour < 18.5: return 'PM_peak'
    if 9 <= hour < 16:  return 'off_peak'
    return 'other'

stop_times['dep_sec'] = stop_times['departure_time'].apply(parse_gtfs_time)

# Filter to congested bus routes only
congested_ids = set(congested['route_id'])
bus_trips = trips[(trips['route_id'].isin(congested_ids)) &
                   (trips['route_type'] == 3) if 'route_type' in trips.columns
                   else trips['route_id'].isin(congested_ids)]

# Tag periods using first stop departure of each trip
first_dep = (stop_times[stop_times['trip_id'].isin(bus_trips['trip_id'])]
             .sort_values('stop_sequence').groupby('trip_id')['dep_sec'].first())
trip_period = first_dep.apply(lambda s: tag_period(s // 3600)).rename('period')

# Build segment table
st = (stop_times[stop_times['trip_id'].isin(bus_trips['trip_id'])]
      .sort_values(['trip_id','stop_sequence']).copy())
st['arr_sec'] = st['arrival_time'].apply(parse_gtfs_time)
st['next_stop'] = st.groupby('trip_id')['stop_id'].shift(-1)
st['next_arr']  = st.groupby('trip_id')['arr_sec'].shift(-1)
st = st.dropna(subset=['next_stop'])
st['segment_time_sec'] = st['next_arr'] - st['dep_sec']
st['next_stop'] = st['next_stop'].astype(int)

# Join period
st = st.join(trip_period, on='trip_id')
st = st.merge(trips[['trip_id','route_id']], on='trip_id')
st = st.merge(routes[['route_id','route_short_name']], on='route_id')

# Segment-level peak vs off-peak
seg_stats = (st[st['period'].isin(['AM_peak','PM_peak','off_peak'])]
             .groupby(['stop_id','next_stop','route_short_name','period'])
             ['segment_time_sec'].mean().reset_index())

seg_pivot = seg_stats.pivot_table(
    index=['stop_id','next_stop','route_short_name'],
    columns='period', values='segment_time_sec').reset_index()
seg_pivot.columns.name = None
seg_pivot = seg_pivot.rename(columns={
    'AM_peak':'AM_sec', 'PM_peak':'PM_sec', 'off_peak':'offpeak_sec'})
seg_pivot = seg_pivot.dropna(subset=['AM_sec','PM_sec','offpeak_sec'])
seg_pivot['AM_inflation']  = seg_pivot['AM_sec']  / seg_pivot['offpeak_sec']
seg_pivot['PM_inflation']  = seg_pivot['PM_sec']  / seg_pivot['offpeak_sec']
seg_pivot['max_inflation'] = seg_pivot[['AM_inflation','PM_inflation']].max(axis=1)

# Join centrality of from-stop
seg_pivot = seg_pivot.merge(
    centrality[['stopnumber','stopname','suburb','lat','lon','influence_score']],
    left_on='stop_id', right_on='stopnumber', how='left')

# Priority score: high inflation AND high centrality of from-stop
seg_pivot['priority'] = seg_pivot['max_inflation'] * seg_pivot['influence_score'].fillna(0)

hotspots = seg_pivot[seg_pivot['offpeak_sec'] >= 30].sort_values(
    'priority', ascending=False)

print("=== TOP 25 CONGESTION HOTSPOT SEGMENTS ===")
print("(High peak inflation on high-centrality corridors)")
print(hotspots.head(25)[['route_short_name','stopname','suburb',
                          'offpeak_sec','AM_sec','PM_sec',
                          'AM_inflation','PM_inflation','priority']
].round(2).to_string(index=False))

hotspots.to_csv(os.path.join(OUT, "segment_hotspots.csv"), index=False)
print("\nSaved: output/segment_hotspots.csv")