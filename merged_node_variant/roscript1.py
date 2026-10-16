"""
Script A: Compute trip durations and identify congestion-affected routes
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
os.makedirs(OUT, exist_ok=True)

# Load
stop_times = pd.read_csv(GTFS + "stop_times.txt",
                         usecols=['trip_id','arrival_time',
                                  'departure_time','stop_id','stop_sequence'])
trips   = pd.read_csv(GTFS + "trips.txt")
routes  = pd.read_csv(GTFS + "routes.txt")
calendar = pd.read_csv(GTFS + "calendar.txt")

# -- 1. Parse times -----------------------------------------------------------
# GTFS allows times > 24:00:00 for post-midnight trips — handle this
def parse_gtfs_time(t):
    h, m, s = str(t).strip().split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

stop_times['arr_sec'] = stop_times['arrival_time'].apply(parse_gtfs_time)
stop_times['dep_sec'] = stop_times['departure_time'].apply(parse_gtfs_time)

# -- 2. Compute per-trip: duration, stop count, first departure hour ----------
trip_first = (stop_times.sort_values('stop_sequence')
              .groupby('trip_id').first()
              .rename(columns={'dep_sec':'start_sec', 'stop_id':'first_stop'}))

trip_last  = (stop_times.sort_values('stop_sequence')
              .groupby('trip_id').last()
              .rename(columns={'arr_sec':'end_sec', 'stop_id':'last_stop'}))

stop_counts = stop_times.groupby('trip_id')['stop_sequence'].max().rename('stop_count')

trip_stats = trip_first[['start_sec','first_stop']].join(
             trip_last[['end_sec','last_stop']]).join(stop_counts)
trip_stats['duration_min'] = (trip_stats['end_sec'] - trip_stats['start_sec']) / 60
trip_stats['start_hour']   = trip_stats['start_sec'] // 3600

# -- 3. Tag time period -------------------------------------------------------
def tag_period(hour):
    if 7 <= hour < 9:   return 'AM_peak'
    if 16.5 <= hour < 18.5: return 'PM_peak'
    if 9 <= hour < 16:  return 'off_peak'
    return 'other'

trip_stats['period'] = trip_stats['start_hour'].apply(tag_period)

# -- 4. Join route info -------------------------------------------------------
trip_stats = trip_stats.join(
    trips.set_index('trip_id')[['route_id','service_id']])
trip_stats = trip_stats.merge(
    routes[['route_id','route_short_name','route_type']], on='route_id')

# Filter: weekday services, bus only (route_type=3), min 6 stops
weekday_svc = calendar[calendar['monday']==1]['service_id']
trip_stats = trip_stats[
    (trip_stats['service_id'].isin(weekday_svc)) &
    (trip_stats['route_type'] == 3) &
    (trip_stats['stop_count'] >= 6) &
    (trip_stats['duration_min'] > 0)
]
print(f"Weekday bus trips for analysis: {len(trip_stats):,}")

# -- 5. Route-level congestion ratio ------------------------------------------
route_period = (trip_stats[trip_stats['period'].isin(['AM_peak','PM_peak','off_peak'])]
                .groupby(['route_id','route_short_name','period'])['duration_min']
                .agg(['mean','std','count'])
                .reset_index())

# Pivot to get peak vs off-peak side by side
pivot = route_period.pivot_table(
    index=['route_id','route_short_name'],
    columns='period',
    values='mean'
).reset_index()

pivot.columns.name = None
pivot = pivot.rename(columns={
    'AM_peak':'mean_AM_min', 'PM_peak':'mean_PM_min', 'off_peak':'mean_offpeak_min'
})

# Congestion ratio: how much longer is peak vs off-peak?
pivot['AM_ratio']  = pivot['mean_AM_min']  / pivot['mean_offpeak_min']
pivot['PM_ratio']  = pivot['mean_PM_min']  / pivot['mean_offpeak_min']
pivot['max_ratio'] = pivot[['AM_ratio','PM_ratio']].max(axis=1)

# Filter: must have data in all three periods and off-peak > 15 min
# (short routes have high ratio variance due to small absolute differences)
pivot = pivot.dropna(subset=['AM_ratio','PM_ratio','mean_offpeak_min'])
pivot = pivot[pivot['mean_offpeak_min'] >= 15]

# Flag congested routes: >15% longer in peak
congested = pivot[pivot['max_ratio'] >= 1.15].sort_values('max_ratio', ascending=False)

print(f"\nRoutes with peak/off-peak ratio >= 1.15: {len(congested)}")
print("\n=== TOP 20 CONGESTION-AFFECTED ROUTES ===")
print(congested.head(20)[['route_short_name','mean_offpeak_min',
                           'mean_AM_min','mean_PM_min',
                           'AM_ratio','PM_ratio']].to_string(index=False))

# Save
trip_stats.to_csv(os.path.join(OUT, "trip_durations.csv"), index=False)
congested.to_csv(os.path.join(OUT, "congested_routes.csv"), index=False)
pivot.to_csv(os.path.join(OUT, "route_congestion_index.csv"), index=False)
print("\nSaved: output/trip_durations.csv")
print("Saved: output/congested_routes.csv")
print("Saved: output/route_congestion_index.csv")