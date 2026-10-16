"""
Polarising stops + interesting routes
Perth Public Transport Network Analysis

Definition (deliberately simple, easy to defend):
  For each weekday route, look at the SII of every ordinary bus stop it
  serves. A stop is "polarising" if its SII deviates from that route's
  mean SII by more than 1.5 route-relative standard deviations. High
  positive deviation = an unusually valuable stop for its route
  (candidate: keep, prioritise for shelter/investment). High negative
  deviation = a stop that is dragging its route down (candidate: skip
  as express, consolidate).

A route is "interesting" if it (a) contains many polarising stops or
(b) has high SD of SII along the route or (c) serves many Q1 fix-first
stops. We rank on all three and take the union of the top-N.

Outputs (all under output/):
  - polarising_stops.csv       stops with route-relative deviation and flag
  - route_scores.csv           per-route summary metrics
  - interesting_routes.csv     top-N routes ranked by each metric, union
"""

import os
import numpy as np
import pandas as pd

BASE       = os.path.dirname(os.path.abspath(__file__))
GTFS       = os.path.join(BASE, "google_transit")
OUTPUT_DIR = os.path.join(BASE, "output")

DEVIATION_THRESHOLD = 1.5
TOP_N_ROUTES        = 20


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading SII (ordinary bus stops) and GTFS route membership...")
sii = pd.read_csv(os.path.join(OUTPUT_DIR, "sii_bus_full.csv"))
sii = sii.rename(columns={"stopnumber": "stop_id"})[
    ["stop_id", "stopname", "suburb", "SII", "G", "quadrant"]]

cal   = pd.read_csv(os.path.join(GTFS, "calendar.txt"))
trips = pd.read_csv(os.path.join(GTFS, "trips.txt"),
                    usecols=["trip_id","route_id","service_id"])
st    = pd.read_csv(os.path.join(GTFS, "stop_times.txt"),
                    usecols=["trip_id","stop_id"])
routes = pd.read_csv(os.path.join(GTFS, "routes.txt"),
                     usecols=["route_id","route_short_name","route_long_name",
                              "route_type"])

weekday_svc = set(cal.loc[cal[["monday","tuesday","wednesday","thursday","friday"]]
                          .sum(axis=1) == 5, "service_id"])
weekday_trips = trips[trips["service_id"].isin(weekday_svc)]

# route ↔ stop membership on ordinary bus stops
stop_routes = (st.merge(weekday_trips[["trip_id","route_id"]], on="trip_id")
                 .drop_duplicates(subset=["stop_id","route_id"]))
stop_routes = stop_routes[stop_routes["stop_id"].isin(sii["stop_id"])]
print(f"  Route-stop pairs (weekday, ordinary stops): {len(stop_routes):,}")
print(f"  Unique routes: {stop_routes['route_id'].nunique():,}")
print(f"  Unique stops in memberships: {stop_routes['stop_id'].nunique():,}")


# ---------------------------------------------------------------------------
# Per-route: SII statistics + polarising flag
# ---------------------------------------------------------------------------
merged = stop_routes.merge(sii, on="stop_id")

# Route-level mean and SD of SII
route_stats = (merged.groupby("route_id")["SII"]
               .agg(["mean","std","count"])
               .rename(columns={"mean":"route_mean_SII",
                                "std":"route_sd_SII",
                                "count":"n_stops"}))

merged = merged.join(route_stats, on="route_id")
# Route-relative z (guard against zero SD)
merged["route_sd_SII_safe"] = merged["route_sd_SII"].replace(0, np.nan)
merged["route_rel_z"] = (merged["SII"] - merged["route_mean_SII"]) / merged["route_sd_SII_safe"]
merged["polarising"] = merged["route_rel_z"].abs() >= DEVIATION_THRESHOLD
merged["polarising_direction"] = np.where(
    merged["route_rel_z"] >= DEVIATION_THRESHOLD, "high",
    np.where(merged["route_rel_z"] <= -DEVIATION_THRESHOLD, "low", ""))

# Attach route names
merged = merged.merge(routes[["route_id","route_short_name","route_long_name"]],
                      on="route_id", how="left")

# Persist polarising_stops.csv (only polarising, sorted by absolute deviation)
polar = merged[merged["polarising"]].copy()
polar["abs_rel_z"] = polar["route_rel_z"].abs()
polar = polar.sort_values(["polarising_direction","abs_rel_z"],
                          ascending=[True, False])
polar_out = polar[["stop_id","stopname","suburb","route_id","route_short_name",
                   "route_long_name","SII","route_mean_SII","route_rel_z",
                   "polarising_direction","G","quadrant"]]
polar_out.to_csv(os.path.join(OUTPUT_DIR, "polarising_stops.csv"), index=False)
print(f"\n  Polarising stops flagged: {len(polar):,}")
print(f"    high (SII >> route mean): {(polar['polarising_direction']=='high').sum():,}")
print(f"    low  (SII << route mean): {(polar['polarising_direction']=='low').sum():,}")


# ---------------------------------------------------------------------------
# Route-level scoring
# ---------------------------------------------------------------------------
q1_per_route = (merged[merged["quadrant"]=="Q1_fix_first"]
                .groupby("route_id").size().rename("n_q1_stops"))
polar_per_route_h = (merged[merged["polarising_direction"]=="high"]
                     .groupby("route_id").size().rename("n_polarising_high"))
polar_per_route_l = (merged[merged["polarising_direction"]=="low"]
                     .groupby("route_id").size().rename("n_polarising_low"))
polar_per_route = (polar_per_route_h.reindex(route_stats.index).fillna(0)
                   + polar_per_route_l.reindex(route_stats.index).fillna(0)).rename("n_polarising")

route_scores = (route_stats
                .join(q1_per_route, how="left")
                .join(polar_per_route_h, how="left")
                .join(polar_per_route_l, how="left")
                .join(polar_per_route, how="left")
                .fillna(0))
route_scores["n_q1_stops"]        = route_scores["n_q1_stops"].astype(int)
route_scores["n_polarising_high"] = route_scores["n_polarising_high"].astype(int)
route_scores["n_polarising_low"]  = route_scores["n_polarising_low"].astype(int)
route_scores["n_polarising"]      = route_scores["n_polarising"].astype(int)

# Join route names/types
route_scores = route_scores.join(
    routes.set_index("route_id")[["route_short_name","route_long_name","route_type"]],
    how="left")

# Only routes with meaningful sample size for SD (>= 5 stops)
route_scores = route_scores[route_scores["n_stops"] >= 5].copy()

route_scores.to_csv(os.path.join(OUTPUT_DIR, "route_scores.csv"))
print(f"\n  Route scores computed for {len(route_scores):,} routes (>=5 stops).")


# ---------------------------------------------------------------------------
# Interesting routes: union of top-N by three metrics
# ---------------------------------------------------------------------------
top_by_polar = route_scores.nlargest(TOP_N_ROUTES, "n_polarising")
top_by_sd    = route_scores.nlargest(TOP_N_ROUTES, "route_sd_SII")
top_by_q1    = route_scores.nlargest(TOP_N_ROUTES, "n_q1_stops")

top_by_polar["reason"] = "many_polarising_stops"
top_by_sd["reason"]    = "high_sii_variance"
top_by_q1["reason"]    = "many_q1_stops"

interesting = pd.concat([top_by_polar, top_by_sd, top_by_q1]).reset_index()
# Collapse duplicates, join reasons
interesting = (interesting.groupby("route_id")
               .agg({"route_short_name":"first",
                     "route_long_name":"first",
                     "route_type":"first",
                     "n_stops":"first",
                     "route_mean_SII":"first",
                     "route_sd_SII":"first",
                     "n_polarising_high":"first",
                     "n_polarising_low":"first",
                     "n_polarising":"first",
                     "n_q1_stops":"first",
                     "reason": lambda s: "+".join(sorted(set(s)))})
               .reset_index())
interesting = interesting.sort_values(
    ["n_q1_stops","n_polarising","route_sd_SII"], ascending=False)
interesting.to_csv(os.path.join(OUTPUT_DIR, "interesting_routes.csv"), index=False)

print(f"\n  Interesting routes (union of top-{TOP_N_ROUTES} on 3 metrics): "
      f"{len(interesting):,}")
print("\n=== TOP INTERESTING ROUTES ===")
print(interesting.head(25)[
    ["route_short_name","route_long_name","n_stops",
     "route_mean_SII","route_sd_SII",
     "n_polarising_high","n_polarising_low","n_q1_stops","reason"]
].round(3).to_string(index=False))
