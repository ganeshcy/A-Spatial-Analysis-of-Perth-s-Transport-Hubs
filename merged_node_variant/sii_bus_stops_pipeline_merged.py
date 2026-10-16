"""
SII Pipeline v2 — Ordinary bus stops only
Perth Public Transport Network Analysis

Rationale for the filter (from the researcher):
  Major bus interchanges and rail platforms dominate any centrality-based
  index because they are, by construction, the network's crossings. They
  are also fixed infrastructure that cannot realistically be relocated
  or resized. The interesting policy question is which ORDINARY roadside
  bus stops punch above their weight — they can be dynamically placed
  and may not always meet initial standards. We therefore restrict the
  SII universe and re-normalise WITHIN that subset.

Methodology note:
  Interchange nodes REMAIN in the underlying graph and GTFS timetable —
  trips still route through them, transfer edges still exist, centrality
  values from script4.py still reflect the real network topology. What
  changes is (1) the set of stops we score, and (2) the mean/std used
  for z-scoring, both of which are computed from ordinary bus stops only.

Outputs (all under output/):
  - sii_bus_full.csv             all ordinary bus stops with components, SII, G, quadrant
  - sii_bus_collinearity.csv     Pearson matrix on the filtered subset
  - sii_bus_validation.csv       per-period agreement with the PTA scheduled-trip
                                 fields. NOT a validation against patronage: those
                                 fields are trip counts, not boardings, and this is
                                 a GTFS-vs-shapefile consistency check only.
                                 See AUDIT_CHECKLIST.md item 8.
  - sii_bus_top.csv              top 50 ordinary bus stops
  - sii_bus_collinearity.png     heatmap
  - sii_bus_validation.png       scatter panel
  - sii_bus_g_matrix.png         quadrant chart
"""

import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE       = os.path.dirname(os.path.abspath(__file__))
PARENT     = os.path.dirname(BASE)
GTFS       = os.path.join(PARENT, "google_transit")
SHP_PATH   = os.path.join(PARENT, "Spatial_Data", "Stops", "Stops.shp")
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- merged-node variant -----------------------------------------------------
_nm = pd.read_csv(os.path.join(OUTPUT_DIR, "node_map.csv"))
NODE_MAP = dict(zip(_nm.stop_id, _nm.node_id))
def to_node(series):
    return series.map(NODE_MAP).fillna(series).astype(int)

# Whole-network scope. The ordinary-stop filter exists in the baseline to stop
# multi-stand interchanges dominating the standardisation. Merging removes the
# reason for it: an interchange is now one node, comparable in kind to a stop.
WHOLE_NETWORK = True

AM_PEAK  = (7, 9)
PM_PEAK  = (16, 18)
OFF_PEAK = (9, 16)
# Everything the three windows above leave out: the pre-07:00 shoulder and the
# post-18:00 evening. Together these carry 195,914 departures, 25.3% of all
# weekday service, which earlier versions of this pipeline simply discarded.
# They are NOT folded into F_off: 00:00-07:00 and 18:00-24:00 are not the
# inter-peak, and folding them in would leave F_off spanning 17 hours against
# two for each peak. Carried as a fourth frequency term instead, which costs
# nothing because the block is collapsed by CRITIC before it reaches the index.
REM_PERIODS = ((0, 7), (18, 24))

# Filter: which stops count as "ordinary bus stops"
INTERCHANGE_PATTERN = re.compile(
    r"\b(Stn|Busport|Bus Stn|Platform|Interchange|Terminal)\b",
    re.IGNORECASE,
)
EXCLUDED_STOPTYPES = {"Rail", "School"}
RAIL_STOPID_THRESHOLD = 99000

# Same weight vector as v1 for direct comparability
WEIGHTS = {
    "F_AM_z":  0.10,
    "F_PM_z":  0.10,
    "F_off_z": 0.05,
    "R_z":     0.20,
    "T_z":     0.20,
    "C_z":     0.35,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

Z_CLIP = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_gtfs_seconds(s):
    parts = s.astype(str).str.split(":", expand=True)
    return (parts[0].astype(int) * 3600
            + parts[1].astype(int) * 60
            + parts[2].astype(int))


# Log-standardisation. Merging concentrates the heterogeneity it was meant to
# resolve: 112 facility nodes carry a median 510 scheduled weekday trips against 35
# elsewhere, and reach 37 routes against a single-stand median of 1. Linear
# z-scoring then pushes them into the +/-3 clip, so the index preserves their
# RANK (Spearman 0.799 on those nodes) but truncates their MAGNITUDE, which is
# what collapses Pearson. Taking logs first compresses the tail so the clip
# stops binding, at the cost of making the components multiplicative in spirit.
# Must match sii_assemble.py (which asserts the two agree). Set False on
# 30 Sep 2026 so the two stages share one frame: assemble recomputes every _z
# column from the raw values with its own zc(), so this constant governs only
# the _z columns and the legacy SII written into sii_bus_full.csv — it does NOT
# feed the HRI. Leaving it True would have left this file carrying a log-frame
# legacy SII alongside plain-frame results, which is how the original mixed-frame
# CRITIC bug went unnoticed.
LOG_TRANSFORM = False

def zscore_clip(series, clip=Z_CLIP):
    x = series.astype(float)
    if LOG_TRANSFORM and (x.min() >= 0):
        # log1p keeps zeros at zero and is monotone, so ranks are untouched.
        x = np.log1p(x)
    mean, std = x.mean(), x.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x - mean) / std).clip(-clip, clip)


def is_ordinary_bus_stop(stopname, stoptype, stopnumber):
    """Filter retained for comparability; disabled when WHOLE_NETWORK is set."""
    if WHOLE_NETWORK:
        return True
    if stopnumber >= RAIL_STOPID_THRESHOLD:
        return False
    if stoptype in EXCLUDED_STOPTYPES:
        return False
    if pd.isna(stopname):
        return False
    if INTERCHANGE_PATTERN.search(str(stopname)):
        return False
    return True


# ---------------------------------------------------------------------------
# 1. Load centrality output and identify the ordinary bus stop universe
# ---------------------------------------------------------------------------
print("=" * 72)
print("STEP 1: Filter to ordinary bus stops")
print("=" * 72)

centrality = pd.read_csv(os.path.join(OUTPUT_DIR, "centrality_results.csv"))
centrality = centrality.drop_duplicates(subset="stopnumber").set_index("stopnumber")
print(f"  Centrality rows: {len(centrality):,}")

# Apply the filter based on stopname/stoptype/stopnumber
mask = [
    is_ordinary_bus_stop(centrality.at[n, "stopname"],
                          centrality.at[n, "stoptype"],
                          n)
    for n in centrality.index
]
ordinary = centrality[mask].copy()
print(f"  Ordinary bus stops (filter applied): {len(ordinary):,}")
print(f"  Excluded: {len(centrality) - len(ordinary):,}")
print(f"  Excluded stoptype breakdown:")
excluded = centrality[[not m for m in mask]]
print(excluded["stoptype"].value_counts().to_string())

ordinary_ids = set(ordinary.index)


# ---------------------------------------------------------------------------
# 2. GTFS load + weekday service filter
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 2: Weekday service, F_AM/F_PM/F_off, R  (measured on ordinary stops)")
print("=" * 72)

cal        = pd.read_csv(os.path.join(GTFS, "calendar.txt"))
trips      = pd.read_csv(os.path.join(GTFS, "trips.txt"),
                         usecols=["trip_id", "route_id", "service_id"])
stop_times = pd.read_csv(os.path.join(GTFS, "stop_times.txt"),
                         usecols=["trip_id", "departure_time", "stop_id",
                                  "stop_sequence"])
stop_times["stop_id"] = to_node(stop_times["stop_id"])

# Any service operating on at least one weekday. The former "== 5" test kept
# only services running all five days, silently dropping 122 services / 480
# trips (0.9%) — Friday-only and school-day patterns vanished entirely.
_wk = cal[["monday","tuesday","wednesday","thursday","friday"]].sum(axis=1)
weekday_svc = set(cal.loc[_wk >= 1, "service_id"])
_days = dict(zip(cal["service_id"], _wk))
weekday_trips = trips[trips["service_id"].isin(weekday_svc)].copy()
weekday_st = stop_times[stop_times["trip_id"].isin(weekday_trips["trip_id"])].copy()
# A trip running only on Fridays contributes 1/5 of a weekday-average departure;
# one running Mon-Fri contributes a full one. Without this, part-week services
# would be over-counted relative to the five-day ones they sit alongside.
_svc = dict(zip(weekday_trips["trip_id"], weekday_trips["service_id"]))
weekday_st["day_w"] = (weekday_st["trip_id"].map(_svc).map(_days) / 5.0).fillna(1.0)
weekday_st["dep_sec"]  = parse_gtfs_seconds(weekday_st["departure_time"])
weekday_st["dep_hour"] = (weekday_st["dep_sec"] // 3600) % 24

# Restrict measurement to ordinary bus stops
weekday_st_ord = weekday_st[weekday_st["stop_id"].isin(ordinary_ids)].copy()
print(f"  Weekday stop_times at ordinary stops: {len(weekday_st_ord):,}"
      f"  (of {len(weekday_st):,} total weekday rows)")

def period_count(df, lo, hi):
    """Departures per weekday in [lo, hi).

    NOT divided by 5. calendar.txt uses a single service_id flagged for all five
    weekdays (510 of them), so a Mon-Fri trip appears ONCE in trips.txt and once
    in stop_times: the raw row count is already a per-weekday figure. Verified —
    the busiest stop has 273 AM rows across 273 DISTINCT trip_ids. The former
    .div(5.0) understated every frequency fivefold. It left the index unchanged
    (a constant divisor cancels in a z-score) but corrupted every reported raw
    value and every comparison against an absolute headway standard.
    """
    sub = df[(df["dep_hour"] >= lo) & (df["dep_hour"] < hi)]
    if "day_w" in sub.columns:
        return sub.groupby("stop_id")["day_w"].sum()
    return sub.groupby("stop_id").size().astype(float)

F_AM  = period_count(weekday_st_ord, *AM_PEAK).rename("F_AM")
F_PM  = period_count(weekday_st_ord, *PM_PEAK).rename("F_PM")
F_off = period_count(weekday_st_ord, *OFF_PEAK).rename("F_off")
F_rem = sum(period_count(weekday_st_ord, lo, hi) for lo, hi in REM_PERIODS)
F_rem = F_rem.rename("F_rem")

weekday_st_ord_routes = weekday_st_ord.merge(
    weekday_trips[["trip_id", "route_id"]], on="trip_id")
R = weekday_st_ord_routes.groupby("stop_id")["route_id"].nunique().rename("R")

print(f"  Ordinary stops with weekday service: F_AM={len(F_AM):,}, "
      f"F_PM={len(F_PM):,}, F_off={len(F_off):,}, R={len(R):,}")
print(f"  Max trips/day (was 111 with interchanges): AM {F_AM.max():.1f}, "
      f"PM {F_PM.max():.1f}, off {F_off.max():.1f}")
print(f"  Max routes/stop (was 30):  {R.max()}")


# ---------------------------------------------------------------------------
# 3. T — transfer potential (routes reachable via a transfer, ordinary stops only)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 3: Transfer potential (T)")
print("=" * 72)

transfers = pd.read_csv(os.path.join(GTFS, "transfers.txt"), skipinitialspace=True)
transfers["from_stop_id"] = to_node(transfers["from_stop_id"])
transfers["to_stop_id"]   = to_node(transfers["to_stop_id"])
transfers = transfers[transfers["from_stop_id"] != transfers["to_stop_id"]]

# Bidirectional
transfers_bi = pd.concat([
    transfers[["from_stop_id", "to_stop_id"]],
    transfers.rename(columns={"from_stop_id": "to_stop_id",
                              "to_stop_id":   "from_stop_id"})
             [["from_stop_id", "to_stop_id"]],
], ignore_index=True).drop_duplicates()

# Routes at each stop (across the FULL weekday graph — including interchanges
# because that's still the physical reality of what you can transfer to)
stop_routes = (weekday_st.merge(weekday_trips[["trip_id","route_id"]], on="trip_id")
               .groupby("stop_id")["route_id"].agg(set).rename("routes_at_stop"))

transfers_bi = transfers_bi.join(stop_routes, on="to_stop_id")
transfers_bi["routes_at_stop"] = transfers_bi["routes_at_stop"].apply(
    lambda x: x if isinstance(x, set) else set())

transfer_routes = (transfers_bi.groupby("from_stop_id")["routes_at_stop"]
                   .apply(lambda s: set().union(*s)))

def transfer_gain(stop_id):
    own = stop_routes.get(stop_id, set())
    via = transfer_routes.get(stop_id, set())
    return len(via - own)

# ---------------------------------------------------------------------------
# T: spatial transfer potential, replacing the transfers.txt definition.
#
# transfers.txt cannot support this component. 95.4% of its 5,086 pairs have both
# ends under one parent_station and its median min_transfer_time is 179 s: it
# documents walking between stands INSIDE an interchange, not opportunity to
# reach a different route. Under the merged representation those walks become
# intra-node and vanish, leaving T non-zero at 0.17% of nodes and a 0.20 weight
# doing nothing. It is a data limitation, not an implementation fault, so the
# component is redefined rather than repaired:
#
#   T = | routes at stops within TRANSFER_WALK_M metres' walk - routes here |
#
# computed by sii_transfer_potential_merged.py, which needs the walk catchments.
# The chain therefore bootstraps: run this once (falling back to transfers.txt),
# then the catchment stage, then the transfer stage, then re-run this.
# ---------------------------------------------------------------------------
# Behavioural: a mid-journey walk, not an access walk. Raised 200 -> 500 after the
# threshold sweep: at 200 m Elizabeth Quay Bus Stn scored T = 0 despite standing
# ~150 m from Elizabeth Quay rail station, a transfer made routinely in practice.
# It first registers at 300 m (T = 4) and reaches its full set at 500 m (T = 7);
# Perth Busport likewise saturates at 500 m (T = 19). Both are unchanged at 600 m,
# so 500 m is where the CBD interchanges have found every route reachable on foot
# and nothing further is bought by widening. corr(T, R) is flat at 0.14-0.18 across
# the whole sweep, so the added coverage is not redundancy with the route count.
TRANSFER_WALK_M = 500

_tp = os.path.join(OUTPUT_DIR, "transfer_potential_spatial.csv")
_col = f"T_spatial_{TRANSFER_WALK_M}"
if os.path.exists(_tp) and _col in pd.read_csv(_tp, nrows=0).columns:
    _t = pd.read_csv(_tp, usecols=["stopnumber", _col])
    T_ord = _t.set_index("stopnumber")[_col].reindex(sorted(ordinary_ids)).fillna(0.0)
    T_ord.name = "T"
    print(f"  T = spatial transfer potential @ {TRANSFER_WALK_M} m "
          f"(non-zero at {(T_ord > 0).mean() * 100:.1f}% of nodes)")
else:
    print("  WARNING: transfer_potential_spatial.csv not found — falling back to")
    print("           the transfers.txt definition, which is non-zero at ~0.2% of")
    print("           nodes. Re-run this stage after sii_transfer_potential_merged.py.")
    T_ord = pd.Series({sid: transfer_gain(sid) for sid in ordinary_ids}, name="T")



# ---------------------------------------------------------------------------
# 4. C — take from centrality_results.csv (computed on the full graph in script4)
# ---------------------------------------------------------------------------
C = ordinary["influence_score"].rename("C")
print(f"\n  C loaded for {len(C):,} ordinary stops "
      f"(range {C.min():.3f}–{C.max():.3f})")


# ---------------------------------------------------------------------------
# 5. G — congestion, aggregated to stop level (ordinary stops only)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 4: G (congestion, ordinary stops only)")
print("=" * 72)

hotspots = pd.read_csv(os.path.join(OUTPUT_DIR, "reliable_hotspots.csv"))
hotspots_ord = hotspots[hotspots["stop_id"].isin(ordinary_ids)]
G = hotspots_ord.groupby("stop_id")["priority"].sum().rename("G")
G_max = hotspots_ord.groupby("stop_id")["max_inflation"].max().rename("G_max_inflation")
print(f"  Ordinary stops flagged as congested: {len(G):,} "
      f"(of {hotspots['stop_id'].nunique():,} in raw hotspots)")


# ---------------------------------------------------------------------------
# 6. Assemble, z-score WITHIN the ordinary subset, collinearity, SII
# ---------------------------------------------------------------------------
# centrality_results.csv predates the rename; accept either header so the
# pipeline runs without re-doing the betweenness computation. script4_merged.py
# emits the new name, so this shim retires itself on the next centrality run.
ordinary = ordinary.rename(columns={"weekday_total": "sched_trips_weekday",
                                    "weekend_total": "sched_trips_weekend"})
for _c in ("sched_trips_weekday", "sched_trips_weekend"):
    if _c not in ordinary.columns:
        ordinary[_c] = 0.0

print("\n" + "=" * 72)
print("STEP 5: Assemble, z-score within ordinary subset, SII")
print("=" * 72)

feat = pd.DataFrame(index=pd.Index(sorted(ordinary_ids), name="stopnumber"))
feat["F_AM"]  = F_AM.reindex(feat.index).fillna(0.0)
feat["F_PM"]  = F_PM.reindex(feat.index).fillna(0.0)
feat["F_off"] = F_off.reindex(feat.index).fillna(0.0)
feat["F_rem"] = F_rem.reindex(feat.index).fillna(0.0)
feat["R"]     = R.reindex(feat.index).fillna(0).astype(int)
feat["T"]     = T_ord.reindex(feat.index).fillna(0).astype(int)
feat["C"]     = C.reindex(feat.index).fillna(0.0)

for col in ["F_AM", "F_PM", "F_off", "F_rem", "R", "T", "C"]:
    feat[f"{col}_z"] = zscore_clip(feat[col])

collin_cols = ["F_AM", "F_PM", "F_off", "F_rem", "R", "T", "C"]
collin = feat[collin_cols].corr(method="pearson")
collin.to_csv(os.path.join(OUTPUT_DIR, "sii_bus_collinearity.csv"))
print(f"\n  Collinearity matrix (ordinary bus stops, n={len(feat):,}):")
print(collin.round(3).to_string())

# SII retains the legacy three-term frequency block and its inherited weights,
# so that it continues to reproduce the values reported in Chapter 4 and the
# comparison in sii_critic_weights.py still checks out. F_rem enters the index
# through S_hat in sii_assemble.py, which is CRITIC-weighted over the collapsed
# four-term block. Do not add F_rem here without regenerating Chapter 4.
feat["SII"] = sum(WEIGHTS[c] * feat[c] for c in WEIGHTS)
print(f"\n  SII range: [{feat['SII'].min():.3f}, {feat['SII'].max():.3f}]")
print(f"  SII mean:  {feat['SII'].mean():.3f}, std: {feat['SII'].std():.3f}")

# Join G and metadata
feat = (feat.join(G, how="left").join(G_max, how="left")
        .join(ordinary[["lon","lat","stopname","stoptype","suburb",
                        "sched_trips_weekday","sched_trips_weekend"]], how="left"))
feat["G"] = feat["G"].fillna(0.0)
feat["G_max_inflation"] = feat["G_max_inflation"].fillna(1.0)


# ---------------------------------------------------------------------------
# 7. Validation against boarding counts
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 6: Validation vs boarding counts (ordinary bus stops)")
print("=" * 72)

shp = gpd.read_file(SHP_PATH)
shp = shp[shp["STATUS"] == "Active"].copy()
shp["sched_trips_weekday"] = (shp["WeekPAM"] + shp["WeekPPM"]
                        + shp["WeekOFF"] + shp["WeekREM"])
shp["sched_trips_weekend"] = (shp["SatPAM"] + shp["SatPPM"]
                        + shp["SatOFF"] + shp["SatREM"]
                        + shp["SunPHPAM"] + shp["SunPHPPM"]
                        + shp["SunPHOFF"] + shp["SunPHREM"])

# These are COUNTS OF SCHEDULED TRIPS, not boardings. The PTA data dictionary
# defines WeekPAM et al. as "the number of trips scheduled" in each window; the
# stop layer carries no patronage field at all. They correlate with departures
# counted from stop_times.txt at 0.988 Pearson because they are the same
# timetable recorded twice. See AUDIT_CHECKLIST.md item 8. A merged node's
# scheduled service is the sum across its stands.
shp["node_id"] = to_node(shp["STOPNUMBER"])
val = (shp.groupby("node_id")[["WeekPAM","WeekPPM","WeekOFF","WeekREM",
                               "sched_trips_weekday","sched_trips_weekend"]].sum())
val.index.name = "STOPNUMBER"
val = val.join(feat[["SII"]], how="inner")

val_results = []
for target in ["WeekPAM","WeekPPM","WeekOFF","WeekREM",
               "sched_trips_weekday","sched_trips_weekend"]:
    s = val.dropna(subset=[target,"SII"])
    val_results.append({
        "target":       target,
        "n":            len(s),
        "pearson_r":    round(s["SII"].corr(s[target]), 4),
        "spearman_rho": round(s["SII"].corr(s[target], method="spearman"), 4),
    })
val_df = pd.DataFrame(val_results)
val_df.to_csv(os.path.join(OUTPUT_DIR, "sii_bus_validation.csv"), index=False)
print(val_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 8. SII × G quadrant
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 7: Quadrant assignment (ordinary bus stops)")
print("=" * 72)

sii_thresh = feat["SII"].median()
g_thresh   = feat.loc[feat["G"] > 0, "G"].median() if (feat["G"] > 0).any() else 0.0

def quadrant(row):
    hi_sii = row["SII"] >= sii_thresh
    hi_g   = row["G"]   >= g_thresh and row["G"] > 0
    if hi_sii and hi_g:      return "Q1_fix_first"
    if hi_sii and not hi_g:  return "Q2_protect"
    if not hi_sii and hi_g:  return "Q3_review_low_value"
    return "Q4_low_priority"

feat["quadrant"] = feat.apply(quadrant, axis=1)
print(f"  SII median threshold: {sii_thresh:.3f}")
print(f"  G median (of nonzero): {g_thresh:.3f}")
print(feat["quadrant"].value_counts().to_string())


# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 8: Save outputs")
print("=" * 72)

feat.reset_index().to_csv(os.path.join(OUTPUT_DIR, "sii_bus_full.csv"), index=False)
top = (feat.sort_values("SII", ascending=False).head(50)
       [["stopname","stoptype","suburb",
         "F_AM","F_PM","F_off","F_rem","R","T","C",
         "SII","G","sched_trips_weekday","quadrant"]])
top.to_csv(os.path.join(OUTPUT_DIR, "sii_bus_top.csv"))

print(f"\n  Top 20 ORDINARY bus stops by SII:")
print(top.head(20).round(3).to_string())


# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collinearity heatmap
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(collin.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(collin_cols))); ax.set_yticks(range(len(collin_cols)))
    ax.set_xticklabels(collin_cols, rotation=45, ha="right")
    ax.set_yticklabels(collin_cols)
    for i in range(len(collin_cols)):
        for j in range(len(collin_cols)):
            ax.text(j, i, f"{collin.values[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(collin.values[i,j])>0.5 else "black",
                    fontsize=9)
    ax.set_title(f"SII Collinearity — ordinary bus stops (n={len(feat):,})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sii_bus_collinearity.png"), dpi=140)
    plt.close(fig)

    # Validation panel
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, tgt in zip(axes.flat, ["WeekPAM","WeekPPM","WeekOFF","WeekREM",
                                    "sched_trips_weekday","sched_trips_weekend"]):
        s = val.dropna(subset=[tgt, "SII"])
        ax.scatter(s["SII"], s[tgt], s=4, alpha=0.25)
        r = s["SII"].corr(s[tgt])
        rho = s["SII"].corr(s[tgt], method="spearman")
        ax.set_title(f"{tgt}\nr={r:.3f}, rho={rho:.3f}, n={len(s):,}")
        ax.set_xlabel("SII"); ax.set_ylabel(tgt)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sii_bus_validation.png"), dpi=140)
    plt.close(fig)

    # Quadrant chart
    fig, ax = plt.subplots(figsize=(8,7))
    sub = feat[feat["G"] > 0]
    colors = {"Q1_fix_first":"#d62728", "Q2_protect":"#2ca02c",
              "Q3_review_low_value":"#ff7f0e", "Q4_low_priority":"#7f7f7f"}
    for q, c in colors.items():
        m = sub[sub["quadrant"]==q]
        ax.scatter(m["SII"], m["G"], s=8, alpha=0.6, c=c, label=q)
    ax.axvline(sii_thresh, color="black", ls="--", lw=0.7)
    ax.axhline(g_thresh,   color="black", ls="--", lw=0.7)
    ax.set_xlabel("SII"); ax.set_ylabel("G")
    ax.set_title("SII × G  (ordinary bus stops with G>0)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sii_bus_g_matrix.png"), dpi=140)
    plt.close(fig)

    print("  Plots saved: sii_bus_collinearity.png, sii_bus_validation.png, "
          "sii_bus_g_matrix.png")
except ImportError:
    print("  matplotlib not available")

print("\n" + "=" * 72)
print("Done. Filtered SII outputs in:", OUTPUT_DIR)
print("=" * 72)
