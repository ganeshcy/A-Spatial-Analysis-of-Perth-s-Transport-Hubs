"""
Congestion exposure (G), rebuilt
Perth Public Transport Network Analysis — merged_node_variant

WHY THIS REPLACES roscript3's G

    The G carried in sii_bus_full.csv came from reliable_hotspots.csv and had
    three defects, two of them fatal.

    1. IT WAS 97.8% CENTRALITY. roscript3 sets

           priority = max_inflation * influence_score

       and influence_score IS the centrality component C. Aggregating priority
       to stop level gave a G whose rank correlation with C, across the nodes
       where it was non-zero, was 0.978. The Chapter 4 cross-tabulation of SII
       against G therefore compared an index containing C against a variable that
       was very nearly C, and its headline result — that no stop falls in the
       high-congestion / low-influence quadrant — was arithmetic, not evidence.
       Recomputed on delay alone that quadrant is populated.

    2. IT COVERED 7% OF THE NETWORK'S ROUTES. roscript3 restricts the segment
       analysis to routes already flagged in congested_routes.csv: 30 of 429.
       Everything downstream inherits that ceiling, which is why G was non-zero
       at 121 of 11,773 nodes — 1.0%. A criterion that is identically zero for
       99% of cases cannot discriminate between them.

    3. IT POOLED WEEKENDS INTO THE WEEKDAY PEAKS. roscript3 never filters on
       service_id, so Saturday and Sunday traversals entered the AM, PM and
       off-peak means. Weekend running is faster, which dilutes the off-peak
       denominator unevenly across routes.

    Plus: stop ids were never remapped to merged nodes, so 26 of 147 hotspot
    origins were raw stands that no longer exist as nodes.

WHAT G MEASURES HERE

    A segment is an ordered pair of consecutive nodes on one route. Mean
    traversal seconds are computed per period from weekday services only, and a
    segment's excess delay is the seconds its worst peak loses against the
    inter-peak mean, floored at zero — a segment that runs FASTER in the peak is
    uncongested, not negatively congested:

        excess(s) = max( max(AM_sec, PM_sec) - off_sec , 0 )

    A node's exposure is the mean excess across the segments departing it,
    weighted by peak traversals so that delay on a segment carrying sixty peak
    departures counts for more than the same delay on one carrying three, then
    shrunk toward the trip-weighted network mean in proportion to the evidence
    behind it:

        G_i = ( SUM_s n_peak(s) e(s) + k * gbar ) / ( SUM_s n_peak(s) + k )

    k is in peak traversals and set at 50, about three times the median node's
    evidence. G is read directly: G = 30 means departures from this node lose,
    on average, thirty seconds to the peak.

WHY SECONDS AND NOT A RATIO

    An earlier formulation used peak inflation, the ratio of peak to inter-peak
    traversal time. Three things argue against it.

    1. THE FEED HAS NO RESOLUTION FOR IT. 59.8% of inter-peak segment times are
       exactly sixty seconds and 70.4% are whole multiples of sixty: Transperth
       schedules to the minute. On a sixty-second segment a ratio can only take
       1.00, 1.50, 2.00 and so on, so the measure is quantised by the timetable
       — and most coarsely where segments are shortest and most numerous.

    2. A RATIO IS BLIND TO SCALE. Twenty per cent of a sixty-second hop is
       twelve seconds; twenty per cent of a ten-minute run is two minutes. A
       passenger experiences the seconds, not the proportion, and a ratio-based
       node average lets very short segments dominate.

    3. SECONDS ARE ADDITIVE ALONG A JOURNEY; RATIOS ARE NOT. Delay summed over
       the segments of a trip is a meaningful quantity. A mean of ratios is not.

    The ratio is retained as G_max_inflation, a descriptive column, because it
    remains the conventional way to state congestion on a single segment.

WHY THE FILTER IS ON THE PEAK, NOT THE INTER-PEAK

    The reliability filter requires the worst peak traversal to be at least two
    minutes. A segment too short to lose meaningful time cannot be congested,
    whatever its ratio says, and this tests that directly.

    The previous filter instead required the INTER-PEAK mean to be at least
    ninety seconds, which is a floor on the denominator rather than on the thing
    being measured. It selected badly in both directions. It admitted 420
    segments whose peak runs under two minutes, whose median inflation is 0.926
    — they run FASTER in the peak — and whose largest excess is 23 seconds; they
    contributed nothing but denominator. And it excluded 312 segments whose peak
    exceeds two minutes but whose inter-peak is under ninety seconds, with a
    median inflation of 1.625 and a median excess of 46 seconds — real delay,
    discarded. Filtering on the peak keeps the second group and drops the first.

WHAT G IS NOT
    It is not part of the Hub Reliance Index and must not be added to it. HRI
    asks how much a place is depended upon; G asks how badly the service there
    is degraded. Combining them would conflate the diagnosis with the symptom.
    G is carried alongside, for cross-tabulation and for the map.

Usage:
    python sii_congestion_merged.py
    python sii_congestion_merged.py --measure ratio     # previous formulation
    python sii_congestion_merged.py --min-peak-sec 180 --min-trips 5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
GTFS = os.path.join(PARENT, "google_transit")
OUT = os.path.join(BASE, "output")

AM_PEAK, PM_PEAK, OFF_PEAK = (7, 9), (16, 18), (9, 16)
# Winsorisation is DISABLED. It was introduced before the shrinkage above and
# the two do the same job; run together, the p99.5 cap of 0.2106 flattened the
# entire top twelve into a tie. Shrinkage is the principled control — it pulls
# the maximum from 0.702 to 0.235 by discounting thin evidence, rather than by
# truncating values that may be real. Set a percentile here to re-enable.
WINSOR_PCTL = None


def tag(h):
    if AM_PEAK[0] <= h < AM_PEAK[1]:
        return "AM"
    if PM_PEAK[0] <= h < PM_PEAK[1]:
        return "PM"
    if OFF_PEAK[0] <= h < OFF_PEAK[1]:
        return "off"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trips", type=int, default=3,
                    help="minimum traversals per period for a segment to count")
    ap.add_argument("--k", type=float, default=50.0, metavar="PEAK_TRIPS",
                    help="empirical-Bayes shrinkage strength, in peak traversals")
    ap.add_argument("--min-peak-sec", type=float, default=120.0,
                    help="a segment counts only if its worst peak traversal is at "
                         "least this long. A segment too short to lose time in "
                         "cannot be congested, whatever its ratio says.")
    ap.add_argument("--measure", choices=["seconds", "ratio"], default="seconds",
                    help="excess delay in seconds (default) or as a ratio to the "
                         "inter-peak mean")
    ap.add_argument("--min-offpeak-sec", type=float, default=0.0,
                    help="legacy denominator floor; superseded by --min-peak-sec")
    args = ap.parse_args()

    nm = pd.read_csv(os.path.join(OUT, "node_map.csv"))
    node = dict(zip(nm.stop_id, nm.node_id))

    print("=" * 74)
    print("CONGESTION EXPOSURE (G) — REBUILT")
    print("=" * 74)

    cal = pd.read_csv(os.path.join(GTFS, "calendar.txt"))
    wk = cal[["monday", "tuesday", "wednesday", "thursday", "friday"]].sum(axis=1)
    weekday_svc = set(cal.loc[wk >= 1, "service_id"])
    trips = pd.read_csv(os.path.join(GTFS, "trips.txt"),
                        usecols=["trip_id", "route_id", "service_id"])
    trips = trips[trips.service_id.isin(weekday_svc)]
    print(f"  weekday services {len(weekday_svc):,}   weekday trips {len(trips):,}")
    print(f"  routes considered: {trips.route_id.nunique():,} "
          f"(roscript3 used 30)")

    st = pd.read_csv(os.path.join(GTFS, "stop_times.txt"),
                     usecols=["trip_id", "arrival_time", "departure_time",
                              "stop_id", "stop_sequence"], low_memory=False)
    st = st[st.trip_id.isin(set(trips.trip_id))].copy()

    def secs(s):
        p = s.astype(str).str.split(":", expand=True)
        return (p[0].astype(int) * 3600 + p[1].astype(int) * 60
                + p[2].astype(int))
    st["dep"] = secs(st.departure_time)
    st["arr"] = secs(st.arrival_time)
    st["node"] = st.stop_id.map(node).fillna(st.stop_id).astype(int)

    st = st.sort_values(["trip_id", "stop_sequence"])
    st["next_node"] = st.groupby("trip_id")["node"].shift(-1)
    st["next_arr"] = st.groupby("trip_id")["arr"].shift(-1)
    st = st.dropna(subset=["next_node"])
    st["next_node"] = st["next_node"].astype(int)
    st["seg_sec"] = st["next_arr"] - st["dep"]

    # A merged facility absorbs the stand-to-stand hops inside it; those are not
    # segments and their traversal time is meaningless.
    st = st[st.node != st.next_node]
    st = st[st.seg_sec > 0]

    st["period"] = ((st.dep // 3600) % 24).map(tag)
    st = st[st.period != "other"]
    st = st.merge(trips[["trip_id", "route_id"]], on="trip_id")
    print(f"  segment traversals: {len(st):,}")

    agg = (st.groupby(["node", "next_node", "route_id", "period"])
             .agg(sec=("seg_sec", "mean"), n=("seg_sec", "size")).reset_index())
    w = agg.pivot_table(index=["node", "next_node", "route_id"],
                        columns="period", values=["sec", "n"]).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    w = w.rename(columns={"sec_AM": "AM_sec", "sec_PM": "PM_sec",
                          "sec_off": "off_sec", "n_AM": "n_AM",
                          "n_PM": "n_PM", "n_off": "n_off"})
    need = ["AM_sec", "PM_sec", "off_sec"]
    w = w.dropna(subset=need)
    print(f"  segments with all three periods: {len(w):,}")

    w["peak_sec"] = w[["AM_sec", "PM_sec"]].max(axis=1)
    ok = ((w.peak_sec >= args.min_peak_sec)
          & (w.off_sec >= args.min_offpeak_sec)
          & (w.n_AM >= args.min_trips) & (w.n_PM >= args.min_trips)
          & (w.n_off >= args.min_trips))
    seg = w[ok].copy()
    print(f"  passing the reliability filter: {len(seg):,} "
          f"(worst-peak traversal >= {args.min_peak_sec:.0f}s, "
          f">= {args.min_trips} traversals per period)")

    seg["infl_AM"] = seg.AM_sec / seg.off_sec
    seg["infl_PM"] = seg.PM_sec / seg.off_sec
    seg["infl"] = seg[["infl_AM", "infl_PM"]].max(axis=1)
    if args.measure == "seconds":
        seg["excess"] = (seg["peak_sec"] - seg["off_sec"]).clip(lower=0.0)
    else:
        seg["excess"] = (seg["infl"] - 1.0).clip(lower=0.0)
    seg["worst_period"] = np.where(seg.infl_AM >= seg.infl_PM, "AM", "PM")
    seg["n_peak"] = seg.n_AM + seg.n_PM

    print(f"\n  excess delay ({args.measure}): median {seg.excess.median():.3f}  "
          f"p90 {seg.excess.quantile(.9):.3f}  max {seg.excess.max():.3f}")
    print(f"  inflation: median {seg.infl.median():.3f}  "
          f"p90 {seg.infl.quantile(.9):.3f}  max {seg.infl.max():.3f}")
    print(f"  segments actually inflated (>1.00): {int((seg.infl > 1).sum()):,} "
          f"({(seg.infl > 1).mean() * 100:.1f}%)")
    print(f"  segments above 1.15: {int((seg.infl > 1.15).sum()):,}")

    # ------------------------------------------------------ small-sample shrinkage
    # The unshrunk weighted mean has the Welshpool problem that empirical-Bayes
    # shrinkage was introduced to fix for V. The median node contributes ONE
    # segment and 17 peak traversals, and the fifteen highest-G nodes had a
    # BELOW-average evidence base (median 13 peak trips) — they ranked top not
    # because they are the most congested but because a thin sample can land
    # anywhere. G is therefore shrunk toward the trip-weighted network mean in
    # proportion to how little evidence supports it:
    #
    #     G_i = ( SUM n_peak*excess + k*gbar ) / ( SUM n_peak + k )
    #
    # k is in units of peak traversals: at k the prior and the local data carry
    # equal weight. Swept below, not asserted.
    seg["wx"] = seg.n_peak * seg.excess
    acc = seg.groupby("node").agg(wx=("wx", "sum"), n_peak=("n_peak", "sum"))
    gbar = float(seg.wx.sum() / seg.n_peak.sum())
    unit = "s" if args.measure == "seconds" else ""
    print(f"\n  trip-weighted network mean excess (shrinkage prior) {gbar:.4f}{unit}")
    print("    {:>6}{:>12}{:>14}{:>26}".format(
        "k", "max G", "median G>0", "peak trips of top 5 (med)"))
    for k in (0, 20, 50, 100, 200):
        gk = (acc.wx + k * gbar) / (acc.n_peak + k)
        print("    {:>6}{:>12.3f}{:>14.3f}{:>26.0f}".format(
            k, gk.max(), gk[gk > 0].median(),
            acc.loc[gk.nlargest(5).index, "n_peak"].median()))
    g = ((acc.wx + args.k * gbar) / (acc.n_peak + args.k)).rename("G")
    print(f"    adopted k = {args.k:g}  (network median evidence is "
          f"{acc.n_peak.median():.0f} peak traversals)")

    if WINSOR_PCTL is not None and (g > 0).any():
        cap = float(np.percentile(g[g > 0], WINSOR_PCTL))
        n_cap = int((g > cap).sum())
        g = g.clip(upper=cap)
        print(f"  winsorised at p{WINSOR_PCTL}: {cap:.4f} ({n_cap} nodes capped)")
    else:
        print("  winsorisation disabled; shrinkage controls the tail")

    extra = (seg.groupby("node")
                .agg(G_max_inflation=("infl", "max"),
                     G_segments=("infl", "size"),
                     G_worst_period=("worst_period",
                                     lambda s: s.mode().iat[0] if len(s) else ""))
             )
    out = pd.concat([g, extra], axis=1).reset_index().rename(
        columns={"node": "stopnumber"})
    out.to_csv(os.path.join(OUT, "congestion_exposure.csv"), index=False)
    seg.to_csv(os.path.join(OUT, "congestion_segments.csv"), index=False)

    # ------------------------------------------------------------- reporting
    master = os.path.join(OUT, "node_master.csv")
    if os.path.exists(master):
        d = pd.read_csv(master, low_memory=False)
        j = d.merge(out, on="stopnumber", how="left", suffixes=("_old", ""))
        j["G"] = j["G"].fillna(0.0)
        n = len(j)
        print(f"\n  nodes with G > 0: {int((j.G > 0).sum()):,} of {n:,} "
              f"({(j.G > 0).mean() * 100:.1f}%)   was 1.0%")
        print(f"  among those: median G {j.loc[j.G > 0, 'G'].median():.2f}{unit}  "
              f"max {j.G.max():.2f}{unit}")

        print("\n  INDEPENDENCE FROM THE INDEX  (the point of the rebuild)")
        print("    {:<14}{:>12}{:>12}".format("vs", "Pearson", "Spearman"))
        for c in ["C", "S_plus", "R", "T", "A", "HRI"]:
            if c in j.columns:
                print("    {:<14}{:>12.3f}{:>12.3f}".format(
                    c, j.G.corr(j[c]), j.G.corr(j[c], method="spearman")))
        print("    (old G vs C on its support was 0.978 Spearman)")

        print("\n  TOP 12 BY CONGESTION EXPOSURE")
        print("    {:<38}{:<14}{:>8}{:>8}{:>7}".format(
            "stop", "suburb", "G", "maxinfl", "segs"))
        j = j[j["stopname"].notna()]
        for _, r in j.nlargest(12, "G").iterrows():
            print("    {:<38}{:<14}{:>8.2f}{:>8.3f}{:>7.0f}".format(
                str(r.stopname)[:37], str(r.suburb)[:13], r.G,
                r.G_max_inflation, r.G_segments))
    print(f"\n  wrote congestion_exposure.csv ({len(out):,} nodes) "
          f"and congestion_segments.csv ({len(seg):,} segments)")


if __name__ == "__main__":
    main()
